// Auto-tuner: dado o hardware + o modelo GGUF, calcula os flags do
// llama-server que extraem o maximo de uma CPU/iGPU (foco: Ryzen 5 5500U,
// Zen 2, AVX2, sem AVX-512, banda de memoria DDR4-2667 como gargalo).

use crate::gguf::ModelInfo;
use crate::hardware::HardwareInfo;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub model_path: String,
    pub model_name: String,
    pub ctx_size: u32,
    pub threads: usize,
    pub threads_batch: usize,
    pub batch: u32,
    pub ubatch: u32,
    pub n_gpu_layers: u32,
    pub mlock: bool,
    pub no_mmap: bool,
    /// "on" | "off" | "auto"
    pub flash_attn: String,
    /// "f16" | "q8_0" | "q4_0"
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub mmproj: Option<String>,
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TuneOverrides {
    pub ctx_size: Option<u32>,
    pub gpu_offload: Option<bool>,
    pub n_gpu_layers: Option<u32>,
    pub kv_quant: Option<bool>,
    pub port: Option<u16>,
    /// Carregar o encoder de visao (mmproj). Desligado por padrao.
    pub use_mmproj: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub config: LlamaConfig,
    pub rationale: Vec<String>,
    pub warnings: Vec<String>,
    pub est_ram_gb: f64,
    pub fits_in_ram: bool,
    pub max_gpu_layers: u32,
    pub gpu_recommended: bool,
}

// Estima a RAM do KV cache em GB.
// bytes = 2 (K e V) * n_layers * ctx * kv_embd * bytes_por_elem
// kv_embd ja considera GQA (n_kv_heads * head_dim), bem menor que embedding.
fn estimate_kv_gb(ctx: u32, block_count: Option<u32>, kv_embd: u32, bytes_per_elem: f64) -> f64 {
    let layers = block_count.unwrap_or(32) as f64;
    let embd = kv_embd as f64;
    let bytes = 2.0 * layers * (ctx as f64) * embd * bytes_per_elem;
    bytes / 1e9
}

// Dimensao efetiva do KV considerando GQA: embedding * (head_count_kv / head_count).
fn kv_embd_of(model: &ModelInfo) -> u32 {
    let embd = model.embedding_length.unwrap_or(4096);
    match (model.head_count, model.head_count_kv) {
        (Some(h), Some(hkv)) if h > 0 => {
            ((embd as u64 * hkv as u64) / h as u64) as u32
        }
        _ => embd,
    }
}

fn kv_bytes_per_elem(kv_quant: bool) -> (f64, &'static str) {
    if kv_quant {
        (1.0, "q8_0") // ~1 byte/elemento
    } else {
        (2.0, "f16")
    }
}

pub fn recommend(
    hw: &HardwareInfo,
    model: &ModelInfo,
    ov: &TuneOverrides,
) -> Recommendation {
    let mut rationale: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // --- Threads ---
    let threads = hw.recommended_gen_threads;
    let threads_batch = hw.recommended_batch_threads;
    rationale.push(format!(
        "Threads de geracao = {} (nucleos fisicos). Em geracao o gargalo e a banda de memoria; usar os {} threads logicos (SMT) normalmente piora por contencao.",
        threads, hw.logical_cores
    ));
    rationale.push(format!(
        "Threads de prompt (batch) = {} (todos os logicos). Processar o prompt e compute-bound e escala com mais threads.",
        threads_batch
    ));

    // --- ISA ---
    rationale.push(format!(
        "SIMD detectado: {}. O llama.cpp seleciona a DLL ggml-cpu correspondente em runtime.",
        hw.features.best_cpu_isa
    ));

    // --- Contexto ---
    let model_ctx_max = model.context_length.unwrap_or(8192);
    let ctx = ov
        .ctx_size
        .unwrap_or(8192)
        .min(model_ctx_max.max(2048));
    rationale.push(format!(
        "Contexto = {} tokens (teto do modelo: {}). Contexto maior = mais RAM de KV cache e geracao mais lenta no fim do contexto.",
        ctx, model_ctx_max
    ));

    // --- KV cache quant ---
    let kv_quant = ov.kv_quant.unwrap_or(false);
    let (bpe, kv_label) = kv_bytes_per_elem(kv_quant);
    if kv_quant {
        rationale.push(
            "KV cache em q8_0: ~metade da RAM e da banda do KV, ajudando em contextos longos nesta CPU memory-bound (custo: leve perda de qualidade).".to_string(),
        );
    } else {
        rationale.push(
            "KV cache em f16 (qualidade). Ative o KV quantizado para ganhar RAM/velocidade em contextos longos.".to_string(),
        );
    }

    // --- KV cache (RAM) ---
    let kv_gb = estimate_kv_gb(ctx, model.block_count, kv_embd_of(model), bpe);

    // --- GPU offload (Vulkan / Vega) ---
    // Medido neste hardware (Qwen 9B Q6): offload TOTAL deu ~+40% de geracao e
    // ~+23% de prompt vs CPU puro; offload PARCIAL ficou ABAIXO do CPU.
    // Conclusao: na Vega e tudo-ou-nada.
    let max_gpu_layers = model.block_count.map(|b| b + 1).unwrap_or(0);
    // Orcamento real da iGPU (detectado via Vulkan), com fallback ~metade da RAM.
    let gpu_budget_gb = hw.gpu_budget_gb;
    let fits_gpu = max_gpu_layers > 0 && (model.size_gb + kv_gb) < gpu_budget_gb * 0.92;
    let gpu_recommended = fits_gpu;
    let gpu_on = ov.gpu_offload.unwrap_or(fits_gpu);
    let n_gpu_layers = if gpu_on {
        let n = ov.n_gpu_layers.unwrap_or(max_gpu_layers).min(max_gpu_layers);
        if n >= max_gpu_layers {
            rationale.push(format!(
                "Offload Vulkan TOTAL: {} camadas na Vega. No benchmark desta maquina rendeu ~40% mais geracao e ~23% mais prompt que CPU puro (o modelo cabe no orcamento ~{:.1} GB da iGPU).",
                n, gpu_budget_gb
            ));
        } else {
            rationale.push(format!(
                "Offload Vulkan PARCIAL: {} de {} camadas.",
                n, max_gpu_layers
            ));
            warnings.push(
                "Offload PARCIAL nesta APU costuma render MENOS que CPU puro (overhead de split + banda compartilhada). Prefira TOTAL (todas as camadas) ou ngl=0.".to_string(),
            );
        }
        n
    } else {
        if fits_gpu {
            rationale.push(
                "CPU puro (ngl=0) por sua escolha. Dica: neste modelo o offload TOTAL na Vega tende a ser mais rapido.".to_string(),
            );
        } else {
            rationale.push(format!(
                "CPU puro (ngl=0): o modelo ({:.1} GB) nao cabe no orcamento da iGPU (~{:.1} GB); offload total nao e viavel aqui.",
                model.size_gb, gpu_budget_gb
            ));
        }
        0
    };

    // --- Estimativa de RAM ---
    let overhead_gb = 0.8;
    let est_ram_gb = model.size_gb + kv_gb + overhead_gb;
    let fits_in_ram = est_ram_gb < hw.total_ram_gb * 0.92;

    rationale.push(format!(
        "RAM estimada: {:.1} GB = modelo {:.1} GB + KV {} {:.1} GB + overhead {:.1} GB (total da maquina: {:.1} GB).",
        est_ram_gb, model.size_gb, kv_label, kv_gb, overhead_gb, hw.total_ram_gb
    ));
    if !fits_in_ram {
        warnings.push(format!(
            "A estimativa ({:.1} GB) esta perto/acima da RAM total ({:.1} GB). Reduza o contexto, use KV q8_0 ou uma quantizacao menor para evitar swap (que mata o desempenho).",
            est_ram_gb, hw.total_ram_gb
        ));
    }

    // --- mlock / mmap ---
    // CRITICO: mlock SO no modo CPU. Com offload o modelo vai para a memoria da
    // iGPU (GTT = mesma RAM fisica); travar a copia da CPU dobra o uso de RAM e
    // trava o carregamento (medido: timeout vs 8s sem mlock).
    let mlock = n_gpu_layers == 0 && fits_in_ram && est_ram_gb < hw.total_ram_gb * 0.72;
    if mlock {
        rationale.push(
            "--mlock ligado: trava o modelo na RAM e impede o Windows de paginar para o disco (evita engasgos). So no modo CPU, onde sobra RAM.".to_string(),
        );
    } else if n_gpu_layers > 0 {
        rationale.push(
            "--mlock desligado: com offload na iGPU, o modelo vai para o GTT (mesma RAM fisica). Travar a copia da CPU dobraria o uso de memoria e travaria o load.".to_string(),
        );
    } else {
        rationale.push(
            "--mlock desligado: a margem de RAM esta apertada para travar tudo na memoria.".to_string(),
        );
    }

    // --- Flash attention ---
    let flash_attn = "on".to_string();
    rationale.push(
        "Flash attention = on: reduz a RAM do KV cache e costuma acelerar; suportado no backend de CPU.".to_string(),
    );

    // --- Batch / ubatch ---
    let batch = 2048u32;
    let ubatch = 512u32;
    rationale.push(format!(
        "batch={} / ubatch={}: bom equilibrio de throughput de prompt sem estourar memoria nesta classe de CPU.",
        batch, ubatch
    ));

    // --- Dica de energia ---
    warnings.push(
        "Dica: rode no modo de energia 'Melhor desempenho' e na tomada — o boost do 5500U cai bastante na bateria.".to_string(),
    );
    // --- Dica de RAM assimetrica ---
    warnings.push(
        "Sua RAM e assimetrica (16+4 GB): parte roda em single-channel, limitando a banda. E o teto fisico de tok/s nesta maquina.".to_string(),
    );

    // --- Visao (mmproj) opt-in ---
    let use_mmproj = ov.use_mmproj.unwrap_or(false);
    let mmproj = if use_mmproj {
        model.mmproj_path.clone()
    } else {
        None
    };
    if model.mmproj_path.is_some() {
        if use_mmproj {
            rationale.push(
                "Visao (mmproj) LIGADA: carrega o encoder de imagens (+RAM e +tempo de load).".to_string(),
            );
        } else {
            rationale.push(
                "Visao (mmproj) desligada por padrao: carrega mais rapido e leve. Ligue se for enviar imagens.".to_string(),
            );
        }
    }

    let port = ov.port.unwrap_or(8080);

    let config = LlamaConfig {
        model_path: model.path.clone(),
        model_name: model.name.clone(),
        ctx_size: ctx,
        threads,
        threads_batch,
        batch,
        ubatch,
        n_gpu_layers,
        mlock,
        no_mmap: false,
        flash_attn,
        cache_type_k: kv_label.to_string(),
        cache_type_v: kv_label.to_string(),
        mmproj,
        host: "127.0.0.1".to_string(),
        port,
    };

    Recommendation {
        config,
        rationale,
        warnings,
        est_ram_gb,
        fits_in_ram,
        max_gpu_layers,
        gpu_recommended,
    }
}
