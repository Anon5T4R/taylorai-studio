// Deteccao de hardware: CPU, nucleos, flags SIMD e RAM.
// Usado pelo auto-tuner para escolher os flags do llama-server.

use serde::Serialize;
use sysinfo::System;

#[derive(Debug, Clone, Serialize)]
pub struct Features {
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    /// melhor variante do ggml-cpu que o seu CPU consegue usar
    pub best_cpu_isa: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareInfo {
    pub cpu_brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub features: Features,
    pub recommended_gen_threads: usize,
    pub recommended_batch_threads: usize,
    /// Memoria enderecavel pela iGPU via Vulkan (GTT/UMA), em GB.
    pub gpu_budget_gb: f64,
}

// Em ARM (Apple Silicon / ARM Linux) nao ha SIMD x86 nem CPUID; usa NEON.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_features() -> Features {
    Features {
        sse42: false,
        avx: false,
        avx2: false,
        avx512f: false,
        fma: false,
        best_cpu_isa: "ARM NEON".to_string(),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_features() -> Features {
    use raw_cpuid::CpuId;
    let cpuid = CpuId::new();
    let fi = cpuid.get_feature_info();
    let efi = cpuid.get_extended_feature_info();

    let sse42 = fi.as_ref().map(|f| f.has_sse42()).unwrap_or(false);
    let avx = fi.as_ref().map(|f| f.has_avx()).unwrap_or(false);
    let fma = fi.as_ref().map(|f| f.has_fma()).unwrap_or(false);
    let avx2 = efi.as_ref().map(|f| f.has_avx2()).unwrap_or(false);
    let avx512f = efi.as_ref().map(|f| f.has_avx512f()).unwrap_or(false);

    // qual DLL ggml-cpu-* o runtime vai conseguir usar (do mais novo ao mais antigo)
    let best_cpu_isa = if avx512f {
        "AVX-512 (ggml-cpu-skylakex/zen4)".to_string()
    } else if avx2 && fma {
        "AVX2 + FMA (ggml-cpu-haswell)".to_string()
    } else if avx {
        "AVX (ggml-cpu-sandybridge)".to_string()
    } else if sse42 {
        "SSE4.2".to_string()
    } else {
        "x64 base".to_string()
    };

    Features {
        sse42,
        avx,
        avx2,
        avx512f,
        fma,
        best_cpu_isa,
    }
}

pub fn get_hardware(gpu_budget_gb: Option<f64>) -> HardwareInfo {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let logical_cores = sys.cpus().len().max(1);
    let physical_cores = sys.physical_core_count().unwrap_or(logical_cores).max(1);

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "CPU desconhecida".to_string());

    // sysinfo 0.33 retorna bytes
    let total_ram_gb = (sys.total_memory() as f64) / 1e9;
    let available_ram_gb = (sys.available_memory() as f64) / 1e9;

    // Em inferencia de geracao (memory-bound), usar nucleos fisicos costuma
    // render mais que usar todos os threads logicos (SMT contende a banda).
    let recommended_gen_threads = physical_cores;
    // O processamento de prompt e compute-bound: aproveita os threads logicos.
    let recommended_batch_threads = logical_cores;

    // Sem deteccao real, estima ~metade da RAM (comportamento tipico de UMA/GTT no AMD APU).
    let gpu_budget_gb = gpu_budget_gb.unwrap_or(total_ram_gb * 0.5);

    HardwareInfo {
        cpu_brand,
        physical_cores,
        logical_cores,
        total_ram_gb,
        available_ram_gb,
        features: detect_features(),
        recommended_gen_threads,
        recommended_batch_threads,
        gpu_budget_gb,
    }
}
