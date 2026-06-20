# TaylorAI Studio

Um "clone do LM Studio" enxuto, focado em **extrair o máximo de CPUs e iGPUs**
rodando modelos **GGUF** via [llama.cpp](https://github.com/ggml-org/llama.cpp).
O diferencial não é a interface — é o **auto-tuner**: ele lê o seu hardware e o
metadado do modelo e escolhe os flags do `llama-server` que rendem mais nesta
máquina específica.

Afinado para o notebook-alvo: **Lenovo IdeaPad 3 (82MF0003BR)** com
**Ryzen 5 5500U** (Zen 2, 6c/12t, AVX2, sem AVX-512), **iGPU Radeon Vega 7**
(Vulkan) e **20 GB DDR4-2667 assimétrica** (16+4).

## Arquitetura

```
┌─────────────────────────────────────────────┐
│  Janela Tauri (WebView2, ~tens de MB de RAM) │
│  ┌─────────────┐   ┌──────────────────────┐  │
│  │  Frontend   │   │  Backend Rust         │  │
│  │  (TS puro)  │←→ │  • detecção de HW     │  │
│  │  chat/UI    │   │  • parser GGUF        │  │
│  └──────┬──────┘   │  • AUTO-TUNER         │  │
│         │          │  • gerência de proc.  │  │
│         │          └──────────┬────────────┘  │
└─────────┼─────────────────────┼───────────────┘
          │ HTTP (OpenAI API)   │ spawn
          ▼                     ▼
   ┌──────────────────────────────────┐
   │  llama-server.exe (b9723, Vulkan)│
   │  backend CPU AVX2 + offload Vega │
   └──────────────────────────────────┘
```

O motor de inferência é o próprio `llama-server` (API compatível com OpenAI).
O app **não reescreve inferência** — ele a orquestra de forma ótima.

## O que o auto-tuner decide (e por quê, neste hardware)

| Flag | Valor no 5500U | Motivo |
|------|----------------|--------|
| `-t` (threads geração) | 6 (núcleos físicos) | Geração é *memory-bound*; usar os 12 threads SMT contende a banda e piora. |
| `-tb` (threads prompt) | 12 (lógicos) | Prompt é *compute-bound* e escala com mais threads. |
| `-ngl` (camadas na GPU) | 0 (padrão) / opcional Vulkan | A Vega compartilha a RAM do sistema: offload ajuda no prompt, nem sempre na geração. |
| `--flash-attn on` | sempre | Reduz RAM do KV cache e acelera. |
| `--mlock` | se couber folgado | Trava o modelo na RAM, evita paginação para o disco. |
| `--cache-type-k/v q8_0` | opcional | Metade da banda/RAM do KV em contextos longos. |
| `-c` (contexto) | 8192 padrão | Maior = mais RAM de KV e geração mais lenta no fim. |

A aba **Ajustes & Auto-tuner** mostra a estimativa de RAM, a linha de comando
final e a justificativa de cada escolha.

## Pré-requisitos (já resolvidos neste setup)

- Node.js + npm
- Rust (toolchain MSVC) — usa o linker do Visual Studio
- WebView2 Runtime (vem no Windows 11)
- Binários do llama.cpp em `src-tauri/binaries/` (build **win-vulkan-x64**)

## Rodar em desenvolvimento

```powershell
npm install
npm run tauri dev
```

## Gerar o executável

```powershell
npm run tauri build
# saída em: <target>/release/bundle/
```

> O `target/` do Rust é redirecionado para fora do OneDrive via
> `src-tauri/.cargo/config.toml` (evita sincronizar GBs de artefatos).

## Dicas de desempenho para o 5500U

1. **Na tomada + modo "Melhor desempenho"** — o boost cai muito na bateria.
2. **RAM assimétrica é o teto físico**: parte dos seus 20 GB roda em
   single-channel (16+4). É o maior limitador de tok/s. Dois pentes iguais
   (ex.: 2×8 ou 2×16) dariam dual-channel pleno e mais velocidade.
3. **Q4_K_M** costuma ser o melhor custo-benefício de velocidade/qualidade;
   Q6/Q8 dão mais qualidade ao custo de banda.
4. Para contexto longo, ligue o **KV q8_0**.
5. Teste o **offload Vulkan** com o botão de benchmark e compare com `ngl=0`.

## Modelos detectados na máquina

Os GGUF ficam em `D:\LocalAIModels\.lmstudio\hub\models\` (layout do LM Studio,
lido automaticamente). Modelos com arquivo `mmproj-*` são multimodais (visão) e
o app passa `--mmproj` quando você os carrega.
