import "./styles.css";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  streamChat,
  type ChatMessage,
  type SamplingParams,
} from "./llama";

// ---------- Tipos espelhando o backend Rust ----------
interface Features {
  sse42: boolean;
  avx: boolean;
  avx2: boolean;
  avx512f: boolean;
  fma: boolean;
  best_cpu_isa: string;
}
interface HardwareInfo {
  cpu_brand: string;
  physical_cores: number;
  logical_cores: number;
  total_ram_gb: number;
  available_ram_gb: number;
  features: Features;
  recommended_gen_threads: number;
  recommended_batch_threads: number;
  gpu_budget_gb: number;
}
interface ModelInfo {
  path: string;
  name: string;
  folder: string;
  arch: string;
  quant: string;
  size_bytes: number;
  size_gb: number;
  block_count: number | null;
  context_length: number | null;
  embedding_length: number | null;
  head_count: number | null;
  head_count_kv: number | null;
  size_label: string | null;
  has_mmproj: boolean;
  mmproj_path: string | null;
}
interface LlamaConfig {
  model_path: string;
  model_name: string;
  ctx_size: number;
  threads: number;
  threads_batch: number;
  batch: number;
  ubatch: number;
  n_gpu_layers: number;
  mlock: boolean;
  no_mmap: boolean;
  flash_attn: string;
  cache_type_k: string;
  cache_type_v: string;
  mmproj: string | null;
  host: string;
  port: number;
  draft_model: string | null;
  draft_n_gpu_layers: number;
  draft_max: number;
}
interface Recommendation {
  config: LlamaConfig;
  rationale: string[];
  warnings: string[];
  est_ram_gb: number;
  fits_in_ram: boolean;
  max_gpu_layers: number;
  gpu_recommended: boolean;
}
interface StatusReport {
  running: boolean;
  healthy: boolean;
  port: number | null;
  model_name: string | null;
}

// ---------- Estado ----------
const state = {
  hw: null as HardwareInfo | null,
  dirs: [] as string[],
  models: [] as ModelInfo[],
  selected: null as ModelInfo | null,
  rec: null as Recommendation | null,
  overrides: {
    ctx_size: 8192,
    gpu_offload: null as boolean | null, // null = auto (backend decide)
    n_gpu_layers: null as number | null, // null = total quando offload ligado
    kv_quant: false,
    port: 8080,
    use_mmproj: false, // visao desligada por padrao
    use_speculative: null as boolean | null, // auto
    draft_path: null as string | null,
    draft_size_gb: null as number | null,
  },
  draftCandidate: null as ModelInfo | null,
  ready: false,
  busy: false,
  messages: [] as ChatMessage[],
  sampling: {
    temperature: 0.7,
    top_p: 0.95,
    top_k: 40,
    min_p: 0.05,
    repeat_penalty: 1.1,
    // alto por padrao: modelos de reasoning gastam muitos tokens "pensando"
    // antes de responder; baixo demais corta a resposta antes de comecar.
    max_tokens: 2048,
  } as SamplingParams,
  systemPrompt: "", // vazio por padrao
  think: false, // opt-in: reasoning desligado por padrao (injeta /no_think)
  abort: null as AbortController | null,
};

// ---------- Helpers de DOM ----------
function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, any> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      el.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== false && v != null) {
      el.setAttribute(k, String(v));
    }
  }
  for (const c of children) {
    el.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return el;
}
const $ = <T extends HTMLElement>(sel: string) =>
  document.querySelector(sel) as T;

const DIRS_KEY = "taylorai.dirs";
const SAMPLING_KEY = "taylorai.sampling";

// ---------- Shell / layout ----------
function buildShell() {
  const app = $("#app");
  app.innerHTML = "";
  app.append(
    h("div", { class: "layout" }, [
      h("aside", { class: "sidebar" }, [
        h("div", { class: "brand" }, [
          h("div", { class: "logo" }, ["⚡"]),
          h("div", {}, [
            h("div", { class: "brand-title" }, ["TaylorAI Studio"]),
            h("div", { class: "brand-sub" }, ["GGUF na CPU/iGPU"]),
          ]),
        ]),
        h("div", { id: "hw-card", class: "card" }, ["Detectando hardware…"]),
        h("div", { class: "section-head" }, [
          h("span", {}, ["Pastas de modelos"]),
          h("button", { class: "mini", id: "add-dir" }, ["+ pasta"]),
        ]),
        h("div", { id: "dirs", class: "dirs" }, []),
        h("div", { class: "section-head" }, [
          h("span", {}, ["Modelos GGUF"]),
          h("button", { class: "mini", id: "rescan" }, ["⟳"]),
        ]),
        h("div", { id: "models", class: "models" }, ["—"]),
      ]),
      h("main", { class: "main" }, [
        h("header", { class: "topbar" }, [
          h("div", { id: "sel-model", class: "sel-model" }, [
            "Nenhum modelo selecionado",
          ]),
          h("div", { class: "topbar-right" }, [
            h("div", { id: "hud", class: "hud" }, []),
            h("div", { id: "status-pill", class: "pill off" }, ["parado"]),
            h("button", { id: "load-btn", class: "primary", disabled: true }, [
              "Carregar",
            ]),
          ]),
        ]),
        h("nav", { class: "tabs" }, [
          h("button", { class: "tab active", "data-view": "chat" }, ["Chat"]),
          h("button", { class: "tab", "data-view": "tuner" }, [
            "Ajustes & Auto-tuner",
          ]),
          h("button", { class: "tab", "data-view": "logs" }, ["Logs"]),
        ]),
        h("section", { id: "view-chat", class: "view" }, []),
        h("section", { id: "view-tuner", class: "view hidden" }, []),
        h("section", { id: "view-logs", class: "view hidden" }, [
          h("pre", { id: "logs", class: "logs" }, []),
        ]),
      ]),
    ]),
  );

  $("#add-dir").addEventListener("click", addDir);
  $("#rescan").addEventListener("click", () => scan());
  $("#load-btn").addEventListener("click", toggleServer);
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () =>
      switchView((t as HTMLElement).dataset.view!),
    ),
  );

  buildChatView();
}

function switchView(view: string) {
  document
    .querySelectorAll(".tab")
    .forEach((t) =>
      t.classList.toggle(
        "active",
        (t as HTMLElement).dataset.view === view,
      ),
    );
  for (const v of ["chat", "tuner", "logs"]) {
    $(`#view-${v}`).classList.toggle("hidden", v !== view);
  }
}

// ---------- Hardware ----------
async function loadHardware() {
  const hw = await invoke<HardwareInfo>("get_hardware");
  state.hw = hw;
  const f = hw.features;
  const simd = [
    f.avx512f && "AVX-512",
    f.avx2 && "AVX2",
    f.fma && "FMA",
    f.avx && "AVX",
    f.sse42 && "SSE4.2",
  ]
    .filter(Boolean)
    .join(" · ");
  $("#hw-card").innerHTML = "";
  $("#hw-card").append(
    h("div", { class: "hw-cpu" }, [hw.cpu_brand]),
    h("div", { class: "hw-grid" }, [
      kv("Nucleos", `${hw.physical_cores} fis / ${hw.logical_cores} log`),
      kv("RAM", `${hw.total_ram_gb.toFixed(1)} GB`),
      kv("SIMD", simd || "—"),
      kv("Backend", hw.features.best_cpu_isa.split(" (")[0]),
    ]),
    h("div", { class: "hw-threads" }, [
      `Tuner: ${hw.recommended_gen_threads} threads p/ geracao, ${hw.recommended_batch_threads} p/ prompt`,
    ]),
    h("div", { class: "hw-threads" }, [
      `iGPU Vulkan: ~${hw.gpu_budget_gb.toFixed(1)} GB endereçáveis p/ offload`,
    ]),
  );
}
function kv(k: string, v: string) {
  return h("div", { class: "kv" }, [
    h("span", { class: "k" }, [k]),
    h("span", { class: "v" }, [v]),
  ]);
}

// ---------- Diretorios & scan ----------
async function loadDirs() {
  const saved = JSON.parse(localStorage.getItem(DIRS_KEY) || "null");
  if (saved && Array.isArray(saved) && saved.length) {
    state.dirs = saved;
  } else {
    state.dirs = await invoke<string[]>("default_model_dirs");
    saveDirs();
  }
  renderDirs();
}
function saveDirs() {
  localStorage.setItem(DIRS_KEY, JSON.stringify(state.dirs));
}
function renderDirs() {
  const box = $("#dirs");
  box.innerHTML = "";
  if (!state.dirs.length) {
    box.append(h("div", { class: "muted" }, ["Nenhuma pasta. Adicione uma."]));
    return;
  }
  for (const d of state.dirs) {
    box.append(
      h("div", { class: "dir" }, [
        h("span", { class: "dir-path", title: d }, [d]),
        h(
          "button",
          {
            class: "x",
            onClick: () => {
              state.dirs = state.dirs.filter((x) => x !== d);
              saveDirs();
              renderDirs();
              scan();
            },
          },
          ["×"],
        ),
      ]),
    );
  }
}
async function addDir() {
  const picked = await invoke<string | null>("pick_folder");
  if (picked && !state.dirs.includes(picked)) {
    state.dirs.push(picked);
    saveDirs();
    renderDirs();
    scan();
  }
}
async function scan() {
  const box = $("#models");
  box.innerHTML = "Procurando…";
  const models = await invoke<ModelInfo[]>("scan_models", { dirs: state.dirs });
  state.models = models;
  box.innerHTML = "";
  if (!models.length) {
    box.append(h("div", { class: "muted" }, ["Nenhum .gguf encontrado."]));
    return;
  }
  for (const m of models) {
    const card = h(
      "div",
      {
        class: "model",
        onClick: () => selectModel(m),
      },
      [
        h("div", { class: "model-name", title: m.path }, [m.name]),
        h("div", { class: "model-meta" }, [
          tag(m.quant),
          tag(`${m.size_gb.toFixed(1)} GB`),
          tag(m.arch),
          ...(m.has_mmproj ? [tag("👁 visao")] : []),
        ]),
      ],
    );
    card.dataset.path = m.path;
    box.append(card);
  }
}
function tag(t: string) {
  return h("span", { class: "tag" }, [t]);
}

// ---------- Selecao de modelo + recomendacao ----------
async function selectModel(m: ModelInfo) {
  state.selected = m;
  state.ready = false;
  document
    .querySelectorAll(".model")
    .forEach((c) =>
      c.classList.toggle(
        "active",
        (c as HTMLElement).dataset.path === m.path,
      ),
    );
  $("#sel-model").textContent = `${m.name}  ·  ${m.quant}`;
  // ajusta limites de contexto ao modelo
  if (m.context_length) {
    state.overrides.ctx_size = Math.min(8192, m.context_length);
  }
  // reseta offload para "auto": o backend decide por modelo (cabe na iGPU?)
  state.overrides.gpu_offload = null;
  state.overrides.n_gpu_layers = null;
  state.overrides.use_mmproj = false; // visao sempre comeca desligada
  // detecta um modelo-rascunho (mesma familia, bem menor) p/ speculative
  state.draftCandidate = findDraft(m);
  state.overrides.use_speculative = null; // auto
  state.overrides.draft_path = state.draftCandidate?.path ?? null;
  state.overrides.draft_size_gb = state.draftCandidate?.size_gb ?? null;
  await refreshRecommendation();
  switchView("tuner");
}

// Acha um modelo-rascunho p/ speculative: mesma arquitetura, bem menor que o alvo.
function findDraft(target: ModelInfo): ModelInfo | null {
  const candidates = state.models.filter(
    (m) =>
      m.path !== target.path &&
      m.arch === target.arch &&
      m.size_gb < target.size_gb * 0.4 &&
      m.size_gb < 2.0,
  );
  candidates.sort((a, b) => a.size_gb - b.size_gb);
  return candidates[0] ?? null;
}

async function refreshRecommendation() {
  if (!state.selected) return;
  const rec = await invoke<Recommendation>("recommend_config", {
    model: state.selected,
    overrides: state.overrides,
  });
  state.rec = rec;
  $("#load-btn").removeAttribute("disabled");
  renderTuner();
}

// ---------- Painel do Tuner ----------
function renderTuner() {
  const v = $("#view-tuner");
  v.innerHTML = "";
  const rec = state.rec;
  if (!rec) {
    v.append(h("div", { class: "muted pad" }, ["Selecione um modelo."]));
    return;
  }
  const c = rec.config;
  const m = state.selected!;
  const maxCtx = m.context_length || 32768;

  const ramClass = rec.fits_in_ram ? "ok" : "warn";

  v.append(
    h("div", { class: "tuner" }, [
      // Controles
      h("div", { class: "panel" }, [
        h("h3", {}, ["Controles"]),
        ctrlSelect(
          "Contexto (tokens)",
          [2048, 4096, 8192, 16384, 32768]
            .filter((x) => x <= Math.max(2048, maxCtx))
            .map((x) => [String(x), String(x)]),
          String(state.overrides.ctx_size),
          (val) => {
            state.overrides.ctx_size = parseInt(val);
            refreshRecommendation();
          },
        ),
        ctrlToggle(
          `Offload Vulkan total (Vega)${rec.gpu_recommended ? " — recomendado" : ""}`,
          c.n_gpu_layers > 0,
          (on) => {
            state.overrides.gpu_offload = on;
            state.overrides.n_gpu_layers = null; // total quando ligado
            refreshRecommendation();
          },
        ),
        ...(c.n_gpu_layers > 0
          ? [
              ctrlRange(
                `Camadas na GPU (${c.n_gpu_layers}/${rec.max_gpu_layers}) — parcial costuma piorar`,
                0,
                rec.max_gpu_layers,
                c.n_gpu_layers,
                (val) => {
                  state.overrides.n_gpu_layers = val;
                  state.overrides.gpu_offload = val > 0;
                  refreshRecommendation();
                },
              ),
            ]
          : []),
        ctrlToggle(
          "KV cache quantizado (q8_0)",
          state.overrides.kv_quant,
          (on) => {
            state.overrides.kv_quant = on;
            refreshRecommendation();
          },
        ),
        ...(m.has_mmproj
          ? [
              ctrlToggle(
                "Visão / multimodal (mmproj) — mais lento",
                state.overrides.use_mmproj,
                (on) => {
                  state.overrides.use_mmproj = on;
                  refreshRecommendation();
                },
              ),
            ]
          : []),
        ...(state.draftCandidate
          ? [
              ctrlToggle(
                `Decodificação especulativa (rascunho: ${state.draftCandidate.name})`,
                c.draft_model != null,
                (on) => {
                  state.overrides.use_speculative = on;
                  refreshRecommendation();
                },
              ),
            ]
          : []),
        ctrlNumber("Porta", state.overrides.port, (val) => {
          state.overrides.port = val;
          refreshRecommendation();
        }),
      ]),
      // Resumo + RAM
      h("div", { class: "panel" }, [
        h("h3", {}, ["Configuracao resultante"]),
        h("div", { class: `ram ${ramClass}` }, [
          h("div", { class: "ram-num" }, [`${rec.est_ram_gb.toFixed(1)} GB`]),
          h("div", { class: "ram-lbl" }, [
            rec.fits_in_ram ? "cabe na RAM" : "perto do limite de RAM",
            ` (de ${state.hw?.total_ram_gb.toFixed(1)} GB)`,
          ]),
        ]),
        h("div", { class: "flags" }, [
          fl("threads (geracao)", String(c.threads)),
          fl("threads (prompt)", String(c.threads_batch)),
          fl("contexto", String(c.ctx_size)),
          fl("n-gpu-layers", String(c.n_gpu_layers)),
          fl("flash-attn", c.flash_attn),
          fl("KV cache", `${c.cache_type_k}`),
          fl("mlock", c.mlock ? "sim" : "nao"),
          fl("batch/ubatch", `${c.batch}/${c.ubatch}`),
          fl("speculative", c.draft_model ? "ligado" : "off"),
        ]),
        h("div", { class: "cmd-label" }, ["Linha de comando:"]),
        h("pre", { class: "cmd" }, [argsPreview(c)]),
      ]),
      // Racional
      h("div", { class: "panel wide" }, [
        h("h3", {}, ["Por que essas escolhas (para o seu hardware)"]),
        h(
          "ul",
          { class: "why" },
          rec.rationale.map((r) => h("li", {}, [r])),
        ),
        ...(rec.warnings.length
          ? [
              h("h3", { class: "warnh" }, ["Avisos"]),
              h(
                "ul",
                { class: "warnlist" },
                rec.warnings.map((w) => h("li", {}, [w])),
              ),
            ]
          : []),
      ]),
    ]),
  );
}

function argsPreview(c: LlamaConfig): string {
  const a: string[] = ["llama-server"];
  a.push("-m", `"${c.model_path}"`);
  a.push("-c", String(c.ctx_size));
  a.push("-t", String(c.threads), "-tb", String(c.threads_batch));
  a.push("-b", String(c.batch), "-ub", String(c.ubatch));
  a.push("-ngl", String(c.n_gpu_layers));
  a.push("--flash-attn", c.flash_attn);
  if (c.cache_type_k !== "f16")
    a.push("--cache-type-k", c.cache_type_k, "--cache-type-v", c.cache_type_v);
  if (c.mlock) a.push("--mlock");
  if (c.mmproj) a.push("--mmproj", `"${c.mmproj}"`);
  if (c.draft_model)
    a.push(
      "-md",
      `"${c.draft_model}"`,
      "-ngld",
      String(c.draft_n_gpu_layers),
      "--draft-max",
      String(c.draft_max),
    );
  a.push("--host", c.host, "--port", String(c.port));
  return a.join(" ");
}

function fl(k: string, v: string) {
  return h("div", { class: "fl" }, [
    h("span", { class: "fk" }, [k]),
    h("span", { class: "fv" }, [v]),
  ]);
}

// Controles reutilizaveis
function ctrlSelect(
  label: string,
  opts: [string, string][],
  value: string,
  on: (v: string) => void,
) {
  const sel = h(
    "select",
    { onChange: (e: Event) => on((e.target as HTMLSelectElement).value) },
    opts.map(([v, l]) =>
      h("option", { value: v, ...(v === value ? { selected: true } : {}) }, [
        l,
      ]),
    ),
  );
  return h("label", { class: "ctrl" }, [
    h("span", {}, [label]),
    sel,
  ]);
}
function ctrlToggle(label: string, value: boolean, on: (v: boolean) => void) {
  const cb = h("input", {
    type: "checkbox",
    ...(value ? { checked: true } : {}),
    onChange: (e: Event) => on((e.target as HTMLInputElement).checked),
  });
  return h("label", { class: "ctrl toggle" }, [
    h("span", {}, [label]),
    cb,
  ]);
}
function ctrlRange(
  label: string,
  min: number,
  max: number,
  value: number,
  on: (v: number) => void,
) {
  const r = h("input", {
    type: "range",
    min: String(min),
    max: String(max),
    value: String(value),
    onInput: (e: Event) => on(parseInt((e.target as HTMLInputElement).value)),
  });
  return h("label", { class: "ctrl" }, [h("span", {}, [label]), r]);
}
function ctrlNumber(label: string, value: number, on: (v: number) => void) {
  const n = h("input", {
    type: "number",
    value: String(value),
    onChange: (e: Event) => on(parseInt((e.target as HTMLInputElement).value)),
  });
  return h("label", { class: "ctrl" }, [h("span", {}, [label]), n]);
}

// ---------- Servidor (carregar/parar) ----------
async function toggleServer() {
  if (state.busy) return;
  const status = await invoke<StatusReport>("server_status");
  if (status.running) {
    await stopServer();
  } else {
    await loadServer();
  }
}

async function loadServer() {
  if (!state.rec) return;
  state.busy = true;
  state.ready = false;
  setPill("subindo", "loading");
  $("#load-btn").textContent = "Subindo…";
  $("#load-btn").setAttribute("disabled", "true");
  addLog(`\n=== Carregando ${state.rec.config.model_name} ===`);
  try {
    await invoke("start_server", { config: state.rec.config });
    switchView("logs");
    await waitForHealthy(state.rec.config.port);
    state.ready = true;
    setPill("pronto", "on");
    $("#load-btn").textContent = "Parar";
    switchView("chat");
  } catch (e) {
    addLog(`ERRO: ${e}`);
    // encerra o processo travado para liberar a porta/RAM
    try {
      await invoke("stop_server");
    } catch {}
    setPill("erro", "off");
    $("#load-btn").textContent = "Carregar";
  } finally {
    state.busy = false;
    $("#load-btn").removeAttribute("disabled");
  }
}

async function stopServer() {
  state.busy = true;
  $("#load-btn").setAttribute("disabled", "true");
  await invoke("stop_server");
  state.ready = false;
  setPill("parado", "off");
  $("#load-btn").textContent = "Carregar";
  state.busy = false;
  $("#load-btn").removeAttribute("disabled");
}

async function waitForHealthy(port: number, timeoutMs = 240000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const s = await invoke<StatusReport>("server_status");
    if (!s.running) throw new Error("processo encerrou durante o carregamento");
    if (s.healthy) return;
    await new Promise((r) => setTimeout(r, 600));
  }
  throw new Error("timeout esperando o servidor ficar saudavel");
}

function setPill(text: string, cls: string) {
  const p = $("#status-pill");
  p.textContent = text;
  p.className = `pill ${cls}`;
}

// ---------- Logs ----------
function addLog(line: string) {
  const pre = $("#logs");
  pre.textContent += (pre.textContent ? "\n" : "") + line;
  pre.scrollTop = pre.scrollHeight;
}

// ---------- Chat ----------
function buildChatView() {
  const v = $("#view-chat");
  v.innerHTML = "";
  v.append(
    h("div", { class: "chat-wrap" }, [
      h("div", { id: "messages", class: "messages" }, [
        h("div", { class: "empty" }, [
          "Carregue um modelo e comece a conversar. As metricas de tok/s aparecem no topo.",
        ]),
      ]),
      h("details", { class: "sampling" }, [
        h("summary", {}, ["Amostragem & system prompt"]),
        h("div", { class: "samp-grid" }, [
          sampField("Temperatura", "temperature", 0, 2, 0.05),
          sampField("top_p", "top_p", 0, 1, 0.01),
          sampField("top_k", "top_k", 0, 200, 1),
          sampField("min_p", "min_p", 0, 1, 0.01),
          sampField("repeat_penalty", "repeat_penalty", 1, 2, 0.01),
          sampField("max_tokens", "max_tokens", 16, 8192, 16),
        ]),
        h("textarea", {
          id: "sysprompt",
          class: "sysprompt",
          rows: "2",
          placeholder: "System prompt (opcional) — vazio por padrão",
          onChange: (e: Event) => {
            state.systemPrompt = (e.target as HTMLTextAreaElement).value;
          },
        }, [state.systemPrompt]),
        h("label", { class: "ctrl toggle", style: "margin-top:8px" }, [
          h("span", {}, [
            "Modo raciocínio (pensar) — desligar envia /no_think (nem todo modelo respeita)",
          ]),
          h("input", {
            type: "checkbox",
            checked: state.think, // false por padrao (opt-in)
            onChange: (e: Event) => {
              state.think = (e.target as HTMLInputElement).checked;
            },
          }),
        ]),
      ]),
      h("div", { class: "composer" }, [
        h("textarea", {
          id: "input",
          placeholder: "Escreva sua mensagem…  (Enter envia, Shift+Enter quebra linha)",
          rows: "1",
        }, []),
        h("button", { id: "send", class: "primary" }, ["Enviar"]),
      ]),
    ]),
  );

  const input = $<HTMLTextAreaElement>("#input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  });
  $("#send").addEventListener("click", onSendOrStop);
}

// Durante o streaming o botao vira "Parar" e deve ABORTAR; caso contrario, envia.
function onSendOrStop() {
  if (state.busy) {
    state.abort?.abort();
  } else {
    send();
  }
}

function sampField(
  label: string,
  key: keyof SamplingParams,
  min: number,
  max: number,
  step: number,
) {
  return h("label", { class: "samp" }, [
    h("span", {}, [label]),
    h("input", {
      type: "number",
      min: String(min),
      max: String(max),
      step: String(step),
      value: String(state.sampling[key]),
      onChange: (e: Event) => {
        state.sampling[key] = parseFloat(
          (e.target as HTMLInputElement).value,
        ) as any;
        localStorage.setItem(SAMPLING_KEY, JSON.stringify(state.sampling));
      },
    }),
  ]);
}

function addMessage(role: "user" | "assistant", content: string): HTMLElement {
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  const msg = h("div", { class: `msg ${role}` }, [
    h("div", { class: "role" }, [role === "user" ? "Voce" : "Assistente"]),
    h("div", { class: "bubble" }, [content]),
  ]);
  box.append(msg);
  box.scrollTop = box.scrollHeight;
  return msg.querySelector(".bubble") as HTMLElement;
}

async function send() {
  if (state.busy) return;
  if (!state.ready) {
    setPill("carregue um modelo", "off");
    return;
  }
  const input = $<HTMLTextAreaElement>("#input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";

  addMessage("user", text);
  state.messages.push({ role: "user", content: text });

  const a = addAssistantMessage();
  a.answer.classList.add("streaming");

  const sysContent = (
    state.think ? state.systemPrompt : `${state.systemPrompt} /no_think`
  ).trim();
  const msgs: ChatMessage[] = [
    // so envia system message se houver conteudo
    ...(sysContent
      ? [{ role: "system" as const, content: sysContent }]
      : []),
    ...state.messages,
  ];

  const port = state.rec!.config.port;
  state.abort = new AbortController();
  state.busy = true;
  $("#send").textContent = "Parar";
  const t0 = performance.now();
  let acc = "";
  let think = "";

  try {
    for await (const chunk of streamChat(
      port,
      msgs,
      state.sampling,
      state.abort.signal,
    )) {
      if (chunk.reasoning) {
        think += chunk.reasoning;
        a.thinkingWrap.style.display = "";
        a.thinking.textContent = think;
      }
      if (chunk.delta) {
        acc += chunk.delta;
        a.answer.textContent = acc;
        $("#messages").scrollTop = $("#messages").scrollHeight;
      }
      if (chunk.timings?.predicted_per_second) {
        showHud(chunk.timings);
      }
      if (chunk.done) break;
    }
  } catch (e) {
    // abort do usuario (botao Parar) nao e erro: so finaliza com o parcial
    const aborted = (e as { name?: string })?.name === "AbortError";
    if (!aborted) {
      acc += `\n\n[erro: ${e}]`;
      a.answer.textContent = acc;
    }
  } finally {
    a.answer.classList.remove("streaming");
    // modelo de reasoning que so "pensou" e nao deu resposta limpa
    if (!acc.trim() && think.trim()) {
      a.answer.textContent =
        "(o modelo respondeu apenas no canal de pensamento — abra 'Pensando' acima)";
      a.answer.classList.add("muted");
    }
    state.messages.push({ role: "assistant", content: acc });
    state.busy = false;
    state.abort = null;
    $("#send").textContent = "Enviar";
    const secs = (performance.now() - t0) / 1000;
    if (!$("#hud").textContent) {
      $("#hud").textContent = `${secs.toFixed(1)}s`;
    }
  }
}

// Cria uma mensagem de assistente com secao recolhivel de "pensamento" + resposta.
function addAssistantMessage() {
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  const thinking = h("pre", { class: "thinking-body" }, []);
  const thinkingWrap = h(
    "details",
    { class: "thinking" },
    [h("summary", {}, ["💭 Pensando…"]), thinking],
  );
  (thinkingWrap as HTMLElement).style.display = "none";
  const answer = h("div", { class: "bubble" }, []);
  const msg = h("div", { class: "msg assistant" }, [
    h("div", { class: "role" }, ["Assistente"]),
    thinkingWrap,
    answer,
  ]);
  box.append(msg);
  box.scrollTop = box.scrollHeight;
  return { answer, thinking, thinkingWrap: thinkingWrap as HTMLElement };
}

function showHud(t: {
  predicted_per_second?: number;
  prompt_per_second?: number;
}) {
  const gen = t.predicted_per_second
    ? `${t.predicted_per_second.toFixed(1)} tok/s`
    : "";
  const pp = t.prompt_per_second
    ? `prompt ${t.prompt_per_second.toFixed(0)} tok/s`
    : "";
  $("#hud").innerHTML = `<b>${gen}</b>${pp ? " · " + pp : ""}`;
}

// ---------- Init ----------
async function init() {
  buildShell();
  const savedSamp = JSON.parse(localStorage.getItem(SAMPLING_KEY) || "null");
  if (savedSamp) state.sampling = { ...state.sampling, ...savedSamp };

  await listen<string>("server-log", (e) => addLog(e.payload));
  await listen<boolean>("server-ready", () =>
    addLog("[taylorai] servidor sinalizou pronto"),
  );

  await loadHardware();
  await loadDirs();
  await scan();
}

init();
