// Gerencia o processo llama-server: monta os argumentos a partir do
// LlamaConfig, sobe o processo, transmite os logs para o front via eventos,
// faz health-check e encerra o processo.

use crate::tuner::LlamaConfig;
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use tauri::{AppHandle, Emitter, Manager};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Default)]
pub struct ServerState {
    pub child: Mutex<Option<Child>>,
    pub running: Mutex<Option<RunningInfo>>,
}

#[derive(Clone, Serialize)]
pub struct RunningInfo {
    pub port: u16,
    pub model_name: String,
    pub args: Vec<String>,
}

#[derive(Clone, Serialize)]
pub struct StatusReport {
    pub running: bool,
    pub healthy: bool,
    pub port: Option<u16>,
    pub model_name: Option<String>,
}

/// Resolve o diretorio onde estao llama-server.exe + DLLs.
/// Ordem: variavel de ambiente -> ao lado do exe -> resource_dir -> arvore de dev.
pub fn resolve_binaries_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(env_dir) = std::env::var("TAYLORAI_LLAMA_DIR") {
        let p = PathBuf::from(env_dir);
        if p.join("llama-server.exe").exists() {
            return Some(p);
        }
    }
    // ao lado do executavel (instalado)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("binaries");
            if p.join("llama-server.exe").exists() {
                return Some(p);
            }
        }
    }
    // resource_dir (bundle)
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("binaries");
        if p.join("llama-server.exe").exists() {
            return Some(p);
        }
    }
    // arvore de desenvolvimento: <crate>/binaries
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("binaries");
    if dev.join("llama-server.exe").exists() {
        return Some(dev);
    }
    None
}

fn build_args(cfg: &LlamaConfig) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let push = |a: &mut Vec<String>, k: &str, v: String| {
        a.push(k.to_string());
        a.push(v);
    };

    push(&mut a, "-m", cfg.model_path.clone());
    push(&mut a, "-c", cfg.ctx_size.to_string());
    push(&mut a, "-t", cfg.threads.to_string());
    push(&mut a, "-tb", cfg.threads_batch.to_string());
    push(&mut a, "-b", cfg.batch.to_string());
    push(&mut a, "-ub", cfg.ubatch.to_string());
    push(&mut a, "-ngl", cfg.n_gpu_layers.to_string());
    push(&mut a, "--flash-attn", cfg.flash_attn.clone());
    // Desliga o auto-fit do llama.cpp b9723: como ja definimos ngl/ctx/batch
    // explicitamente, o auto-fit nao tem o que ajustar e TRAVA o load sob
    // pressao de RAM (fica preso em "fitting params to device memory").
    push(&mut a, "-fit", "off".to_string());

    if cfg.cache_type_k != "f16" {
        push(&mut a, "--cache-type-k", cfg.cache_type_k.clone());
    }
    if cfg.cache_type_v != "f16" {
        push(&mut a, "--cache-type-v", cfg.cache_type_v.clone());
    }
    if cfg.mlock {
        a.push("--mlock".to_string());
    }
    if cfg.no_mmap {
        a.push("--no-mmap".to_string());
    }
    if let Some(mm) = &cfg.mmproj {
        push(&mut a, "--mmproj", mm.clone());
    }

    push(&mut a, "--host", cfg.host.clone());
    push(&mut a, "--port", cfg.port.to_string());
    // metricas detalhadas no /props e nas respostas
    a.push("--metrics".to_string());
    a
}

pub fn start(app: &AppHandle, cfg: LlamaConfig) -> Result<RunningInfo, String> {
    let state = app.state::<ServerState>();
    // ja rodando?
    {
        let guard = state.child.lock().unwrap();
        if guard.is_some() {
            return Err("Ja existe um servidor rodando. Pare-o antes de iniciar outro.".into());
        }
    }

    let bin_dir = resolve_binaries_dir(app)
        .ok_or_else(|| "Nao encontrei llama-server.exe (pasta binaries).".to_string())?;
    let exe = bin_dir.join("llama-server.exe");
    let args = build_args(&cfg);

    let mut cmd = Command::new(&exe);
    cmd.args(&args)
        .current_dir(&bin_dir) // para as DLLs ggml-*/llama.dll resolverem
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Falha ao iniciar llama-server: {e}"))?;

    // transmite stdout e stderr como eventos "server-log"
    let pump = |app: AppHandle, stream: Option<Box<dyn std::io::Read + Send>>| {
        if let Some(s) = stream {
            std::thread::spawn(move || {
                let reader = BufReader::new(s);
                for line in reader.lines().map_while(Result::ok) {
                    // sinaliza prontidao
                    if line.contains("server is listening")
                        || line.contains("HTTP server is listening")
                        || line.contains("starting the main loop")
                    {
                        let _ = app.emit("server-ready", true);
                    }
                    let _ = app.emit("server-log", line);
                }
            });
        }
    };
    if let Some(out) = child.stdout.take() {
        pump(app.clone(), Some(Box::new(out)));
    }
    if let Some(err) = child.stderr.take() {
        pump(app.clone(), Some(Box::new(err)));
    }

    let info = RunningInfo {
        port: cfg.port,
        model_name: cfg.model_name.clone(),
        args: args.clone(),
    };

    *state.child.lock().unwrap() = Some(child);
    *state.running.lock().unwrap() = Some(info.clone());

    let _ = app.emit(
        "server-log",
        format!("[taylorai] iniciando: llama-server {}", args.join(" ")),
    );

    Ok(info)
}

pub fn stop(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<ServerState>();
    let mut guard = state.child.lock().unwrap();
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    *state.running.lock().unwrap() = None;
    let _ = app.emit("server-log", "[taylorai] servidor encerrado.".to_string());
    Ok(())
}

pub fn status(app: &AppHandle) -> StatusReport {
    let state = app.state::<ServerState>();
    let running_info = state.running.lock().unwrap().clone();
    let mut child_alive = false;
    {
        let mut guard = state.child.lock().unwrap();
        if let Some(child) = guard.as_mut() {
            match child.try_wait() {
                Ok(Some(_)) => {
                    // processo morreu
                    *guard = None;
                }
                Ok(None) => child_alive = true,
                Err(_) => child_alive = true,
            }
        }
    }

    let port = running_info.as_ref().map(|r| r.port);
    let healthy = match port {
        Some(p) if child_alive => health_ok(p),
        _ => false,
    };

    StatusReport {
        running: child_alive,
        healthy,
        port,
        model_name: running_info.map(|r| r.model_name),
    }
}

pub fn health_ok(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    match ureq::get(&url)
        .timeout(std::time::Duration::from_millis(800))
        .call()
    {
        Ok(resp) => resp.status() == 200,
        Err(_) => false,
    }
}

// Orcamento de memoria enderecavel pela iGPU (Vulkan), detectado uma vez via
// `llama-server --list-devices` e cacheado. Ex.: Vega -> ~10.9 GB (GTT/UMA).
static GPU_BUDGET_GB: OnceLock<Option<f64>> = OnceLock::new();

pub fn vulkan_budget_gb(app: &AppHandle) -> Option<f64> {
    *GPU_BUDGET_GB.get_or_init(|| detect_vulkan_budget_gb(app))
}

fn detect_vulkan_budget_gb(app: &AppHandle) -> Option<f64> {
    let bin_dir = resolve_binaries_dir(app)?;
    let mut cmd = Command::new(bin_dir.join("llama-server.exe"));
    cmd.arg("--list-devices").current_dir(&bin_dir);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.output().ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // Procura "(<total> MiB, <free> MiB free)" numa linha de device Vulkan e
    // usa o "free" como orcamento.
    for line in text.lines() {
        if !line.to_lowercase().contains("vulkan") {
            continue;
        }
        if let Some(idx) = line.find("MiB free") {
            let prefix = &line[..idx];
            let digits: String = prefix
                .chars()
                .rev()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            if let Ok(mib) = digits.parse::<f64>() {
                return Some(mib * 1_048_576.0 / 1e9); // MiB -> GB
            }
        }
    }
    None
}

/// Encerra o servidor no fim do app (chamado no RunEvent::Exit).
pub fn kill_on_exit(app: &AppHandle) {
    let state = app.state::<ServerState>();
    let mut guard = match state.child.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}
