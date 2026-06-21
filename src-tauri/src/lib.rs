mod gguf;
mod hardware;
mod server;
mod tuner;

use gguf::ModelInfo;
use hardware::HardwareInfo;
use server::{RunningInfo, ServerState, StatusReport};
use tauri::{AppHandle, RunEvent};
use tuner::{LlamaConfig, Recommendation, TuneOverrides};

#[tauri::command]
fn get_hardware(app: AppHandle) -> HardwareInfo {
    let budget = server::vulkan_budget_gb(&app);
    hardware::get_hardware(budget)
}

#[tauri::command]
fn scan_models(dirs: Vec<String>) -> Vec<ModelInfo> {
    gguf::scan_dirs(&dirs)
}

#[tauri::command]
fn recommend_config(
    app: AppHandle,
    model: ModelInfo,
    overrides: TuneOverrides,
) -> Recommendation {
    let budget = server::vulkan_budget_gb(&app);
    let hw = hardware::get_hardware(budget);
    tuner::recommend(&hw, &model, &overrides)
}

#[tauri::command]
fn start_server(app: AppHandle, config: LlamaConfig) -> Result<RunningInfo, String> {
    server::start(&app, config)
}

#[tauri::command]
fn stop_server(app: AppHandle) -> Result<(), String> {
    server::stop(&app)
}

#[tauri::command]
fn server_status(app: AppHandle) -> StatusReport {
    server::status(&app)
}

#[tauri::command]
fn pick_folder() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Escolha a pasta com modelos GGUF")
        .pick_folder()
        .map(|p| p.to_string_lossy().into_owned())
}

/// Diretorios candidatos onde costumam existir modelos GGUF.
#[tauri::command]
fn default_model_dirs() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push_if_exists = |p: std::path::PathBuf| {
        if p.exists() {
            out.push(p.to_string_lossy().into_owned());
        }
    };

    // LM Studio em varias unidades (so Windows)
    #[cfg(windows)]
    for drive in ["C", "D", "E"] {
        push_if_exists(std::path::PathBuf::from(format!(
            "{drive}:\\LocalAIModels\\.lmstudio\\hub\\models"
        )));
    }
    // home: USERPROFILE no Windows, HOME no Linux/macOS
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let home = std::path::PathBuf::from(home);
        push_if_exists(home.join(".lmstudio").join("hub").join("models"));
        push_if_exists(home.join(".cache").join("lm-studio").join("models"));
        push_if_exists(
            home.join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }
    out
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ServerState::default())
        .invoke_handler(tauri::generate_handler![
            get_hardware,
            scan_models,
            recommend_config,
            start_server,
            stop_server,
            server_status,
            pick_folder,
            default_model_dirs
        ])
        .build(tauri::generate_context!())
        .expect("erro ao inicializar o TaylorAI Studio")
        .run(|app: &AppHandle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                server::kill_on_exit(app);
            }
            if let RunEvent::Exit = event {
                server::kill_on_exit(app);
            }
        });
}
