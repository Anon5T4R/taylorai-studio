// Esconde o console no Windows em build release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    taylorai_studio_lib::run()
}
