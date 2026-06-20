import { defineConfig } from "vite";

// Tauri espera um dev server fixo. Mantemos leve: sem framework, so Vite.
export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "127.0.0.1",
    watch: {
      // nao vigiar o backend Rust
      ignored: ["**/src-tauri/**"],
    },
  },
  build: {
    target: "esnext",
    minify: "esbuild",
    sourcemap: false,
  },
});
