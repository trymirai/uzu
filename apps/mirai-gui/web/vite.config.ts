import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
import { defineConfig } from "vite";

// Match the demo's browser isolation so workers can use SharedArrayBuffer.
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  root: __dirname,
  publicDir: path.resolve(__dirname, "../client/public"),
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "../client/src") } },
  define: { __PLATFORM__: JSON.stringify("web") },
  build: { outDir: "dist", emptyOutDir: true },
  server: { port: 3001, strictPort: true, headers },
  preview: { port: 3001, strictPort: true, headers },
  worker: { format: "es" },
  assetsInclude: ["**/*.wasm"],
});
