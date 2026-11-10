import path from "node:path";
import { defineConfig, mergeConfig } from "vite";
import { sharedClientConfig } from "../vite.shared";

// Desktop (Tauri) frontend: the shared client with the Tauri platform services.
export default defineConfig(({ mode }) =>
  mergeConfig(sharedClientConfig("tauri", mode), {
    root: path.resolve(__dirname, "../client"),
    build: { outDir: path.resolve(__dirname, "dist"), emptyOutDir: true },
    server: { port: 3002, strictPort: true },
  }),
);
