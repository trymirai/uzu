import path from "node:path";
import { defineConfig, mergeConfig } from "vitest/config";
import { sharedClientConfig } from "../vite.shared";

// Match the demo's browser isolation so workers can use SharedArrayBuffer.
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig(
  mergeConfig(sharedClientConfig("web"), {
    root: __dirname,
    publicDir: path.resolve(__dirname, "../client/public"),
    resolve: {
      alias: {
        "@uzu/wasm": path.resolve(__dirname, "../../../crates/legacy/uzu/bindings/wasm/src/wasm/uzu.js"),
      },
    },
    build: { outDir: "dist", emptyOutDir: true },
    server: {
      port: 3001,
      strictPort: true,
      headers,
      fs: {
        allow: [
          path.resolve(__dirname, ".."),
          path.resolve(__dirname, "../../../crates/legacy/uzu/bindings/wasm/src/wasm"),
        ],
      },
    },
    preview: { port: 3001, strictPort: true, headers },
    worker: { format: "es" },
    test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
  }),
);
