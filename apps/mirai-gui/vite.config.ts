import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { readFileSync } from "node:fs";
import { defineConfig } from "vitest/config";

// `--mode web` builds the browser variant; vitest runs with mode "test".
export default defineConfig(({ mode }) => {
  const platform = mode === "web" || mode === "test" ? "web" : "tauri";
  return {
    plugins: [
      react(),
      tailwindcss(),
      ...(mode === "test"
        ? []
        : [
            tanstackRouter({
              routesDirectory: path.join(__dirname, "client/src/routes"),
              generatedRouteTree: path.join(__dirname, "client/src/route-tree.gen.ts"),
              routeFileIgnorePattern: "\\.test\\..*$",
            }),
          ]),
    ],
    root: path.join(__dirname, "client"),
    build: {
      outDir: path.join(__dirname, `client/dist-${platform}`),
      emptyOutDir: true,
    },
    // The engine worker synchronizes GPU batches using SharedArrayBuffer.
    server: {
      port: platform === "web" ? 3001 : 3002,
      strictPort: true,
      headers:
        platform === "web"
          ? {
              "Cross-Origin-Opener-Policy": "same-origin",
              "Cross-Origin-Embedder-Policy": "require-corp",
            }
          : {},
    },
    preview: {
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    },
    worker: { format: "es" },
    resolve: { alias: { "@": path.resolve(__dirname, "client/src") } },
    define: {
      __PLATFORM__: JSON.stringify(platform),
      __MODEL_TOOLCHAIN_VERSION__: JSON.stringify(
        readFileSync(path.resolve(__dirname, "../../crates/uzu-engine/src/utils/version.rs"), "utf8").match(
          /TOOLCHAIN_VERSION: &str = "([^"]+)"/,
        )?.[1] ??
          (() => {
            throw new Error("Engine toolchain version is missing");
          })(),
      ),
    },
    optimizeDeps: { include: ["shiki"] },
    assetsInclude: ["**/*.wasm"],
    test: {
      environment: "jsdom",
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
    },
  };
});
