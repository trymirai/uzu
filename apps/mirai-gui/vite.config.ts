import { tanstackRouter } from "@tanstack/router-plugin/vite";
import path from "path";
import { defineConfig, mergeConfig } from "vitest/config";
import { sharedClientConfig } from "./vite.shared";

// Desktop (Tauri) frontend; the browser app is built from web/vite.config.ts.
// Vitest runs with mode "test" against the browser platform services.
export default defineConfig(({ mode }) => {
  const isTest = mode === "test";
  return mergeConfig(sharedClientConfig(isTest ? "web" : "tauri"), {
    plugins: isTest
      ? []
      : [
          tanstackRouter({
            routesDirectory: path.join(__dirname, "client/src/routes"),
            generatedRouteTree: path.join(__dirname, "client/src/route-tree.gen.ts"),
            routeFileIgnorePattern: "\\.test\\..*$",
          }),
        ],
    root: path.join(__dirname, "client"),
    build: {
      outDir: path.join(__dirname, "client/dist-tauri"),
      emptyOutDir: true,
    },
    server: { port: 3002, strictPort: true },
    test: {
      environment: "jsdom",
      poolOptions: {
        forks: { execArgv: ["--no-experimental-webstorage"] },
      },
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
    },
  });
});
