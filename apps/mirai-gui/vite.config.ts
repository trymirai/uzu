import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import path from "path";
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
    server: { port: platform === "web" ? 3001 : 3002, strictPort: true },
    resolve: { alias: { "@": path.resolve(__dirname, "client/src") } },
    define: { __PLATFORM__: JSON.stringify(platform) },
    optimizeDeps: { include: ["shiki"] },
    assetsInclude: ["**/*.wasm"],
    test: {
      environment: "jsdom",
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
    },
  };
});
