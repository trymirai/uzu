import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import autoprefixer from "autoprefixer";
import path from "path";
import tailwindcss from "tailwindcss";
import { defineConfig } from "vitest/config";

// `--mode web` builds the browser variant; vitest runs with mode "test".
export default defineConfig(({ mode }) => {
  const platform = mode === "web" || mode === "test" ? "web" : "tauri";
  return {
    plugins: [
      react(),
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
    css: { postcss: { plugins: [tailwindcss(), autoprefixer()] } },
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
