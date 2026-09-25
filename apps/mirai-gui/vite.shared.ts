import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

type Platform = "tauri" | "web";

export const clientAlias = { "@": path.resolve(__dirname, "client/src") };

export const clientDefine = (platform: Platform) => ({
  __PLATFORM__: JSON.stringify(platform),
});

export const createClientConfig = ({ platform, port }: { platform: Platform; port: number }) =>
  defineConfig({
    plugins: [
      react(),
      tanstackRouter({
        routesDirectory: path.join(__dirname, "client/src/routes"),
        generatedRouteTree: path.join(__dirname, "client/src/routeTree.gen.ts"),
        routeFileIgnorePattern: "\\.test\\..*$",
      }),
    ],
    root: path.join(__dirname, "client"),
    // Absolute base so `createBrowserHistory` deep links (e.g. /chat/abc) resolve
    // asset URLs correctly.
    base: "/",
    build: {
      outDir: path.join(__dirname, `client/dist-${platform}`),
      emptyOutDir: true,
    },
    server: { port, strictPort: true },
    resolve: { alias: clientAlias },
    define: clientDefine(platform),
    optimizeDeps: { include: ["shiki"] },
    assetsInclude: ["**/*.wasm"],
  });
