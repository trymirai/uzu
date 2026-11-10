import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// Settings shared by the desktop and web apps. Both regenerate the client route
// tree while serving or building; tests (mode "test") leave it alone.
export function sharedClientConfig(platform: "web" | "tauri", mode: string) {
  const routeTree =
    mode === "test"
      ? []
      : [
          tanstackRouter({
            routesDirectory: path.join(__dirname, "client/src/routes"),
            generatedRouteTree: path.join(__dirname, "client/src/route-tree.gen.ts"),
            routeFileIgnorePattern: "\\.test\\..*$",
          }),
        ];
  return {
    plugins: [...routeTree, react(), tailwindcss()],
    resolve: { alias: { "@": path.resolve(__dirname, "client/src") } },
    define: { __PLATFORM__: JSON.stringify(platform) },
    optimizeDeps: { include: ["shiki"] },
    assetsInclude: ["**/*.wasm"],
  };
}
