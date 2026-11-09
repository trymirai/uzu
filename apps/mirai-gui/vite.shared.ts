import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

export function sharedClientConfig(platform: "web" | "tauri") {
  return {
    plugins: [react(), tailwindcss()],
    resolve: { alias: { "@": path.resolve(__dirname, "client/src") } },
    define: { __PLATFORM__: JSON.stringify(platform) },
    optimizeDeps: { include: ["shiki"] },
    assetsInclude: ["**/*.wasm"],
  };
}
