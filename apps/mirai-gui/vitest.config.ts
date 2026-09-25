import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { clientAlias, clientDefine } from "./vite.shared";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: clientAlias },
  define: clientDefine("web"),
  test: {
    environment: "jsdom",
    include: ["client/src/**/*.test.{ts,tsx}"],
    css: false,
  },
});
