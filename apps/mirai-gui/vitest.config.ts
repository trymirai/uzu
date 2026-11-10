import path from "node:path";
import { defineConfig, mergeConfig } from "vitest/config";
import { sharedClientConfig } from "./vite.shared";

// Tests for the shared client, run against the browser platform services.
// The apps are served and built from desktop/vite.config.ts and web/vite.config.ts.
export default defineConfig(
  mergeConfig(sharedClientConfig("web", "test"), {
    root: path.join(__dirname, "client"),
    test: {
      environment: "jsdom",
      poolOptions: {
        forks: { execArgv: ["--no-experimental-webstorage"] },
      },
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
    },
  }),
);
