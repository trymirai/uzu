import js from "@eslint/js";
import { defineConfig } from "eslint/config";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default defineConfig([
  // eslint does not read .gitignore.
  { ignores: ["client/dist-*/**", "src-tauri/target/**", "src-tauri/gen/**", "client/src/route-tree.gen.ts"] },
  js.configs.recommended,
  {
    files: ["vite.config.ts"],
    languageOptions: {
      parser: tseslint.parser,
      globals: {
        process: "readonly",
        __dirname: "readonly",
      },
    },
  },
  {
    files: ["tailwind.config.js"],
    languageOptions: {
      sourceType: "commonjs",
      globals: {
        require: "readonly",
        module: "writable",
        exports: "writable",
        process: "readonly",
        __dirname: "readonly",
      },
    },
  },
  {
    files: ["client/src/**/*.{ts,tsx}"],
    extends: [tseslint.configs.recommended],
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/exhaustive-deps": "error",
      "react-hooks/rules-of-hooks": "error",
      "@typescript-eslint/consistent-type-definitions": ["error", "type"],
    },
  },
  // Module augmentation only merges into an interface.
  {
    files: ["client/src/router.ts"],
    rules: { "@typescript-eslint/consistent-type-definitions": "off" },
  },
  {
    files: ["client/src/**/*.{ts,tsx}"],
    ignores: ["client/src/platform/**"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@tauri-apps/api", "@tauri-apps/api/*", "@tauri-apps/plugin-*"],
              message: "Tauri APIs belong to platform/services/*/tauri.ts. Everywhere else go through getPlatform().",
            },
          ],
        },
      ],
    },
  },
]);
