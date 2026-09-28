import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default [
  // eslint does not read .gitignore.
  { ignores: ["client/dist-*/**", "src-tauri/target/**", "src-tauri/gen/**", "client/src/route-tree.gen.ts"] },
  js.configs.recommended,
  {
    files: ["vite*.config.ts", "vite.shared.ts"],
    languageOptions: {
      parser: tseslint.parser,
      globals: {
        process: "readonly",
        __dirname: "readonly",
      },
    },
  },
  {
    files: ["tailwind.config.js", "postcss.config.js"],
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
  ...tseslint.config({
    files: ["client/src/**/*.{ts,tsx}"],
    extends: [tseslint.configs.recommended],
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/exhaustive-deps": "error",
      "react-hooks/rules-of-hooks": "error",
      "@typescript-eslint/consistent-type-definitions": ["error", "type"],
    },
  }),
  // Module augmentation only merges into an interface.
  {
    files: ["client/src/router.ts"],
    rules: { "@typescript-eslint/consistent-type-definitions": "off" },
  },
  // Platform bridges (Tauri invoke, web fetch) belong to platform/services/*;
  // everything else goes through getPlatform().
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
];
