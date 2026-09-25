/** @type {import('tailwindcss').Config} */
const defaultTheme = require("tailwindcss/defaultTheme");
const plugin = require("tailwindcss/plugin");
const fs = require("fs");
const path = require("path");

// Semantic colors and shadows are scraped from the ui-kit token CSS so
// Tailwind keys stay in sync with the variables.
const tokensCss = fs.readFileSync(path.join(__dirname, "client/src/ui-kit/tokens/semantic.css"), "utf8");
const tokenVars = (family) =>
  Object.fromEntries(
    [...tokensCss.matchAll(new RegExp(`^\\s*--ui-${family}-([a-z0-9-]+)\\s*:`, "gm"))].map((m) => [
      m[1],
      `var(--ui-${family}-${m[1]})`,
    ]),
  );
const kitColors = tokenVars("color");
const kitShadows = tokenVars("shadow");
const kitGray = Object.fromEntries(
  [50, 100, 200, 300, 500, 600, 700, 1000, 1100, 1200].map((n) => [n, `var(--ui-color-gray-${n})`]),
);

// Kit utilities live in a plugin so the app's Tailwind generates them.
const kitUtilities = plugin(({ addUtilities }) => {
  addUtilities({
    ".thin-scrollbar::-webkit-scrollbar": { width: "6px", height: "6px" },
    ".thin-scrollbar::-webkit-scrollbar-track": { "background-color": "transparent" },
    ".thin-scrollbar::-webkit-scrollbar-thumb": {
      "background-color": "var(--ui-color-gray-500)",
      "border-radius": "9999px",
    },
  });
});

module.exports = {
  // Scoped to sources so the build output in client/dist-* is never scanned.
  content: ["./client/index.html", "./client/src/**/*.{js,ts,jsx,tsx}", "./node_modules/streamdown/dist/*.js"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["InterVariable", "Inter", ...defaultTheme.fontFamily.sans],
        mono: [
          "Geist Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          '"Liberation Mono"',
          '"Courier New"',
          "monospace",
        ],
      },
      colors: {
        ...kitColors,
        gray: kitGray,
        green: { 500: "rgb(from var(--ui-color-green-500) r g b / <alpha-value>)" },
        amber: { 500: "rgb(from var(--ui-color-amber-500) r g b / <alpha-value>)" },
        red: { 500: "rgb(from var(--ui-color-red-500) r g b / <alpha-value>)" },
        "bg-sidebar": {
          DEFAULT: "#F5F5F5",
          dark: "#050506",
        },
        bg: {
          DEFAULT: "#FCFCFC",
          dark: "#0A0A0A",
        },
        "bg-hover": {
          DEFAULT: "#E5E5E5",
          dark: "#181A1B",
        },
        "bg-sub": {
          DEFAULT: "#F4F5F5",
          dark: "#1F1F1F",
        },
        "card-hover": {
          DEFAULT: "#F6F8F8",
          dark: "#1F1F1F",
        },
        "card-modal-hover": {
          DEFAULT: "#F6F8F8",
          dark: "#1F1F1F",
        },
        "card-modal": {
          DEFAULT: "#FFFFFF",
          dark: "#141414",
        },
        "button-border": {
          DEFAULT: "#DFE1E2",
          dark: "#343434",
        },
        "label-title": {
          DEFAULT: "#000000",
          dark: "#FFFFFF",
        },
        "label-title-dark": {
          DEFAULT: "#FFFFFF",
          dark: "#000000",
        },
        "label-muted": {
          DEFAULT: "#526168",
          dark: "#9198A1",
        },
        "bg-modal": {
          DEFAULT: "#FFFFFF",
          dark: "#141414",
        },
        "cell-border": {
          DEFAULT: "#E5E5E5",
          dark: "#212227",
        },
        success: "rgb(from var(--ui-color-success) r g b / <alpha-value>)",
        error: "#FF2020",
        // DEFAULT is the app accent; 500 keeps the kit entry this key shadows, in
        // rgb(from …) form so opacity variants keep generating.
        blue: {
          DEFAULT: "#33BBFF",
          500: "rgb(from var(--ui-color-blue-500) r g b / <alpha-value>)",
        },
        progress: "#FF6A20",
      },
      screens: {
        lg: "1050px",
      },
      transitionTimingFunction: { spring: "cubic-bezier(0.23, 1, 0.32, 1)" },
      boxShadow: kitShadows,
    },
  },
  plugins: [kitUtilities],
};
