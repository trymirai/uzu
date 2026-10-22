export type SystemUiService = {
  setWindowTheme(theme: "system" | "light" | "dark"): Promise<boolean>;
};
