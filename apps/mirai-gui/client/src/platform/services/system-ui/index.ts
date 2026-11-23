export type ZoomStep = "in" | "out" | "reset";

export type SystemUiService = {
  setWindowTheme(theme: "system" | "light" | "dark"): Promise<boolean>;
  /** The interface zoom (1 is 100%). Always 1 where the browser owns zoom (web). */
  getZoom(): Promise<number>;
  /** Steps the zoom like View ▸ Zoom In / Zoom Out / Actual Size, and resolves to the new level. */
  stepZoom(step: ZoomStep): Promise<number>;
  /** Every zoom change, from the menu, its shortcuts or stepZoom. */
  onZoomChange(cb: (level: number) => void): () => void;
  menuTitles(): Promise<string[]>;
  openMenu(index: number, left: number, bottom: number): Promise<void>;
  runMenuItem(id: string): Promise<void>;
  minimizeWindow(): Promise<void>;
  toggleMaximizeWindow(): Promise<void>;
  closeWindow(): Promise<void>;
  onWindowMaximizedChange(cb: (maximized: boolean) => void): () => void;
};
