export type ZoomStep = "in" | "out" | "reset";

export type SystemUiService = {
  setWindowTheme(theme: "system" | "light" | "dark"): Promise<boolean>;
  /** The interface zoom (1 is 100%). Always 1 where the browser owns zoom (web). */
  getZoom(): Promise<number>;
  /** Steps the zoom like View ▸ Zoom In / Zoom Out / Actual Size, and resolves to the new level. */
  stepZoom(step: ZoomStep): Promise<number>;
  /** Every zoom change, from the menu, its shortcuts or stepZoom. */
  onZoomChange(cb: (level: number) => void): () => void;
  /** The app menus' titles, for a title bar the page draws (the frameless Windows and Linux window). */
  menuTitles(): Promise<string[]>;
  /** Opens app menu `index` under its title, whose left and bottom edges are in page pixels. */
  openMenu(index: number, left: number, bottom: number): Promise<void>;
  /** Runs an app menu item, by its id in desktop/src/system_ui.rs, for its shortcut. */
  runMenuItem(id: string): Promise<void>;
  minimizeWindow(): Promise<void>;
  toggleMaximizeWindow(): Promise<void>;
  closeWindow(): Promise<void>;
  /** Whether the window is maximized now and after every resize. */
  onWindowMaximizedChange(cb: (maximized: boolean) => void): () => void;
};
