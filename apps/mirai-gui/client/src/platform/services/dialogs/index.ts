export type FileDialogFilter = { name: string; extensions: string[] };

export type SaveDialogOptions = {
  title?: string;
  defaultPath?: string;
  filters?: FileDialogFilter[];
};

export type OpenDialogOptions = {
  title?: string;
  filters?: FileDialogFilter[];
};

export type DialogsService = {
  /** Resolves to the chosen absolute path, or null when cancelled/unsupported. */
  showSaveDialog(options: SaveDialogOptions): Promise<string | null>;
  /** Resolves to the chosen absolute path, or null when cancelled/unsupported. */
  showOpenDialog(options: OpenDialogOptions): Promise<string | null>;
  readTextFile(absolutePath: string): Promise<string | null>;
};
