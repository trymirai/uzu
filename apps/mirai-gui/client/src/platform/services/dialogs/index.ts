export type FileDialogFilter = { name: string; extensions: string[] };

export type SaveDialogOptions = {
  title?: string;
  defaultPath?: string;
  filters?: FileDialogFilter[];
};

export type DialogsService = {
  /** Resolves to the chosen absolute path; null when cancelled, unsupported or failed. */
  showSaveDialog(options: SaveDialogOptions): Promise<string | null>;
  readTextFile(absolutePath: string): Promise<string | null>;
};
