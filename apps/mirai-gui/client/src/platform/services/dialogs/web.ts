import type { DialogsService } from ".";

export const webDialogs: DialogsService = {
  showSaveDialog: () => Promise.resolve(null),
  showOpenDialog: () => Promise.resolve(null),
  readTextFile: () => Promise.resolve(null),
};
