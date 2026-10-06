export const SUPPORTED_FILE_TYPES: {
  text: string[];
} = {
  text: [
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
  ],
};

export const MAX_SINGLE_FILE_SIZE = 256 * 1024;
export const MAX_TOTAL_ATTACHMENTS_SIZE = 512 * 1024;
export const MAX_FILES_PER_MESSAGE = 5;
