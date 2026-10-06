import type { AttachedFile } from "@/types/files";

const ATTACHMENTS_KEY = "mirai-attached-files";

type AttachmentsById = {
  [fileId: string]: AttachedFile;
};

class AttachmentStorage {
  private storage: AttachmentsById = {};

  constructor() {
    this.loadFromStorage();
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(ATTACHMENTS_KEY);
      if (stored) {
        this.storage = JSON.parse(stored);
      }
    } catch (error) {
      console.error("Error loading attachment storage:", error);
      this.storage = {};
    }
  }

  private saveToStorage(): void {
    try {
      localStorage.setItem(ATTACHMENTS_KEY, JSON.stringify(this.storage));
    } catch (error) {
      console.error("Error saving attachment storage:", error);
    }
  }

  saveFile(file: AttachedFile): void {
    this.storage[file.id] = file;
    this.saveToStorage();
  }

  getFiles(fileIds: string[]): AttachedFile[] {
    return fileIds.map((id) => this.storage[id]).filter((file) => file !== undefined);
  }

  deleteFiles(fileIds: string[]): void {
    fileIds.forEach((id) => delete this.storage[id]);
    this.saveToStorage();
  }

  cleanup(): void {
    const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
    const fileIdsToDelete: string[] = [];

    Object.values(this.storage).forEach((file) => {
      const timestamp = Number.parseInt(file.id.split("-")[0] ?? "", 10);
      if (Number.isFinite(timestamp) && timestamp < sevenDaysAgo) {
        fileIdsToDelete.push(file.id);
      }
    });

    this.deleteFiles(fileIdsToDelete);
  }
}

export const attachmentStorage = new AttachmentStorage();
