// Keep conversations in IndexedDB rather than localStorage: writes are atomic,
// failures are visible, and lengthy transcripts do not exhaust a small sync quota.
// The name and version match the earlier uzu.run build, so its chats remain readable.
const DATABASE = "mirai.web";
const VERSION = 1;
const FILES = "files";

let opening: Promise<IDBDatabase> | undefined;

const database = (): Promise<IDBDatabase> => {
  opening ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE, VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(FILES)) request.result.createObjectStore(FILES);
    };
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => {
        db.close();
        opening = undefined;
      };
      resolve(db);
    };
    request.onerror = () => reject(request.error ?? new Error("Could not open browser storage"));
    request.onblocked = () => reject(new Error("Close other Mirai tabs to update browser storage"));
  }).catch((error: unknown) => {
    opening = undefined;
    throw error;
  });
  return opening;
};

async function transact<T>(mode: IDBTransactionMode, operation: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await database();
  return new Promise<T>((resolve, reject) => {
    const transaction = db.transaction(FILES, mode);
    const request = operation(transaction.objectStore(FILES));
    transaction.oncomplete = () => resolve(request.result);
    transaction.onabort = () => reject(transaction.error ?? request.error ?? new Error("Browser storage write failed"));
    transaction.onerror = () => reject(transaction.error ?? request.error ?? new Error("Browser storage failed"));
  });
}

/** Text files keyed by path, the browser counterpart of the desktop data directory. */
export const browserFiles = {
  get: (key: string): Promise<string | undefined> =>
    transact<string | undefined>("readonly", (store) => store.get(key) as IDBRequest<string | undefined>),
  put: async (key: string, value: string): Promise<void> => {
    await transact("readwrite", (store) => store.put(value, key));
  },
  delete: async (key: string): Promise<void> => {
    await transact("readwrite", (store) => store.delete(key));
  },
  keys: async (): Promise<string[]> => (await transact("readonly", (store) => store.getAllKeys())).map(String),
};
