// Keep conversations in IndexedDB rather than localStorage: writes are atomic,
// failures are visible, and lengthy transcripts do not exhaust a small sync quota.
const DATABASE = "mirai.web";
const VERSION = 1;
export type BrowserStore = "files" | "modelStates";

let opening: Promise<IDBDatabase> | undefined;

const database = (): Promise<IDBDatabase> => {
  opening ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE, VERSION);
    request.onupgradeneeded = () => {
      for (const name of ["files", "modelStates"]) {
        if (!request.result.objectStoreNames.contains(name)) request.result.createObjectStore(name);
      }
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

async function transact<T>(
  store: BrowserStore,
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await database();
  return new Promise<T>((resolve, reject) => {
    const transaction = db.transaction(store, mode);
    const request = operation(transaction.objectStore(store));
    transaction.oncomplete = () => resolve(request.result);
    transaction.onabort = () => reject(transaction.error ?? request.error ?? new Error("Browser storage write failed"));
    transaction.onerror = () => reject(transaction.error ?? request.error ?? new Error("Browser storage failed"));
  });
}

export const browserDb = {
  get: <T>(store: BrowserStore, key: string): Promise<T | undefined> =>
    transact<T | undefined>(store, "readonly", (table) => table.get(key)),
  put: async (store: BrowserStore, key: string, value: unknown): Promise<void> => {
    await transact(store, "readwrite", (table) => table.put(value, key));
  },
  update: async <T>(store: BrowserStore, key: string, change: (value: T | undefined) => T): Promise<void> => {
    let failure: unknown;
    try {
      await transact(store, "readwrite", (table) => {
        const request = table.get(key);
        request.onsuccess = () => {
          try {
            table.put(change(request.result as T | undefined), key);
          } catch (error) {
            failure = error;
            table.transaction.abort();
          }
        };
        return request;
      });
    } catch (error) {
      throw failure ?? error;
    }
  },
  delete: async (store: BrowserStore, key: string): Promise<void> => {
    await transact(store, "readwrite", (table) => table.delete(key));
  },
  keys: async (store: BrowserStore): Promise<string[]> =>
    (await transact(store, "readonly", (table) => table.getAllKeys())).map(String),
};
