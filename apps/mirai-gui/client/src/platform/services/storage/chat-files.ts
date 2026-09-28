import { invoke } from "../shared/invoke";

// Chat files are relative names; the Rust side confines them to the chats dir.
export const listChatFiles = () => invoke<string[]>("chat_list_files");
export const loadChatFile = (path: string) => invoke<string | null>("chat_load_file", { path });
export const saveChatFile = (path: string, content: string) => invoke<void>("chat_save_file", { path, content });
export const deleteChatFile = (path: string) => invoke<void>("chat_delete_file", { path });
