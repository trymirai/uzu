import { mountApp } from "./mount-app";
import { createAppRouter } from "./router";
import "./index.css";
import { TauriPlatformClient } from "./platform/tauri-platform-client";

mountApp(new TauriPlatformClient(), createAppRouter);
