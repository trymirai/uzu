import { mountApp } from "@/mount-app";
import { WebPlatformClient } from "@/platform/web-platform-client";
import { initializeWebAnalytics } from "./analytics";
import { browserChat, browserSession } from "./chat";
import { browserModelManager } from "./models/browser-models";
import { createWebRouter } from "./router";
import "./styles.css";

const router = mountApp(new WebPlatformClient(browserChat, browserSession, browserModelManager), createWebRouter);
initializeWebAnalytics(router);
