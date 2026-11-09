import { mountApp } from "@/mount-app";
import { WebPlatformClient } from "@/platform/web-platform-client";
import { browserModelManager } from "./models/browser-models";
import { createWebRouter } from "./router";
import "./styles.css";

mountApp({ ...new WebPlatformClient(), models: browserModelManager }, createWebRouter);
