import type { ModelsService } from ".";
import {
  browserModelManager,
  findBrowserModel,
  setBrowserEncodingInspector,
  setBrowserModelConfigInspector,
} from "./browser-models";
import { browserRuntime, inspectBrowserEncoding, inspectBrowserModelConfig } from "../chat/browser-runtime";

setBrowserEncodingInspector(inspectBrowserEncoding);
setBrowserModelConfigInspector(inspectBrowserModelConfig);

export const webModels: ModelsService = {
  ...browserModelManager,
  resumeDownload: (identifier) => browserModelManager.startDownload(identifier),
  async deleteModel(identifier) {
    const model = await findBrowserModel(identifier);
    await browserRuntime.eject(model.repoId ?? model.identifier);
    await browserModelManager.deleteModel(identifier);
  },
};
