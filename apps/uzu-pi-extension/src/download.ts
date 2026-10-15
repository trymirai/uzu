import {
  DownloadPhaseDownloaded,
  DownloadPhaseError,
  DownloadPhaseLocked,
  DownloadPhasePaused,
  type Engine,
  type Model,
} from "@trymirai/uzu";
import { abortable } from "./abort.js";

export async function prepareModel(
  engine: Pick<
    Engine,
    "modelPath" | "downloader" | "download" | "chatModelInfo"
  >,
  model: Model,
  signal?: AbortSignal,
  progress: (message?: string) => void = () => {},
) {
  signal?.throwIfAborted();
  if (!(await engine.modelPath(model))) {
    if (!model.isDownloadable)
      throw new Error(`Uzu model files are unavailable: ${model.name}`);
    const downloader = engine.downloader(model);
    if ((await downloader.state)?.phase instanceof DownloadPhaseLocked)
      throw new Error(`Another process is downloading ${model.name}`);
    signal?.throwIfAborted();
    progress(`Downloading ${model.name}`);
    try {
      // Await startup before pausing a cancelled request, so resume cannot race pause.
      const stream = await engine.download(model);
      signal?.throwIfAborted();
      while (true) {
        const update = await abortable(stream.next(), signal);
        if (!update) break;
        if (update.phase instanceof DownloadPhaseError)
          throw new Error(update.phase.message);
        if (update.phase instanceof DownloadPhaseLocked)
          throw new Error(`Another process is downloading ${model.name}`);
        if (update.phase instanceof DownloadPhasePaused)
          throw new Error(`Download paused: ${model.name}`);
        progress(
          `Downloading ${model.name}: ${Math.round(update.progress * 100)}%`,
        );
      }
      signal?.throwIfAborted();
      const state = await downloader.state;
      if (state?.phase instanceof DownloadPhaseError)
        throw new Error(state.phase.message);
      if (!(state?.phase instanceof DownloadPhaseDownloaded))
        throw new Error(`Download did not complete: ${model.name}`);
    } catch (error) {
      if (!((await downloader.state)?.phase instanceof DownloadPhaseLocked))
        await downloader.pause();
      throw error;
    } finally {
      progress();
    }
  }
  signal?.throwIfAborted();
  const info = await engine.chatModelInfo(model);
  signal?.throwIfAborted();
  if (!info) throw new Error(`Unsupported Uzu chat model: ${model.name}`);
  return info;
}
