import { describe, expect, it } from "vitest";
import { modelDownloadPhases as phases, type ModelDownloadState } from "@/types/modelManager";
import { downloadStatePatch } from "./modelDownloadState";

const identifier = "vendor/model";
const MB = 1024 * 1024;

const state = (patch: Partial<ModelDownloadState>): ModelDownloadState => ({
  totalKbytes: 4096,
  downloadedKbytes: 1024,
  phase: phases.downloading,
  seq: 0,
  ...patch,
});

const progress = (completedBytes: number, totalBytes: number | null = 4 * MB) =>
  ({ kind: "progress", identifier, seq: 0, completedBytes, totalBytes }) as const;

describe("downloadStatePatch", () => {
  it("starts tracking a model it has not seen before", () => {
    expect(downloadStatePatch(undefined, progress(MB))).toEqual({
      downloadedKbytes: 1024,
      totalKbytes: 4096,
      phase: phases.downloading,
    });
  });

  it("keeps the known total when a tick carries none", () => {
    expect(downloadStatePatch(state({}), progress(2 * MB, null))).toEqual({
      downloadedKbytes: 2048,
      phase: phases.downloading,
    });
  });

  it("does not reactivate a paused or finished download on a late tick", () => {
    expect(downloadStatePatch(state({ phase: phases.paused }), progress(2 * MB))).toEqual({
      downloadedKbytes: 2048,
      totalKbytes: 4096,
    });
    expect(downloadStatePatch(state({ phase: phases.downloaded }), progress(4 * MB))).toEqual({
      downloadedKbytes: 4096,
      totalKbytes: 4096,
    });
  });

  it("keeps an error until bytes actually grow", () => {
    const failed = state({ phase: phases.error, error: "disk full" });
    expect(downloadStatePatch(failed, progress(MB))).toEqual({ downloadedKbytes: 1024, totalKbytes: 4096 });
    expect(downloadStatePatch(failed, progress(2 * MB))).toEqual({
      downloadedKbytes: 2048,
      totalKbytes: 4096,
      phase: phases.downloading,
      error: "",
    });
  });

  it("leaves a locked download once progress arrives", () => {
    expect(downloadStatePatch(state({ phase: phases.locked }), progress(2 * MB))).toMatchObject({
      phase: phases.downloading,
      error: "",
    });
  });

  it("ignores resumed while an error is showing", () => {
    expect(
      downloadStatePatch(state({ phase: phases.error, error: "x" }), { kind: "resumed", identifier, seq: 0 }),
    ).toBeNull();
    expect(downloadStatePatch(state({ phase: phases.paused }), { kind: "resumed", identifier, seq: 0 })).toEqual({
      phase: phases.downloading,
      error: "",
    });
  });

  it("maps terminal events", () => {
    expect(downloadStatePatch(state({}), { kind: "done", identifier, seq: 0 })).toEqual({
      phase: phases.downloaded,
      error: "",
    });
    expect(downloadStatePatch(state({}), { kind: "error", identifier, seq: 0, error: "boom" })).toEqual({
      phase: phases.error,
      error: "boom",
    });
    expect(downloadStatePatch(state({}), { kind: "locked", identifier, seq: 0, lockedBy: "cli" })).toEqual({
      phase: phases.locked,
      error: "",
    });
    expect(downloadStatePatch(state({}), { kind: "paused", identifier, seq: 0 })).toEqual({ phase: phases.paused });
    expect(downloadStatePatch(state({}), { kind: "deleted", identifier, seq: 0 })).toEqual({
      phase: phases.notDownloaded,
      downloadedKbytes: 0,
      totalKbytes: 0,
      error: "",
    });
  });
});
