import { describe, expect, it } from "vitest";
import { ModelKind, type PlatformModel } from "@/types/models";
import { modelDownloadPhases } from "@/types/model-manager";
import { buildFamilyDetailView } from "./view";

const model = (repoId: string, size: number): PlatformModel => ({
  repoId,
  kind: ModelKind.Text,
  name: repoId,
  vendor: "Alibaba",
  reasoning: { kind: "unsupported" },
  supportsTools: true,
  size,
  familyIdentifier: "qwen3.8",
});

describe("buildFamilyDetailView", () => {
  // The API's order for Qwen3.8 27B: the recommended checkpoint first, not the smallest.
  const catalog = [
    model("trymirai/Qwen3.8-27B-M", 15.7e9),
    model("trymirai/Qwen3.8-27B-S", 9.9e9),
    model("trymirai/Qwen3.8-27B-XS", 7.2e9),
    model("trymirai/Qwen3.8-27B-L", 28.9e9),
  ];

  it("keeps each family in the catalog's order instead of sorting by size", () => {
    const view = buildFamilyDetailView({
      models: catalog,
      modelPhasesById: {},
      familyIdentifier: "qwen3.8",
      searchLower: "",
    });
    expect(view.available.map((m) => m.repoId)).toEqual(catalog.map((m) => m.repoId));
  });

  it("keeps that order within Downloaded and Available", () => {
    const view = buildFamilyDetailView({
      models: catalog,
      modelPhasesById: {
        "trymirai/Qwen3.8-27B-L": modelDownloadPhases.downloaded,
        "trymirai/Qwen3.8-27B-S": modelDownloadPhases.downloaded,
      },
      familyIdentifier: "qwen3.8",
      searchLower: "",
    });
    expect(view.installed.map((m) => m.repoId)).toEqual(["trymirai/Qwen3.8-27B-S", "trymirai/Qwen3.8-27B-L"]);
    expect(view.available.map((m) => m.repoId)).toEqual(["trymirai/Qwen3.8-27B-M", "trymirai/Qwen3.8-27B-XS"]);
  });
});
