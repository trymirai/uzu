import type { DownloadState, Model } from "@uzu/wasm";
import type { EngineModel, ModelDownloadState, ModelDownloadPhase } from "@/types/model-manager";
import { toDownloadProgressKbytes } from "@/utils/download-progress";
import type { UzuBindings } from "../uzu-runtime";

export type CatalogModel = EngineModel & {
  files: { name: string; url: string; sizeBytes: number }[];
  browserUnsupportedReason?: string;
};
export type ModelDescriptor = Omit<CatalogModel, "state">;

// Copy UI metadata once; retain only the native Model handle for engine operations.
export function describeModel(model: Model, api: UzuBindings): ModelDescriptor {
  const family = model.family;
  const properties = model.properties;
  const quantization = model.quantization;
  const vendor = family?.vendor ?? quantization?.vendor;
  const metadata = vendor?.metadata;
  const icons = metadata?.icons ?? [];
  const accessibility = model.accessibility;
  const source = accessibility instanceof api.ModelAccessibilityOnDevice ? accessibility.source : undefined;
  const files = source instanceof api.ModelSourceRegistry ? source.files : [];
  try {
    const light = icons.find((icon) => icon.theme === api.ImageTheme.Light)?.url ?? icons[0]?.url;
    const dark = icons.find((icon) => icon.theme === api.ImageTheme.Dark)?.url ?? light;
    return {
      identifier: model.identifier,
      repoId: model.repoIds[0] ?? model.identifier,
      name: model.name,
      vendor: vendor?.name ?? "",
      ...(light && dark ? { vendorIcons: { light, dark } } : {}),
      ...(family ? { familyIdentifier: family.identifier, familyName: family.name } : {}),
      ...(properties ? { paramSize: properties.size } : {}),
      quantization: quantization?.method ?? null,
      ...(quantization ? { quantizationBits: quantization.bitsPerWeight } : {}),
      reasoning: { kind: "unsupported" },
      supportsTools: false,
      files: files.map((file) => ({ name: file.name, url: file.url, sizeBytes: file.size })),
    };
  } finally {
    for (const file of files) file.free();
    for (const icon of icons) icon.free();
    source?.free();
    accessibility.free();
    metadata?.free();
    vendor?.free();
    quantization?.free();
    properties?.free();
    family?.free();
  }
}

const phaseNames: Record<string, ModelDownloadPhase> = {
  Loading: "Initializing",
  "Not Downloaded": "NotDownloaded",
  Downloading: "Downloading",
  Paused: "Paused",
  Downloaded: "Downloaded",
  Locked: "Locked",
  Error: "Error",
};

export function describeState(state: DownloadState, api: UzuBindings): Omit<ModelDownloadState, "seq"> {
  const phase = state.phase;
  try {
    return {
      phase: phaseNames[state.name] ?? "Initializing",
      ...toDownloadProgressKbytes(state.downloadedBytes, state.totalBytes),
      ...(phase instanceof api.DownloadPhaseError ? { error: phase.message } : {}),
    };
  } finally {
    phase.free();
    state.free();
  }
}
