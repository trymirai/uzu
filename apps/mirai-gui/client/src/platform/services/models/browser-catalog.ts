import type { EngineModel } from "@/types/model-manager";

// Same registry protocol and toolchain as the desktop engine's Mirai registry.
export const REGISTRY_URL = "https://sdk.trymirai.com/api/v1/fetch/models";
export const MODEL_TOOLCHAIN_VERSION = __MODEL_TOOLCHAIN_VERSION__;
export const CATALOG_CACHE_KEY = `https://sdk.trymirai.com/api/v1/browser-catalog-v2?toolchain=${MODEL_TOOLCHAIN_VERSION}`;
export type ModelFileHash = { method: "sha256" | "git_blob_sha1" | "crc32c"; value: string };
export type BrowserModelFile = { name: string; url: string; sizeBytes: number; hashes: ModelFileHash[] };
const inferenceFileNames = new Set(["config.json", "tokenizer.json", "model.safetensors"]);
// The browser runs the base checkpoint. Repository documentation and optional
// speculator checkpoints are not inputs to this session and need no download.
export function browserInferenceFiles(files: BrowserModelFile[]): BrowserModelFile[] {
  return files.filter((file) => inferenceFileNames.has(file.name));
}
type CheckpointCapabilities = { reasoning?: boolean; tools?: boolean };
export type BrowserModel = Omit<EngineModel, "state"> & {
  files: BrowserModelFile[];
  maximumContextLength?: number;
  encoding?: Record<string, unknown>;
  encodingVerified?: boolean;
  checkpointCapabilities?: CheckpointCapabilities;
  repository?: { identifier: string; commit_hash?: string; paths?: string[] };
  description?: string;
};

// An encoding can represent features that a particular checkpoint was not trained
// to use. Explicit registry restrictions take precedence over its capabilities.
export function restrictCheckpointCapabilities(
  support: Pick<BrowserModel, "reasoning" | "supportsTools">,
  checkpoint?: CheckpointCapabilities,
): Pick<BrowserModel, "reasoning" | "supportsTools"> {
  return {
    reasoning: checkpoint?.reasoning === false ? { kind: "unsupported" } : support.reasoning,
    supportsTools: checkpoint?.tools === false ? false : support.supportsTools,
  };
}

type RecordValue = Record<string, unknown>;
const record = (value: unknown): RecordValue => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid model registry response");
  return value as RecordValue;
};
const text = (value: unknown): string => {
  if (typeof value !== "string" || !value) throw new Error("Invalid model registry text");
  return value;
};
const positive = (value: unknown): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error("Invalid model file size");
  return value as number;
};

export function secureModelUrl(value: unknown): string {
  const url = new URL(text(value));
  if (url.protocol !== "https:" || url.username || url.password) throw new Error("Model files must use HTTPS");
  return url.href;
}
export function modelFileName(value: unknown): string {
  const name = text(value);
  if (
    name.includes("\\") ||
    name.includes(":") ||
    name.split("/").some((part) => !part || part === "." || part === "..")
  )
    throw new Error(`Invalid model file name: ${name}`);
  return name;
}
export function parseFile(value: unknown): BrowserModelFile {
  const file = record(value);
  const hashes = (Array.isArray(file.hashes) ? file.hashes : []).flatMap((value): ModelFileHash[] => {
    const hash = record(value);
    if (hash.method === "sha256" || hash.method === "git_blob_sha1") {
      const digest = text(hash.value);
      if (!new RegExp(`^[a-f0-9]{${hash.method === "sha256" ? 64 : 40}}$`).test(digest))
        throw new Error("Invalid model file checksum");
      return [{ method: hash.method, value: digest }];
    }
    if (hash.method === "crc32c") {
      const digest = text(hash.value);
      if (!/^[A-Za-z0-9+/]{6}==$/.test(digest)) throw new Error("Invalid model file checksum");
      return [{ method: "crc32c", value: digest }];
    }
    return [];
  });
  return { name: modelFileName(file.name), url: secureModelUrl(file.url), sizeBytes: positive(file.size), hashes };
}

export function browserModelUnsupportedReason(model: Pick<BrowserModel, "files">): string | undefined {
  for (const name of ["config.json", "tokenizer.json", "model.safetensors"]) {
    if (!model.files.some((file) => file.name === name))
      return "This model's checkpoint format is not supported in the browser yet.";
  }
  if (model.files.some((file) => file.name === "config.json" && file.sizeBytes > 16 * 1024 * 1024))
    return "This model's configuration exceeds the browser engine limit.";
  if (model.files.some((file) => file.name === "tokenizer.json" && file.sizeBytes > 128 * 1024 * 1024))
    return "This model's tokenizer exceeds the browser engine limit.";
  return undefined;
}

export function parseRegistry(value: unknown): BrowserModel[] {
  const response = record(value);
  if (!Array.isArray(response.models) || !Array.isArray(response.metadatas))
    throw new Error("Invalid model registry catalog");
  const metadatas = new Map(
    response.metadatas.map((value) => {
      const metadata = record(value);
      return [text(metadata.id), metadata];
    }),
  );
  const metadata = (value: unknown): RecordValue | undefined => {
    if (value == null) return undefined;
    const reference = record(value);
    const result = metadatas.get(text(reference.metadata_id));
    if (!result) throw new Error("The model registry is missing referenced metadata");
    return result;
  };
  const seen = new Set<string>();
  return response.models.flatMap((value): BrowserModel[] => {
    const model = record(value);
    const access = record(model.accessibility);
    if (
      access.type !== "local" ||
      !Array.isArray(model.specializations) ||
      !model.specializations.some((value) => record(value).type === "chat")
    )
      return [];
    const source = record(access.reference);
    if (source.type !== "mirai") return [];
    const identifier = text(model.id);
    if (seen.has(identifier)) throw new Error("Duplicate model in registry catalog");
    seen.add(identifier);
    const family = model.family == null ? undefined : record(model.family);
    const quantization = model.quantization == null ? undefined : record(model.quantization);
    const properties = model.properties == null ? undefined : record(model.properties);
    const vendor = metadata(family?.vendor ?? quantization?.vendor);
    const familyMetadata = metadata(family);
    const propertiesMetadata = metadata(properties);
    const quantMetadata = metadata(quantization);
    const modelMetadata = typeof model.metadata_id === "string" ? metadatas.get(model.metadata_id) : undefined;
    const repository = source.repository == null ? undefined : record(source.repository);
    const repoId = repository ? text(repository.identifier) : identifier;
    const icons = Array.isArray(vendor?.icons) ? vendor.icons.map(record) : [];
    const icon = (theme: string): string | undefined => {
      const matching = icons.filter((value) => value.theme === theme);
      const found = ["svg", "png", "webp"].flatMap((format) => matching.filter((value) => value.format === format))[0];
      return found ? secureModelUrl(found.url) : undefined;
    };
    const light = icon("light") ?? icon("dark");
    const dark = icon("dark") ?? light;
    const specialization = record(model.specializations.find((value) => record(value).type === "chat"));
    const encodings = Array.isArray(model.encodings) ? model.encodings.map(record) : [];
    const encoding =
      encodings.find((value) => value.type === "hanashi") ?? encodings.find((value) => value.type === "harmony");
    const files = (Array.isArray(source.files) ? source.files : []).map(parseFile);
    if (new Set(files.map((file) => file.name)).size !== files.length)
      throw new Error("Duplicate model file in registry");
    const result: BrowserModel = {
      identifier,
      repoId,
      vendor: typeof vendor?.name === "string" ? vendor.name : "",
      ...(light && dark ? { vendorIcons: { light, dark } } : {}),
      name:
        [familyMetadata?.name, propertiesMetadata?.name, quantMetadata?.name]
          .filter((value) => typeof value === "string" && value)
          .join(" ") || identifier,
      ...(family ? { familyIdentifier: text(family.id), familyName: text(familyMetadata?.name) } : {}),
      ...(properties ? { paramSize: positive(properties.size) } : {}),
      quantization: typeof quantization?.method === "string" ? quantization.method : null,
      ...(quantization ? { quantizationBits: positive(quantization.bits_per_weight) } : {}),
      // Precise choices are supplied by the shared encoding inspector when present.
      reasoning: specialization.supports_reasoning ? { kind: "alwaysOn" } : { kind: "unsupported" },
      supportsTools: specialization.supports_tools === true,
      checkpointCapabilities: {
        ...(typeof specialization.supports_reasoning === "boolean"
          ? { reasoning: specialization.supports_reasoning }
          : {}),
        ...(typeof specialization.supports_tools === "boolean" ? { tools: specialization.supports_tools } : {}),
      },
      ...(typeof specialization.context_length === "number"
        ? { maximumContextLength: specialization.context_length }
        : {}),
      ...(encoding ? { encoding } : {}),
      ...(repository
        ? {
            repository: {
              identifier: repoId,
              ...(typeof repository.commit_hash === "string" ? { commit_hash: repository.commit_hash } : {}),
              ...(Array.isArray(repository.paths) ? { paths: repository.paths.map(text) } : {}),
            },
          }
        : {}),
      ...(typeof modelMetadata?.description === "string" ? { description: modelMetadata.description } : {}),
      files: browserInferenceFiles(files),
    };
    result.browserUnsupportedReason = browserModelUnsupportedReason(result);
    return [result];
  });
}

export async function resolveRepository(model: BrowserModel, previous?: BrowserModel): Promise<BrowserModel> {
  const repository = model.repository;
  if (!repository?.commit_hash) return model;
  const revision = repository.commit_hash;
  if (
    /^[a-f0-9]{40}$/.test(revision) &&
    previous?.repository?.identifier === repository.identifier &&
    previous.repository.commit_hash === revision &&
    JSON.stringify(previous.repository.paths ?? null) === JSON.stringify(repository.paths ?? null) &&
    previous.files.length > 0 &&
    previous.files.every((file) => file.url.startsWith("https://huggingface.co/"))
  ) {
    const files = browserInferenceFiles(previous.files);
    return { ...model, files, browserUnsupportedReason: browserModelUnsupportedReason({ files }) };
  }
  const repoPath = repository.identifier.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(
    `https://huggingface.co/api/models/${repoPath}/revision/${encodeURIComponent(revision)}?blobs=true`,
    { signal: AbortSignal.timeout(30_000) },
  );
  if (!response.ok) throw new Error(`Could not resolve ${repository.identifier} (HTTP ${response.status})`);
  const info = record(await response.json());
  const commit = text(info.sha);
  if (!/^[a-f0-9]{40}$/.test(commit) || (/^[a-f0-9]{40}$/.test(revision) && revision !== commit))
    throw new Error("The model repository returned a different revision");
  if (!Array.isArray(info.siblings)) throw new Error("The model repository returned no files");
  const files = info.siblings
    .map(record)
    .filter((entry) => !repository.paths || repository.paths.includes(text(entry.rfilename)))
    .filter((entry) => inferenceFileNames.has(text(entry.rfilename)))
    .map((entry) => {
      const name = modelFileName(entry.rfilename);
      const lfs = entry.lfs == null ? undefined : record(entry.lfs);
      return parseFile({
        name,
        size: lfs?.size ?? entry.size,
        url: `https://huggingface.co/${repoPath}/resolve/${commit}/${name.split("/").map(encodeURIComponent).join("/")}`,
        hashes: lfs?.sha256
          ? [{ method: "sha256", value: lfs.sha256 }]
          : entry.blobId
            ? [{ method: "git_blob_sha1", value: entry.blobId }]
            : [],
      });
    });
  if (!files.length) throw new Error("The model repository returned no files");
  return {
    ...model,
    repository: { ...repository, commit_hash: commit },
    files,
    browserUnsupportedReason: browserModelUnsupportedReason({ files }),
  };
}

export async function fetchRegistry(): Promise<BrowserModel[]> {
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const response = await fetch(REGISTRY_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(30_000),
    body: JSON.stringify({
      device: { os_name: "browser", cpu_name: null, memory_total: (memory ?? 4) * 1024 ** 3 },
      backends: [{ identifier: "uzu", version: MODEL_TOOLCHAIN_VERSION }],
      include_traces: false,
      show_all: false,
    }),
  });
  if (!response.ok) throw new Error(`Could not load the model catalog (HTTP ${response.status})`);
  return parseRegistry(await response.json());
}
