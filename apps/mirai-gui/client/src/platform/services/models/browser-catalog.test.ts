import { afterEach, expect, it, vi } from "vitest";
import {
  browserModelUnsupportedReason,
  parseRegistry,
  resolveRepository,
  restrictCheckpointCapabilities,
  type BrowserModel,
} from "./browser-catalog";

const digest = "a".repeat(64);
const source = () => ({
  metadatas: [
    {
      id: "vendor",
      name: "Vendor",
      icons: [
        { theme: "dark", format: "png", url: "https://assets.example/vendor.png" },
        { theme: "light", format: "svg", url: "https://assets.example/vendor.svg" },
      ],
    },
    { id: "family", name: "Family" },
    { id: "properties", name: "Tiny" },
    { id: "quant", name: "Mirai M" },
    { id: "model", name: "Full model", description: "Model description" },
  ],
  models: [
    {
      id: "vendor:family:tiny:m4",
      metadata_id: "model",
      family: { id: "vendor:family", metadata_id: "family", vendor: { metadata_id: "vendor" } },
      properties: { metadata_id: "properties", size: 230000000 },
      quantization: { metadata_id: "quant", method: "mirai-m", bits_per_weight: 4 },
      specializations: [{ type: "chat", supports_tools: true, supports_reasoning: true, context_length: 128000 }],
      encodings: [
        { type: "unknown", name: "unused" },
        { type: "hanashi", name: "qwen3.5" },
      ],
      accessibility: {
        type: "local",
        reference: {
          type: "mirai",
          repository: { identifier: "vendor/model", commit_hash: "b".repeat(40) },
          files: ["config.json", "tokenizer.json", "model.safetensors"].map((name) => ({
            name,
            url: `https://assets.example/${name}`,
            size: 100,
            hashes: [{ method: "sha256", value: digest }],
          })),
        },
      },
    },
  ],
});
afterEach(() => vi.unstubAllGlobals());
it("preserves desktop catalog metadata and selects the supported chat encoding", () => {
  expect(parseRegistry(source())[0]).toMatchObject({
    identifier: "vendor:family:tiny:m4",
    repoId: "vendor/model",
    vendor: "Vendor",
    name: "Family Tiny Mirai M",
    familyIdentifier: "vendor:family",
    familyName: "Family",
    paramSize: 230000000,
    vendorIcons: { light: "https://assets.example/vendor.svg", dark: "https://assets.example/vendor.png" },
    quantization: "mirai-m",
    quantizationBits: 4,
    encoding: { type: "hanashi", name: "qwen3.5" },
    supportsTools: true,
    checkpointCapabilities: { reasoning: true, tools: true },
    maximumContextLength: 128000,
    description: "Model description",
    browserUnsupportedReason: undefined,
  });
});
it("preserves explicit checkpoint restrictions without treating missing flags as false", () => {
  const registry = source();
  registry.models[0]!.specializations[0]!.supports_tools = false;
  registry.models[0]!.specializations[0]!.supports_reasoning = false;
  expect(parseRegistry(registry)[0]?.checkpointCapabilities).toEqual({ reasoning: false, tools: false });
  Reflect.deleteProperty(registry.models[0]!.specializations[0]!, "supports_tools");
  Reflect.deleteProperty(registry.models[0]!.specializations[0]!, "supports_reasoning");
  expect(parseRegistry(registry)[0]?.checkpointCapabilities).toEqual({});
});
it("intersects checkpoint and encoding capabilities without inventing unsupported features", () => {
  const encoding = { reasoning: { kind: "alwaysOn" as const }, supportsTools: true };
  expect(restrictCheckpointCapabilities(encoding, { reasoning: false, tools: false })).toEqual({
    reasoning: { kind: "unsupported" },
    supportsTools: false,
  });
  expect(restrictCheckpointCapabilities(encoding)).toEqual(encoding);
  expect(restrictCheckpointCapabilities(encoding, { reasoning: true, tools: true })).toEqual(encoding);
  const unsupported = { reasoning: { kind: "unsupported" as const }, supportsTools: false };
  expect(restrictCheckpointCapabilities(unsupported, { reasoning: true, tools: true })).toEqual(unsupported);
});
it("excludes cloud models and models without chat specialization", () => {
  const registry = source();
  registry.models[0]!.accessibility.type = "remote";
  expect(parseRegistry(registry)).toHaveLength(0);
  registry.models[0]!.accessibility.type = "local";
  registry.models[0]!.specializations[0]!.type = "classification";
  expect(parseRegistry(registry)).toHaveLength(0);
});
it("rejects invalid URLs, traversal paths, and missing metadata", () => {
  const registry = source();
  registry.models[0]!.accessibility.reference.files[0]!.url = "http://insecure.example/config.json";
  expect(() => parseRegistry(registry)).toThrow("HTTPS");
  registry.models[0]!.accessibility.reference.files[0]!.url = "https://assets.example/config.json";
  registry.models[0]!.accessibility.reference.files[0]!.name = "../config.json";
  expect(() => parseRegistry(registry)).toThrow("Invalid model file name");
  registry.models[0]!.accessibility.reference.files[0]!.name = "config.json";
  registry.metadatas = [];
  expect(() => parseRegistry(registry)).toThrow("missing referenced metadata");
});
it("keeps unsupported formats and sizes visible with a concrete reason", () => {
  const model = parseRegistry(source())[0]!;
  model.files[0]!.sizeBytes = 16 * 1024 * 1024 + 1;
  expect(browserModelUnsupportedReason(model)).toContain("configuration");
  model.files = model.files.filter((file) => file.name !== "model.safetensors");
  expect(browserModelUnsupportedReason(model)).toContain("checkpoint format");
});
it("accepts ranged weight sources beyond the former 2 GB limit and 4 GiB offsets", () => {
  const model = parseRegistry(source())[0]!;
  for (const size of [2_000_000_001, 2_441_821_768, 5_189_121_136]) {
    model.files[2]!.sizeBytes = size;
    expect(browserModelUnsupportedReason(model)).toBeUndefined();
  }
});
it("resolves immutable Hugging Face file URLs and their authoritative hashes", async () => {
  const model = parseRegistry(source())[0]!;
  const commit = model.repository!.commit_hash!;
  const fetchMock = vi.fn<(url: string) => Promise<Response>>(async () =>
    Response.json({
      sha: commit,
      siblings: [
        { rfilename: "config.json", size: 100, blobId: "c".repeat(40) },
        { rfilename: "model.safetensors", lfs: { size: 1000, sha256: digest } },
      ],
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  const result = await resolveRepository(model);
  expect(fetchMock.mock.calls[0]?.[0]).toContain(`/revision/${commit}?blobs=true`);
  expect(result.files[1]).toEqual({
    name: "model.safetensors",
    url: `https://huggingface.co/vendor/model/resolve/${commit}/model.safetensors`,
    sizeBytes: 1000,
    hashes: [{ method: "sha256", value: digest }],
  });
  expect(result.files[0]!.hashes[0]?.method).toBe("git_blob_sha1");
});
it("selects only the base checkpoint inputs from registry and repository manifests", async () => {
  const registry = source();
  registry.models[0]!.accessibility.reference.files.push({
    name: "speculator/model.safetensors",
    url: "https://assets.example/speculator/model.safetensors",
    size: 755_000_000,
    hashes: [{ method: "sha256", value: digest }],
  });
  const model = parseRegistry(registry)[0]!;
  expect(model.files.map((file) => file.name)).toEqual(["config.json", "tokenizer.json", "model.safetensors"]);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      Response.json({
        sha: model.repository!.commit_hash,
        siblings: [
          ...model.files.map((file) => ({ rfilename: file.name, size: file.sizeBytes, blobId: "c".repeat(40) })),
          { rfilename: "speculator/model.safetensors", lfs: { size: 755_000_000, sha256: digest } },
          { rfilename: "README.md" },
        ],
      }),
    ),
  );
  expect((await resolveRepository(model)).files.map((file) => file.name)).toEqual([
    "config.json",
    "tokenizer.json",
    "model.safetensors",
  ]);
});
it("rejects a repository that resolves a pinned commit to a different revision", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => Response.json({ sha: "c".repeat(40), siblings: [] })),
  );
  await expect(resolveRepository(parseRegistry(source())[0]!)).rejects.toThrow("different revision");
});
it("reuses a pinned descriptor only when the selected repository paths still match", async () => {
  const model = parseRegistry(source())[0]!;
  const cached: BrowserModel = {
    ...model,
    files: model.files.map((file) => ({ ...file, url: file.url.replace("assets.example", "huggingface.co") })),
  };
  const fetchMock = vi.fn<(url: string) => Promise<Response>>(async () =>
    Response.json({
      sha: model.repository!.commit_hash,
      siblings: [{ rfilename: "config.json", size: 100, blobId: "c".repeat(40) }],
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  expect((await resolveRepository(model, cached)).files).toEqual(cached.files);
  expect(fetchMock).not.toHaveBeenCalled();
  await resolveRepository({ ...model, repository: { ...model.repository!, paths: ["config.json"] } }, cached);
  expect(fetchMock).toHaveBeenCalledOnce();
});
