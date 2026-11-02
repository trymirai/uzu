/* tslint:disable */
/* eslint-disable */
/**
 * The `ReadableStreamType` enum.
 *
 * *This API requires the following crate features to be activated: `ReadableStreamType`*
 */

export type ReadableStreamType = "bytes";
export interface ChatMessageMetadata {
    values: Map<string, Value>;
}

export interface ClassificationOutputProbabilities {
    values: Map<string, number>;
}

export type ChatContentBlock = { type: "identity"; value: string } | { type: "reasoning_effort"; value: ReasoningEffort } | { type: "conversation_start_date"; value: string } | { type: "knowledge_cutoff"; value: string } | { type: "builtin_tools"; names: string[] } | { type: "tools"; namespaces: ToolNamespace[] } | { type: "text"; value: string } | { type: "image"; url: string } | { type: "video"; url: string } | { type: "audio"; url: string } | { type: "file"; url: string } | { type: "reasoning"; value: string } | { type: "tool_call"; value: ToolCall } | { type: "tool_call_candidate"; value: Value; identifier?: string } | { type: "tool_call_result"; id: string | undefined; name: string | undefined; value: Value } | ({ type: "translation" } & { source_language_code: string; target_language_code: string } & TranslationPayload) | { type: "custom"; value: Value };

export type ChatContentBlockType = "Identity" | "ReasoningEffort" | "ConversationStartDate" | "KnowledgeCutoff" | "BuiltinTools" | "Tools" | "Text" | "Image" | "Video" | "Audio" | "File" | "Reasoning" | "ToolCall" | "ToolCallCandidate" | "ToolCallResult" | "Translation" | "Custom";

export type ChatReplyEnergy = { type: "total"; total: number } | { type: "components"; cpu: number; gpu: number; ane: number; dram: number };

export type ChatReplyFinishReason = "Stop" | "Length" | "Cancelled" | "ContextLimitReached" | "ToolCalls" | "Rejected";

export type ChatReplyJoulesPerToken = { type: "total"; total: number } | { type: "components"; cpu: number; gpu: number; ane: number; dram: number };

export type ChatRole = { User: {} } | { Assistant: {} } | { System: {} } | { Developer: {} } | { Tool: {} } | { Custom: { name: string } };

export type ChatSessionState = "Idle" | "Generation" | "ToolCalling" | "Resetting";

export type ChatSessionStreamChunk = { Replies: { replies: ChatReply[] } } | { Error: { error: ChatSessionError } } | { ToolResults: { messages: ChatMessage[] } };

export type ClassificationRole = "User" | "Assistant";

export type ContextLength = { Default: {} } | { Maximal: {} } | { Custom: { length: number } };

export type DecisionAnswer = { type: "choice"; index: number; key: string; probabilities: number[] } | { type: "score"; probabilities: number[]; score: number } | { type: "yes_no"; yes_probability: number };

export type DecisionQuestion = { type: "choice"; instructions: string | undefined; options: DecisionOption[] } | { type: "score"; instructions: string | undefined; levels: string[] } | { type: "yes_no"; instructions: string | undefined; no_description: string | undefined; yes_description: string | undefined };

export type DownloadManagerType = "universal";

export type DownloadPhase = { NotDownloaded: {} } | { Downloading: {} } | { Paused: {} } | { Downloaded: {} } | { Locked: { manager_id: string } } | { Error: { message: string } } | { Initializing: {} };

export type Grammar = { JsonAny: {} } | { JsonSchema: { schema: string } } | { Regex: { pattern: string } };

export type HashMethod = "crc32c" | "sha256" | "git_blob_sha1";

export type ImageFormat = "png" | "webp" | "svg";

export type ImageTheme = "light" | "dark";

export type ModelAccessibility = { type: "local"; reference: ModelSource } | { type: "remote"; repository: Repository | undefined };

export type ModelSource = { type: "mirai"; toolchain_version: string; repository: Repository | undefined; source_repository: Repository | undefined; files: File[] } | { type: "local"; path: string };

export type ModelSpecialization = { type: "chat" } | { type: "classification" } | { type: "decision" } | { type: "text_to_speech" } | { type: "translation" } | { type: "speculation" };

export type ReasoningEffort = "Disabled" | "Default" | "Low" | "Medium" | "High" | "XHigh";

export type SamplingMethod = { Greedy: {} } | { Stochastic: { temperature: number | undefined; top_k: number | undefined; top_p: number | undefined; min_p: number | undefined; repetition_penalty: number | undefined; suffix_repetition_length: number | undefined } };

export type SamplingPolicy = { Default: {} } | { Custom: { method: SamplingMethod } };

export type SamplingSeed = { Default: {} } | { Custom: { seed: number } };

export type ToolDescription = { type: "function"; function: ToolFunction };

export type TranslationPayload = { type: "text"; text: string } | { type: "image"; url: string };


export class CancelToken {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    cancel(): void;
    static create(): CancelToken;
    readonly isCancelled: boolean;
}

export class ChatConfig {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(): ChatConfig;
    withContextLength(context_length: ContextLength): ChatConfig;
    withSamplingSeed(sampling_seed: SamplingSeed): ChatConfig;
    context_length: ContextLength;
    sampling_seed: SamplingSeed;
}

/**
 * A loaded chat backend instance (model weights, tokenizer, configuration).
 * Cloning is cheap and shares the underlying instance, so multiple [`ChatSession`]s
 * can be created from one [`ChatInstance`] without loading the model again.
 */
export class ChatInstance {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly model: Model;
}

export class ChatMessage {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static assistant(): ChatMessage;
    static developer(): ChatMessage;
    static forRole(role: ChatRole): ChatMessage;
    static system(): ChatMessage;
    static tool(): ChatMessage;
    static user(): ChatMessage;
    withBlock(block: ChatContentBlock): ChatMessage;
    withReasoningEffort(reasoning_effort: ReasoningEffort): ChatMessage;
    withReasoning(reasoning: string): ChatMessage;
    withText(text: string): ChatMessage;
    withToolCallCandidate(value: Value): ChatMessage;
    withToolCall(tool_call: ToolCall): ChatMessage;
    withToolNamespaces(tool_namespaces: ToolNamespace[]): ChatMessage;
    readonly reasoningEffort: ReasoningEffort | undefined;
    readonly reasoning: string | undefined;
    readonly text: string | undefined;
    readonly toolCalls: ToolCall[];
    readonly toolNamespaces: ToolNamespace[];
    content: ChatContentBlock[];
    metadata: ChatMessageMetadata;
    role: ChatRole;
}

export class ChatModelCapabilities {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    reasoning_efforts: ReasoningEffort[];
    requires_tools: boolean;
    supports_disable_reasoning: boolean;
    supports_multiple_tool_calls: boolean;
    supports_reasoning: boolean;
    supports_tools: boolean;
}

/**
 * Chat metadata available without loading model weights or allocating a session.
 */
export class ChatModelInfo {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    capabilities: ChatModelCapabilities;
    /**
     * Architectural limit; absent before download or for models without a finite limit.
     */
    get context_length(): number | undefined;
    /**
     * Architectural limit; absent before download or for models without a finite limit.
     */
    set context_length(value: number | null | undefined);
    get default_reasoning_effort(): ReasoningEffort | undefined;
    set default_reasoning_effort(value: ReasoningEffort | null | undefined);
}

export class ChatReply {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get finish_reason(): ChatReplyFinishReason | undefined;
    set finish_reason(value: ChatReplyFinishReason | null | undefined);
    message: ChatMessage;
    stats: ChatReplyStats;
}

export class ChatReplyConfig {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(): ChatReplyConfig;
    withGrammar(grammar?: Grammar | null): ChatReplyConfig;
    withSamplingMethod(sampling_method: SamplingMethod): ChatReplyConfig;
    withSamplingPolicy(sampling_policy: SamplingPolicy): ChatReplyConfig;
    withTokenLimit(token_limit?: number | null): ChatReplyConfig;
    withToolTurnLimit(tool_turn_limit?: number | null): ChatReplyConfig;
    get grammar(): Grammar | undefined;
    set grammar(value: Grammar | null | undefined);
    sampling_policy: SamplingPolicy;
    get token_limit(): number | undefined;
    set token_limit(value: number | null | undefined);
    /**
     * Maximum number of automatic tool-call turns per reply.
     * `None` allows unlimited turns; `Some(0)` prevents automatic tool execution.
     */
    get tool_turn_limit(): number | undefined;
    /**
     * Maximum number of automatic tool-call turns per reply.
     * `None` allows unlimited turns; `Some(0)` prevents automatic tool execution.
     */
    set tool_turn_limit(value: number | null | undefined);
}

export class ChatReplySpeculatorStats {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    num_decode_forward_passes: number;
    tokens_per_forward_pass: number;
}

export class ChatReplyStats {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly inputJoulesPerToken: ChatReplyJoulesPerToken | undefined;
    readonly outputJoulesPerToken: ChatReplyJoulesPerToken | undefined;
    readonly tokensCount: number | undefined;
    readonly totalJoules: number | undefined;
    duration: number;
    get generate_tokens_per_second(): number | undefined;
    set generate_tokens_per_second(value: number | null | undefined);
    get input_energy(): ChatReplyEnergy | undefined;
    set input_energy(value: ChatReplyEnergy | null | undefined);
    get memory_used_bytes(): bigint | undefined;
    set memory_used_bytes(value: bigint | null | undefined);
    get output_energy(): ChatReplyEnergy | undefined;
    set output_energy(value: ChatReplyEnergy | null | undefined);
    get prefill_tokens_per_second(): number | undefined;
    set prefill_tokens_per_second(value: number | null | undefined);
    get speculator_stats(): ChatReplySpeculatorStats | undefined;
    set speculator_stats(value: ChatReplySpeculatorStats | null | undefined);
    get time_to_first_token(): number | undefined;
    set time_to_first_token(value: number | null | undefined);
    get tokens_count_input_cached(): number | undefined;
    set tokens_count_input_cached(value: number | null | undefined);
    get tokens_count_input(): number | undefined;
    set tokens_count_input(value: number | null | undefined);
    get tokens_count_output(): number | undefined;
    set tokens_count_output(value: number | null | undefined);
}

export class ChatSession {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    addTool(tool: NativeTool): Promise<void>;
    /**
     * Consumes the tool handles; the session retains their callbacks.
     */
    addTools(tools: NativeTool[]): Promise<void>;
    replyWithStream(input: ChatMessage[], config: ChatReplyConfig): Promise<ChatSessionStream>;
    reply(input: ChatMessage[], config: ChatReplyConfig): Promise<ChatReply[]>;
    reset(): Promise<void>;
    readonly messages: Promise<ChatMessage[]>;
    readonly samplingDefaults: Promise<SamplingParameters | undefined>;
    readonly state: Promise<ChatSessionState>;
    readonly supportsToolCalls: Promise<boolean>;
}

export class ChatSessionStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    next(): Promise<ChatSessionStreamChunk | undefined>;
    readonly cancelToken: CancelToken;
}

export class ClassificationLabel {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    description: string;
    index: bigint;
    label: string;
    recommended_threshold: number;
    title: string;
}

export class ClassificationMessage {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static assistant(content: string): ClassificationMessage;
    static user(content: string): ClassificationMessage;
    content: string;
    role: ClassificationRole;
}

export class ClassificationOutput {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    logits: Float64Array;
    probabilities: ClassificationOutputProbabilities;
    stats: ClassificationStats;
}

export class ClassificationStats {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    confidence: number;
    forward_pass_duration: number;
    postprocessing_duration: number;
    predicted_label: string;
    preprocessing_duration: number;
    tokens_count: bigint;
    tokens_per_second: number;
    total_duration: number;
}

export class DecisionInput {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    questions: DecisionQuestion[];
    state: string;
}

export class DecisionOption {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get description(): string | undefined;
    set description(value: string | null | undefined);
    key: string;
}

export class DecisionOutput {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    results: DecisionResult[];
}

export class DecisionResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    answer: DecisionAnswer;
    stats: DecisionStats;
}

export class DecisionStats {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    tokens_count: bigint;
    total_duration: number;
}

export class Device {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(): Device;
    get cpu_name(): string | undefined;
    set cpu_name(value: string | null | undefined);
    home_path: string;
    get memory_total(): bigint | undefined;
    set memory_total(value: bigint | null | undefined);
    get os_name(): string | undefined;
    set os_name(value: string | null | undefined);
}

export class DownloadState {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly canDelete: boolean;
    readonly canPause: boolean;
    readonly isInProgress: boolean;
    readonly name: string;
    readonly progress: number;
    downloaded_bytes: bigint;
    phase: DownloadPhase;
    total_bytes: bigint;
}

export class Downloader {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    delete(): Promise<void>;
    pause(): Promise<void>;
    progress(): Promise<DownloaderStream>;
    resume(): Promise<void>;
    readonly state: Promise<DownloadState | undefined>;
}

export class DownloaderStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    next(): Promise<DownloadState | undefined>;
}

export class Engine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    chatInstance(model: Model, config: ChatConfig): Promise<ChatInstance>;
    /**
     * Returns metadata for a Uzu chat model, including models not yet downloaded.
     * Encoding capabilities come from the catalog; the context limit needs local configuration.
     * Remote and non-chat models return `None`.
     */
    chatModelInfo(model: Model): Promise<ChatModelInfo | undefined>;
    chatWithInstance(instance: ChatInstance): Promise<ChatSession>;
    chat(model: Model, config: ChatConfig): Promise<ChatSession>;
    static create(config: EngineConfig): Promise<Engine>;
    downloadState(model: Model): Promise<DownloadState | undefined>;
    download(model: Model): Promise<DownloaderStream>;
    downloader(model: Model): Downloader;
    modelByIdentifier(identifier: string): Promise<Model | undefined>;
    modelByPath(path: string): Promise<Model | undefined>;
    modelByRepoId(repo_id: string): Promise<Model | undefined>;
    modelPath(model: Model): Promise<string | undefined>;
    model(identifier: string): Promise<Model | undefined>;
    modelsByFamily(family_identifier: string): Promise<Model[]>;
    modelsByVendor(vendor_identifier: string): Promise<Model[]>;
    /**
     * Refresh registries and wait until the catalog has been published to storage.
     * Registries that fail to refresh retain their last known entries.
     */
    refreshModels(): Promise<Model[]>;
    registerCallback(callback: EngineCallback): Promise<void>;
    removeBackend(identifier: string): Promise<void>;
    removeRegistry(registry_identifier: string): Promise<void>;
    readonly downloadStates: Promise<Map<any, any>>;
    readonly modelFamilies: Promise<ModelFamily[]>;
    readonly modelRegistries: Promise<ModelRegistry[]>;
    readonly modelVendors: Promise<ModelVendor[]>;
    readonly modelsDownloadable: Promise<Model[]>;
    readonly modelsForChat: Promise<Model[]>;
    readonly modelsForClassification: Promise<Model[]>;
    readonly modelsForSpeculation: Promise<Model[]>;
    readonly modelsForTextToSpeech: Promise<Model[]>;
    readonly modelsForTranslation: Promise<Model[]>;
    readonly modelsOnDevice: Promise<Model[]>;
    readonly modelsRemote: Promise<Model[]>;
    readonly models: Promise<Model[]>;
}

export class EngineCallback {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(callback: Function): EngineCallback;
}

export class EngineConfig {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(): EngineConfig;
    withAllowLmstudioUsage(allow_lmstudio_usage: boolean): EngineConfig;
    withAllowOllamaUsage(allow_ollama_usage: boolean): EngineConfig;
    withAnthropicApiKey(anthropic_api_key: string): EngineConfig;
    withBasetenApiKey(baseten_api_key: string): EngineConfig;
    withDownloadManagerType(download_manager_type: DownloadManagerType): EngineConfig;
    withGeminiApiKey(gemini_api_key: string): EngineConfig;
    withHuggingfaceApiKey(huggingface_api_key: string): EngineConfig;
    withLocalPath(local_path: string): EngineConfig;
    withMiraiApiKey(mirai_api_key: string): EngineConfig;
    withOpenaiApiKey(openai_api_key: string): EngineConfig;
    withOpenrouterApiKey(openrouter_api_key: string): EngineConfig;
    withXaiApiKey(xai_api_key: string): EngineConfig;
    allow_lmstudio_usage: boolean;
    allow_ollama_usage: boolean;
    get anthropic_api_key(): string | undefined;
    set anthropic_api_key(value: string | null | undefined);
    get baseten_api_key(): string | undefined;
    set baseten_api_key(value: string | null | undefined);
    download_manager_type: DownloadManagerType;
    get gemini_api_key(): string | undefined;
    set gemini_api_key(value: string | null | undefined);
    get huggingface_api_key(): string | undefined;
    set huggingface_api_key(value: string | null | undefined);
    get local_path(): string | undefined;
    set local_path(value: string | null | undefined);
    get mirai_api_key(): string | undefined;
    set mirai_api_key(value: string | null | undefined);
    get openai_api_key(): string | undefined;
    set openai_api_key(value: string | null | undefined);
    get openrouter_api_key(): string | undefined;
    set openrouter_api_key(value: string | null | undefined);
    get xai_api_key(): string | undefined;
    set xai_api_key(value: string | null | undefined);
}

export class Feature {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    name: string;
    values: string[];
}

export class File {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly crc32c: string | undefined;
    hashes: Hash[];
    name: string;
    size: bigint;
    url: string;
}

export class Hash {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    method: HashMethod;
    value: string;
}

export class Image {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get description(): string | undefined;
    set description(value: string | null | undefined);
    format: ImageFormat;
    get height(): number | undefined;
    set height(value: number | null | undefined);
    id: string;
    theme: ImageTheme;
    url: string;
    get width(): number | undefined;
    set width(value: number | null | undefined);
}

export class IntoUnderlyingByteSource {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    cancel(): void;
    pull(controller: ReadableByteStreamController): Promise<any>;
    start(controller: ReadableByteStreamController): void;
    readonly autoAllocateChunkSize: number;
    readonly type: ReadableStreamType;
}

export class IntoUnderlyingSink {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    abort(reason: any): Promise<any>;
    close(): Promise<any>;
    write(chunk: any): Promise<any>;
}

export class IntoUnderlyingSource {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    cancel(): void;
    pull(controller: ReadableStreamDefaultController): Promise<any>;
}

export class Metadata {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static external(name: string): Metadata;
    get description(): string | undefined;
    set description(value: string | null | undefined);
    icons: Image[];
    identifier: string;
    name: string;
}

export class Model {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static external(identifier: string, registry_identifier: string, registry_name: string, backend_identifier: string, backend_name: string, backend_version: string, specializations: ModelSpecialization[], accessibility: ModelAccessibility, encoding?: Value | null): Model;
    accessibility: ModelAccessibility;
    backends: ModelBackend[];
    get encoding(): Value | undefined;
    set encoding(value: Value | null | undefined);
    get family(): ModelFamily | undefined;
    set family(value: ModelFamily | null | undefined);
    identifier: string;
    get properties(): ModelProperties | undefined;
    set properties(value: ModelProperties | null | undefined);
    get quantization(): ModelQuantization | undefined;
    set quantization(value: ModelQuantization | null | undefined);
    registry: ModelRegistry;
    specializations: ModelSpecialization[];
    readonly cacheIdentifier: string;
    readonly checkpointVersion: string | undefined;
    readonly filesystemPath: string | undefined;
    readonly isChatCapable: boolean;
    readonly isClassificationCapable: boolean;
    readonly isDownloadable: boolean;
    readonly isOnDevice: boolean;
    readonly isQuantized: boolean;
    readonly isRemote: boolean;
    readonly isSpeculationCapable: boolean;
    readonly isTextToSpeechCapable: boolean;
    readonly isTranslationCapable: boolean;
    readonly name: string;
    readonly repoIds: string[];
}

export class ModelBackend {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    identifier: string;
    metadata: Metadata;
    version: string;
    readonly name: string;
}

export class ModelFamily {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    identifier: string;
    metadata: Metadata;
    vendor: ModelVendor;
    readonly name: string;
}

export class ModelProperties {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    identifier: string;
    metadata: Metadata;
    size: bigint;
    get version(): string | undefined;
    set version(value: string | null | undefined);
    readonly name: string;
}

export class ModelQuantization {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    bits_per_weight: number;
    identifier: string;
    metadata: Metadata;
    method: string;
    vendor: ModelVendor;
    readonly name: string;
}

export class ModelRegistry {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    identifier: string;
    metadata: Metadata;
    readonly name: string;
}

export class ModelVendor {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    identifier: string;
    metadata: Metadata;
    readonly name: string;
}

export class NativeTool {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The definition uses the serialized ToolFunction shape. The callback receives
     * JSON arguments and an invocation ID, and returns JSON or a promise of JSON.
     */
    constructor(definition_json: string, invoke_json: Function, cancel: Function);
}

export class PcmBatch {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    saveAsWav(path: string): void;
    channels: number;
    lengths: Uint32Array;
    sample_rate: number;
    samples: Float64Array;
    readonly batchSize: number;
    readonly duration: number;
    readonly totalFrames: number;
}

export class Repository {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get commit_hash(): string | undefined;
    set commit_hash(value: string | null | undefined);
    identifier: string;
    get paths(): string[] | undefined;
    set paths(value: string[] | null | undefined);
}

export class SamplingParameters {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get min_p(): number | undefined;
    set min_p(value: number | null | undefined);
    get repetition_penalty(): number | undefined;
    set repetition_penalty(value: number | null | undefined);
    get suffix_repetition_length(): bigint | undefined;
    set suffix_repetition_length(value: bigint | null | undefined);
    get temperature(): number | undefined;
    set temperature(value: number | null | undefined);
    get top_k(): bigint | undefined;
    set top_k(value: bigint | null | undefined);
    get top_p(): number | undefined;
    set top_p(value: number | null | undefined);
}

export class TextToSpeechOutput {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    pcm_batch: PcmBatch;
    stats: TextToSpeechStats;
}

export class TextToSpeechStats {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    audio_duration: number;
    first_chunk_seconds: number;
    generation_duration: number;
    text_length: number;
}

export class Token {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    id: number;
    is_special: boolean;
    value: string;
}

export class ToolCall {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    arguments: Value;
    get identifier(): string | undefined;
    set identifier(value: string | null | undefined);
    name: string;
}

export class ToolCallArgumentDelta {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    arguments: string;
    name: string;
}

/**
 * Converts model tool-call candidates into append-only JSON argument fragments.
 */
export class ToolCallArgumentStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(tools: ToolNamespace[]): ToolCallArgumentStream;
    finish(call: ToolCall): ToolCallArgumentDelta;
    update(raw: string): ToolCallArgumentDelta;
}

export class ToolFunction {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    description: string;
    name: string;
    get parameters(): Value | undefined;
    set parameters(value: Value | null | undefined);
    get return_definition(): Value | undefined;
    set return_definition(value: Value | null | undefined);
}

export class ToolNamespace {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get description(): string | undefined;
    set description(value: string | null | undefined);
    name: string;
    tools: ToolDescription[];
}

export class Value {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    json: string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_canceltoken_free: (a: number, b: number) => void;
    readonly __wbg_chatconfig_free: (a: number, b: number) => void;
    readonly __wbg_chatinstance_free: (a: number, b: number) => void;
    readonly __wbg_chatmessage_free: (a: number, b: number) => void;
    readonly __wbg_chatmodelcapabilities_free: (a: number, b: number) => void;
    readonly __wbg_chatmodelinfo_free: (a: number, b: number) => void;
    readonly __wbg_chatreply_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyconfig_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyspeculatorstats_free: (a: number, b: number) => void;
    readonly __wbg_chatreplystats_free: (a: number, b: number) => void;
    readonly __wbg_chatsession_free: (a: number, b: number) => void;
    readonly __wbg_chatsessionstream_free: (a: number, b: number) => void;
    readonly __wbg_classificationlabel_free: (a: number, b: number) => void;
    readonly __wbg_classificationmessage_free: (a: number, b: number) => void;
    readonly __wbg_classificationoutput_free: (a: number, b: number) => void;
    readonly __wbg_classificationstats_free: (a: number, b: number) => void;
    readonly __wbg_decisioninput_free: (a: number, b: number) => void;
    readonly __wbg_decisionoption_free: (a: number, b: number) => void;
    readonly __wbg_decisionoutput_free: (a: number, b: number) => void;
    readonly __wbg_decisionresult_free: (a: number, b: number) => void;
    readonly __wbg_decisionstats_free: (a: number, b: number) => void;
    readonly __wbg_device_free: (a: number, b: number) => void;
    readonly __wbg_downloader_free: (a: number, b: number) => void;
    readonly __wbg_downloaderstream_free: (a: number, b: number) => void;
    readonly __wbg_downloadstate_free: (a: number, b: number) => void;
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly __wbg_enginecallback_free: (a: number, b: number) => void;
    readonly __wbg_engineconfig_free: (a: number, b: number) => void;
    readonly __wbg_feature_free: (a: number, b: number) => void;
    readonly __wbg_file_free: (a: number, b: number) => void;
    readonly __wbg_get_chatconfig_context_length: (a: number) => any;
    readonly __wbg_get_chatconfig_sampling_seed: (a: number) => any;
    readonly __wbg_get_chatmessage_content: (a: number) => [number, number];
    readonly __wbg_get_chatmessage_metadata: (a: number) => any;
    readonly __wbg_get_chatmessage_role: (a: number) => any;
    readonly __wbg_get_chatmodelcapabilities_reasoning_efforts: (a: number) => [number, number];
    readonly __wbg_get_chatmodelcapabilities_requires_tools: (a: number) => number;
    readonly __wbg_get_chatmodelcapabilities_supports_disable_reasoning: (a: number) => number;
    readonly __wbg_get_chatmodelcapabilities_supports_multiple_tool_calls: (a: number) => number;
    readonly __wbg_get_chatmodelcapabilities_supports_reasoning: (a: number) => number;
    readonly __wbg_get_chatmodelcapabilities_supports_tools: (a: number) => number;
    readonly __wbg_get_chatmodelinfo_capabilities: (a: number) => number;
    readonly __wbg_get_chatmodelinfo_context_length: (a: number) => number;
    readonly __wbg_get_chatmodelinfo_default_reasoning_effort: (a: number) => any;
    readonly __wbg_get_chatreply_finish_reason: (a: number) => any;
    readonly __wbg_get_chatreply_message: (a: number) => number;
    readonly __wbg_get_chatreply_stats: (a: number) => number;
    readonly __wbg_get_chatreplyconfig_grammar: (a: number) => any;
    readonly __wbg_get_chatreplyconfig_sampling_policy: (a: number) => any;
    readonly __wbg_get_chatreplyconfig_token_limit: (a: number) => number;
    readonly __wbg_get_chatreplyconfig_tool_turn_limit: (a: number) => number;
    readonly __wbg_get_chatreplyspeculatorstats_num_decode_forward_passes: (a: number) => number;
    readonly __wbg_get_chatreplyspeculatorstats_tokens_per_forward_pass: (a: number) => number;
    readonly __wbg_get_chatreplystats_duration: (a: number) => number;
    readonly __wbg_get_chatreplystats_generate_tokens_per_second: (a: number) => [number, number];
    readonly __wbg_get_chatreplystats_input_energy: (a: number) => any;
    readonly __wbg_get_chatreplystats_memory_used_bytes: (a: number) => [number, bigint];
    readonly __wbg_get_chatreplystats_output_energy: (a: number) => any;
    readonly __wbg_get_chatreplystats_prefill_tokens_per_second: (a: number) => [number, number];
    readonly __wbg_get_chatreplystats_speculator_stats: (a: number) => number;
    readonly __wbg_get_chatreplystats_time_to_first_token: (a: number) => [number, number];
    readonly __wbg_get_chatreplystats_tokens_count_input: (a: number) => number;
    readonly __wbg_get_chatreplystats_tokens_count_input_cached: (a: number) => number;
    readonly __wbg_get_chatreplystats_tokens_count_output: (a: number) => number;
    readonly __wbg_get_classificationlabel_description: (a: number) => [number, number];
    readonly __wbg_get_classificationlabel_index: (a: number) => bigint;
    readonly __wbg_get_classificationlabel_label: (a: number) => [number, number];
    readonly __wbg_get_classificationlabel_recommended_threshold: (a: number) => number;
    readonly __wbg_get_classificationlabel_title: (a: number) => [number, number];
    readonly __wbg_get_classificationmessage_content: (a: number) => [number, number];
    readonly __wbg_get_classificationmessage_role: (a: number) => any;
    readonly __wbg_get_classificationoutput_logits: (a: number) => [number, number];
    readonly __wbg_get_classificationoutput_probabilities: (a: number) => any;
    readonly __wbg_get_classificationoutput_stats: (a: number) => number;
    readonly __wbg_get_classificationstats_confidence: (a: number) => number;
    readonly __wbg_get_classificationstats_forward_pass_duration: (a: number) => number;
    readonly __wbg_get_classificationstats_postprocessing_duration: (a: number) => number;
    readonly __wbg_get_classificationstats_predicted_label: (a: number) => [number, number];
    readonly __wbg_get_classificationstats_preprocessing_duration: (a: number) => number;
    readonly __wbg_get_classificationstats_tokens_count: (a: number) => bigint;
    readonly __wbg_get_classificationstats_tokens_per_second: (a: number) => number;
    readonly __wbg_get_classificationstats_total_duration: (a: number) => number;
    readonly __wbg_get_decisioninput_questions: (a: number) => [number, number];
    readonly __wbg_get_decisioninput_state: (a: number) => [number, number];
    readonly __wbg_get_decisionoption_description: (a: number) => [number, number];
    readonly __wbg_get_decisionoption_key: (a: number) => [number, number];
    readonly __wbg_get_decisionoutput_results: (a: number) => [number, number];
    readonly __wbg_get_decisionresult_answer: (a: number) => any;
    readonly __wbg_get_decisionresult_stats: (a: number) => number;
    readonly __wbg_get_decisionstats_tokens_count: (a: number) => bigint;
    readonly __wbg_get_decisionstats_total_duration: (a: number) => number;
    readonly __wbg_get_device_cpu_name: (a: number) => [number, number];
    readonly __wbg_get_device_home_path: (a: number) => [number, number];
    readonly __wbg_get_device_memory_total: (a: number) => [number, bigint];
    readonly __wbg_get_device_os_name: (a: number) => [number, number];
    readonly __wbg_get_downloadstate_downloaded_bytes: (a: number) => bigint;
    readonly __wbg_get_downloadstate_phase: (a: number) => any;
    readonly __wbg_get_downloadstate_total_bytes: (a: number) => bigint;
    readonly __wbg_get_engineconfig_allow_lmstudio_usage: (a: number) => number;
    readonly __wbg_get_engineconfig_allow_ollama_usage: (a: number) => number;
    readonly __wbg_get_engineconfig_anthropic_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_baseten_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_download_manager_type: (a: number) => any;
    readonly __wbg_get_engineconfig_gemini_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_huggingface_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_local_path: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_mirai_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_openai_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_openrouter_api_key: (a: number) => [number, number];
    readonly __wbg_get_engineconfig_xai_api_key: (a: number) => [number, number];
    readonly __wbg_get_feature_name: (a: number) => [number, number];
    readonly __wbg_get_feature_values: (a: number) => [number, number];
    readonly __wbg_get_file_hashes: (a: number) => [number, number];
    readonly __wbg_get_file_name: (a: number) => [number, number];
    readonly __wbg_get_file_size: (a: number) => bigint;
    readonly __wbg_get_file_url: (a: number) => [number, number];
    readonly __wbg_get_hash_method: (a: number) => any;
    readonly __wbg_get_hash_value: (a: number) => [number, number];
    readonly __wbg_get_image_description: (a: number) => [number, number];
    readonly __wbg_get_image_format: (a: number) => any;
    readonly __wbg_get_image_height: (a: number) => number;
    readonly __wbg_get_image_id: (a: number) => [number, number];
    readonly __wbg_get_image_theme: (a: number) => any;
    readonly __wbg_get_image_url: (a: number) => [number, number];
    readonly __wbg_get_image_width: (a: number) => number;
    readonly __wbg_get_metadata_description: (a: number) => [number, number];
    readonly __wbg_get_metadata_icons: (a: number) => [number, number];
    readonly __wbg_get_metadata_identifier: (a: number) => [number, number];
    readonly __wbg_get_metadata_name: (a: number) => [number, number];
    readonly __wbg_get_model_accessibility: (a: number) => any;
    readonly __wbg_get_model_backends: (a: number) => [number, number];
    readonly __wbg_get_model_encoding: (a: number) => number;
    readonly __wbg_get_model_family: (a: number) => number;
    readonly __wbg_get_model_identifier: (a: number) => [number, number];
    readonly __wbg_get_model_properties: (a: number) => number;
    readonly __wbg_get_model_quantization: (a: number) => number;
    readonly __wbg_get_model_registry: (a: number) => number;
    readonly __wbg_get_model_specializations: (a: number) => [number, number];
    readonly __wbg_get_modelbackend_identifier: (a: number) => [number, number];
    readonly __wbg_get_modelbackend_metadata: (a: number) => number;
    readonly __wbg_get_modelbackend_version: (a: number) => [number, number];
    readonly __wbg_get_modelfamily_identifier: (a: number) => [number, number];
    readonly __wbg_get_modelfamily_metadata: (a: number) => number;
    readonly __wbg_get_modelfamily_vendor: (a: number) => number;
    readonly __wbg_get_modelproperties_identifier: (a: number) => [number, number];
    readonly __wbg_get_modelproperties_metadata: (a: number) => number;
    readonly __wbg_get_modelproperties_size: (a: number) => bigint;
    readonly __wbg_get_modelproperties_version: (a: number) => [number, number];
    readonly __wbg_get_modelquantization_bits_per_weight: (a: number) => number;
    readonly __wbg_get_modelquantization_identifier: (a: number) => [number, number];
    readonly __wbg_get_modelquantization_metadata: (a: number) => number;
    readonly __wbg_get_modelquantization_method: (a: number) => [number, number];
    readonly __wbg_get_modelquantization_vendor: (a: number) => number;
    readonly __wbg_get_modelregistry_identifier: (a: number) => [number, number];
    readonly __wbg_get_modelregistry_metadata: (a: number) => number;
    readonly __wbg_get_modelvendor_identifier: (a: number) => [number, number];
    readonly __wbg_get_modelvendor_metadata: (a: number) => number;
    readonly __wbg_get_pcmbatch_channels: (a: number) => number;
    readonly __wbg_get_pcmbatch_lengths: (a: number) => [number, number];
    readonly __wbg_get_pcmbatch_sample_rate: (a: number) => number;
    readonly __wbg_get_pcmbatch_samples: (a: number) => [number, number];
    readonly __wbg_get_repository_commit_hash: (a: number) => [number, number];
    readonly __wbg_get_repository_identifier: (a: number) => [number, number];
    readonly __wbg_get_repository_paths: (a: number) => [number, number];
    readonly __wbg_get_samplingparameters_min_p: (a: number) => [number, number];
    readonly __wbg_get_samplingparameters_repetition_penalty: (a: number) => [number, number];
    readonly __wbg_get_samplingparameters_suffix_repetition_length: (a: number) => [number, bigint];
    readonly __wbg_get_samplingparameters_temperature: (a: number) => [number, number];
    readonly __wbg_get_samplingparameters_top_k: (a: number) => [number, bigint];
    readonly __wbg_get_samplingparameters_top_p: (a: number) => [number, number];
    readonly __wbg_get_texttospeechoutput_pcm_batch: (a: number) => number;
    readonly __wbg_get_texttospeechoutput_stats: (a: number) => number;
    readonly __wbg_get_texttospeechstats_audio_duration: (a: number) => number;
    readonly __wbg_get_texttospeechstats_first_chunk_seconds: (a: number) => number;
    readonly __wbg_get_texttospeechstats_generation_duration: (a: number) => number;
    readonly __wbg_get_texttospeechstats_text_length: (a: number) => number;
    readonly __wbg_get_token_id: (a: number) => number;
    readonly __wbg_get_token_is_special: (a: number) => number;
    readonly __wbg_get_token_value: (a: number) => [number, number];
    readonly __wbg_get_toolcall_arguments: (a: number) => number;
    readonly __wbg_get_toolcall_identifier: (a: number) => [number, number];
    readonly __wbg_get_toolcall_name: (a: number) => [number, number];
    readonly __wbg_get_toolcallargumentdelta_arguments: (a: number) => [number, number];
    readonly __wbg_get_toolcallargumentdelta_name: (a: number) => [number, number];
    readonly __wbg_get_toolfunction_description: (a: number) => [number, number];
    readonly __wbg_get_toolfunction_name: (a: number) => [number, number];
    readonly __wbg_get_toolfunction_parameters: (a: number) => number;
    readonly __wbg_get_toolfunction_return_definition: (a: number) => number;
    readonly __wbg_get_toolnamespace_description: (a: number) => [number, number];
    readonly __wbg_get_toolnamespace_name: (a: number) => [number, number];
    readonly __wbg_get_toolnamespace_tools: (a: number) => [number, number];
    readonly __wbg_get_value_json: (a: number) => [number, number];
    readonly __wbg_hash_free: (a: number, b: number) => void;
    readonly __wbg_image_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingbytesource_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingsink_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingsource_free: (a: number, b: number) => void;
    readonly __wbg_metadata_free: (a: number, b: number) => void;
    readonly __wbg_model_free: (a: number, b: number) => void;
    readonly __wbg_modelbackend_free: (a: number, b: number) => void;
    readonly __wbg_modelfamily_free: (a: number, b: number) => void;
    readonly __wbg_modelproperties_free: (a: number, b: number) => void;
    readonly __wbg_modelquantization_free: (a: number, b: number) => void;
    readonly __wbg_modelregistry_free: (a: number, b: number) => void;
    readonly __wbg_modelvendor_free: (a: number, b: number) => void;
    readonly __wbg_nativetool_free: (a: number, b: number) => void;
    readonly __wbg_pcmbatch_free: (a: number, b: number) => void;
    readonly __wbg_repository_free: (a: number, b: number) => void;
    readonly __wbg_samplingparameters_free: (a: number, b: number) => void;
    readonly __wbg_set_chatconfig_context_length: (a: number, b: any) => void;
    readonly __wbg_set_chatconfig_sampling_seed: (a: number, b: any) => void;
    readonly __wbg_set_chatmessage_content: (a: number, b: number, c: number) => void;
    readonly __wbg_set_chatmessage_metadata: (a: number, b: any) => void;
    readonly __wbg_set_chatmessage_role: (a: number, b: any) => void;
    readonly __wbg_set_chatmodelcapabilities_reasoning_efforts: (a: number, b: number, c: number) => void;
    readonly __wbg_set_chatmodelcapabilities_requires_tools: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelcapabilities_supports_disable_reasoning: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelcapabilities_supports_multiple_tool_calls: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelcapabilities_supports_reasoning: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelcapabilities_supports_tools: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelinfo_capabilities: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelinfo_context_length: (a: number, b: number) => void;
    readonly __wbg_set_chatmodelinfo_default_reasoning_effort: (a: number, b: number) => void;
    readonly __wbg_set_chatreply_finish_reason: (a: number, b: number) => void;
    readonly __wbg_set_chatreply_message: (a: number, b: number) => void;
    readonly __wbg_set_chatreply_stats: (a: number, b: number) => void;
    readonly __wbg_set_chatreplyconfig_grammar: (a: number, b: number) => void;
    readonly __wbg_set_chatreplyconfig_sampling_policy: (a: number, b: any) => void;
    readonly __wbg_set_chatreplyconfig_token_limit: (a: number, b: number) => void;
    readonly __wbg_set_chatreplyconfig_tool_turn_limit: (a: number, b: number) => void;
    readonly __wbg_set_chatreplyspeculatorstats_num_decode_forward_passes: (a: number, b: number) => void;
    readonly __wbg_set_chatreplyspeculatorstats_tokens_per_forward_pass: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_duration: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_generate_tokens_per_second: (a: number, b: number, c: number) => void;
    readonly __wbg_set_chatreplystats_input_energy: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_memory_used_bytes: (a: number, b: number, c: bigint) => void;
    readonly __wbg_set_chatreplystats_output_energy: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_prefill_tokens_per_second: (a: number, b: number, c: number) => void;
    readonly __wbg_set_chatreplystats_speculator_stats: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_time_to_first_token: (a: number, b: number, c: number) => void;
    readonly __wbg_set_chatreplystats_tokens_count_input: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_tokens_count_input_cached: (a: number, b: number) => void;
    readonly __wbg_set_chatreplystats_tokens_count_output: (a: number, b: number) => void;
    readonly __wbg_set_classificationlabel_description: (a: number, b: number, c: number) => void;
    readonly __wbg_set_classificationlabel_index: (a: number, b: bigint) => void;
    readonly __wbg_set_classificationlabel_label: (a: number, b: number, c: number) => void;
    readonly __wbg_set_classificationlabel_recommended_threshold: (a: number, b: number) => void;
    readonly __wbg_set_classificationlabel_title: (a: number, b: number, c: number) => void;
    readonly __wbg_set_classificationmessage_content: (a: number, b: number, c: number) => void;
    readonly __wbg_set_classificationmessage_role: (a: number, b: any) => void;
    readonly __wbg_set_classificationoutput_logits: (a: number, b: number, c: number) => void;
    readonly __wbg_set_classificationoutput_probabilities: (a: number, b: any) => void;
    readonly __wbg_set_classificationoutput_stats: (a: number, b: number) => void;
    readonly __wbg_set_classificationstats_confidence: (a: number, b: number) => void;
    readonly __wbg_set_classificationstats_forward_pass_duration: (a: number, b: number) => void;
    readonly __wbg_set_classificationstats_postprocessing_duration: (a: number, b: number) => void;
    readonly __wbg_set_classificationstats_predicted_label: (a: number, b: number, c: number) => void;
    readonly __wbg_set_classificationstats_preprocessing_duration: (a: number, b: number) => void;
    readonly __wbg_set_classificationstats_tokens_count: (a: number, b: bigint) => void;
    readonly __wbg_set_classificationstats_tokens_per_second: (a: number, b: number) => void;
    readonly __wbg_set_classificationstats_total_duration: (a: number, b: number) => void;
    readonly __wbg_set_decisioninput_questions: (a: number, b: number, c: number) => void;
    readonly __wbg_set_decisioninput_state: (a: number, b: number, c: number) => void;
    readonly __wbg_set_decisionoption_description: (a: number, b: number, c: number) => void;
    readonly __wbg_set_decisionoption_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_decisionoutput_results: (a: number, b: number, c: number) => void;
    readonly __wbg_set_decisionresult_answer: (a: number, b: any) => void;
    readonly __wbg_set_decisionresult_stats: (a: number, b: number) => void;
    readonly __wbg_set_decisionstats_tokens_count: (a: number, b: bigint) => void;
    readonly __wbg_set_decisionstats_total_duration: (a: number, b: number) => void;
    readonly __wbg_set_device_cpu_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_device_home_path: (a: number, b: number, c: number) => void;
    readonly __wbg_set_device_memory_total: (a: number, b: number, c: bigint) => void;
    readonly __wbg_set_device_os_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_downloadstate_downloaded_bytes: (a: number, b: bigint) => void;
    readonly __wbg_set_downloadstate_phase: (a: number, b: any) => void;
    readonly __wbg_set_downloadstate_total_bytes: (a: number, b: bigint) => void;
    readonly __wbg_set_engineconfig_allow_lmstudio_usage: (a: number, b: number) => void;
    readonly __wbg_set_engineconfig_allow_ollama_usage: (a: number, b: number) => void;
    readonly __wbg_set_engineconfig_anthropic_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_baseten_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_download_manager_type: (a: number, b: any) => void;
    readonly __wbg_set_engineconfig_gemini_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_huggingface_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_local_path: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_mirai_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_openai_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_openrouter_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_engineconfig_xai_api_key: (a: number, b: number, c: number) => void;
    readonly __wbg_set_feature_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_feature_values: (a: number, b: number, c: number) => void;
    readonly __wbg_set_file_hashes: (a: number, b: number, c: number) => void;
    readonly __wbg_set_file_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_file_size: (a: number, b: bigint) => void;
    readonly __wbg_set_file_url: (a: number, b: number, c: number) => void;
    readonly __wbg_set_hash_method: (a: number, b: any) => void;
    readonly __wbg_set_hash_value: (a: number, b: number, c: number) => void;
    readonly __wbg_set_image_description: (a: number, b: number, c: number) => void;
    readonly __wbg_set_image_format: (a: number, b: any) => void;
    readonly __wbg_set_image_height: (a: number, b: number) => void;
    readonly __wbg_set_image_id: (a: number, b: number, c: number) => void;
    readonly __wbg_set_image_theme: (a: number, b: any) => void;
    readonly __wbg_set_image_url: (a: number, b: number, c: number) => void;
    readonly __wbg_set_image_width: (a: number, b: number) => void;
    readonly __wbg_set_metadata_description: (a: number, b: number, c: number) => void;
    readonly __wbg_set_metadata_icons: (a: number, b: number, c: number) => void;
    readonly __wbg_set_metadata_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_metadata_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_model_accessibility: (a: number, b: any) => void;
    readonly __wbg_set_model_backends: (a: number, b: number, c: number) => void;
    readonly __wbg_set_model_encoding: (a: number, b: number) => void;
    readonly __wbg_set_model_family: (a: number, b: number) => void;
    readonly __wbg_set_model_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_model_properties: (a: number, b: number) => void;
    readonly __wbg_set_model_quantization: (a: number, b: number) => void;
    readonly __wbg_set_model_registry: (a: number, b: number) => void;
    readonly __wbg_set_model_specializations: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelbackend_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelbackend_metadata: (a: number, b: number) => void;
    readonly __wbg_set_modelbackend_version: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelfamily_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelfamily_metadata: (a: number, b: number) => void;
    readonly __wbg_set_modelfamily_vendor: (a: number, b: number) => void;
    readonly __wbg_set_modelproperties_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelproperties_metadata: (a: number, b: number) => void;
    readonly __wbg_set_modelproperties_size: (a: number, b: bigint) => void;
    readonly __wbg_set_modelproperties_version: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelquantization_bits_per_weight: (a: number, b: number) => void;
    readonly __wbg_set_modelquantization_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelquantization_metadata: (a: number, b: number) => void;
    readonly __wbg_set_modelquantization_method: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelquantization_vendor: (a: number, b: number) => void;
    readonly __wbg_set_modelregistry_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelregistry_metadata: (a: number, b: number) => void;
    readonly __wbg_set_modelvendor_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_modelvendor_metadata: (a: number, b: number) => void;
    readonly __wbg_set_pcmbatch_channels: (a: number, b: number) => void;
    readonly __wbg_set_pcmbatch_lengths: (a: number, b: number, c: number) => void;
    readonly __wbg_set_pcmbatch_sample_rate: (a: number, b: number) => void;
    readonly __wbg_set_pcmbatch_samples: (a: number, b: number, c: number) => void;
    readonly __wbg_set_repository_commit_hash: (a: number, b: number, c: number) => void;
    readonly __wbg_set_repository_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_repository_paths: (a: number, b: number, c: number) => void;
    readonly __wbg_set_samplingparameters_min_p: (a: number, b: number, c: number) => void;
    readonly __wbg_set_samplingparameters_repetition_penalty: (a: number, b: number, c: number) => void;
    readonly __wbg_set_samplingparameters_suffix_repetition_length: (a: number, b: number, c: bigint) => void;
    readonly __wbg_set_samplingparameters_temperature: (a: number, b: number, c: number) => void;
    readonly __wbg_set_samplingparameters_top_k: (a: number, b: number, c: bigint) => void;
    readonly __wbg_set_samplingparameters_top_p: (a: number, b: number, c: number) => void;
    readonly __wbg_set_texttospeechoutput_pcm_batch: (a: number, b: number) => void;
    readonly __wbg_set_texttospeechoutput_stats: (a: number, b: number) => void;
    readonly __wbg_set_texttospeechstats_audio_duration: (a: number, b: number) => void;
    readonly __wbg_set_texttospeechstats_first_chunk_seconds: (a: number, b: number) => void;
    readonly __wbg_set_texttospeechstats_generation_duration: (a: number, b: number) => void;
    readonly __wbg_set_texttospeechstats_text_length: (a: number, b: number) => void;
    readonly __wbg_set_token_id: (a: number, b: number) => void;
    readonly __wbg_set_token_is_special: (a: number, b: number) => void;
    readonly __wbg_set_token_value: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolcall_arguments: (a: number, b: number) => void;
    readonly __wbg_set_toolcall_identifier: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolcall_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolcallargumentdelta_arguments: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolcallargumentdelta_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolfunction_description: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolfunction_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolfunction_parameters: (a: number, b: number) => void;
    readonly __wbg_set_toolfunction_return_definition: (a: number, b: number) => void;
    readonly __wbg_set_toolnamespace_description: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolnamespace_name: (a: number, b: number, c: number) => void;
    readonly __wbg_set_toolnamespace_tools: (a: number, b: number, c: number) => void;
    readonly __wbg_set_value_json: (a: number, b: number, c: number) => void;
    readonly __wbg_texttospeechoutput_free: (a: number, b: number) => void;
    readonly __wbg_texttospeechstats_free: (a: number, b: number) => void;
    readonly __wbg_token_free: (a: number, b: number) => void;
    readonly __wbg_toolcall_free: (a: number, b: number) => void;
    readonly __wbg_toolcallargumentdelta_free: (a: number, b: number) => void;
    readonly __wbg_toolcallargumentstream_free: (a: number, b: number) => void;
    readonly __wbg_toolfunction_free: (a: number, b: number) => void;
    readonly __wbg_toolnamespace_free: (a: number, b: number) => void;
    readonly __wbg_value_free: (a: number, b: number) => void;
    readonly canceltoken_cancel: (a: number) => void;
    readonly canceltoken_create: () => number;
    readonly canceltoken_isCancelled: (a: number) => number;
    readonly chatconfig_create: () => number;
    readonly chatconfig_withContextLength: (a: number, b: any) => number;
    readonly chatconfig_withSamplingSeed: (a: number, b: any) => number;
    readonly chatinstance_model: (a: number) => number;
    readonly chatmessage_assistant: () => number;
    readonly chatmessage_developer: () => number;
    readonly chatmessage_forRole: (a: any) => number;
    readonly chatmessage_reasoning: (a: number) => [number, number];
    readonly chatmessage_reasoningEffort: (a: number) => any;
    readonly chatmessage_system: () => number;
    readonly chatmessage_text: (a: number) => [number, number];
    readonly chatmessage_tool: () => number;
    readonly chatmessage_toolCalls: (a: number) => [number, number];
    readonly chatmessage_toolNamespaces: (a: number) => [number, number];
    readonly chatmessage_user: () => number;
    readonly chatmessage_withBlock: (a: number, b: any) => number;
    readonly chatmessage_withReasoning: (a: number, b: number, c: number) => number;
    readonly chatmessage_withReasoningEffort: (a: number, b: any) => number;
    readonly chatmessage_withText: (a: number, b: number, c: number) => number;
    readonly chatmessage_withToolCall: (a: number, b: number) => number;
    readonly chatmessage_withToolCallCandidate: (a: number, b: number) => number;
    readonly chatmessage_withToolNamespaces: (a: number, b: number, c: number) => number;
    readonly chatreplyconfig_create: () => number;
    readonly chatreplyconfig_withGrammar: (a: number, b: number) => number;
    readonly chatreplyconfig_withSamplingMethod: (a: number, b: any) => number;
    readonly chatreplyconfig_withSamplingPolicy: (a: number, b: any) => number;
    readonly chatreplyconfig_withTokenLimit: (a: number, b: number) => number;
    readonly chatreplyconfig_withToolTurnLimit: (a: number, b: number) => number;
    readonly chatreplystats_inputJoulesPerToken: (a: number) => any;
    readonly chatreplystats_outputJoulesPerToken: (a: number) => any;
    readonly chatreplystats_tokensCount: (a: number) => number;
    readonly chatreplystats_totalJoules: (a: number) => [number, number];
    readonly chatsession_addTool: (a: number, b: number) => any;
    readonly chatsession_addTools: (a: number, b: number, c: number) => any;
    readonly chatsession_messages: (a: number) => any;
    readonly chatsession_reply: (a: number, b: number, c: number, d: number) => any;
    readonly chatsession_replyWithStream: (a: number, b: number, c: number, d: number) => any;
    readonly chatsession_reset: (a: number) => any;
    readonly chatsession_samplingDefaults: (a: number) => any;
    readonly chatsession_state: (a: number) => any;
    readonly chatsession_supportsToolCalls: (a: number) => any;
    readonly chatsessionstream_cancelToken: (a: number) => number;
    readonly chatsessionstream_next: (a: number) => any;
    readonly classificationmessage_assistant: (a: number, b: number) => number;
    readonly classificationmessage_user: (a: number, b: number) => number;
    readonly device_create: () => [number, number, number];
    readonly downloader_delete: (a: number) => any;
    readonly downloader_pause: (a: number) => any;
    readonly downloader_progress: (a: number) => any;
    readonly downloader_resume: (a: number) => any;
    readonly downloader_state: (a: number) => any;
    readonly downloaderstream_next: (a: number) => any;
    readonly downloadstate_canDelete: (a: number) => number;
    readonly downloadstate_canPause: (a: number) => number;
    readonly downloadstate_isInProgress: (a: number) => number;
    readonly downloadstate_name: (a: number) => [number, number];
    readonly downloadstate_progress: (a: number) => number;
    readonly engine_chat: (a: number, b: number, c: number) => any;
    readonly engine_chatInstance: (a: number, b: number, c: number) => any;
    readonly engine_chatModelInfo: (a: number, b: number) => any;
    readonly engine_chatWithInstance: (a: number, b: number) => any;
    readonly engine_create: (a: number) => any;
    readonly engine_download: (a: number, b: number) => any;
    readonly engine_downloadState: (a: number, b: number) => any;
    readonly engine_downloadStates: (a: number) => any;
    readonly engine_downloader: (a: number, b: number) => number;
    readonly engine_model: (a: number, b: number, c: number) => any;
    readonly engine_modelByIdentifier: (a: number, b: number, c: number) => any;
    readonly engine_modelByPath: (a: number, b: number, c: number) => any;
    readonly engine_modelByRepoId: (a: number, b: number, c: number) => any;
    readonly engine_modelFamilies: (a: number) => any;
    readonly engine_modelPath: (a: number, b: number) => any;
    readonly engine_modelRegistries: (a: number) => any;
    readonly engine_modelVendors: (a: number) => any;
    readonly engine_models: (a: number) => any;
    readonly engine_modelsByFamily: (a: number, b: number, c: number) => any;
    readonly engine_modelsByVendor: (a: number, b: number, c: number) => any;
    readonly engine_modelsDownloadable: (a: number) => any;
    readonly engine_modelsForChat: (a: number) => any;
    readonly engine_modelsForClassification: (a: number) => any;
    readonly engine_modelsForSpeculation: (a: number) => any;
    readonly engine_modelsForTextToSpeech: (a: number) => any;
    readonly engine_modelsForTranslation: (a: number) => any;
    readonly engine_modelsOnDevice: (a: number) => any;
    readonly engine_modelsRemote: (a: number) => any;
    readonly engine_refreshModels: (a: number) => any;
    readonly engine_registerCallback: (a: number, b: number) => any;
    readonly engine_removeBackend: (a: number, b: number, c: number) => any;
    readonly engine_removeRegistry: (a: number, b: number, c: number) => any;
    readonly enginecallback_create: (a: any) => number;
    readonly engineconfig_create: () => number;
    readonly engineconfig_withAllowLmstudioUsage: (a: number, b: number) => number;
    readonly engineconfig_withAllowOllamaUsage: (a: number, b: number) => number;
    readonly engineconfig_withAnthropicApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withBasetenApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withDownloadManagerType: (a: number, b: any) => number;
    readonly engineconfig_withGeminiApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withHuggingfaceApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withLocalPath: (a: number, b: number, c: number) => number;
    readonly engineconfig_withMiraiApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withOpenaiApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withOpenrouterApiKey: (a: number, b: number, c: number) => number;
    readonly engineconfig_withXaiApiKey: (a: number, b: number, c: number) => number;
    readonly file_crc32c: (a: number) => [number, number];
    readonly intounderlyingbytesource_autoAllocateChunkSize: (a: number) => number;
    readonly intounderlyingbytesource_cancel: (a: number) => void;
    readonly intounderlyingbytesource_pull: (a: number, b: any) => any;
    readonly intounderlyingbytesource_start: (a: number, b: any) => void;
    readonly intounderlyingbytesource_type: (a: number) => number;
    readonly intounderlyingsink_abort: (a: number, b: any) => any;
    readonly intounderlyingsink_close: (a: number) => any;
    readonly intounderlyingsink_write: (a: number, b: any) => any;
    readonly intounderlyingsource_cancel: (a: number) => void;
    readonly intounderlyingsource_pull: (a: number, b: any) => any;
    readonly metadata_external: (a: number, b: number) => number;
    readonly model_cacheIdentifier: (a: number) => [number, number];
    readonly model_checkpointVersion: (a: number) => [number, number];
    readonly model_external: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: any, p: number) => number;
    readonly model_filesystemPath: (a: number) => [number, number];
    readonly model_isChatCapable: (a: number) => number;
    readonly model_isClassificationCapable: (a: number) => number;
    readonly model_isDownloadable: (a: number) => number;
    readonly model_isOnDevice: (a: number) => number;
    readonly model_isQuantized: (a: number) => number;
    readonly model_isRemote: (a: number) => number;
    readonly model_isSpeculationCapable: (a: number) => number;
    readonly model_isTextToSpeechCapable: (a: number) => number;
    readonly model_isTranslationCapable: (a: number) => number;
    readonly model_name: (a: number) => [number, number];
    readonly model_repoIds: (a: number) => [number, number];
    readonly modelbackend_name: (a: number) => [number, number];
    readonly modelfamily_name: (a: number) => [number, number];
    readonly modelproperties_name: (a: number) => [number, number];
    readonly modelquantization_name: (a: number) => [number, number];
    readonly modelregistry_name: (a: number) => [number, number];
    readonly modelvendor_name: (a: number) => [number, number];
    readonly nativetool_new: (a: number, b: number, c: any, d: any) => [number, number, number];
    readonly pcmbatch_batchSize: (a: number) => number;
    readonly pcmbatch_duration: (a: number) => number;
    readonly pcmbatch_saveAsWav: (a: number, b: number, c: number) => [number, number];
    readonly pcmbatch_totalFrames: (a: number) => number;
    readonly toolcallargumentstream_create: (a: number, b: number) => number;
    readonly toolcallargumentstream_finish: (a: number, b: number) => [number, number, number];
    readonly toolcallargumentstream_update: (a: number, b: number, c: number) => [number, number, number];
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__476: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__wasm_bindgen_6c51f868a643a6ac___JsValue__true_: (a: number, b: number, c: any) => any;
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true_: (a: number, b: number) => void;
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true__1_: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
