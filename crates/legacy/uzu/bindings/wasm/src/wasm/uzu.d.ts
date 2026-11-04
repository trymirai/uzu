/* tslint:disable */
/* eslint-disable */
/**
 * The `ReadableStreamType` enum.
 *
 * *This API requires the following crate features to be activated: `ReadableStreamType`*
 */

export type ReadableStreamType = "bytes";
export declare enum ChatContentBlockType {
    Identity = 'Identity',
    ReasoningEffort = 'ReasoningEffort',
    ConversationStartDate = 'ConversationStartDate',
    KnowledgeCutoff = 'KnowledgeCutoff',
    BuiltinTools = 'BuiltinTools',
    Tools = 'Tools',
    Text = 'Text',
    Image = 'Image',
    Video = 'Video',
    Audio = 'Audio',
    File = 'File',
    Reasoning = 'Reasoning',
    ToolCall = 'ToolCall',
    ToolCallCandidate = 'ToolCallCandidate',
    ToolCallResult = 'ToolCallResult',
    Translation = 'Translation',
    Custom = 'Custom',
}

export declare enum ChatReplyFinishReason {
    Stop = 'Stop',
    Length = 'Length',
    Cancelled = 'Cancelled',
    ContextLimitReached = 'ContextLimitReached',
    ToolCalls = 'ToolCalls',
    Rejected = 'Rejected',
}

export declare enum ChatSessionState {
    Idle = 'Idle',
    Generation = 'Generation',
    ToolCalling = 'ToolCalling',
    Resetting = 'Resetting',
}

export declare enum ClassificationRole {
    User = 'User',
    Assistant = 'Assistant',
}

export declare enum DecisionSessionState {
    Idle = 'Idle',
    Deciding = 'Deciding',
}

export declare enum DownloadManagerType {
    Universal = 'Universal',
}

export declare enum HashMethod {
    CRC32C = 'CRC32C',
    Sha256 = 'Sha256',
    GitBlobSha1 = 'GitBlobSha1',
}

export declare enum ImageFormat {
    Png = 'Png',
    Webp = 'Webp',
    Svg = 'Svg',
}

export declare enum ImageTheme {
    Light = 'Light',
    Dark = 'Dark',
}

export declare enum ReasoningEffort {
    Disabled = 'Disabled',
    Default = 'Default',
    Low = 'Low',
    Medium = 'Medium',
    High = 'High',
    XHigh = 'XHigh',
}

export interface ChatConfig {
    contextLength: ContextLength;
    samplingSeed: SamplingSeed;
}

export interface ChatContentBlockAudio {
    url: string;
}

export interface ChatContentBlockBuiltinTools {
    names: Array<string>;
}

export interface ChatContentBlockConversationStartDate {
    value: string;
}

export interface ChatContentBlockCustom {
    value: Value;
}

export interface ChatContentBlockFile {
    url: string;
}

export interface ChatContentBlockIdentity {
    value: string;
}

export interface ChatContentBlockImage {
    url: string;
}

export interface ChatContentBlockKnowledgeCutoff {
    value: string;
}

export interface ChatContentBlockReasoning {
    value: string;
}

export interface ChatContentBlockReasoningEffort {
    value: ReasoningEffort;
}

export interface ChatContentBlockText {
    value: string;
}

export interface ChatContentBlockToolCall {
    value: ToolCall;
}

export interface ChatContentBlockToolCallCandidate {
    value: Value;
    identifier?: string;
}

export interface ChatContentBlockToolCallResult {
    identifier?: string;
    name?: string;
    value: Value;
}

export interface ChatContentBlockTools {
    namespaces: Array<ToolNamespace>;
}

export interface ChatContentBlockTranslation {
    payload: TranslationPayload;
    sourceLanguageCode: string;
    targetLanguageCode: string;
}

export interface ChatContentBlockVideo {
    url: string;
}

export interface ChatMessage {
    role: ChatRole;
    content: Array<ChatContentBlock>;
    metadata: ChatMessageMetadata;
}

export interface ChatMessageMetadata {
    values: Record<string, Value>;
}

export interface ChatModelCapabilities {
    supportsReasoning: boolean;
    supportsDisableReasoning: boolean;
    reasoningEfforts: Array<ReasoningEffort>;
    supportsTools: boolean;
    supportsMultipleToolCalls: boolean;
    requiresTools: boolean;
}

export interface ChatModelInfo {
    contextLength?: number;
    capabilities: ChatModelCapabilities;
    defaultReasoningEffort?: ReasoningEffort;
}

export interface ChatReply {
    message: ChatMessage;
    stats: ChatReplyStats;
    finishReason?: ChatReplyFinishReason;
}

export interface ChatReplyConfig {
    tokenLimit?: number;
    samplingPolicy: SamplingPolicy;
    grammar?: Grammar;
    toolTurnLimit?: number;
}

export interface ChatReplyJoulesPerTokenComponents {
    cpu: number;
    gpu: number;
    ane: number;
    dram: number;
}

export interface ChatReplyJoulesPerTokenTotal {
    total: number;
}

export interface ChatReplySpeculatorStats {
    tokensPerForwardPass: number;
    numDecodeForwardPasses: number;
}

export interface ChatReplyStats {
    duration: number;
    timeToFirstToken?: number;
    prefillTokensPerSecond?: number;
    generateTokensPerSecond?: number;
    tokensCountInput?: number;
    tokensCountInputCached?: number;
    tokensCountOutput?: number;
    memoryUsedBytes?: number;
    speculatorStats?: ChatReplySpeculatorStats;
    inputEnergy?: ReplyEnergy;
    outputEnergy?: ReplyEnergy;
}

export interface ChatRoleAssistant {
}

export interface ChatRoleCustom {
    name: string;
}

export interface ChatRoleDeveloper {
}

export interface ChatRoleSystem {
}

export interface ChatRoleTool {
}

export interface ChatRoleUser {
}

export interface ChatSessionStream {
    [Symbol.asyncIterator](): AsyncGenerator<ChatSessionStreamChunk, void, undefined>
}

export interface ChatSessionStreamChunkError {
    error: ChatSessionError;
}

export interface ChatSessionStreamChunkReplies {
    replies: Array<ChatReply>;
}

export interface ChatSessionStreamChunkToolResults {
    messages: Array<ChatMessage>;
}

export interface ClassificationLabel {
    index: number;
    label: string;
    title: string;
    description: string;
    recommendedThreshold: number;
}

export interface ClassificationMessage {
    role: ClassificationRole;
    content: string;
}

export interface ClassificationOutput {
    logits: Array<number>;
    probabilities: ClassificationOutputProbabilities;
    stats: ClassificationStats;
}

export interface ClassificationOutputProbabilities {
    values: Record<string, number>;
}

export interface ClassificationStats {
    preprocessingDuration: number;
    forwardPassDuration: number;
    postprocessingDuration: number;
    totalDuration: number;
    tokensCount: number;
    tokensPerSecond: number;
    predictedLabel: string;
    confidence: number;
}

export interface ContextLengthCustom {
    length: number;
}

export interface ContextLengthDefault {
}

export interface ContextLengthMaximal {
}

export interface DecisionAnswerChoice {
    index: number;
    key: string;
    probabilities: Array<number>;
}

export interface DecisionAnswerScore {
    probabilities: Array<number>;
    score: number;
}

export interface DecisionAnswerYesNo {
    yesProbability: number;
}

export interface DecisionInput {
    state: string;
    questions: Array<DecisionQuestion>;
}

export interface DecisionOption {
    key: string;
    description?: string;
}

export interface DecisionOutput {
    results: Array<DecisionResult>;
}

export interface DecisionQuestionChoice {
    instructions?: string;
    options: Array<DecisionOption>;
}

export interface DecisionQuestionScore {
    instructions?: string;
    levels: Array<string>;
}

export interface DecisionQuestionYesNo {
    instructions?: string;
    noDescription?: string;
    yesDescription?: string;
}

export interface DecisionResult {
    answer: DecisionAnswer;
    stats: DecisionStats;
}

export interface DecisionStats {
    tokensCount: number;
    prefillDuration: number;
    totalDuration: number;
    energy?: ReplyEnergy;
}

export interface Device {
    osName?: string;
    cpuName?: string;
    memoryTotal?: number;
    homePath: string;
}

export interface DownloadPhaseDownloaded {
}

export interface DownloadPhaseDownloading {
}

export interface DownloadPhaseError {
    message: string;
}

export interface DownloadPhaseInitializing {
}

export interface DownloadPhaseLocked {
    managerId: string;
}

export interface DownloadPhaseNotDownloaded {
}

export interface DownloadPhasePaused {
}

export interface DownloadState {
    totalBytes: number;
    downloadedBytes: number;
    phase: DownloadPhase;
}

export interface DownloaderStream {
    [Symbol.asyncIterator](): AsyncGenerator<DownloadState, void, undefined>
}

export interface EngineConfig {
    miraiApiKey?: string;
    localPath?: string;
    huggingfaceApiKey?: string;
    openaiApiKey?: string;
    anthropicApiKey?: string;
    geminiApiKey?: string;
    xaiApiKey?: string;
    basetenApiKey?: string;
    openrouterApiKey?: string;
    allowOllamaUsage: boolean;
    allowLmstudioUsage: boolean;
    downloadManagerType: DownloadManagerType;
}

export interface Feature {
    name: string;
    values: Array<string>;
}

export interface File {
    url: string;
    name: string;
    size: number;
    hashes: Array<Hash>;
}

export interface GrammarJsonAny {
}

export interface GrammarJsonSchema {
    schema: string;
}

export interface GrammarRegex {
    pattern: string;
}

export interface Hash {
    method: HashMethod;
    value: string;
}

export interface Image {
    id: string;
    url: string;
    theme: ImageTheme;
    format: ImageFormat;
    description?: string;
    width?: number;
    height?: number;
}

export interface Metadata {
    identifier: string;
    name: string;
    description?: string;
    icons: Array<Image>;
}

export interface Model {
    identifier: ModelIdentifier;
    registry: ModelRegistry;
    backends: Array<ModelBackend>;
    family?: ModelFamily;
    properties?: ModelProperties;
    quantization?: ModelQuantization;
    specializations: Array<ModelSpecialization>;
    accessibility: ModelAccessibility;
    encoding?: Value;
}

export interface ModelAccessibilityOnDevice {
    source: ModelSource;
}

export interface ModelAccessibilityRemote {
    repository?: Repository;
}

export interface ModelBackend {
    identifier: string;
    version: string;
    metadata: Metadata;
}

export interface ModelFamily {
    identifier: string;
    vendor: ModelVendor;
    metadata: Metadata;
}

export interface ModelProperties {
    identifier: string;
    size: number;
    version?: string;
    metadata: Metadata;
}

export interface ModelQuantization {
    identifier: string;
    method: string;
    bitsPerWeight: number;
    vendor: ModelVendor;
    metadata: Metadata;
}

export interface ModelRegistry {
    identifier: string;
    metadata: Metadata;
}

export interface ModelSourceFilesystem {
    path: string;
}

export interface ModelSourceRegistry {
    toolchainVersion: string;
    repository?: Repository;
    sourceRepository?: Repository;
    files: Array<File>;
}

export interface ModelSpecializationChat {
}

export interface ModelSpecializationClassification {
}

export interface ModelSpecializationDecision {
}

export interface ModelSpecializationSpeculation {
}

export interface ModelSpecializationTextToSpeech {
}

export interface ModelSpecializationTranslation {
}

export interface ModelVendor {
    identifier: string;
    metadata: Metadata;
}

export interface PcmBatch {
    samples: Array<number>;
    sampleRate: number;
    channels: number;
    lengths: Array<number>;
}

export interface ReplyEnergyComponents {
    cpu: number;
    gpu: number;
    ane: number;
    dram: number;
}

export interface ReplyEnergyTotal {
    total: number;
}

export interface Repository {
    identifier: string;
    commitHash?: string;
    paths?: Array<string>;
}

export interface SamplingMethodGreedy {
}

export interface SamplingMethodStochastic {
    temperature?: number;
    topK?: number;
    topP?: number;
    minP?: number;
    repetitionPenalty?: number;
    suffixRepetitionLength?: number;
}

export interface SamplingParameters {
    temperature?: number;
    topK?: number;
    topP?: number;
    minP?: number;
    repetitionPenalty?: number;
    suffixRepetitionLength?: number;
}

export interface SamplingPolicyCustom {
    method: SamplingMethod;
}

export interface SamplingPolicyDefault {
}

export interface SamplingSeedCustom {
    seed: number;
}

export interface SamplingSeedDefault {
}

export interface TextToSpeechOutput {
    pcmBatch: PcmBatch;
    stats: TextToSpeechStats;
}

export interface TextToSpeechStats {
    textLength: number;
    firstChunkSeconds: number;
    generationDuration: number;
    audioDuration: number;
}

export interface Token {
    id: TokenId;
    value: TokenValue;
    isSpecial: boolean;
}

export interface ToolCall {
    identifier?: string;
    name: string;
    arguments: Value;
}

export interface ToolCallArgumentDelta {
    name: string;
    arguments: string;
}

export interface ToolDescriptionFunction {
    toolFunction: ToolFunction;
}

export interface ToolFunction {
    name: string;
    description: string;
    parameters?: Value;
    returnDefinition?: Value;
}

export interface ToolNamespace {
    name: string;
    description?: string;
    tools: Array<ToolDescription>;
}

export interface TranslationPayloadImage {
    url: string;
}

export interface TranslationPayloadText {
    text: string;
}

export interface Value {
    json: string;
}

export type ChatContentBlock = ChatContentBlockIdentity | ChatContentBlockReasoningEffort | ChatContentBlockConversationStartDate | ChatContentBlockKnowledgeCutoff | ChatContentBlockBuiltinTools | ChatContentBlockTools | ChatContentBlockText | ChatContentBlockImage | ChatContentBlockVideo | ChatContentBlockAudio | ChatContentBlockFile | ChatContentBlockReasoning | ChatContentBlockToolCall | ChatContentBlockToolCallCandidate | ChatContentBlockToolCallResult | ChatContentBlockTranslation | ChatContentBlockCustom;

export type ChatReplyJoulesPerToken = ChatReplyJoulesPerTokenTotal | ChatReplyJoulesPerTokenComponents;

export type ChatRole = ChatRoleUser | ChatRoleAssistant | ChatRoleSystem | ChatRoleDeveloper | ChatRoleTool | ChatRoleCustom;

export type ChatSessionError =
| { type: 'Backend'; message: string }
| { type: 'Loading'; message: string }
| { type: 'UnsupportedModel' }
| { type: 'UnableToPerformOperationInCurrentState' }
| { type: 'NoResponse' }
| { type: 'ToolTurnLimitExceeded'; limit: number };

export type ChatSessionStreamChunk = ChatSessionStreamChunkReplies | ChatSessionStreamChunkError | ChatSessionStreamChunkToolResults;

export type ContextLength = ContextLengthDefault | ContextLengthMaximal | ContextLengthCustom;

export type DecisionAnswer = DecisionAnswerChoice | DecisionAnswerScore | DecisionAnswerYesNo;

export type DecisionQuestion = DecisionQuestionChoice | DecisionQuestionScore | DecisionQuestionYesNo;

export type DecisionSessionError =
| { type: 'Backend'; message: string }
| { type: 'Encoding'; message: string }
| { type: 'InvalidQuestion'; index: number; message: string }
| { type: 'NoQuestions' }
| { type: 'UnsupportedModel' }
| { type: 'UnableToPerformOperationInCurrentState' }
| { type: 'NoResponse' };

export type DeviceError =
| { type: 'UnsupportedDevice' };

export type DownloadPhase = DownloadPhaseNotDownloaded | DownloadPhaseDownloading | DownloadPhasePaused | DownloadPhaseDownloaded | DownloadPhaseLocked | DownloadPhaseError | DownloadPhaseInitializing;

export type EngineError =
| { type: 'TokioError'; message: string }
| { type: 'Device'; field0: DeviceError }
| { type: 'Storage'; field0: StorageError }
| { type: 'Registry'; field0: RegistryError }
| { type: 'UnableToCreateBackend' }
| { type: 'BackendNotFound' }
| { type: 'ChatSession'; field0: ChatSessionError }
| { type: 'DecisionSession'; field0: DecisionSessionError }
| { type: 'ModelConfig'; message: string };

export type Grammar = GrammarJsonAny | GrammarJsonSchema | GrammarRegex;

export type ModelAccessibility = ModelAccessibilityOnDevice | ModelAccessibilityRemote;

export type ModelIdentifier = string;

export type ModelSource = ModelSourceRegistry | ModelSourceFilesystem;

export type ModelSpecialization = ModelSpecializationChat | ModelSpecializationClassification | ModelSpecializationDecision | ModelSpecializationTextToSpeech | ModelSpecializationTranslation | ModelSpecializationSpeculation;

export type PcmBatchError =
| { type: 'UnableToSaveAsWav'; message: string };

export type RegistryError =
| { type: 'UnableToCreate'; message: string }
| { type: 'UnableToGetModels'; message: string }
| { type: 'UnableToAddRegistry'; identifier: string };

export type ReplyEnergy = ReplyEnergyTotal | ReplyEnergyComponents;

export type SamplingMethod = SamplingMethodGreedy | SamplingMethodStochastic;

export type SamplingPolicy = SamplingPolicyDefault | SamplingPolicyCustom;

export type SamplingSeed = SamplingSeedDefault | SamplingSeedCustom;

export type StorageError =
| { type: 'UnableToCreateDirectory'; path: string }
| { type: 'DownloadManager'; message: string }
| { type: 'Locked' }
| { type: 'ModelNotFound'; identifier: ModelIdentifier }
| { type: 'UnsupportedModel'; identifier: ModelIdentifier };

export type TokenId = number;

export type TokenValue = string;

export type ToolDescription = ToolDescriptionFunction;

export type TranslationPayload = TranslationPayloadText | TranslationPayloadImage;


export class CancelToken {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    cancel(): void;
    static create(): CancelToken;
    readonly isCancelled: boolean;
}

export class ChatConfig {
    free(): void;
    [Symbol.dispose](): void;
    constructor(contextLength: ContextLength, samplingSeed: SamplingSeed);
    static create(): ChatConfig;
    withContextLength(contextLength: ContextLength): ChatConfig;
    withSamplingSeed(samplingSeed: SamplingSeed): ChatConfig;
}

export class ChatContentBlockAudio {
    free(): void;
    [Symbol.dispose](): void;
    constructor(url: string);
}

export class ChatContentBlockBuiltinTools {
    free(): void;
    [Symbol.dispose](): void;
    constructor(names: Array<string>);
}

export class ChatContentBlockConversationStartDate {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: string);
}

export class ChatContentBlockCustom {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: Value);
}

export class ChatContentBlockFile {
    free(): void;
    [Symbol.dispose](): void;
    constructor(url: string);
}

export class ChatContentBlockIdentity {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: string);
}

export class ChatContentBlockImage {
    free(): void;
    [Symbol.dispose](): void;
    constructor(url: string);
}

export class ChatContentBlockKnowledgeCutoff {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: string);
}

export class ChatContentBlockReasoning {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: string);
}

export class ChatContentBlockReasoningEffort {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: ReasoningEffort);
}

export class ChatContentBlockText {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: string);
}

export class ChatContentBlockToolCall {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: ToolCall);
}

export class ChatContentBlockToolCallCandidate {
    free(): void;
    [Symbol.dispose](): void;
    constructor(value: Value, identifier?: string | null);
}

export class ChatContentBlockToolCallResult {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string | undefined | null, name: string | undefined | null, value: Value);
}

export class ChatContentBlockTools {
    free(): void;
    [Symbol.dispose](): void;
    constructor(namespaces: Array<ToolNamespace>);
}

export class ChatContentBlockTranslation {
    free(): void;
    [Symbol.dispose](): void;
    constructor(payload: TranslationPayload, sourceLanguageCode: string, targetLanguageCode: string);
}

export class ChatContentBlockVideo {
    free(): void;
    [Symbol.dispose](): void;
    constructor(url: string);
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
    free(): void;
    [Symbol.dispose](): void;
    constructor(role: ChatRole, content: Array<ChatContentBlock>, metadata: ChatMessageMetadata);
    static assistant(): ChatMessage;
    static developer(): ChatMessage;
    static for_role(role: ChatRole): ChatMessage;
    static system(): ChatMessage;
    static tool(): ChatMessage;
    static user(): ChatMessage;
    withBlock(block: ChatContentBlock): ChatMessage;
    withReasoningEffort(reasoningEffort: ReasoningEffort): ChatMessage;
    withReasoning(reasoning: string): ChatMessage;
    withText(text: string): ChatMessage;
    withToolCallCandidate(value: Value): ChatMessage;
    withToolCall(toolCall: ToolCall): ChatMessage;
    withToolNamespaces(toolNamespaces: Array<ToolNamespace>): ChatMessage;
    readonly reasoning: string | null;
    readonly reasoningEffort: ReasoningEffort | null;
    readonly text: string | null;
    readonly toolCalls: Array<ToolCall>;
    readonly toolNamespaces: Array<ToolNamespace>;
}

export class ChatModelCapabilities {
    free(): void;
    [Symbol.dispose](): void;
    constructor(supportsReasoning: boolean, supportsDisableReasoning: boolean, reasoningEfforts: Array<ReasoningEffort>, supportsTools: boolean, supportsMultipleToolCalls: boolean, requiresTools: boolean);
}

/**
 * Chat metadata available without loading model weights or allocating a session.
 */
export class ChatModelInfo {
    free(): void;
    [Symbol.dispose](): void;
    constructor(contextLength: number | undefined | null, capabilities: ChatModelCapabilities, defaultReasoningEffort?: ReasoningEffort | null);
}

export class ChatReply {
    free(): void;
    [Symbol.dispose](): void;
    constructor(message: ChatMessage, stats: ChatReplyStats, finishReason?: ChatReplyFinishReason | null);
}

export class ChatReplyConfig {
    free(): void;
    [Symbol.dispose](): void;
    constructor(tokenLimit: number | undefined | null, samplingPolicy: SamplingPolicy, grammar?: Grammar | null, toolTurnLimit?: number | null);
    static create(): ChatReplyConfig;
    withGrammar(grammar?: Grammar | null): ChatReplyConfig;
    withSamplingMethod(samplingMethod: SamplingMethod): ChatReplyConfig;
    withSamplingPolicy(samplingPolicy: SamplingPolicy): ChatReplyConfig;
    withTokenLimit(tokenLimit?: number | null): ChatReplyConfig;
    withToolTurnLimit(toolTurnLimit?: number | null): ChatReplyConfig;
}

export class ChatReplyJoulesPerTokenComponents {
    free(): void;
    [Symbol.dispose](): void;
    constructor(cpu: number, gpu: number, ane: number, dram: number);
}

export class ChatReplyJoulesPerTokenTotal {
    free(): void;
    [Symbol.dispose](): void;
    constructor(total: number);
}

export class ChatReplySpeculatorStats {
    free(): void;
    [Symbol.dispose](): void;
    constructor(tokensPerForwardPass: number, numDecodeForwardPasses: number);
}

export class ChatReplyStats {
    free(): void;
    [Symbol.dispose](): void;
    constructor(duration: number, timeToFirstToken?: number | null, prefillTokensPerSecond?: number | null, generateTokensPerSecond?: number | null, tokensCountInput?: number | null, tokensCountInputCached?: number | null, tokensCountOutput?: number | null, memoryUsedBytes?: number | null, speculatorStats?: ChatReplySpeculatorStats | null, inputEnergy?: ReplyEnergy | null, outputEnergy?: ReplyEnergy | null);
    readonly inputJoulesPerToken: ChatReplyJoulesPerToken | null;
    readonly outputJoulesPerToken: ChatReplyJoulesPerToken | null;
    readonly tokensCount: number | null;
    readonly totalJoules: number | null;
}

export class ChatRoleAssistant {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ChatRoleCustom {
    free(): void;
    [Symbol.dispose](): void;
    constructor(name: string);
}

export class ChatRoleDeveloper {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ChatRoleSystem {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ChatRoleTool {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ChatRoleUser {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ChatSession {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    addTool(tool: NativeTool): Promise<void>;
    addTools(tools: Array<NativeTool>): Promise<void>;
    replyWithStream(input: Array<ChatMessage>, config: ChatReplyConfig): Promise<ChatSessionStream>;
    reply(input: Array<ChatMessage>, config: ChatReplyConfig): Promise<Array<ChatReply>>;
    reset(): Promise<void>;
    readonly messages: Promise<Array<ChatMessage>>;
    readonly samplingDefaults: Promise<SamplingParameters | null>;
    readonly state: Promise<ChatSessionState>;
    readonly supportsToolCalls: Promise<boolean>;
}

export class ChatSessionStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    next(): Promise<ChatSessionStreamChunk | null>;
    readonly cancelToken: CancelToken;
}

export class ChatSessionStreamChunkError {
    free(): void;
    [Symbol.dispose](): void;
    constructor(error: ChatSessionError);
}

export class ChatSessionStreamChunkReplies {
    free(): void;
    [Symbol.dispose](): void;
    constructor(replies: Array<ChatReply>);
}

export class ChatSessionStreamChunkToolResults {
    free(): void;
    [Symbol.dispose](): void;
    constructor(messages: Array<ChatMessage>);
}

export class ClassificationLabel {
    free(): void;
    [Symbol.dispose](): void;
    constructor(index: number, label: string, title: string, description: string, recommendedThreshold: number);
}

export class ClassificationMessage {
    free(): void;
    [Symbol.dispose](): void;
    constructor(role: ClassificationRole, content: string);
    static assistant(content: string): ClassificationMessage;
    static user(content: string): ClassificationMessage;
}

export class ClassificationOutput {
    free(): void;
    [Symbol.dispose](): void;
    constructor(logits: Array<number>, probabilities: ClassificationOutputProbabilities, stats: ClassificationStats);
}

export class ClassificationStats {
    free(): void;
    [Symbol.dispose](): void;
    constructor(preprocessingDuration: number, forwardPassDuration: number, postprocessingDuration: number, totalDuration: number, tokensCount: number, tokensPerSecond: number, predictedLabel: string, confidence: number);
}

export class ContextLengthCustom {
    free(): void;
    [Symbol.dispose](): void;
    constructor(length: number);
}

export class ContextLengthDefault {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ContextLengthMaximal {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class DecisionAnswerChoice {
    free(): void;
    [Symbol.dispose](): void;
    constructor(index: number, key: string, probabilities: Array<number>);
}

export class DecisionAnswerScore {
    free(): void;
    [Symbol.dispose](): void;
    constructor(probabilities: Array<number>, score: number);
}

export class DecisionAnswerYesNo {
    free(): void;
    [Symbol.dispose](): void;
    constructor(yesProbability: number);
}

export class DecisionInput {
    free(): void;
    [Symbol.dispose](): void;
    constructor(state: string, questions: Array<DecisionQuestion>);
}

export class DecisionOption {
    free(): void;
    [Symbol.dispose](): void;
    constructor(key: string, description?: string | null);
}

export class DecisionOutput {
    free(): void;
    [Symbol.dispose](): void;
    constructor(results: Array<DecisionResult>);
}

export class DecisionQuestionChoice {
    free(): void;
    [Symbol.dispose](): void;
    constructor(instructions: string | undefined | null, options: Array<DecisionOption>);
}

export class DecisionQuestionScore {
    free(): void;
    [Symbol.dispose](): void;
    constructor(instructions: string | undefined | null, levels: Array<string>);
}

export class DecisionQuestionYesNo {
    free(): void;
    [Symbol.dispose](): void;
    constructor(instructions?: string | null, noDescription?: string | null, yesDescription?: string | null);
}

export class DecisionResult {
    free(): void;
    [Symbol.dispose](): void;
    constructor(answer: DecisionAnswer, stats: DecisionStats);
}

export class DecisionSession {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    decide(input: DecisionInput): Promise<DecisionOutput>;
    readonly state: Promise<DecisionSessionState>;
}

export class DecisionStats {
    free(): void;
    [Symbol.dispose](): void;
    constructor(tokensCount: number, prefillDuration: number, totalDuration: number, energy?: ReplyEnergy | null);
}

export class Device {
    free(): void;
    [Symbol.dispose](): void;
    constructor(osName: string | undefined | null, cpuName: string | undefined | null, memoryTotal: number | undefined | null, homePath: string);
    static create(): Device;
}

export class DownloadPhaseDownloaded {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class DownloadPhaseDownloading {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class DownloadPhaseError {
    free(): void;
    [Symbol.dispose](): void;
    constructor(message: string);
}

export class DownloadPhaseInitializing {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class DownloadPhaseLocked {
    free(): void;
    [Symbol.dispose](): void;
    constructor(managerId: string);
}

export class DownloadPhaseNotDownloaded {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class DownloadPhasePaused {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class DownloadState {
    free(): void;
    [Symbol.dispose](): void;
    constructor(totalBytes: number, downloadedBytes: number, phase: DownloadPhase);
    readonly canDelete: boolean;
    readonly canPause: boolean;
    readonly isInProgress: boolean;
    readonly name: string;
    readonly progress: number;
}

export class Downloader {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    delete(): Promise<void>;
    pause(): Promise<void>;
    progress(): Promise<DownloaderStream>;
    resume(): Promise<void>;
    readonly state: Promise<DownloadState | null>;
}

export class DownloaderStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    next(): Promise<DownloadState | null>;
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
    chatModelInfo(model: Model): Promise<ChatModelInfo | null>;
    chatWithInstance(instance: ChatInstance): Promise<ChatSession>;
    chat(model: Model, config: ChatConfig): Promise<ChatSession>;
    static create(config: EngineConfig): Promise<Engine>;
    decision(model: Model): Promise<DecisionSession>;
    downloadState(model: Model): Promise<DownloadState | null>;
    download(model: Model): Promise<DownloaderStream>;
    downloader(model: Model): Downloader;
    modelByIdentifier(identifier: ModelIdentifier): Promise<Model | null>;
    modelByPath(path: string): Promise<Model | null>;
    modelByRepoId(repoId: string): Promise<Model | null>;
    modelPath(model: Model): Promise<string | null>;
    model(identifier: string): Promise<Model | null>;
    modelsByFamily(familyIdentifier: string): Promise<Array<Model>>;
    modelsByVendor(vendorIdentifier: string): Promise<Array<Model>>;
    /**
     * Refresh registries and wait until the catalog has been published to storage.
     * Registries that fail to refresh retain their last known entries.
     */
    refreshModels(): Promise<Array<Model>>;
    registerCallback(callback: EngineCallback): Promise<void>;
    removeBackend(identifier: string): Promise<void>;
    removeRegistry(registryIdentifier: string): Promise<void>;
    readonly downloadStates: Promise<Record<ModelIdentifier, DownloadState>>;
    readonly modelFamilies: Promise<Array<ModelFamily>>;
    readonly modelRegistries: Promise<Array<ModelRegistry>>;
    readonly modelVendors: Promise<Array<ModelVendor>>;
    readonly models: Promise<Array<Model>>;
    readonly modelsDownloadable: Promise<Array<Model>>;
    readonly modelsForChat: Promise<Array<Model>>;
    readonly modelsForClassification: Promise<Array<Model>>;
    readonly modelsForSpeculation: Promise<Array<Model>>;
    readonly modelsForTextToSpeech: Promise<Array<Model>>;
    readonly modelsForTranslation: Promise<Array<Model>>;
    readonly modelsOnDevice: Promise<Array<Model>>;
    readonly modelsRemote: Promise<Array<Model>>;
}

export class EngineCallback {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(callback: () => void): EngineCallback;
}

export class EngineConfig {
    free(): void;
    [Symbol.dispose](): void;
    constructor(miraiApiKey: string | undefined | null, localPath: string | undefined | null, huggingfaceApiKey: string | undefined | null, openaiApiKey: string | undefined | null, anthropicApiKey: string | undefined | null, geminiApiKey: string | undefined | null, xaiApiKey: string | undefined | null, basetenApiKey: string | undefined | null, openrouterApiKey: string | undefined | null, allowOllamaUsage: boolean, allowLmstudioUsage: boolean, downloadManagerType: DownloadManagerType);
    static create(): EngineConfig;
    withAllowLmstudioUsage(allowLmstudioUsage: boolean): EngineConfig;
    withAllowOllamaUsage(allowOllamaUsage: boolean): EngineConfig;
    withAnthropicApiKey(anthropicApiKey: string): EngineConfig;
    withBasetenApiKey(basetenApiKey: string): EngineConfig;
    withDownloadManagerType(downloadManagerType: DownloadManagerType): EngineConfig;
    withGeminiApiKey(geminiApiKey: string): EngineConfig;
    withHuggingfaceApiKey(huggingfaceApiKey: string): EngineConfig;
    withLocalPath(localPath: string): EngineConfig;
    withMiraiApiKey(miraiApiKey: string): EngineConfig;
    withOpenaiApiKey(openaiApiKey: string): EngineConfig;
    withOpenrouterApiKey(openrouterApiKey: string): EngineConfig;
    withXaiApiKey(xaiApiKey: string): EngineConfig;
}

export class Feature {
    free(): void;
    [Symbol.dispose](): void;
    constructor(name: string, values: Array<string>);
}

export class File {
    free(): void;
    [Symbol.dispose](): void;
    constructor(url: string, name: string, size: number, hashes: Array<Hash>);
    readonly crc32C: string | null;
}

export class GrammarJsonAny {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class GrammarJsonSchema {
    free(): void;
    [Symbol.dispose](): void;
    constructor(schema: string);
}

export class GrammarRegex {
    free(): void;
    [Symbol.dispose](): void;
    constructor(pattern: string);
}

export class Hash {
    free(): void;
    [Symbol.dispose](): void;
    constructor(method: HashMethod, value: string);
}

export class Image {
    free(): void;
    [Symbol.dispose](): void;
    constructor(id: string, url: string, theme: ImageTheme, format: ImageFormat, description?: string | null, width?: number | null, height?: number | null);
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
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, name: string, description: string | undefined | null, icons: Array<Image>);
    static external(name: string): Metadata;
}

export class Model {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: ModelIdentifier, registry: ModelRegistry, backends: Array<ModelBackend>, family: ModelFamily | undefined | null, properties: ModelProperties | undefined | null, quantization: ModelQuantization | undefined | null, specializations: Array<ModelSpecialization>, accessibility: ModelAccessibility, encoding?: Value | null);
    static external(identifier: ModelIdentifier, registryIdentifier: string, registryName: string, backendIdentifier: string, backendName: string, backendVersion: string, specializations: Array<ModelSpecialization>, accessibility: ModelAccessibility, encoding?: Value | null): Model;
    readonly cacheIdentifier: string;
    readonly checkpointVersion: string | null;
    readonly filesystemPath: string | null;
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
    readonly repoIds: Array<string>;
}

export class ModelAccessibilityOnDevice {
    free(): void;
    [Symbol.dispose](): void;
    constructor(source: ModelSource);
}

export class ModelAccessibilityRemote {
    free(): void;
    [Symbol.dispose](): void;
    constructor(repository?: Repository | null);
}

export class ModelBackend {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, version: string, metadata: Metadata);
    readonly name: string;
}

export class ModelFamily {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, vendor: ModelVendor, metadata: Metadata);
    readonly name: string;
}

export class ModelProperties {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, size: number, version: string | undefined | null, metadata: Metadata);
    readonly name: string;
}

export class ModelQuantization {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, method: string, bitsPerWeight: number, vendor: ModelVendor, metadata: Metadata);
    readonly name: string;
}

export class ModelRegistry {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, metadata: Metadata);
    readonly name: string;
}

export class ModelSourceFilesystem {
    free(): void;
    [Symbol.dispose](): void;
    constructor(path: string);
}

export class ModelSourceRegistry {
    free(): void;
    [Symbol.dispose](): void;
    constructor(toolchainVersion: string, repository: Repository | undefined | null, sourceRepository: Repository | undefined | null, files: Array<File>);
}

export class ModelSpecializationChat {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ModelSpecializationClassification {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ModelSpecializationDecision {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ModelSpecializationSpeculation {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ModelSpecializationTextToSpeech {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ModelSpecializationTranslation {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class ModelVendor {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, metadata: Metadata);
    readonly name: string;
}

export class NativeTool {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The callback receives the JSON arguments and an invocation ID, and returns JSON or a promise
     * of JSON. `cancel` receives the invocation ID when the call is cancelled.
     */
    constructor(definition: ToolFunction, invoke_json: (arg0: string, arg1: string) => Promise<string>, cancel: (arg: string) => void);
}

export class PcmBatch {
    free(): void;
    [Symbol.dispose](): void;
    constructor(samples: Array<number>, sampleRate: number, channels: number, lengths: Array<number>);
    saveAsWav(path: string): void;
    readonly batchSize: number;
    readonly duration: number;
    readonly totalFrames: number;
}

export class ReplyEnergyComponents {
    free(): void;
    [Symbol.dispose](): void;
    constructor(cpu: number, gpu: number, ane: number, dram: number);
}

export class ReplyEnergyTotal {
    free(): void;
    [Symbol.dispose](): void;
    constructor(total: number);
}

export class Repository {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string, commitHash?: string | null, paths?: Array<string> | null);
}

export class SamplingMethodGreedy {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class SamplingMethodStochastic {
    free(): void;
    [Symbol.dispose](): void;
    constructor(temperature?: number | null, topK?: number | null, topP?: number | null, minP?: number | null, repetitionPenalty?: number | null, suffixRepetitionLength?: number | null);
}

export class SamplingParameters {
    free(): void;
    [Symbol.dispose](): void;
    constructor(temperature?: number | null, topK?: number | null, topP?: number | null, minP?: number | null, repetitionPenalty?: number | null, suffixRepetitionLength?: number | null);
}

export class SamplingPolicyCustom {
    free(): void;
    [Symbol.dispose](): void;
    constructor(method: SamplingMethod);
}

export class SamplingPolicyDefault {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class SamplingSeedCustom {
    free(): void;
    [Symbol.dispose](): void;
    constructor(seed: number);
}

export class SamplingSeedDefault {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
}

export class TextToSpeechOutput {
    free(): void;
    [Symbol.dispose](): void;
    constructor(pcmBatch: PcmBatch, stats: TextToSpeechStats);
}

export class TextToSpeechStats {
    free(): void;
    [Symbol.dispose](): void;
    constructor(textLength: number, firstChunkSeconds: number, generationDuration: number, audioDuration: number);
}

export class Token {
    free(): void;
    [Symbol.dispose](): void;
    constructor(id: TokenId, value: TokenValue, isSpecial: boolean);
}

export class ToolCall {
    free(): void;
    [Symbol.dispose](): void;
    constructor(identifier: string | undefined | null, name: string, _arguments: Value);
}

export class ToolCallArgumentDelta {
    free(): void;
    [Symbol.dispose](): void;
    constructor(name: string, _arguments: string);
}

/**
 * Converts model tool-call candidates into append-only JSON argument fragments.
 */
export class ToolCallArgumentStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(tools: Array<ToolNamespace>): ToolCallArgumentStream;
    finish(call: ToolCall): ToolCallArgumentDelta;
    update(raw: string): ToolCallArgumentDelta;
}

export class ToolDescriptionFunction {
    free(): void;
    [Symbol.dispose](): void;
    constructor(toolFunction: ToolFunction);
}

export class ToolFunction {
    free(): void;
    [Symbol.dispose](): void;
    constructor(name: string, description: string, parameters?: Value | null, returnDefinition?: Value | null);
}

export class ToolNamespace {
    free(): void;
    [Symbol.dispose](): void;
    constructor(name: string, description: string | undefined | null, tools: Array<ToolDescription>);
}

export class TranslationPayloadImage {
    free(): void;
    [Symbol.dispose](): void;
    constructor(url: string);
}

export class TranslationPayloadText {
    free(): void;
    [Symbol.dispose](): void;
    constructor(text: string);
}

export class Value {
    free(): void;
    [Symbol.dispose](): void;
    constructor(json: string);
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_canceltoken_free: (a: number, b: number) => void;
    readonly __wbg_chatconfig_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockaudio_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockbuiltintools_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockconversationstartdate_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockcustom_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockfile_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockidentity_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockimage_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockknowledgecutoff_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockreasoning_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockreasoningeffort_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktext_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktoolcall_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktoolcallcandidate_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktoolcallresult_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktools_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktranslation_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblocktype_free: (a: number, b: number) => void;
    readonly __wbg_chatcontentblockvideo_free: (a: number, b: number) => void;
    readonly __wbg_chatinstance_free: (a: number, b: number) => void;
    readonly __wbg_chatmessage_free: (a: number, b: number) => void;
    readonly __wbg_chatmodelcapabilities_free: (a: number, b: number) => void;
    readonly __wbg_chatmodelinfo_free: (a: number, b: number) => void;
    readonly __wbg_chatreply_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyconfig_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyfinishreason_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyjoulespertokencomponents_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyjoulespertokentotal_free: (a: number, b: number) => void;
    readonly __wbg_chatreplyspeculatorstats_free: (a: number, b: number) => void;
    readonly __wbg_chatreplystats_free: (a: number, b: number) => void;
    readonly __wbg_chatroleassistant_free: (a: number, b: number) => void;
    readonly __wbg_chatrolecustom_free: (a: number, b: number) => void;
    readonly __wbg_chatroledeveloper_free: (a: number, b: number) => void;
    readonly __wbg_chatrolesystem_free: (a: number, b: number) => void;
    readonly __wbg_chatroletool_free: (a: number, b: number) => void;
    readonly __wbg_chatroleuser_free: (a: number, b: number) => void;
    readonly __wbg_chatsession_free: (a: number, b: number) => void;
    readonly __wbg_chatsessionstate_free: (a: number, b: number) => void;
    readonly __wbg_chatsessionstream_free: (a: number, b: number) => void;
    readonly __wbg_chatsessionstreamchunkerror_free: (a: number, b: number) => void;
    readonly __wbg_chatsessionstreamchunkreplies_free: (a: number, b: number) => void;
    readonly __wbg_chatsessionstreamchunktoolresults_free: (a: number, b: number) => void;
    readonly __wbg_classificationlabel_free: (a: number, b: number) => void;
    readonly __wbg_classificationmessage_free: (a: number, b: number) => void;
    readonly __wbg_classificationoutput_free: (a: number, b: number) => void;
    readonly __wbg_classificationrole_free: (a: number, b: number) => void;
    readonly __wbg_classificationstats_free: (a: number, b: number) => void;
    readonly __wbg_contextlengthcustom_free: (a: number, b: number) => void;
    readonly __wbg_contextlengthdefault_free: (a: number, b: number) => void;
    readonly __wbg_contextlengthmaximal_free: (a: number, b: number) => void;
    readonly __wbg_decisionanswerchoice_free: (a: number, b: number) => void;
    readonly __wbg_decisionanswerscore_free: (a: number, b: number) => void;
    readonly __wbg_decisionansweryesno_free: (a: number, b: number) => void;
    readonly __wbg_decisioninput_free: (a: number, b: number) => void;
    readonly __wbg_decisionoption_free: (a: number, b: number) => void;
    readonly __wbg_decisionoutput_free: (a: number, b: number) => void;
    readonly __wbg_decisionquestionchoice_free: (a: number, b: number) => void;
    readonly __wbg_decisionquestionscore_free: (a: number, b: number) => void;
    readonly __wbg_decisionquestionyesno_free: (a: number, b: number) => void;
    readonly __wbg_decisionresult_free: (a: number, b: number) => void;
    readonly __wbg_decisionsession_free: (a: number, b: number) => void;
    readonly __wbg_decisionsessionstate_free: (a: number, b: number) => void;
    readonly __wbg_decisionstats_free: (a: number, b: number) => void;
    readonly __wbg_device_free: (a: number, b: number) => void;
    readonly __wbg_downloader_free: (a: number, b: number) => void;
    readonly __wbg_downloaderstream_free: (a: number, b: number) => void;
    readonly __wbg_downloadmanagertype_free: (a: number, b: number) => void;
    readonly __wbg_downloadphasedownloaded_free: (a: number, b: number) => void;
    readonly __wbg_downloadphasedownloading_free: (a: number, b: number) => void;
    readonly __wbg_downloadphaseerror_free: (a: number, b: number) => void;
    readonly __wbg_downloadphaseinitializing_free: (a: number, b: number) => void;
    readonly __wbg_downloadphaselocked_free: (a: number, b: number) => void;
    readonly __wbg_downloadphasenotdownloaded_free: (a: number, b: number) => void;
    readonly __wbg_downloadphasepaused_free: (a: number, b: number) => void;
    readonly __wbg_downloadstate_free: (a: number, b: number) => void;
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly __wbg_enginecallback_free: (a: number, b: number) => void;
    readonly __wbg_engineconfig_free: (a: number, b: number) => void;
    readonly __wbg_feature_free: (a: number, b: number) => void;
    readonly __wbg_file_free: (a: number, b: number) => void;
    readonly __wbg_grammarjsonany_free: (a: number, b: number) => void;
    readonly __wbg_grammarjsonschema_free: (a: number, b: number) => void;
    readonly __wbg_grammarregex_free: (a: number, b: number) => void;
    readonly __wbg_hash_free: (a: number, b: number) => void;
    readonly __wbg_hashmethod_free: (a: number, b: number) => void;
    readonly __wbg_image_free: (a: number, b: number) => void;
    readonly __wbg_imageformat_free: (a: number, b: number) => void;
    readonly __wbg_imagetheme_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingbytesource_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingsink_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingsource_free: (a: number, b: number) => void;
    readonly __wbg_metadata_free: (a: number, b: number) => void;
    readonly __wbg_model_free: (a: number, b: number) => void;
    readonly __wbg_modelaccessibilityondevice_free: (a: number, b: number) => void;
    readonly __wbg_modelaccessibilityremote_free: (a: number, b: number) => void;
    readonly __wbg_modelbackend_free: (a: number, b: number) => void;
    readonly __wbg_modelfamily_free: (a: number, b: number) => void;
    readonly __wbg_modelproperties_free: (a: number, b: number) => void;
    readonly __wbg_modelquantization_free: (a: number, b: number) => void;
    readonly __wbg_modelregistry_free: (a: number, b: number) => void;
    readonly __wbg_modelsourcefilesystem_free: (a: number, b: number) => void;
    readonly __wbg_modelsourceregistry_free: (a: number, b: number) => void;
    readonly __wbg_modelspecializationchat_free: (a: number, b: number) => void;
    readonly __wbg_modelspecializationclassification_free: (a: number, b: number) => void;
    readonly __wbg_modelspecializationdecision_free: (a: number, b: number) => void;
    readonly __wbg_modelspecializationspeculation_free: (a: number, b: number) => void;
    readonly __wbg_modelspecializationtexttospeech_free: (a: number, b: number) => void;
    readonly __wbg_modelspecializationtranslation_free: (a: number, b: number) => void;
    readonly __wbg_modelvendor_free: (a: number, b: number) => void;
    readonly __wbg_nativetool_free: (a: number, b: number) => void;
    readonly __wbg_pcmbatch_free: (a: number, b: number) => void;
    readonly __wbg_reasoningeffort_free: (a: number, b: number) => void;
    readonly __wbg_replyenergycomponents_free: (a: number, b: number) => void;
    readonly __wbg_replyenergytotal_free: (a: number, b: number) => void;
    readonly __wbg_repository_free: (a: number, b: number) => void;
    readonly __wbg_samplingmethodgreedy_free: (a: number, b: number) => void;
    readonly __wbg_samplingmethodstochastic_free: (a: number, b: number) => void;
    readonly __wbg_samplingparameters_free: (a: number, b: number) => void;
    readonly __wbg_samplingpolicycustom_free: (a: number, b: number) => void;
    readonly __wbg_samplingpolicydefault_free: (a: number, b: number) => void;
    readonly __wbg_samplingseedcustom_free: (a: number, b: number) => void;
    readonly __wbg_samplingseeddefault_free: (a: number, b: number) => void;
    readonly __wbg_texttospeechoutput_free: (a: number, b: number) => void;
    readonly __wbg_texttospeechstats_free: (a: number, b: number) => void;
    readonly __wbg_token_free: (a: number, b: number) => void;
    readonly __wbg_toolcall_free: (a: number, b: number) => void;
    readonly __wbg_toolcallargumentdelta_free: (a: number, b: number) => void;
    readonly __wbg_toolcallargumentstream_free: (a: number, b: number) => void;
    readonly __wbg_tooldescriptionfunction_free: (a: number, b: number) => void;
    readonly __wbg_toolfunction_free: (a: number, b: number) => void;
    readonly __wbg_toolnamespace_free: (a: number, b: number) => void;
    readonly __wbg_translationpayloadimage_free: (a: number, b: number) => void;
    readonly __wbg_translationpayloadtext_free: (a: number, b: number) => void;
    readonly __wbg_value_free: (a: number, b: number) => void;
    readonly canceltoken___wasm_is_cancelled: (a: number) => [number, number, number];
    readonly canceltoken_cancel: (a: number) => [number, number];
    readonly canceltoken_create: () => [number, number, number];
    readonly chatconfig___uzuCloneChatConfig: (a: number) => number;
    readonly chatconfig___wasm_get_context_length: (a: number) => any;
    readonly chatconfig___wasm_get_sampling_seed: (a: number) => any;
    readonly chatconfig___wasm_new: (a: any, b: any) => [number, number, number];
    readonly chatconfig___wasm_set_context_length: (a: number, b: any) => void;
    readonly chatconfig___wasm_set_sampling_seed: (a: number, b: any) => void;
    readonly chatconfig_create: () => [number, number, number];
    readonly chatconfig_withContextLength: (a: number, b: any) => [number, number, number];
    readonly chatconfig_withSamplingSeed: (a: number, b: any) => [number, number, number];
    readonly chatcontentblockaudio___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockaudio___wasm_get_url: (a: number) => any;
    readonly chatcontentblockaudio___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockaudio___wasm_set_url: (a: number, b: any) => void;
    readonly chatcontentblockbuiltintools___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockbuiltintools___wasm_get_names: (a: number) => any;
    readonly chatcontentblockbuiltintools___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockbuiltintools___wasm_set_names: (a: number, b: any) => void;
    readonly chatcontentblockconversationstartdate___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockconversationstartdate___wasm_get_value: (a: number) => any;
    readonly chatcontentblockconversationstartdate___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockconversationstartdate___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblockcustom___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockcustom___wasm_get_value: (a: number) => any;
    readonly chatcontentblockcustom___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockcustom___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblockfile___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockfile___wasm_get_url: (a: number) => any;
    readonly chatcontentblockfile___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockfile___wasm_set_url: (a: number, b: any) => void;
    readonly chatcontentblockidentity___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockidentity___wasm_get_value: (a: number) => any;
    readonly chatcontentblockidentity___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockidentity___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblockimage___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockimage___wasm_get_url: (a: number) => any;
    readonly chatcontentblockimage___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockimage___wasm_set_url: (a: number, b: any) => void;
    readonly chatcontentblockknowledgecutoff___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockknowledgecutoff___wasm_get_value: (a: number) => any;
    readonly chatcontentblockknowledgecutoff___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockknowledgecutoff___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblockreasoning___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockreasoning___wasm_get_value: (a: number) => any;
    readonly chatcontentblockreasoning___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockreasoning___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblockreasoningeffort___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockreasoningeffort___wasm_get_value: (a: number) => any;
    readonly chatcontentblockreasoningeffort___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockreasoningeffort___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblocktext___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblocktext___wasm_get_value: (a: number) => any;
    readonly chatcontentblocktext___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblocktext___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblocktoolcall___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblocktoolcall___wasm_get_value: (a: number) => any;
    readonly chatcontentblocktoolcall___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblocktoolcall___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblocktoolcallcandidate___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblocktoolcallcandidate___wasm_get_identifier: (a: number) => any;
    readonly chatcontentblocktoolcallcandidate___wasm_get_value: (a: number) => any;
    readonly chatcontentblocktoolcallcandidate___wasm_new: (a: any, b: any) => [number, number, number];
    readonly chatcontentblocktoolcallcandidate___wasm_set_identifier: (a: number, b: any) => void;
    readonly chatcontentblocktoolcallcandidate___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblocktoolcallresult___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblocktoolcallresult___wasm_get_identifier: (a: number) => any;
    readonly chatcontentblocktoolcallresult___wasm_get_name: (a: number) => any;
    readonly chatcontentblocktoolcallresult___wasm_get_value: (a: number) => any;
    readonly chatcontentblocktoolcallresult___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly chatcontentblocktoolcallresult___wasm_set_identifier: (a: number, b: any) => void;
    readonly chatcontentblocktoolcallresult___wasm_set_name: (a: number, b: any) => void;
    readonly chatcontentblocktoolcallresult___wasm_set_value: (a: number, b: any) => void;
    readonly chatcontentblocktools___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblocktools___wasm_get_namespaces: (a: number) => any;
    readonly chatcontentblocktools___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblocktools___wasm_set_namespaces: (a: number, b: any) => void;
    readonly chatcontentblocktranslation___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblocktranslation___wasm_get_payload: (a: number) => any;
    readonly chatcontentblocktranslation___wasm_get_source_language_code: (a: number) => any;
    readonly chatcontentblocktranslation___wasm_get_target_language_code: (a: number) => any;
    readonly chatcontentblocktranslation___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly chatcontentblocktranslation___wasm_set_payload: (a: number, b: any) => void;
    readonly chatcontentblocktranslation___wasm_set_source_language_code: (a: number, b: any) => void;
    readonly chatcontentblocktranslation___wasm_set_target_language_code: (a: number, b: any) => void;
    readonly chatcontentblocktype___wasm_variant_Audio: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_BuiltinTools: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_ConversationStartDate: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Custom: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_File: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Identity: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Image: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_KnowledgeCutoff: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Reasoning: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_ReasoningEffort: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Text: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_ToolCall: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_ToolCallCandidate: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_ToolCallResult: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Tools: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Translation: () => [number, number];
    readonly chatcontentblocktype___wasm_variant_Video: () => [number, number];
    readonly chatcontentblockvideo___uzuCloneChatContentBlock: (a: number) => number;
    readonly chatcontentblockvideo___wasm_get_url: (a: number) => any;
    readonly chatcontentblockvideo___wasm_new: (a: any) => [number, number, number];
    readonly chatcontentblockvideo___wasm_set_url: (a: number, b: any) => void;
    readonly chatinstance___wasm_model: (a: number) => [number, number, number];
    readonly chatmessage___uzuCloneChatMessage: (a: number) => number;
    readonly chatmessage___wasm_get_content: (a: number) => any;
    readonly chatmessage___wasm_get_metadata: (a: number) => any;
    readonly chatmessage___wasm_get_role: (a: number) => any;
    readonly chatmessage___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly chatmessage___wasm_reasoning: (a: number) => [number, number, number];
    readonly chatmessage___wasm_reasoning_effort: (a: number) => [number, number, number];
    readonly chatmessage___wasm_set_content: (a: number, b: any) => void;
    readonly chatmessage___wasm_set_metadata: (a: number, b: any) => void;
    readonly chatmessage___wasm_set_role: (a: number, b: any) => void;
    readonly chatmessage___wasm_text: (a: number) => [number, number, number];
    readonly chatmessage___wasm_tool_calls: (a: number) => [number, number, number];
    readonly chatmessage___wasm_tool_namespaces: (a: number) => [number, number, number];
    readonly chatmessage_assistant: () => [number, number, number];
    readonly chatmessage_developer: () => [number, number, number];
    readonly chatmessage_for_role: (a: any) => [number, number, number];
    readonly chatmessage_system: () => [number, number, number];
    readonly chatmessage_tool: () => [number, number, number];
    readonly chatmessage_user: () => [number, number, number];
    readonly chatmessage_withBlock: (a: number, b: any) => [number, number, number];
    readonly chatmessage_withReasoning: (a: number, b: any) => [number, number, number];
    readonly chatmessage_withReasoningEffort: (a: number, b: any) => [number, number, number];
    readonly chatmessage_withText: (a: number, b: any) => [number, number, number];
    readonly chatmessage_withToolCall: (a: number, b: any) => [number, number, number];
    readonly chatmessage_withToolCallCandidate: (a: number, b: any) => [number, number, number];
    readonly chatmessage_withToolNamespaces: (a: number, b: any) => [number, number, number];
    readonly chatmodelcapabilities___uzuCloneChatModelCapabilities: (a: number) => number;
    readonly chatmodelcapabilities___wasm_get_reasoning_efforts: (a: number) => any;
    readonly chatmodelcapabilities___wasm_get_requires_tools: (a: number) => any;
    readonly chatmodelcapabilities___wasm_get_supports_disable_reasoning: (a: number) => any;
    readonly chatmodelcapabilities___wasm_get_supports_multiple_tool_calls: (a: number) => any;
    readonly chatmodelcapabilities___wasm_get_supports_reasoning: (a: number) => any;
    readonly chatmodelcapabilities___wasm_get_supports_tools: (a: number) => any;
    readonly chatmodelcapabilities___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any) => [number, number, number];
    readonly chatmodelcapabilities___wasm_set_reasoning_efforts: (a: number, b: any) => void;
    readonly chatmodelcapabilities___wasm_set_requires_tools: (a: number, b: any) => void;
    readonly chatmodelcapabilities___wasm_set_supports_disable_reasoning: (a: number, b: any) => void;
    readonly chatmodelcapabilities___wasm_set_supports_multiple_tool_calls: (a: number, b: any) => void;
    readonly chatmodelcapabilities___wasm_set_supports_reasoning: (a: number, b: any) => void;
    readonly chatmodelcapabilities___wasm_set_supports_tools: (a: number, b: any) => void;
    readonly chatmodelinfo___uzuCloneChatModelInfo: (a: number) => number;
    readonly chatmodelinfo___wasm_get_capabilities: (a: number) => any;
    readonly chatmodelinfo___wasm_get_context_length: (a: number) => any;
    readonly chatmodelinfo___wasm_get_default_reasoning_effort: (a: number) => any;
    readonly chatmodelinfo___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly chatmodelinfo___wasm_set_capabilities: (a: number, b: any) => void;
    readonly chatmodelinfo___wasm_set_context_length: (a: number, b: any) => void;
    readonly chatmodelinfo___wasm_set_default_reasoning_effort: (a: number, b: any) => void;
    readonly chatreply___uzuCloneChatReply: (a: number) => number;
    readonly chatreply___wasm_get_finish_reason: (a: number) => any;
    readonly chatreply___wasm_get_message: (a: number) => any;
    readonly chatreply___wasm_get_stats: (a: number) => any;
    readonly chatreply___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly chatreply___wasm_set_finish_reason: (a: number, b: any) => void;
    readonly chatreply___wasm_set_message: (a: number, b: any) => void;
    readonly chatreply___wasm_set_stats: (a: number, b: any) => void;
    readonly chatreplyconfig___uzuCloneChatReplyConfig: (a: number) => number;
    readonly chatreplyconfig___wasm_get_grammar: (a: number) => any;
    readonly chatreplyconfig___wasm_get_sampling_policy: (a: number) => any;
    readonly chatreplyconfig___wasm_get_token_limit: (a: number) => any;
    readonly chatreplyconfig___wasm_get_tool_turn_limit: (a: number) => any;
    readonly chatreplyconfig___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly chatreplyconfig___wasm_set_grammar: (a: number, b: any) => void;
    readonly chatreplyconfig___wasm_set_sampling_policy: (a: number, b: any) => void;
    readonly chatreplyconfig___wasm_set_token_limit: (a: number, b: any) => void;
    readonly chatreplyconfig___wasm_set_tool_turn_limit: (a: number, b: any) => void;
    readonly chatreplyconfig_create: () => [number, number, number];
    readonly chatreplyconfig_withGrammar: (a: number, b: any) => [number, number, number];
    readonly chatreplyconfig_withSamplingMethod: (a: number, b: any) => [number, number, number];
    readonly chatreplyconfig_withSamplingPolicy: (a: number, b: any) => [number, number, number];
    readonly chatreplyconfig_withTokenLimit: (a: number, b: any) => [number, number, number];
    readonly chatreplyconfig_withToolTurnLimit: (a: number, b: any) => [number, number, number];
    readonly chatreplyfinishreason___wasm_variant_Cancelled: () => [number, number];
    readonly chatreplyfinishreason___wasm_variant_ContextLimitReached: () => [number, number];
    readonly chatreplyfinishreason___wasm_variant_Length: () => [number, number];
    readonly chatreplyfinishreason___wasm_variant_Rejected: () => [number, number];
    readonly chatreplyfinishreason___wasm_variant_Stop: () => [number, number];
    readonly chatreplyfinishreason___wasm_variant_ToolCalls: () => [number, number];
    readonly chatreplyjoulespertokencomponents___uzuCloneChatReplyJoulesPerToken: (a: number) => number;
    readonly chatreplyjoulespertokencomponents___wasm_get_ane: (a: number) => any;
    readonly chatreplyjoulespertokencomponents___wasm_get_cpu: (a: number) => any;
    readonly chatreplyjoulespertokencomponents___wasm_get_dram: (a: number) => any;
    readonly chatreplyjoulespertokencomponents___wasm_get_gpu: (a: number) => any;
    readonly chatreplyjoulespertokencomponents___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly chatreplyjoulespertokencomponents___wasm_set_ane: (a: number, b: any) => void;
    readonly chatreplyjoulespertokencomponents___wasm_set_cpu: (a: number, b: any) => void;
    readonly chatreplyjoulespertokencomponents___wasm_set_dram: (a: number, b: any) => void;
    readonly chatreplyjoulespertokencomponents___wasm_set_gpu: (a: number, b: any) => void;
    readonly chatreplyjoulespertokentotal___uzuCloneChatReplyJoulesPerToken: (a: number) => number;
    readonly chatreplyjoulespertokentotal___wasm_get_total: (a: number) => any;
    readonly chatreplyjoulespertokentotal___wasm_new: (a: any) => [number, number, number];
    readonly chatreplyjoulespertokentotal___wasm_set_total: (a: number, b: any) => void;
    readonly chatreplyspeculatorstats___uzuCloneChatReplySpeculatorStats: (a: number) => number;
    readonly chatreplyspeculatorstats___wasm_get_num_decode_forward_passes: (a: number) => any;
    readonly chatreplyspeculatorstats___wasm_get_tokens_per_forward_pass: (a: number) => any;
    readonly chatreplyspeculatorstats___wasm_new: (a: any, b: any) => [number, number, number];
    readonly chatreplyspeculatorstats___wasm_set_num_decode_forward_passes: (a: number, b: any) => void;
    readonly chatreplyspeculatorstats___wasm_set_tokens_per_forward_pass: (a: number, b: any) => void;
    readonly chatreplystats___uzuCloneChatReplyStats: (a: number) => number;
    readonly chatreplystats___wasm_get_duration: (a: number) => any;
    readonly chatreplystats___wasm_get_generate_tokens_per_second: (a: number) => any;
    readonly chatreplystats___wasm_get_input_energy: (a: number) => any;
    readonly chatreplystats___wasm_get_memory_used_bytes: (a: number) => any;
    readonly chatreplystats___wasm_get_output_energy: (a: number) => any;
    readonly chatreplystats___wasm_get_prefill_tokens_per_second: (a: number) => any;
    readonly chatreplystats___wasm_get_speculator_stats: (a: number) => any;
    readonly chatreplystats___wasm_get_time_to_first_token: (a: number) => any;
    readonly chatreplystats___wasm_get_tokens_count_input: (a: number) => any;
    readonly chatreplystats___wasm_get_tokens_count_input_cached: (a: number) => any;
    readonly chatreplystats___wasm_get_tokens_count_output: (a: number) => any;
    readonly chatreplystats___wasm_input_joules_per_token: (a: number) => [number, number, number];
    readonly chatreplystats___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any, g: any, h: any, i: any, j: any, k: any) => [number, number, number];
    readonly chatreplystats___wasm_output_joules_per_token: (a: number) => [number, number, number];
    readonly chatreplystats___wasm_set_duration: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_generate_tokens_per_second: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_input_energy: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_memory_used_bytes: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_output_energy: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_prefill_tokens_per_second: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_speculator_stats: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_time_to_first_token: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_tokens_count_input: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_tokens_count_input_cached: (a: number, b: any) => void;
    readonly chatreplystats___wasm_set_tokens_count_output: (a: number, b: any) => void;
    readonly chatreplystats___wasm_tokens_count: (a: number) => [number, number, number];
    readonly chatreplystats___wasm_total_joules: (a: number) => [number, number, number];
    readonly chatroleassistant___uzuCloneChatRole: (a: number) => number;
    readonly chatroleassistant___wasm_new: () => [number, number, number];
    readonly chatrolecustom___uzuCloneChatRole: (a: number) => number;
    readonly chatrolecustom___wasm_get_name: (a: number) => any;
    readonly chatrolecustom___wasm_new: (a: any) => [number, number, number];
    readonly chatrolecustom___wasm_set_name: (a: number, b: any) => void;
    readonly chatroledeveloper___uzuCloneChatRole: (a: number) => number;
    readonly chatroledeveloper___wasm_new: () => [number, number, number];
    readonly chatrolesystem___uzuCloneChatRole: (a: number) => number;
    readonly chatrolesystem___wasm_new: () => [number, number, number];
    readonly chatroletool___uzuCloneChatRole: (a: number) => number;
    readonly chatroletool___wasm_new: () => [number, number, number];
    readonly chatroleuser___uzuCloneChatRole: (a: number) => number;
    readonly chatroleuser___wasm_new: () => [number, number, number];
    readonly chatsession___wasm_messages: (a: number) => any;
    readonly chatsession___wasm_sampling_defaults: (a: number) => any;
    readonly chatsession___wasm_state: (a: number) => any;
    readonly chatsession___wasm_supports_tool_calls: (a: number) => any;
    readonly chatsession_addTool: (a: number, b: number) => any;
    readonly chatsession_addTools: (a: number, b: any) => any;
    readonly chatsession_reply: (a: number, b: any, c: any) => any;
    readonly chatsession_replyWithStream: (a: number, b: any, c: any) => any;
    readonly chatsession_reset: (a: number) => any;
    readonly chatsessionstate___wasm_variant_Generation: () => [number, number];
    readonly chatsessionstate___wasm_variant_Idle: () => [number, number];
    readonly chatsessionstate___wasm_variant_Resetting: () => [number, number];
    readonly chatsessionstate___wasm_variant_ToolCalling: () => [number, number];
    readonly chatsessionstream___wasm_cancel_token: (a: number) => [number, number, number];
    readonly chatsessionstream___wasm_stream_marker: () => number;
    readonly chatsessionstream_next: (a: number) => any;
    readonly chatsessionstreamchunkerror___uzuCloneChatSessionStreamChunk: (a: number) => number;
    readonly chatsessionstreamchunkerror___wasm_get_error: (a: number) => any;
    readonly chatsessionstreamchunkerror___wasm_new: (a: any) => [number, number, number];
    readonly chatsessionstreamchunkerror___wasm_set_error: (a: number, b: any) => void;
    readonly chatsessionstreamchunkreplies___uzuCloneChatSessionStreamChunk: (a: number) => number;
    readonly chatsessionstreamchunkreplies___wasm_get_replies: (a: number) => any;
    readonly chatsessionstreamchunkreplies___wasm_new: (a: any) => [number, number, number];
    readonly chatsessionstreamchunkreplies___wasm_set_replies: (a: number, b: any) => void;
    readonly chatsessionstreamchunktoolresults___uzuCloneChatSessionStreamChunk: (a: number) => number;
    readonly chatsessionstreamchunktoolresults___wasm_get_messages: (a: number) => any;
    readonly chatsessionstreamchunktoolresults___wasm_new: (a: any) => [number, number, number];
    readonly chatsessionstreamchunktoolresults___wasm_set_messages: (a: number, b: any) => void;
    readonly classificationlabel___uzuCloneClassificationLabel: (a: number) => number;
    readonly classificationlabel___wasm_get_description: (a: number) => any;
    readonly classificationlabel___wasm_get_index: (a: number) => any;
    readonly classificationlabel___wasm_get_label: (a: number) => any;
    readonly classificationlabel___wasm_get_recommended_threshold: (a: number) => any;
    readonly classificationlabel___wasm_get_title: (a: number) => any;
    readonly classificationlabel___wasm_new: (a: any, b: any, c: any, d: any, e: any) => [number, number, number];
    readonly classificationlabel___wasm_set_description: (a: number, b: any) => void;
    readonly classificationlabel___wasm_set_index: (a: number, b: any) => void;
    readonly classificationlabel___wasm_set_label: (a: number, b: any) => void;
    readonly classificationlabel___wasm_set_recommended_threshold: (a: number, b: any) => void;
    readonly classificationlabel___wasm_set_title: (a: number, b: any) => void;
    readonly classificationmessage___uzuCloneClassificationMessage: (a: number) => number;
    readonly classificationmessage___wasm_get_content: (a: number) => any;
    readonly classificationmessage___wasm_get_role: (a: number) => any;
    readonly classificationmessage___wasm_new: (a: any, b: any) => [number, number, number];
    readonly classificationmessage___wasm_set_content: (a: number, b: any) => void;
    readonly classificationmessage___wasm_set_role: (a: number, b: any) => void;
    readonly classificationmessage_assistant: (a: any) => [number, number, number];
    readonly classificationmessage_user: (a: any) => [number, number, number];
    readonly classificationoutput___uzuCloneClassificationOutput: (a: number) => number;
    readonly classificationoutput___wasm_get_logits: (a: number) => any;
    readonly classificationoutput___wasm_get_probabilities: (a: number) => any;
    readonly classificationoutput___wasm_get_stats: (a: number) => any;
    readonly classificationoutput___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly classificationoutput___wasm_set_logits: (a: number, b: any) => void;
    readonly classificationoutput___wasm_set_probabilities: (a: number, b: any) => void;
    readonly classificationoutput___wasm_set_stats: (a: number, b: any) => void;
    readonly classificationrole___wasm_variant_Assistant: () => [number, number];
    readonly classificationrole___wasm_variant_User: () => [number, number];
    readonly classificationstats___uzuCloneClassificationStats: (a: number) => number;
    readonly classificationstats___wasm_get_confidence: (a: number) => any;
    readonly classificationstats___wasm_get_forward_pass_duration: (a: number) => any;
    readonly classificationstats___wasm_get_postprocessing_duration: (a: number) => any;
    readonly classificationstats___wasm_get_predicted_label: (a: number) => any;
    readonly classificationstats___wasm_get_preprocessing_duration: (a: number) => any;
    readonly classificationstats___wasm_get_tokens_count: (a: number) => any;
    readonly classificationstats___wasm_get_tokens_per_second: (a: number) => any;
    readonly classificationstats___wasm_get_total_duration: (a: number) => any;
    readonly classificationstats___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any, g: any, h: any) => [number, number, number];
    readonly classificationstats___wasm_set_confidence: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_forward_pass_duration: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_postprocessing_duration: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_predicted_label: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_preprocessing_duration: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_tokens_count: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_tokens_per_second: (a: number, b: any) => void;
    readonly classificationstats___wasm_set_total_duration: (a: number, b: any) => void;
    readonly contextlengthcustom___uzuCloneContextLength: (a: number) => number;
    readonly contextlengthcustom___wasm_get_length: (a: number) => any;
    readonly contextlengthcustom___wasm_new: (a: any) => [number, number, number];
    readonly contextlengthcustom___wasm_set_length: (a: number, b: any) => void;
    readonly contextlengthdefault___uzuCloneContextLength: (a: number) => number;
    readonly contextlengthdefault___wasm_new: () => [number, number, number];
    readonly contextlengthmaximal___uzuCloneContextLength: (a: number) => number;
    readonly contextlengthmaximal___wasm_new: () => [number, number, number];
    readonly decisionanswerchoice___uzuCloneDecisionAnswer: (a: number) => number;
    readonly decisionanswerchoice___wasm_get_index: (a: number) => any;
    readonly decisionanswerchoice___wasm_get_key: (a: number) => any;
    readonly decisionanswerchoice___wasm_get_probabilities: (a: number) => any;
    readonly decisionanswerchoice___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly decisionanswerchoice___wasm_set_index: (a: number, b: any) => void;
    readonly decisionanswerchoice___wasm_set_key: (a: number, b: any) => void;
    readonly decisionanswerchoice___wasm_set_probabilities: (a: number, b: any) => void;
    readonly decisionanswerscore___uzuCloneDecisionAnswer: (a: number) => number;
    readonly decisionanswerscore___wasm_get_probabilities: (a: number) => any;
    readonly decisionanswerscore___wasm_get_score: (a: number) => any;
    readonly decisionanswerscore___wasm_new: (a: any, b: any) => [number, number, number];
    readonly decisionanswerscore___wasm_set_probabilities: (a: number, b: any) => void;
    readonly decisionanswerscore___wasm_set_score: (a: number, b: any) => void;
    readonly decisionansweryesno___uzuCloneDecisionAnswer: (a: number) => number;
    readonly decisionansweryesno___wasm_get_yes_probability: (a: number) => any;
    readonly decisionansweryesno___wasm_new: (a: any) => [number, number, number];
    readonly decisionansweryesno___wasm_set_yes_probability: (a: number, b: any) => void;
    readonly decisioninput___uzuCloneDecisionInput: (a: number) => number;
    readonly decisioninput___wasm_get_questions: (a: number) => any;
    readonly decisioninput___wasm_get_state: (a: number) => any;
    readonly decisioninput___wasm_new: (a: any, b: any) => [number, number, number];
    readonly decisioninput___wasm_set_questions: (a: number, b: any) => void;
    readonly decisioninput___wasm_set_state: (a: number, b: any) => void;
    readonly decisionoption___uzuCloneDecisionOption: (a: number) => number;
    readonly decisionoption___wasm_get_description: (a: number) => any;
    readonly decisionoption___wasm_get_key: (a: number) => any;
    readonly decisionoption___wasm_new: (a: any, b: any) => [number, number, number];
    readonly decisionoption___wasm_set_description: (a: number, b: any) => void;
    readonly decisionoption___wasm_set_key: (a: number, b: any) => void;
    readonly decisionoutput___uzuCloneDecisionOutput: (a: number) => number;
    readonly decisionoutput___wasm_get_results: (a: number) => any;
    readonly decisionoutput___wasm_new: (a: any) => [number, number, number];
    readonly decisionoutput___wasm_set_results: (a: number, b: any) => void;
    readonly decisionquestionchoice___uzuCloneDecisionQuestion: (a: number) => number;
    readonly decisionquestionchoice___wasm_get_instructions: (a: number) => any;
    readonly decisionquestionchoice___wasm_get_options: (a: number) => any;
    readonly decisionquestionchoice___wasm_new: (a: any, b: any) => [number, number, number];
    readonly decisionquestionchoice___wasm_set_instructions: (a: number, b: any) => void;
    readonly decisionquestionchoice___wasm_set_options: (a: number, b: any) => void;
    readonly decisionquestionscore___uzuCloneDecisionQuestion: (a: number) => number;
    readonly decisionquestionscore___wasm_get_instructions: (a: number) => any;
    readonly decisionquestionscore___wasm_get_levels: (a: number) => any;
    readonly decisionquestionscore___wasm_new: (a: any, b: any) => [number, number, number];
    readonly decisionquestionscore___wasm_set_instructions: (a: number, b: any) => void;
    readonly decisionquestionscore___wasm_set_levels: (a: number, b: any) => void;
    readonly decisionquestionyesno___uzuCloneDecisionQuestion: (a: number) => number;
    readonly decisionquestionyesno___wasm_get_instructions: (a: number) => any;
    readonly decisionquestionyesno___wasm_get_no_description: (a: number) => any;
    readonly decisionquestionyesno___wasm_get_yes_description: (a: number) => any;
    readonly decisionquestionyesno___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly decisionquestionyesno___wasm_set_instructions: (a: number, b: any) => void;
    readonly decisionquestionyesno___wasm_set_no_description: (a: number, b: any) => void;
    readonly decisionquestionyesno___wasm_set_yes_description: (a: number, b: any) => void;
    readonly decisionresult___uzuCloneDecisionResult: (a: number) => number;
    readonly decisionresult___wasm_get_answer: (a: number) => any;
    readonly decisionresult___wasm_get_stats: (a: number) => any;
    readonly decisionresult___wasm_new: (a: any, b: any) => [number, number, number];
    readonly decisionresult___wasm_set_answer: (a: number, b: any) => void;
    readonly decisionresult___wasm_set_stats: (a: number, b: any) => void;
    readonly decisionsession___wasm_state: (a: number) => any;
    readonly decisionsession_decide: (a: number, b: any) => any;
    readonly decisionsessionstate___wasm_variant_Deciding: () => [number, number];
    readonly decisionsessionstate___wasm_variant_Idle: () => [number, number];
    readonly decisionstats___uzuCloneDecisionStats: (a: number) => number;
    readonly decisionstats___wasm_get_energy: (a: number) => any;
    readonly decisionstats___wasm_get_prefill_duration: (a: number) => any;
    readonly decisionstats___wasm_get_tokens_count: (a: number) => any;
    readonly decisionstats___wasm_get_total_duration: (a: number) => any;
    readonly decisionstats___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly decisionstats___wasm_set_energy: (a: number, b: any) => void;
    readonly decisionstats___wasm_set_prefill_duration: (a: number, b: any) => void;
    readonly decisionstats___wasm_set_tokens_count: (a: number, b: any) => void;
    readonly decisionstats___wasm_set_total_duration: (a: number, b: any) => void;
    readonly device___uzuCloneDevice: (a: number) => number;
    readonly device___wasm_get_cpu_name: (a: number) => any;
    readonly device___wasm_get_home_path: (a: number) => any;
    readonly device___wasm_get_memory_total: (a: number) => any;
    readonly device___wasm_get_os_name: (a: number) => any;
    readonly device___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly device___wasm_set_cpu_name: (a: number, b: any) => void;
    readonly device___wasm_set_home_path: (a: number, b: any) => void;
    readonly device___wasm_set_memory_total: (a: number, b: any) => void;
    readonly device___wasm_set_os_name: (a: number, b: any) => void;
    readonly device_create: () => [number, number, number];
    readonly downloader___wasm_state: (a: number) => any;
    readonly downloader_delete: (a: number) => any;
    readonly downloader_pause: (a: number) => any;
    readonly downloader_progress: (a: number) => any;
    readonly downloader_resume: (a: number) => any;
    readonly downloaderstream___wasm_stream_marker: () => number;
    readonly downloaderstream_next: (a: number) => any;
    readonly downloadmanagertype___wasm_variant_Universal: () => [number, number];
    readonly downloadphasedownloaded___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphasedownloaded___wasm_new: () => [number, number, number];
    readonly downloadphasedownloading___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphasedownloading___wasm_new: () => [number, number, number];
    readonly downloadphaseerror___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphaseerror___wasm_get_message: (a: number) => any;
    readonly downloadphaseerror___wasm_new: (a: any) => [number, number, number];
    readonly downloadphaseerror___wasm_set_message: (a: number, b: any) => void;
    readonly downloadphaseinitializing___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphaseinitializing___wasm_new: () => [number, number, number];
    readonly downloadphaselocked___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphaselocked___wasm_get_manager_id: (a: number) => any;
    readonly downloadphaselocked___wasm_new: (a: any) => [number, number, number];
    readonly downloadphaselocked___wasm_set_manager_id: (a: number, b: any) => void;
    readonly downloadphasenotdownloaded___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphasenotdownloaded___wasm_new: () => [number, number, number];
    readonly downloadphasepaused___uzuCloneDownloadPhase: (a: number) => number;
    readonly downloadphasepaused___wasm_new: () => [number, number, number];
    readonly downloadstate___uzuCloneDownloadState: (a: number) => number;
    readonly downloadstate___wasm_can_delete: (a: number) => [number, number, number];
    readonly downloadstate___wasm_can_pause: (a: number) => [number, number, number];
    readonly downloadstate___wasm_get_downloaded_bytes: (a: number) => any;
    readonly downloadstate___wasm_get_phase: (a: number) => any;
    readonly downloadstate___wasm_get_total_bytes: (a: number) => any;
    readonly downloadstate___wasm_is_in_progress: (a: number) => [number, number, number];
    readonly downloadstate___wasm_name: (a: number) => [number, number, number];
    readonly downloadstate___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly downloadstate___wasm_progress: (a: number) => [number, number, number];
    readonly downloadstate___wasm_set_downloaded_bytes: (a: number, b: any) => void;
    readonly downloadstate___wasm_set_phase: (a: number, b: any) => void;
    readonly downloadstate___wasm_set_total_bytes: (a: number, b: any) => void;
    readonly engine___wasm_download_states: (a: number) => any;
    readonly engine___wasm_model_families: (a: number) => any;
    readonly engine___wasm_model_registries: (a: number) => any;
    readonly engine___wasm_model_vendors: (a: number) => any;
    readonly engine___wasm_models: (a: number) => any;
    readonly engine___wasm_models_downloadable: (a: number) => any;
    readonly engine___wasm_models_for_chat: (a: number) => any;
    readonly engine___wasm_models_for_classification: (a: number) => any;
    readonly engine___wasm_models_for_speculation: (a: number) => any;
    readonly engine___wasm_models_for_text_to_speech: (a: number) => any;
    readonly engine___wasm_models_for_translation: (a: number) => any;
    readonly engine___wasm_models_on_device: (a: number) => any;
    readonly engine___wasm_models_remote: (a: number) => any;
    readonly engine_chat: (a: number, b: any, c: any) => any;
    readonly engine_chatInstance: (a: number, b: any, c: any) => any;
    readonly engine_chatModelInfo: (a: number, b: number) => any;
    readonly engine_chatWithInstance: (a: number, b: number) => any;
    readonly engine_create: (a: any) => any;
    readonly engine_decision: (a: number, b: any) => any;
    readonly engine_download: (a: number, b: number) => any;
    readonly engine_downloadState: (a: number, b: number) => any;
    readonly engine_downloader: (a: number, b: number) => [number, number, number];
    readonly engine_model: (a: number, b: any) => any;
    readonly engine_modelByIdentifier: (a: number, b: any) => any;
    readonly engine_modelByPath: (a: number, b: any) => any;
    readonly engine_modelByRepoId: (a: number, b: any) => any;
    readonly engine_modelPath: (a: number, b: number) => any;
    readonly engine_modelsByFamily: (a: number, b: any) => any;
    readonly engine_modelsByVendor: (a: number, b: any) => any;
    readonly engine_refreshModels: (a: number) => any;
    readonly engine_registerCallback: (a: number, b: number) => any;
    readonly engine_removeBackend: (a: number, b: any) => any;
    readonly engine_removeRegistry: (a: number, b: any) => any;
    readonly enginecallback_create: (a: any) => number;
    readonly engineconfig___uzuCloneEngineConfig: (a: number) => number;
    readonly engineconfig___wasm_get_allow_lmstudio_usage: (a: number) => any;
    readonly engineconfig___wasm_get_allow_ollama_usage: (a: number) => any;
    readonly engineconfig___wasm_get_anthropic_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_baseten_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_download_manager_type: (a: number) => any;
    readonly engineconfig___wasm_get_gemini_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_huggingface_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_local_path: (a: number) => any;
    readonly engineconfig___wasm_get_mirai_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_openai_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_openrouter_api_key: (a: number) => any;
    readonly engineconfig___wasm_get_xai_api_key: (a: number) => any;
    readonly engineconfig___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any, g: any, h: any, i: any, j: any, k: any, l: any) => [number, number, number];
    readonly engineconfig___wasm_set_allow_lmstudio_usage: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_allow_ollama_usage: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_anthropic_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_baseten_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_download_manager_type: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_gemini_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_huggingface_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_local_path: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_mirai_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_openai_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_openrouter_api_key: (a: number, b: any) => void;
    readonly engineconfig___wasm_set_xai_api_key: (a: number, b: any) => void;
    readonly engineconfig_create: () => [number, number, number];
    readonly engineconfig_withAllowLmstudioUsage: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withAllowOllamaUsage: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withAnthropicApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withBasetenApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withDownloadManagerType: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withGeminiApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withHuggingfaceApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withLocalPath: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withMiraiApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withOpenaiApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withOpenrouterApiKey: (a: number, b: any) => [number, number, number];
    readonly engineconfig_withXaiApiKey: (a: number, b: any) => [number, number, number];
    readonly feature___uzuCloneFeature: (a: number) => number;
    readonly feature___wasm_get_name: (a: number) => any;
    readonly feature___wasm_get_values: (a: number) => any;
    readonly feature___wasm_new: (a: any, b: any) => [number, number, number];
    readonly feature___wasm_set_name: (a: number, b: any) => void;
    readonly feature___wasm_set_values: (a: number, b: any) => void;
    readonly file___uzuCloneFile: (a: number) => number;
    readonly file___wasm_crc32c: (a: number) => [number, number, number];
    readonly file___wasm_get_hashes: (a: number) => any;
    readonly file___wasm_get_name: (a: number) => any;
    readonly file___wasm_get_size: (a: number) => any;
    readonly file___wasm_get_url: (a: number) => any;
    readonly file___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly file___wasm_set_hashes: (a: number, b: any) => void;
    readonly file___wasm_set_name: (a: number, b: any) => void;
    readonly file___wasm_set_size: (a: number, b: any) => void;
    readonly file___wasm_set_url: (a: number, b: any) => void;
    readonly grammarjsonany___uzuCloneGrammar: (a: number) => number;
    readonly grammarjsonany___wasm_new: () => [number, number, number];
    readonly grammarjsonschema___uzuCloneGrammar: (a: number) => number;
    readonly grammarjsonschema___wasm_get_schema: (a: number) => any;
    readonly grammarjsonschema___wasm_new: (a: any) => [number, number, number];
    readonly grammarjsonschema___wasm_set_schema: (a: number, b: any) => void;
    readonly grammarregex___uzuCloneGrammar: (a: number) => number;
    readonly grammarregex___wasm_get_pattern: (a: number) => any;
    readonly grammarregex___wasm_new: (a: any) => [number, number, number];
    readonly grammarregex___wasm_set_pattern: (a: number, b: any) => void;
    readonly hash___uzuCloneHash: (a: number) => number;
    readonly hash___wasm_get_method: (a: number) => any;
    readonly hash___wasm_get_value: (a: number) => any;
    readonly hash___wasm_new: (a: any, b: any) => [number, number, number];
    readonly hash___wasm_set_method: (a: number, b: any) => void;
    readonly hash___wasm_set_value: (a: number, b: any) => void;
    readonly hashmethod___wasm_variant_CRC32C: () => [number, number];
    readonly hashmethod___wasm_variant_GitBlobSha1: () => [number, number];
    readonly hashmethod___wasm_variant_Sha256: () => [number, number];
    readonly image___uzuCloneImage: (a: number) => number;
    readonly image___wasm_get_description: (a: number) => any;
    readonly image___wasm_get_format: (a: number) => any;
    readonly image___wasm_get_height: (a: number) => any;
    readonly image___wasm_get_id: (a: number) => any;
    readonly image___wasm_get_theme: (a: number) => any;
    readonly image___wasm_get_url: (a: number) => any;
    readonly image___wasm_get_width: (a: number) => any;
    readonly image___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any, g: any) => [number, number, number];
    readonly image___wasm_set_description: (a: number, b: any) => void;
    readonly image___wasm_set_format: (a: number, b: any) => void;
    readonly image___wasm_set_height: (a: number, b: any) => void;
    readonly image___wasm_set_id: (a: number, b: any) => void;
    readonly image___wasm_set_theme: (a: number, b: any) => void;
    readonly image___wasm_set_url: (a: number, b: any) => void;
    readonly image___wasm_set_width: (a: number, b: any) => void;
    readonly imageformat___wasm_variant_Png: () => [number, number];
    readonly imageformat___wasm_variant_Svg: () => [number, number];
    readonly imageformat___wasm_variant_Webp: () => [number, number];
    readonly imagetheme___wasm_variant_Dark: () => [number, number];
    readonly imagetheme___wasm_variant_Light: () => [number, number];
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
    readonly metadata___uzuCloneMetadata: (a: number) => number;
    readonly metadata___wasm_get_description: (a: number) => any;
    readonly metadata___wasm_get_icons: (a: number) => any;
    readonly metadata___wasm_get_identifier: (a: number) => any;
    readonly metadata___wasm_get_name: (a: number) => any;
    readonly metadata___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly metadata___wasm_set_description: (a: number, b: any) => void;
    readonly metadata___wasm_set_icons: (a: number, b: any) => void;
    readonly metadata___wasm_set_identifier: (a: number, b: any) => void;
    readonly metadata___wasm_set_name: (a: number, b: any) => void;
    readonly metadata_external: (a: any) => [number, number, number];
    readonly model___uzuCloneModel: (a: number) => number;
    readonly model___wasm_cache_identifier: (a: number) => [number, number, number];
    readonly model___wasm_checkpoint_version: (a: number) => [number, number, number];
    readonly model___wasm_filesystem_path: (a: number) => [number, number, number];
    readonly model___wasm_get_accessibility: (a: number) => any;
    readonly model___wasm_get_backends: (a: number) => any;
    readonly model___wasm_get_encoding: (a: number) => any;
    readonly model___wasm_get_family: (a: number) => any;
    readonly model___wasm_get_identifier: (a: number) => any;
    readonly model___wasm_get_properties: (a: number) => any;
    readonly model___wasm_get_quantization: (a: number) => any;
    readonly model___wasm_get_registry: (a: number) => any;
    readonly model___wasm_get_specializations: (a: number) => any;
    readonly model___wasm_is_chat_capable: (a: number) => [number, number, number];
    readonly model___wasm_is_classification_capable: (a: number) => [number, number, number];
    readonly model___wasm_is_downloadable: (a: number) => [number, number, number];
    readonly model___wasm_is_on_device: (a: number) => [number, number, number];
    readonly model___wasm_is_quantized: (a: number) => [number, number, number];
    readonly model___wasm_is_remote: (a: number) => [number, number, number];
    readonly model___wasm_is_speculation_capable: (a: number) => [number, number, number];
    readonly model___wasm_is_text_to_speech_capable: (a: number) => [number, number, number];
    readonly model___wasm_is_translation_capable: (a: number) => [number, number, number];
    readonly model___wasm_name: (a: number) => [number, number, number];
    readonly model___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any, g: any, h: any, i: any) => [number, number, number];
    readonly model___wasm_repo_ids: (a: number) => [number, number, number];
    readonly model___wasm_set_accessibility: (a: number, b: any) => void;
    readonly model___wasm_set_backends: (a: number, b: any) => void;
    readonly model___wasm_set_encoding: (a: number, b: any) => void;
    readonly model___wasm_set_family: (a: number, b: any) => void;
    readonly model___wasm_set_identifier: (a: number, b: any) => void;
    readonly model___wasm_set_properties: (a: number, b: any) => void;
    readonly model___wasm_set_quantization: (a: number, b: any) => void;
    readonly model___wasm_set_registry: (a: number, b: any) => void;
    readonly model___wasm_set_specializations: (a: number, b: any) => void;
    readonly model_external: (a: any, b: any, c: any, d: any, e: any, f: any, g: any, h: any, i: any) => [number, number, number];
    readonly modelaccessibilityondevice___uzuCloneModelAccessibility: (a: number) => number;
    readonly modelaccessibilityondevice___wasm_get_source: (a: number) => any;
    readonly modelaccessibilityondevice___wasm_new: (a: any) => [number, number, number];
    readonly modelaccessibilityondevice___wasm_set_source: (a: number, b: any) => void;
    readonly modelaccessibilityremote___uzuCloneModelAccessibility: (a: number) => number;
    readonly modelaccessibilityremote___wasm_get_repository: (a: number) => any;
    readonly modelaccessibilityremote___wasm_new: (a: any) => [number, number, number];
    readonly modelaccessibilityremote___wasm_set_repository: (a: number, b: any) => void;
    readonly modelbackend___uzuCloneModelBackend: (a: number) => number;
    readonly modelbackend___wasm_get_identifier: (a: number) => any;
    readonly modelbackend___wasm_get_metadata: (a: number) => any;
    readonly modelbackend___wasm_get_version: (a: number) => any;
    readonly modelbackend___wasm_name: (a: number) => [number, number, number];
    readonly modelbackend___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly modelbackend___wasm_set_identifier: (a: number, b: any) => void;
    readonly modelbackend___wasm_set_metadata: (a: number, b: any) => void;
    readonly modelbackend___wasm_set_version: (a: number, b: any) => void;
    readonly modelfamily___uzuCloneModelFamily: (a: number) => number;
    readonly modelfamily___wasm_get_identifier: (a: number) => any;
    readonly modelfamily___wasm_get_metadata: (a: number) => any;
    readonly modelfamily___wasm_get_vendor: (a: number) => any;
    readonly modelfamily___wasm_name: (a: number) => [number, number, number];
    readonly modelfamily___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly modelfamily___wasm_set_identifier: (a: number, b: any) => void;
    readonly modelfamily___wasm_set_metadata: (a: number, b: any) => void;
    readonly modelfamily___wasm_set_vendor: (a: number, b: any) => void;
    readonly modelproperties___uzuCloneModelProperties: (a: number) => number;
    readonly modelproperties___wasm_get_identifier: (a: number) => any;
    readonly modelproperties___wasm_get_metadata: (a: number) => any;
    readonly modelproperties___wasm_get_size: (a: number) => any;
    readonly modelproperties___wasm_get_version: (a: number) => any;
    readonly modelproperties___wasm_name: (a: number) => [number, number, number];
    readonly modelproperties___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly modelproperties___wasm_set_identifier: (a: number, b: any) => void;
    readonly modelproperties___wasm_set_metadata: (a: number, b: any) => void;
    readonly modelproperties___wasm_set_size: (a: number, b: any) => void;
    readonly modelproperties___wasm_set_version: (a: number, b: any) => void;
    readonly modelquantization___uzuCloneModelQuantization: (a: number) => number;
    readonly modelquantization___wasm_get_bits_per_weight: (a: number) => any;
    readonly modelquantization___wasm_get_identifier: (a: number) => any;
    readonly modelquantization___wasm_get_metadata: (a: number) => any;
    readonly modelquantization___wasm_get_method: (a: number) => any;
    readonly modelquantization___wasm_get_vendor: (a: number) => any;
    readonly modelquantization___wasm_name: (a: number) => [number, number, number];
    readonly modelquantization___wasm_new: (a: any, b: any, c: any, d: any, e: any) => [number, number, number];
    readonly modelquantization___wasm_set_bits_per_weight: (a: number, b: any) => void;
    readonly modelquantization___wasm_set_identifier: (a: number, b: any) => void;
    readonly modelquantization___wasm_set_metadata: (a: number, b: any) => void;
    readonly modelquantization___wasm_set_method: (a: number, b: any) => void;
    readonly modelquantization___wasm_set_vendor: (a: number, b: any) => void;
    readonly modelregistry___uzuCloneModelRegistry: (a: number) => number;
    readonly modelregistry___wasm_get_identifier: (a: number) => any;
    readonly modelregistry___wasm_get_metadata: (a: number) => any;
    readonly modelregistry___wasm_name: (a: number) => [number, number, number];
    readonly modelregistry___wasm_new: (a: any, b: any) => [number, number, number];
    readonly modelregistry___wasm_set_identifier: (a: number, b: any) => void;
    readonly modelregistry___wasm_set_metadata: (a: number, b: any) => void;
    readonly modelsourcefilesystem___uzuCloneModelSource: (a: number) => number;
    readonly modelsourcefilesystem___wasm_get_path: (a: number) => any;
    readonly modelsourcefilesystem___wasm_new: (a: any) => [number, number, number];
    readonly modelsourcefilesystem___wasm_set_path: (a: number, b: any) => void;
    readonly modelsourceregistry___uzuCloneModelSource: (a: number) => number;
    readonly modelsourceregistry___wasm_get_files: (a: number) => any;
    readonly modelsourceregistry___wasm_get_repository: (a: number) => any;
    readonly modelsourceregistry___wasm_get_source_repository: (a: number) => any;
    readonly modelsourceregistry___wasm_get_toolchain_version: (a: number) => any;
    readonly modelsourceregistry___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly modelsourceregistry___wasm_set_files: (a: number, b: any) => void;
    readonly modelsourceregistry___wasm_set_repository: (a: number, b: any) => void;
    readonly modelsourceregistry___wasm_set_source_repository: (a: number, b: any) => void;
    readonly modelsourceregistry___wasm_set_toolchain_version: (a: number, b: any) => void;
    readonly modelspecializationchat___uzuCloneModelSpecialization: (a: number) => number;
    readonly modelspecializationchat___wasm_new: () => [number, number, number];
    readonly modelspecializationclassification___uzuCloneModelSpecialization: (a: number) => number;
    readonly modelspecializationclassification___wasm_new: () => [number, number, number];
    readonly modelspecializationdecision___uzuCloneModelSpecialization: (a: number) => number;
    readonly modelspecializationdecision___wasm_new: () => [number, number, number];
    readonly modelspecializationspeculation___uzuCloneModelSpecialization: (a: number) => number;
    readonly modelspecializationspeculation___wasm_new: () => [number, number, number];
    readonly modelspecializationtexttospeech___uzuCloneModelSpecialization: (a: number) => number;
    readonly modelspecializationtexttospeech___wasm_new: () => [number, number, number];
    readonly modelspecializationtranslation___uzuCloneModelSpecialization: (a: number) => number;
    readonly modelspecializationtranslation___wasm_new: () => [number, number, number];
    readonly modelvendor___uzuCloneModelVendor: (a: number) => number;
    readonly modelvendor___wasm_get_identifier: (a: number) => any;
    readonly modelvendor___wasm_get_metadata: (a: number) => any;
    readonly modelvendor___wasm_name: (a: number) => [number, number, number];
    readonly modelvendor___wasm_new: (a: any, b: any) => [number, number, number];
    readonly modelvendor___wasm_set_identifier: (a: number, b: any) => void;
    readonly modelvendor___wasm_set_metadata: (a: number, b: any) => void;
    readonly nativetool___uzuCloneNativeTool: (a: number) => number;
    readonly nativetool_new: (a: any, b: any, c: any) => [number, number, number];
    readonly pcmbatch___uzuClonePcmBatch: (a: number) => number;
    readonly pcmbatch___wasm_batch_size: (a: number) => [number, number, number];
    readonly pcmbatch___wasm_duration: (a: number) => [number, number, number];
    readonly pcmbatch___wasm_get_channels: (a: number) => any;
    readonly pcmbatch___wasm_get_lengths: (a: number) => any;
    readonly pcmbatch___wasm_get_sample_rate: (a: number) => any;
    readonly pcmbatch___wasm_get_samples: (a: number) => any;
    readonly pcmbatch___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly pcmbatch___wasm_set_channels: (a: number, b: any) => void;
    readonly pcmbatch___wasm_set_lengths: (a: number, b: any) => void;
    readonly pcmbatch___wasm_set_sample_rate: (a: number, b: any) => void;
    readonly pcmbatch___wasm_set_samples: (a: number, b: any) => void;
    readonly pcmbatch___wasm_total_frames: (a: number) => [number, number, number];
    readonly pcmbatch_saveAsWav: (a: number, b: any) => [number, number];
    readonly reasoningeffort___wasm_variant_Default: () => [number, number];
    readonly reasoningeffort___wasm_variant_Disabled: () => [number, number];
    readonly reasoningeffort___wasm_variant_High: () => [number, number];
    readonly reasoningeffort___wasm_variant_Low: () => [number, number];
    readonly reasoningeffort___wasm_variant_Medium: () => [number, number];
    readonly reasoningeffort___wasm_variant_XHigh: () => [number, number];
    readonly replyenergycomponents___uzuCloneReplyEnergy: (a: number) => number;
    readonly replyenergycomponents___wasm_get_ane: (a: number) => any;
    readonly replyenergycomponents___wasm_get_cpu: (a: number) => any;
    readonly replyenergycomponents___wasm_get_dram: (a: number) => any;
    readonly replyenergycomponents___wasm_get_gpu: (a: number) => any;
    readonly replyenergycomponents___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly replyenergycomponents___wasm_set_ane: (a: number, b: any) => void;
    readonly replyenergycomponents___wasm_set_cpu: (a: number, b: any) => void;
    readonly replyenergycomponents___wasm_set_dram: (a: number, b: any) => void;
    readonly replyenergycomponents___wasm_set_gpu: (a: number, b: any) => void;
    readonly replyenergytotal___uzuCloneReplyEnergy: (a: number) => number;
    readonly replyenergytotal___wasm_get_total: (a: number) => any;
    readonly replyenergytotal___wasm_new: (a: any) => [number, number, number];
    readonly replyenergytotal___wasm_set_total: (a: number, b: any) => void;
    readonly repository___uzuCloneRepository: (a: number) => number;
    readonly repository___wasm_get_commit_hash: (a: number) => any;
    readonly repository___wasm_get_identifier: (a: number) => any;
    readonly repository___wasm_get_paths: (a: number) => any;
    readonly repository___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly repository___wasm_set_commit_hash: (a: number, b: any) => void;
    readonly repository___wasm_set_identifier: (a: number, b: any) => void;
    readonly repository___wasm_set_paths: (a: number, b: any) => void;
    readonly samplingmethodgreedy___uzuCloneSamplingMethod: (a: number) => number;
    readonly samplingmethodgreedy___wasm_new: () => [number, number, number];
    readonly samplingmethodstochastic___uzuCloneSamplingMethod: (a: number) => number;
    readonly samplingmethodstochastic___wasm_get_min_p: (a: number) => any;
    readonly samplingmethodstochastic___wasm_get_repetition_penalty: (a: number) => any;
    readonly samplingmethodstochastic___wasm_get_suffix_repetition_length: (a: number) => any;
    readonly samplingmethodstochastic___wasm_get_temperature: (a: number) => any;
    readonly samplingmethodstochastic___wasm_get_top_k: (a: number) => any;
    readonly samplingmethodstochastic___wasm_get_top_p: (a: number) => any;
    readonly samplingmethodstochastic___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any) => [number, number, number];
    readonly samplingmethodstochastic___wasm_set_min_p: (a: number, b: any) => void;
    readonly samplingmethodstochastic___wasm_set_repetition_penalty: (a: number, b: any) => void;
    readonly samplingmethodstochastic___wasm_set_suffix_repetition_length: (a: number, b: any) => void;
    readonly samplingmethodstochastic___wasm_set_temperature: (a: number, b: any) => void;
    readonly samplingmethodstochastic___wasm_set_top_k: (a: number, b: any) => void;
    readonly samplingmethodstochastic___wasm_set_top_p: (a: number, b: any) => void;
    readonly samplingparameters___uzuCloneSamplingParameters: (a: number) => number;
    readonly samplingparameters___wasm_get_min_p: (a: number) => any;
    readonly samplingparameters___wasm_get_repetition_penalty: (a: number) => any;
    readonly samplingparameters___wasm_get_suffix_repetition_length: (a: number) => any;
    readonly samplingparameters___wasm_get_temperature: (a: number) => any;
    readonly samplingparameters___wasm_get_top_k: (a: number) => any;
    readonly samplingparameters___wasm_get_top_p: (a: number) => any;
    readonly samplingparameters___wasm_new: (a: any, b: any, c: any, d: any, e: any, f: any) => [number, number, number];
    readonly samplingparameters___wasm_set_min_p: (a: number, b: any) => void;
    readonly samplingparameters___wasm_set_repetition_penalty: (a: number, b: any) => void;
    readonly samplingparameters___wasm_set_suffix_repetition_length: (a: number, b: any) => void;
    readonly samplingparameters___wasm_set_temperature: (a: number, b: any) => void;
    readonly samplingparameters___wasm_set_top_k: (a: number, b: any) => void;
    readonly samplingparameters___wasm_set_top_p: (a: number, b: any) => void;
    readonly samplingpolicycustom___uzuCloneSamplingPolicy: (a: number) => number;
    readonly samplingpolicycustom___wasm_get_method: (a: number) => any;
    readonly samplingpolicycustom___wasm_new: (a: any) => [number, number, number];
    readonly samplingpolicycustom___wasm_set_method: (a: number, b: any) => void;
    readonly samplingpolicydefault___uzuCloneSamplingPolicy: (a: number) => number;
    readonly samplingpolicydefault___wasm_new: () => [number, number, number];
    readonly samplingseedcustom___uzuCloneSamplingSeed: (a: number) => number;
    readonly samplingseedcustom___wasm_get_seed: (a: number) => any;
    readonly samplingseedcustom___wasm_new: (a: any) => [number, number, number];
    readonly samplingseedcustom___wasm_set_seed: (a: number, b: any) => void;
    readonly samplingseeddefault___uzuCloneSamplingSeed: (a: number) => number;
    readonly samplingseeddefault___wasm_new: () => [number, number, number];
    readonly texttospeechoutput___uzuCloneTextToSpeechOutput: (a: number) => number;
    readonly texttospeechoutput___wasm_get_pcm_batch: (a: number) => any;
    readonly texttospeechoutput___wasm_get_stats: (a: number) => any;
    readonly texttospeechoutput___wasm_new: (a: any, b: any) => [number, number, number];
    readonly texttospeechoutput___wasm_set_pcm_batch: (a: number, b: any) => void;
    readonly texttospeechoutput___wasm_set_stats: (a: number, b: any) => void;
    readonly texttospeechstats___uzuCloneTextToSpeechStats: (a: number) => number;
    readonly texttospeechstats___wasm_get_audio_duration: (a: number) => any;
    readonly texttospeechstats___wasm_get_first_chunk_seconds: (a: number) => any;
    readonly texttospeechstats___wasm_get_generation_duration: (a: number) => any;
    readonly texttospeechstats___wasm_get_text_length: (a: number) => any;
    readonly texttospeechstats___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly texttospeechstats___wasm_set_audio_duration: (a: number, b: any) => void;
    readonly texttospeechstats___wasm_set_first_chunk_seconds: (a: number, b: any) => void;
    readonly texttospeechstats___wasm_set_generation_duration: (a: number, b: any) => void;
    readonly texttospeechstats___wasm_set_text_length: (a: number, b: any) => void;
    readonly token___uzuCloneToken: (a: number) => number;
    readonly token___wasm_get_id: (a: number) => any;
    readonly token___wasm_get_is_special: (a: number) => any;
    readonly token___wasm_get_value: (a: number) => any;
    readonly token___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly token___wasm_set_id: (a: number, b: any) => void;
    readonly token___wasm_set_is_special: (a: number, b: any) => void;
    readonly token___wasm_set_value: (a: number, b: any) => void;
    readonly toolcall___uzuCloneToolCall: (a: number) => number;
    readonly toolcall___wasm_get_arguments: (a: number) => any;
    readonly toolcall___wasm_get_identifier: (a: number) => any;
    readonly toolcall___wasm_get_name: (a: number) => any;
    readonly toolcall___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly toolcall___wasm_set_arguments: (a: number, b: any) => void;
    readonly toolcall___wasm_set_identifier: (a: number, b: any) => void;
    readonly toolcall___wasm_set_name: (a: number, b: any) => void;
    readonly toolcallargumentdelta___uzuCloneToolCallArgumentDelta: (a: number) => number;
    readonly toolcallargumentdelta___wasm_get_arguments: (a: number) => any;
    readonly toolcallargumentdelta___wasm_get_name: (a: number) => any;
    readonly toolcallargumentdelta___wasm_new: (a: any, b: any) => [number, number, number];
    readonly toolcallargumentdelta___wasm_set_arguments: (a: number, b: any) => void;
    readonly toolcallargumentdelta___wasm_set_name: (a: number, b: any) => void;
    readonly toolcallargumentstream_create: (a: any) => [number, number, number];
    readonly toolcallargumentstream_finish: (a: number, b: any) => [number, number, number];
    readonly toolcallargumentstream_update: (a: number, b: any) => [number, number, number];
    readonly tooldescriptionfunction___uzuCloneToolDescription: (a: number) => number;
    readonly tooldescriptionfunction___wasm_get_tool_function: (a: number) => any;
    readonly tooldescriptionfunction___wasm_new: (a: any) => [number, number, number];
    readonly tooldescriptionfunction___wasm_set_tool_function: (a: number, b: any) => void;
    readonly toolfunction___uzuCloneToolFunction: (a: number) => number;
    readonly toolfunction___wasm_get_description: (a: number) => any;
    readonly toolfunction___wasm_get_name: (a: number) => any;
    readonly toolfunction___wasm_get_parameters: (a: number) => any;
    readonly toolfunction___wasm_get_return_definition: (a: number) => any;
    readonly toolfunction___wasm_new: (a: any, b: any, c: any, d: any) => [number, number, number];
    readonly toolfunction___wasm_set_description: (a: number, b: any) => void;
    readonly toolfunction___wasm_set_name: (a: number, b: any) => void;
    readonly toolfunction___wasm_set_parameters: (a: number, b: any) => void;
    readonly toolfunction___wasm_set_return_definition: (a: number, b: any) => void;
    readonly toolnamespace___uzuCloneToolNamespace: (a: number) => number;
    readonly toolnamespace___wasm_get_description: (a: number) => any;
    readonly toolnamespace___wasm_get_name: (a: number) => any;
    readonly toolnamespace___wasm_get_tools: (a: number) => any;
    readonly toolnamespace___wasm_new: (a: any, b: any, c: any) => [number, number, number];
    readonly toolnamespace___wasm_set_description: (a: number, b: any) => void;
    readonly toolnamespace___wasm_set_name: (a: number, b: any) => void;
    readonly toolnamespace___wasm_set_tools: (a: number, b: any) => void;
    readonly translationpayloadimage___uzuCloneTranslationPayload: (a: number) => number;
    readonly translationpayloadimage___wasm_get_url: (a: number) => any;
    readonly translationpayloadimage___wasm_new: (a: any) => [number, number, number];
    readonly translationpayloadimage___wasm_set_url: (a: number, b: any) => void;
    readonly translationpayloadtext___uzuCloneTranslationPayload: (a: number) => number;
    readonly translationpayloadtext___wasm_get_text: (a: number) => any;
    readonly translationpayloadtext___wasm_new: (a: any) => [number, number, number];
    readonly translationpayloadtext___wasm_set_text: (a: number, b: any) => void;
    readonly value___uzuCloneValue: (a: number) => number;
    readonly value___wasm_get_json: (a: number) => any;
    readonly value___wasm_new: (a: any) => [number, number, number];
    readonly value___wasm_set_json: (a: number, b: any) => void;
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__898: (a: number, b: number, c: any) => [number, number];
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
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
