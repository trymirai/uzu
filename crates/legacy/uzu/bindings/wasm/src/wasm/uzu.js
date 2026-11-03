/* @ts-self-types="./uzu.d.ts" */

export class CancelToken {
    static __wrap(ptr) {
        const obj = Object.create(CancelToken.prototype);
        obj.__wbg_ptr = ptr;
        CancelTokenFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CancelTokenFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_canceltoken_free(ptr, 0);
    }
    cancel() {
        wasm.canceltoken_cancel(this.__wbg_ptr);
    }
    /**
     * @returns {CancelToken}
     */
    static create() {
        const ret = wasm.canceltoken_create();
        return CancelToken.__wrap(ret);
    }
    /**
     * @returns {boolean}
     */
    get isCancelled() {
        const ret = wasm.canceltoken_isCancelled(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) CancelToken.prototype[Symbol.dispose] = CancelToken.prototype.free;

export class ChatConfig {
    static __wrap(ptr) {
        const obj = Object.create(ChatConfig.prototype);
        obj.__wbg_ptr = ptr;
        ChatConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatconfig_free(ptr, 0);
    }
    /**
     * @returns {ChatConfig}
     */
    static create() {
        const ret = wasm.chatconfig_create();
        return ChatConfig.__wrap(ret);
    }
    /**
     * @param {ContextLength} context_length
     * @returns {ChatConfig}
     */
    withContextLength(context_length) {
        const ret = wasm.chatconfig_withContextLength(this.__wbg_ptr, context_length);
        return ChatConfig.__wrap(ret);
    }
    /**
     * @param {SamplingSeed} sampling_seed
     * @returns {ChatConfig}
     */
    withSamplingSeed(sampling_seed) {
        const ret = wasm.chatconfig_withSamplingSeed(this.__wbg_ptr, sampling_seed);
        return ChatConfig.__wrap(ret);
    }
    /**
     * @returns {ContextLength}
     */
    get context_length() {
        const ret = wasm.__wbg_get_chatconfig_context_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {SamplingSeed}
     */
    get sampling_seed() {
        const ret = wasm.__wbg_get_chatconfig_sampling_seed(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ContextLength} arg0
     */
    set context_length(arg0) {
        wasm.__wbg_set_chatconfig_context_length(this.__wbg_ptr, arg0);
    }
    /**
     * @param {SamplingSeed} arg0
     */
    set sampling_seed(arg0) {
        wasm.__wbg_set_chatconfig_sampling_seed(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ChatConfig.prototype[Symbol.dispose] = ChatConfig.prototype.free;

/**
 * A loaded chat backend instance (model weights, tokenizer, configuration).
 * Cloning is cheap and shares the underlying instance, so multiple [`ChatSession`]s
 * can be created from one [`ChatInstance`] without loading the model again.
 */
export class ChatInstance {
    static __wrap(ptr) {
        const obj = Object.create(ChatInstance.prototype);
        obj.__wbg_ptr = ptr;
        ChatInstanceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatInstanceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatinstance_free(ptr, 0);
    }
    /**
     * @returns {Model}
     */
    get model() {
        const ret = wasm.chatinstance_model(this.__wbg_ptr);
        return Model.__wrap(ret);
    }
}
if (Symbol.dispose) ChatInstance.prototype[Symbol.dispose] = ChatInstance.prototype.free;

export class ChatMessage {
    static __wrap(ptr) {
        const obj = Object.create(ChatMessage.prototype);
        obj.__wbg_ptr = ptr;
        ChatMessageFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    static __unwrap(jsValue) {
        if (!(jsValue instanceof ChatMessage)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatMessageFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatmessage_free(ptr, 0);
    }
    /**
     * @returns {ChatMessage}
     */
    static assistant() {
        const ret = wasm.chatmessage_assistant();
        return ChatMessage.__wrap(ret);
    }
    /**
     * @returns {ChatMessage}
     */
    static developer() {
        const ret = wasm.chatmessage_developer();
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {ChatRole} role
     * @returns {ChatMessage}
     */
    static forRole(role) {
        const ret = wasm.chatmessage_forRole(role);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @returns {ReasoningEffort | undefined}
     */
    get reasoningEffort() {
        const ret = wasm.chatmessage_reasoningEffort(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string | undefined}
     */
    get reasoning() {
        const ret = wasm.chatmessage_reasoning(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {ChatMessage}
     */
    static system() {
        const ret = wasm.chatmessage_system();
        return ChatMessage.__wrap(ret);
    }
    /**
     * @returns {string | undefined}
     */
    get text() {
        const ret = wasm.chatmessage_text(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {ToolCall[]}
     */
    get toolCalls() {
        const ret = wasm.chatmessage_toolCalls(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {ToolNamespace[]}
     */
    get toolNamespaces() {
        const ret = wasm.chatmessage_toolNamespaces(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {ChatMessage}
     */
    static tool() {
        const ret = wasm.chatmessage_tool();
        return ChatMessage.__wrap(ret);
    }
    /**
     * @returns {ChatMessage}
     */
    static user() {
        const ret = wasm.chatmessage_user();
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {ChatContentBlock} block
     * @returns {ChatMessage}
     */
    withBlock(block) {
        const ret = wasm.chatmessage_withBlock(this.__wbg_ptr, block);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {ReasoningEffort} reasoning_effort
     * @returns {ChatMessage}
     */
    withReasoningEffort(reasoning_effort) {
        const ret = wasm.chatmessage_withReasoningEffort(this.__wbg_ptr, reasoning_effort);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {string} reasoning
     * @returns {ChatMessage}
     */
    withReasoning(reasoning) {
        const ptr0 = passStringToWasm0(reasoning, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.chatmessage_withReasoning(this.__wbg_ptr, ptr0, len0);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {string} text
     * @returns {ChatMessage}
     */
    withText(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.chatmessage_withText(this.__wbg_ptr, ptr0, len0);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {Value} value
     * @returns {ChatMessage}
     */
    withToolCallCandidate(value) {
        _assertClass(value, Value);
        var ptr0 = value.__destroy_into_raw();
        const ret = wasm.chatmessage_withToolCallCandidate(this.__wbg_ptr, ptr0);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {ToolCall} tool_call
     * @returns {ChatMessage}
     */
    withToolCall(tool_call) {
        _assertClass(tool_call, ToolCall);
        var ptr0 = tool_call.__destroy_into_raw();
        const ret = wasm.chatmessage_withToolCall(this.__wbg_ptr, ptr0);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @param {ToolNamespace[]} tool_namespaces
     * @returns {ChatMessage}
     */
    withToolNamespaces(tool_namespaces) {
        const ptr0 = passArrayJsValueToWasm0(tool_namespaces, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.chatmessage_withToolNamespaces(this.__wbg_ptr, ptr0, len0);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @returns {ChatContentBlock[]}
     */
    get content() {
        const ret = wasm.__wbg_get_chatmessage_content(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {ChatMessageMetadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_chatmessage_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ChatRole}
     */
    get role() {
        const ret = wasm.__wbg_get_chatmessage_role(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatContentBlock[]} arg0
     */
    set content(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_chatmessage_content(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ChatMessageMetadata} arg0
     */
    set metadata(arg0) {
        wasm.__wbg_set_chatmessage_metadata(this.__wbg_ptr, arg0);
    }
    /**
     * @param {ChatRole} arg0
     */
    set role(arg0) {
        wasm.__wbg_set_chatmessage_role(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ChatMessage.prototype[Symbol.dispose] = ChatMessage.prototype.free;

export class ChatModelCapabilities {
    static __wrap(ptr) {
        const obj = Object.create(ChatModelCapabilities.prototype);
        obj.__wbg_ptr = ptr;
        ChatModelCapabilitiesFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatModelCapabilitiesFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatmodelcapabilities_free(ptr, 0);
    }
    /**
     * @returns {ReasoningEffort[]}
     */
    get reasoning_efforts() {
        const ret = wasm.__wbg_get_chatmodelcapabilities_reasoning_efforts(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get requires_tools() {
        const ret = wasm.__wbg_get_chatmodelcapabilities_requires_tools(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get supports_disable_reasoning() {
        const ret = wasm.__wbg_get_chatmodelcapabilities_supports_disable_reasoning(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get supports_multiple_tool_calls() {
        const ret = wasm.__wbg_get_chatmodelcapabilities_supports_multiple_tool_calls(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get supports_reasoning() {
        const ret = wasm.__wbg_get_chatmodelcapabilities_supports_reasoning(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get supports_tools() {
        const ret = wasm.__wbg_get_chatmodelcapabilities_supports_tools(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {ReasoningEffort[]} arg0
     */
    set reasoning_efforts(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_chatmodelcapabilities_reasoning_efforts(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {boolean} arg0
     */
    set requires_tools(arg0) {
        wasm.__wbg_set_chatmodelcapabilities_requires_tools(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} arg0
     */
    set supports_disable_reasoning(arg0) {
        wasm.__wbg_set_chatmodelcapabilities_supports_disable_reasoning(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} arg0
     */
    set supports_multiple_tool_calls(arg0) {
        wasm.__wbg_set_chatmodelcapabilities_supports_multiple_tool_calls(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} arg0
     */
    set supports_reasoning(arg0) {
        wasm.__wbg_set_chatmodelcapabilities_supports_reasoning(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} arg0
     */
    set supports_tools(arg0) {
        wasm.__wbg_set_chatmodelcapabilities_supports_tools(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ChatModelCapabilities.prototype[Symbol.dispose] = ChatModelCapabilities.prototype.free;

/**
 * Chat metadata available without loading model weights or allocating a session.
 */
export class ChatModelInfo {
    static __wrap(ptr) {
        const obj = Object.create(ChatModelInfo.prototype);
        obj.__wbg_ptr = ptr;
        ChatModelInfoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatModelInfoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatmodelinfo_free(ptr, 0);
    }
    /**
     * @returns {ChatModelCapabilities}
     */
    get capabilities() {
        const ret = wasm.__wbg_get_chatmodelinfo_capabilities(this.__wbg_ptr);
        return ChatModelCapabilities.__wrap(ret);
    }
    /**
     * Architectural limit; absent before download or for models without a finite limit.
     * @returns {number | undefined}
     */
    get context_length() {
        const ret = wasm.__wbg_get_chatmodelinfo_context_length(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {ReasoningEffort | undefined}
     */
    get default_reasoning_effort() {
        const ret = wasm.__wbg_get_chatmodelinfo_default_reasoning_effort(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatModelCapabilities} arg0
     */
    set capabilities(arg0) {
        _assertClass(arg0, ChatModelCapabilities);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_chatmodelinfo_capabilities(this.__wbg_ptr, ptr0);
    }
    /**
     * Architectural limit; absent before download or for models without a finite limit.
     * @param {number | null} [arg0]
     */
    set context_length(arg0) {
        wasm.__wbg_set_chatmodelinfo_context_length(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
    /**
     * @param {ReasoningEffort | null} [arg0]
     */
    set default_reasoning_effort(arg0) {
        wasm.__wbg_set_chatmodelinfo_default_reasoning_effort(this.__wbg_ptr, isLikeNone(arg0) ? 0 : addToExternrefTable0(arg0));
    }
}
if (Symbol.dispose) ChatModelInfo.prototype[Symbol.dispose] = ChatModelInfo.prototype.free;

export class ChatReply {
    static __wrap(ptr) {
        const obj = Object.create(ChatReply.prototype);
        obj.__wbg_ptr = ptr;
        ChatReplyFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplyFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreply_free(ptr, 0);
    }
    /**
     * @returns {ChatReplyFinishReason | undefined}
     */
    get finish_reason() {
        const ret = wasm.__wbg_get_chatreply_finish_reason(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ChatMessage}
     */
    get message() {
        const ret = wasm.__wbg_get_chatreply_message(this.__wbg_ptr);
        return ChatMessage.__wrap(ret);
    }
    /**
     * @returns {ChatReplyStats}
     */
    get stats() {
        const ret = wasm.__wbg_get_chatreply_stats(this.__wbg_ptr);
        return ChatReplyStats.__wrap(ret);
    }
    /**
     * @param {ChatReplyFinishReason | null} [arg0]
     */
    set finish_reason(arg0) {
        wasm.__wbg_set_chatreply_finish_reason(this.__wbg_ptr, isLikeNone(arg0) ? 0 : addToExternrefTable0(arg0));
    }
    /**
     * @param {ChatMessage} arg0
     */
    set message(arg0) {
        _assertClass(arg0, ChatMessage);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_chatreply_message(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {ChatReplyStats} arg0
     */
    set stats(arg0) {
        _assertClass(arg0, ChatReplyStats);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_chatreply_stats(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ChatReply.prototype[Symbol.dispose] = ChatReply.prototype.free;

export class ChatReplyConfig {
    static __wrap(ptr) {
        const obj = Object.create(ChatReplyConfig.prototype);
        obj.__wbg_ptr = ptr;
        ChatReplyConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplyConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreplyconfig_free(ptr, 0);
    }
    /**
     * @returns {ChatReplyConfig}
     */
    static create() {
        const ret = wasm.chatreplyconfig_create();
        return ChatReplyConfig.__wrap(ret);
    }
    /**
     * @param {Grammar | null} [grammar]
     * @returns {ChatReplyConfig}
     */
    withGrammar(grammar) {
        const ret = wasm.chatreplyconfig_withGrammar(this.__wbg_ptr, isLikeNone(grammar) ? 0 : addToExternrefTable0(grammar));
        return ChatReplyConfig.__wrap(ret);
    }
    /**
     * @param {SamplingMethod} sampling_method
     * @returns {ChatReplyConfig}
     */
    withSamplingMethod(sampling_method) {
        const ret = wasm.chatreplyconfig_withSamplingMethod(this.__wbg_ptr, sampling_method);
        return ChatReplyConfig.__wrap(ret);
    }
    /**
     * @param {SamplingPolicy} sampling_policy
     * @returns {ChatReplyConfig}
     */
    withSamplingPolicy(sampling_policy) {
        const ret = wasm.chatreplyconfig_withSamplingPolicy(this.__wbg_ptr, sampling_policy);
        return ChatReplyConfig.__wrap(ret);
    }
    /**
     * @param {number | null} [token_limit]
     * @returns {ChatReplyConfig}
     */
    withTokenLimit(token_limit) {
        const ret = wasm.chatreplyconfig_withTokenLimit(this.__wbg_ptr, isLikeNone(token_limit) ? Number.MAX_SAFE_INTEGER : (token_limit) >>> 0);
        return ChatReplyConfig.__wrap(ret);
    }
    /**
     * @param {number | null} [tool_turn_limit]
     * @returns {ChatReplyConfig}
     */
    withToolTurnLimit(tool_turn_limit) {
        const ret = wasm.chatreplyconfig_withToolTurnLimit(this.__wbg_ptr, isLikeNone(tool_turn_limit) ? Number.MAX_SAFE_INTEGER : (tool_turn_limit) >>> 0);
        return ChatReplyConfig.__wrap(ret);
    }
    /**
     * @returns {Grammar | undefined}
     */
    get grammar() {
        const ret = wasm.__wbg_get_chatreplyconfig_grammar(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {SamplingPolicy}
     */
    get sampling_policy() {
        const ret = wasm.__wbg_get_chatreplyconfig_sampling_policy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number | undefined}
     */
    get token_limit() {
        const ret = wasm.__wbg_get_chatreplyconfig_token_limit(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * Maximum number of automatic tool-call turns per reply.
     * `None` allows unlimited turns; `Some(0)` prevents automatic tool execution.
     * @returns {number | undefined}
     */
    get tool_turn_limit() {
        const ret = wasm.__wbg_get_chatreplyconfig_tool_turn_limit(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @param {Grammar | null} [arg0]
     */
    set grammar(arg0) {
        wasm.__wbg_set_chatreplyconfig_grammar(this.__wbg_ptr, isLikeNone(arg0) ? 0 : addToExternrefTable0(arg0));
    }
    /**
     * @param {SamplingPolicy} arg0
     */
    set sampling_policy(arg0) {
        wasm.__wbg_set_chatreplyconfig_sampling_policy(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set token_limit(arg0) {
        wasm.__wbg_set_chatreplyconfig_token_limit(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
    /**
     * Maximum number of automatic tool-call turns per reply.
     * `None` allows unlimited turns; `Some(0)` prevents automatic tool execution.
     * @param {number | null} [arg0]
     */
    set tool_turn_limit(arg0) {
        wasm.__wbg_set_chatreplyconfig_tool_turn_limit(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
}
if (Symbol.dispose) ChatReplyConfig.prototype[Symbol.dispose] = ChatReplyConfig.prototype.free;

export class ChatReplySpeculatorStats {
    static __wrap(ptr) {
        const obj = Object.create(ChatReplySpeculatorStats.prototype);
        obj.__wbg_ptr = ptr;
        ChatReplySpeculatorStatsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplySpeculatorStatsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreplyspeculatorstats_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get num_decode_forward_passes() {
        const ret = wasm.__wbg_get_chatreplyspeculatorstats_num_decode_forward_passes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get tokens_per_forward_pass() {
        const ret = wasm.__wbg_get_chatreplyspeculatorstats_tokens_per_forward_pass(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set num_decode_forward_passes(arg0) {
        wasm.__wbg_set_chatreplyspeculatorstats_num_decode_forward_passes(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set tokens_per_forward_pass(arg0) {
        wasm.__wbg_set_chatreplyspeculatorstats_tokens_per_forward_pass(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ChatReplySpeculatorStats.prototype[Symbol.dispose] = ChatReplySpeculatorStats.prototype.free;

export class ChatReplyStats {
    static __wrap(ptr) {
        const obj = Object.create(ChatReplyStats.prototype);
        obj.__wbg_ptr = ptr;
        ChatReplyStatsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplyStatsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreplystats_free(ptr, 0);
    }
    /**
     * @returns {ChatReplyJoulesPerToken | undefined}
     */
    get inputJoulesPerToken() {
        const ret = wasm.chatreplystats_inputJoulesPerToken(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ChatReplyJoulesPerToken | undefined}
     */
    get outputJoulesPerToken() {
        const ret = wasm.chatreplystats_outputJoulesPerToken(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number | undefined}
     */
    get tokensCount() {
        const ret = wasm.chatreplystats_tokensCount(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {number | undefined}
     */
    get totalJoules() {
        const ret = wasm.chatreplystats_totalJoules(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number}
     */
    get duration() {
        const ret = wasm.__wbg_get_chatreplystats_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number | undefined}
     */
    get generate_tokens_per_second() {
        const ret = wasm.__wbg_get_chatreplystats_generate_tokens_per_second(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {ReplyEnergy | undefined}
     */
    get input_energy() {
        const ret = wasm.__wbg_get_chatreplystats_input_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint | undefined}
     */
    get memory_used_bytes() {
        const ret = wasm.__wbg_get_chatreplystats_memory_used_bytes(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {ReplyEnergy | undefined}
     */
    get output_energy() {
        const ret = wasm.__wbg_get_chatreplystats_output_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number | undefined}
     */
    get prefill_tokens_per_second() {
        const ret = wasm.__wbg_get_chatreplystats_prefill_tokens_per_second(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {ChatReplySpeculatorStats | undefined}
     */
    get speculator_stats() {
        const ret = wasm.__wbg_get_chatreplystats_speculator_stats(this.__wbg_ptr);
        return ret === 0 ? undefined : ChatReplySpeculatorStats.__wrap(ret);
    }
    /**
     * @returns {number | undefined}
     */
    get time_to_first_token() {
        const ret = wasm.__wbg_get_chatreplystats_time_to_first_token(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    get tokens_count_input_cached() {
        const ret = wasm.__wbg_get_chatreplystats_tokens_count_input_cached(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {number | undefined}
     */
    get tokens_count_input() {
        const ret = wasm.__wbg_get_chatreplystats_tokens_count_input(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {number | undefined}
     */
    get tokens_count_output() {
        const ret = wasm.__wbg_get_chatreplystats_tokens_count_output(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @param {number} arg0
     */
    set duration(arg0) {
        wasm.__wbg_set_chatreplystats_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set generate_tokens_per_second(arg0) {
        wasm.__wbg_set_chatreplystats_generate_tokens_per_second(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @param {ReplyEnergy | null} [arg0]
     */
    set input_energy(arg0) {
        wasm.__wbg_set_chatreplystats_input_energy(this.__wbg_ptr, isLikeNone(arg0) ? 0 : addToExternrefTable0(arg0));
    }
    /**
     * @param {bigint | null} [arg0]
     */
    set memory_used_bytes(arg0) {
        wasm.__wbg_set_chatreplystats_memory_used_bytes(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? BigInt(0) : arg0);
    }
    /**
     * @param {ReplyEnergy | null} [arg0]
     */
    set output_energy(arg0) {
        wasm.__wbg_set_chatreplystats_output_energy(this.__wbg_ptr, isLikeNone(arg0) ? 0 : addToExternrefTable0(arg0));
    }
    /**
     * @param {number | null} [arg0]
     */
    set prefill_tokens_per_second(arg0) {
        wasm.__wbg_set_chatreplystats_prefill_tokens_per_second(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @param {ChatReplySpeculatorStats | null} [arg0]
     */
    set speculator_stats(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, ChatReplySpeculatorStats);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_chatreplystats_speculator_stats(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set time_to_first_token(arg0) {
        wasm.__wbg_set_chatreplystats_time_to_first_token(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set tokens_count_input_cached(arg0) {
        wasm.__wbg_set_chatreplystats_tokens_count_input_cached(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set tokens_count_input(arg0) {
        wasm.__wbg_set_chatreplystats_tokens_count_input(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set tokens_count_output(arg0) {
        wasm.__wbg_set_chatreplystats_tokens_count_output(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
}
if (Symbol.dispose) ChatReplyStats.prototype[Symbol.dispose] = ChatReplyStats.prototype.free;

export class ChatSession {
    static __wrap(ptr) {
        const obj = Object.create(ChatSession.prototype);
        obj.__wbg_ptr = ptr;
        ChatSessionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatSessionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatsession_free(ptr, 0);
    }
    /**
     * @param {NativeTool} tool
     * @returns {Promise<void>}
     */
    addTool(tool) {
        _assertClass(tool, NativeTool);
        const ret = wasm.chatsession_addTool(this.__wbg_ptr, tool.__wbg_ptr);
        return ret;
    }
    /**
     * Consumes the tool handles; the session retains their callbacks.
     * @param {NativeTool[]} tools
     * @returns {Promise<void>}
     */
    addTools(tools) {
        const ptr0 = passArrayJsValueToWasm0(tools, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.chatsession_addTools(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @returns {Promise<ChatMessage[]>}
     */
    get messages() {
        const ret = wasm.chatsession_messages(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatMessage[]} input
     * @param {ChatReplyConfig} config
     * @returns {Promise<ChatSessionStream>}
     */
    replyWithStream(input, config) {
        const ptr0 = passArrayJsValueToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        _assertClass(config, ChatReplyConfig);
        var ptr1 = config.__destroy_into_raw();
        const ret = wasm.chatsession_replyWithStream(this.__wbg_ptr, ptr0, len0, ptr1);
        return ret;
    }
    /**
     * @param {ChatMessage[]} input
     * @param {ChatReplyConfig} config
     * @returns {Promise<ChatReply[]>}
     */
    reply(input, config) {
        const ptr0 = passArrayJsValueToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        _assertClass(config, ChatReplyConfig);
        var ptr1 = config.__destroy_into_raw();
        const ret = wasm.chatsession_reply(this.__wbg_ptr, ptr0, len0, ptr1);
        return ret;
    }
    /**
     * @returns {Promise<void>}
     */
    reset() {
        const ret = wasm.chatsession_reset(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<SamplingParameters | undefined>}
     */
    get samplingDefaults() {
        const ret = wasm.chatsession_samplingDefaults(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<ChatSessionState>}
     */
    get state() {
        const ret = wasm.chatsession_state(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<boolean>}
     */
    get supportsToolCalls() {
        const ret = wasm.chatsession_supportsToolCalls(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) ChatSession.prototype[Symbol.dispose] = ChatSession.prototype.free;

export class ChatSessionStream {
    static __wrap(ptr) {
        const obj = Object.create(ChatSessionStream.prototype);
        obj.__wbg_ptr = ptr;
        ChatSessionStreamFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatSessionStreamFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatsessionstream_free(ptr, 0);
    }
    /**
     * @returns {CancelToken}
     */
    get cancelToken() {
        const ret = wasm.chatsessionstream_cancelToken(this.__wbg_ptr);
        return CancelToken.__wrap(ret);
    }
    /**
     * @returns {Promise<ChatSessionStreamChunk | undefined>}
     */
    next() {
        const ret = wasm.chatsessionstream_next(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) ChatSessionStream.prototype[Symbol.dispose] = ChatSessionStream.prototype.free;

export class ClassificationLabel {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClassificationLabelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_classificationlabel_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get description() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_classificationlabel_description(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {bigint}
     */
    get index() {
        const ret = wasm.__wbg_get_classificationlabel_index(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get label() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_classificationlabel_label(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get recommended_threshold() {
        const ret = wasm.__wbg_get_classificationlabel_recommended_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get title() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_classificationlabel_title(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set description(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_classificationlabel_description(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {bigint} arg0
     */
    set index(arg0) {
        wasm.__wbg_set_classificationlabel_index(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set label(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_classificationlabel_label(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} arg0
     */
    set recommended_threshold(arg0) {
        wasm.__wbg_set_classificationlabel_recommended_threshold(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set title(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_classificationlabel_title(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ClassificationLabel.prototype[Symbol.dispose] = ClassificationLabel.prototype.free;

export class ClassificationMessage {
    static __wrap(ptr) {
        const obj = Object.create(ClassificationMessage.prototype);
        obj.__wbg_ptr = ptr;
        ClassificationMessageFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClassificationMessageFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_classificationmessage_free(ptr, 0);
    }
    /**
     * @param {string} content
     * @returns {ClassificationMessage}
     */
    static assistant(content) {
        const ptr0 = passStringToWasm0(content, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.classificationmessage_assistant(ptr0, len0);
        return ClassificationMessage.__wrap(ret);
    }
    /**
     * @param {string} content
     * @returns {ClassificationMessage}
     */
    static user(content) {
        const ptr0 = passStringToWasm0(content, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.classificationmessage_user(ptr0, len0);
        return ClassificationMessage.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get content() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_classificationmessage_content(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {ClassificationRole}
     */
    get role() {
        const ret = wasm.__wbg_get_classificationmessage_role(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} arg0
     */
    set content(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_classificationmessage_content(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ClassificationRole} arg0
     */
    set role(arg0) {
        wasm.__wbg_set_classificationmessage_role(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ClassificationMessage.prototype[Symbol.dispose] = ClassificationMessage.prototype.free;

export class ClassificationOutput {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClassificationOutputFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_classificationoutput_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    get logits() {
        const ret = wasm.__wbg_get_classificationoutput_logits(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {ClassificationOutputProbabilities}
     */
    get probabilities() {
        const ret = wasm.__wbg_get_classificationoutput_probabilities(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ClassificationStats}
     */
    get stats() {
        const ret = wasm.__wbg_get_classificationoutput_stats(this.__wbg_ptr);
        return ClassificationStats.__wrap(ret);
    }
    /**
     * @param {Float64Array} arg0
     */
    set logits(arg0) {
        const ptr0 = passArrayF64ToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_classificationoutput_logits(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ClassificationOutputProbabilities} arg0
     */
    set probabilities(arg0) {
        wasm.__wbg_set_classificationoutput_probabilities(this.__wbg_ptr, arg0);
    }
    /**
     * @param {ClassificationStats} arg0
     */
    set stats(arg0) {
        _assertClass(arg0, ClassificationStats);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_classificationoutput_stats(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ClassificationOutput.prototype[Symbol.dispose] = ClassificationOutput.prototype.free;

export class ClassificationStats {
    static __wrap(ptr) {
        const obj = Object.create(ClassificationStats.prototype);
        obj.__wbg_ptr = ptr;
        ClassificationStatsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClassificationStatsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_classificationstats_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get confidence() {
        const ret = wasm.__wbg_get_classificationstats_confidence(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get forward_pass_duration() {
        const ret = wasm.__wbg_get_classificationstats_forward_pass_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get postprocessing_duration() {
        const ret = wasm.__wbg_get_classificationstats_postprocessing_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get predicted_label() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_classificationstats_predicted_label(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get preprocessing_duration() {
        const ret = wasm.__wbg_get_classificationstats_preprocessing_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get tokens_count() {
        const ret = wasm.__wbg_get_classificationstats_tokens_count(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get tokens_per_second() {
        const ret = wasm.__wbg_get_classificationstats_tokens_per_second(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get total_duration() {
        const ret = wasm.__wbg_get_classificationstats_total_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set confidence(arg0) {
        wasm.__wbg_set_classificationstats_confidence(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set forward_pass_duration(arg0) {
        wasm.__wbg_set_classificationstats_forward_pass_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set postprocessing_duration(arg0) {
        wasm.__wbg_set_classificationstats_postprocessing_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set predicted_label(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_classificationstats_predicted_label(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} arg0
     */
    set preprocessing_duration(arg0) {
        wasm.__wbg_set_classificationstats_preprocessing_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set tokens_count(arg0) {
        wasm.__wbg_set_classificationstats_tokens_count(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set tokens_per_second(arg0) {
        wasm.__wbg_set_classificationstats_tokens_per_second(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set total_duration(arg0) {
        wasm.__wbg_set_classificationstats_total_duration(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ClassificationStats.prototype[Symbol.dispose] = ClassificationStats.prototype.free;

export class DecisionInput {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionInputFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisioninput_free(ptr, 0);
    }
    /**
     * @returns {DecisionQuestion[]}
     */
    get questions() {
        const ret = wasm.__wbg_get_decisioninput_questions(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    get state() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_decisioninput_state(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {DecisionQuestion[]} arg0
     */
    set questions(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_decisioninput_questions(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set state(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_decisioninput_state(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) DecisionInput.prototype[Symbol.dispose] = DecisionInput.prototype.free;

export class DecisionOption {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionOptionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionoption_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get description() {
        const ret = wasm.__wbg_get_decisionoption_description(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string}
     */
    get key() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_decisionoption_key(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string | null} [arg0]
     */
    set description(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_decisionoption_description(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set key(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_decisionoption_key(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) DecisionOption.prototype[Symbol.dispose] = DecisionOption.prototype.free;

export class DecisionOutput {
    static __wrap(ptr) {
        const obj = Object.create(DecisionOutput.prototype);
        obj.__wbg_ptr = ptr;
        DecisionOutputFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionOutputFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionoutput_free(ptr, 0);
    }
    /**
     * @returns {DecisionResult[]}
     */
    get results() {
        const ret = wasm.__wbg_get_decisionoutput_results(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {DecisionResult[]} arg0
     */
    set results(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_decisionoutput_results(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) DecisionOutput.prototype[Symbol.dispose] = DecisionOutput.prototype.free;

export class DecisionResult {
    static __wrap(ptr) {
        const obj = Object.create(DecisionResult.prototype);
        obj.__wbg_ptr = ptr;
        DecisionResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    static __unwrap(jsValue) {
        if (!(jsValue instanceof DecisionResult)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionresult_free(ptr, 0);
    }
    /**
     * @returns {DecisionAnswer}
     */
    get answer() {
        const ret = wasm.__wbg_get_decisionresult_answer(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {DecisionStats}
     */
    get stats() {
        const ret = wasm.__wbg_get_decisionresult_stats(this.__wbg_ptr);
        return DecisionStats.__wrap(ret);
    }
    /**
     * @param {DecisionAnswer} arg0
     */
    set answer(arg0) {
        wasm.__wbg_set_decisionresult_answer(this.__wbg_ptr, arg0);
    }
    /**
     * @param {DecisionStats} arg0
     */
    set stats(arg0) {
        _assertClass(arg0, DecisionStats);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_decisionresult_stats(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) DecisionResult.prototype[Symbol.dispose] = DecisionResult.prototype.free;

export class DecisionSession {
    static __wrap(ptr) {
        const obj = Object.create(DecisionSession.prototype);
        obj.__wbg_ptr = ptr;
        DecisionSessionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionSessionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionsession_free(ptr, 0);
    }
    /**
     * @param {DecisionInput} input
     * @returns {Promise<DecisionOutput>}
     */
    decide(input) {
        _assertClass(input, DecisionInput);
        var ptr0 = input.__destroy_into_raw();
        const ret = wasm.decisionsession_decide(this.__wbg_ptr, ptr0);
        return ret;
    }
    /**
     * @returns {Promise<DecisionSessionState>}
     */
    get state() {
        const ret = wasm.decisionsession_state(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) DecisionSession.prototype[Symbol.dispose] = DecisionSession.prototype.free;

export class DecisionStats {
    static __wrap(ptr) {
        const obj = Object.create(DecisionStats.prototype);
        obj.__wbg_ptr = ptr;
        DecisionStatsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionStatsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionstats_free(ptr, 0);
    }
    /**
     * @returns {ReplyEnergy | undefined}
     */
    get energy() {
        const ret = wasm.__wbg_get_decisionstats_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get prefill_duration() {
        const ret = wasm.__wbg_get_decisionstats_prefill_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get tokens_count() {
        const ret = wasm.__wbg_get_decisionstats_tokens_count(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get total_duration() {
        const ret = wasm.__wbg_get_decisionstats_total_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ReplyEnergy | null} [arg0]
     */
    set energy(arg0) {
        wasm.__wbg_set_decisionstats_energy(this.__wbg_ptr, isLikeNone(arg0) ? 0 : addToExternrefTable0(arg0));
    }
    /**
     * @param {number} arg0
     */
    set prefill_duration(arg0) {
        wasm.__wbg_set_decisionstats_prefill_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set tokens_count(arg0) {
        wasm.__wbg_set_decisionstats_tokens_count(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set total_duration(arg0) {
        wasm.__wbg_set_decisionstats_total_duration(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) DecisionStats.prototype[Symbol.dispose] = DecisionStats.prototype.free;

export class Device {
    static __wrap(ptr) {
        const obj = Object.create(Device.prototype);
        obj.__wbg_ptr = ptr;
        DeviceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DeviceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_device_free(ptr, 0);
    }
    /**
     * @returns {Device}
     */
    static create() {
        const ret = wasm.device_create();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Device.__wrap(ret[0]);
    }
    /**
     * @returns {string | undefined}
     */
    get cpu_name() {
        const ret = wasm.__wbg_get_device_cpu_name(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string}
     */
    get home_path() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_device_home_path(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {bigint | undefined}
     */
    get memory_total() {
        const ret = wasm.__wbg_get_device_memory_total(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {string | undefined}
     */
    get os_name() {
        const ret = wasm.__wbg_get_device_os_name(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @param {string | null} [arg0]
     */
    set cpu_name(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_device_cpu_name(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set home_path(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_device_home_path(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {bigint | null} [arg0]
     */
    set memory_total(arg0) {
        wasm.__wbg_set_device_memory_total(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? BigInt(0) : arg0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set os_name(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_device_os_name(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Device.prototype[Symbol.dispose] = Device.prototype.free;

export class DownloadState {
    static __wrap(ptr) {
        const obj = Object.create(DownloadState.prototype);
        obj.__wbg_ptr = ptr;
        DownloadStateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadstate_free(ptr, 0);
    }
    /**
     * @returns {boolean}
     */
    get canDelete() {
        const ret = wasm.downloadstate_canDelete(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get canPause() {
        const ret = wasm.downloadstate_canPause(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isInProgress() {
        const ret = wasm.downloadstate_isInProgress(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.downloadstate_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get progress() {
        const ret = wasm.downloadstate_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get downloaded_bytes() {
        const ret = wasm.__wbg_get_downloadstate_downloaded_bytes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {DownloadPhase}
     */
    get phase() {
        const ret = wasm.__wbg_get_downloadstate_phase(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {bigint}
     */
    get total_bytes() {
        const ret = wasm.__wbg_get_downloadstate_total_bytes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {bigint} arg0
     */
    set downloaded_bytes(arg0) {
        wasm.__wbg_set_downloadstate_downloaded_bytes(this.__wbg_ptr, arg0);
    }
    /**
     * @param {DownloadPhase} arg0
     */
    set phase(arg0) {
        wasm.__wbg_set_downloadstate_phase(this.__wbg_ptr, arg0);
    }
    /**
     * @param {bigint} arg0
     */
    set total_bytes(arg0) {
        wasm.__wbg_set_downloadstate_total_bytes(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) DownloadState.prototype[Symbol.dispose] = DownloadState.prototype.free;

export class Downloader {
    static __wrap(ptr) {
        const obj = Object.create(Downloader.prototype);
        obj.__wbg_ptr = ptr;
        DownloaderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloaderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloader_free(ptr, 0);
    }
    /**
     * @returns {Promise<void>}
     */
    delete() {
        const ret = wasm.downloader_delete(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<void>}
     */
    pause() {
        const ret = wasm.downloader_pause(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<DownloaderStream>}
     */
    progress() {
        const ret = wasm.downloader_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<void>}
     */
    resume() {
        const ret = wasm.downloader_resume(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<DownloadState | undefined>}
     */
    get state() {
        const ret = wasm.downloader_state(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Downloader.prototype[Symbol.dispose] = Downloader.prototype.free;

export class DownloaderStream {
    static __wrap(ptr) {
        const obj = Object.create(DownloaderStream.prototype);
        obj.__wbg_ptr = ptr;
        DownloaderStreamFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloaderStreamFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloaderstream_free(ptr, 0);
    }
    /**
     * @returns {Promise<DownloadState | undefined>}
     */
    next() {
        const ret = wasm.downloaderstream_next(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) DownloaderStream.prototype[Symbol.dispose] = DownloaderStream.prototype.free;

export class Engine {
    static __wrap(ptr) {
        const obj = Object.create(Engine.prototype);
        obj.__wbg_ptr = ptr;
        EngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engine_free(ptr, 0);
    }
    /**
     * @param {Model} model
     * @param {ChatConfig} config
     * @returns {Promise<ChatInstance>}
     */
    chatInstance(model, config) {
        _assertClass(model, Model);
        var ptr0 = model.__destroy_into_raw();
        _assertClass(config, ChatConfig);
        var ptr1 = config.__destroy_into_raw();
        const ret = wasm.engine_chatInstance(this.__wbg_ptr, ptr0, ptr1);
        return ret;
    }
    /**
     * Returns metadata for a Uzu chat model, including models not yet downloaded.
     * Encoding capabilities come from the catalog; the context limit needs local configuration.
     * Remote and non-chat models return `None`.
     * @param {Model} model
     * @returns {Promise<ChatModelInfo | undefined>}
     */
    chatModelInfo(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_chatModelInfo(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatInstance} instance
     * @returns {Promise<ChatSession>}
     */
    chatWithInstance(instance) {
        _assertClass(instance, ChatInstance);
        const ret = wasm.engine_chatWithInstance(this.__wbg_ptr, instance.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @param {ChatConfig} config
     * @returns {Promise<ChatSession>}
     */
    chat(model, config) {
        _assertClass(model, Model);
        var ptr0 = model.__destroy_into_raw();
        _assertClass(config, ChatConfig);
        var ptr1 = config.__destroy_into_raw();
        const ret = wasm.engine_chat(this.__wbg_ptr, ptr0, ptr1);
        return ret;
    }
    /**
     * @param {EngineConfig} config
     * @returns {Promise<Engine>}
     */
    static create(config) {
        _assertClass(config, EngineConfig);
        var ptr0 = config.__destroy_into_raw();
        const ret = wasm.engine_create(ptr0);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {Promise<DecisionSession>}
     */
    decision(model) {
        _assertClass(model, Model);
        var ptr0 = model.__destroy_into_raw();
        const ret = wasm.engine_decision(this.__wbg_ptr, ptr0);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {Promise<DownloadState | undefined>}
     */
    downloadState(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_downloadState(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Map<any, any>>}
     */
    get downloadStates() {
        const ret = wasm.engine_downloadStates(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {Promise<DownloaderStream>}
     */
    download(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_download(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {Downloader}
     */
    downloader(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_downloader(this.__wbg_ptr, model.__wbg_ptr);
        return Downloader.__wrap(ret);
    }
    /**
     * @param {string} identifier
     * @returns {Promise<Model | undefined>}
     */
    modelByIdentifier(identifier) {
        const ptr0 = passStringToWasm0(identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_modelByIdentifier(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {string} path
     * @returns {Promise<Model | undefined>}
     */
    modelByPath(path) {
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_modelByPath(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {string} repo_id
     * @returns {Promise<Model | undefined>}
     */
    modelByRepoId(repo_id) {
        const ptr0 = passStringToWasm0(repo_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_modelByRepoId(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @returns {Promise<ModelFamily[]>}
     */
    get modelFamilies() {
        const ret = wasm.engine_modelFamilies(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {Promise<string | undefined>}
     */
    modelPath(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_modelPath(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<ModelRegistry[]>}
     */
    get modelRegistries() {
        const ret = wasm.engine_modelRegistries(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<ModelVendor[]>}
     */
    get modelVendors() {
        const ret = wasm.engine_modelVendors(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} identifier
     * @returns {Promise<Model | undefined>}
     */
    model(identifier) {
        const ptr0 = passStringToWasm0(identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_model(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {string} family_identifier
     * @returns {Promise<Model[]>}
     */
    modelsByFamily(family_identifier) {
        const ptr0 = passStringToWasm0(family_identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_modelsByFamily(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {string} vendor_identifier
     * @returns {Promise<Model[]>}
     */
    modelsByVendor(vendor_identifier) {
        const ptr0 = passStringToWasm0(vendor_identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_modelsByVendor(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsDownloadable() {
        const ret = wasm.engine_modelsDownloadable(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsForChat() {
        const ret = wasm.engine_modelsForChat(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsForClassification() {
        const ret = wasm.engine_modelsForClassification(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsForSpeculation() {
        const ret = wasm.engine_modelsForSpeculation(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsForTextToSpeech() {
        const ret = wasm.engine_modelsForTextToSpeech(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsForTranslation() {
        const ret = wasm.engine_modelsForTranslation(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsOnDevice() {
        const ret = wasm.engine_modelsOnDevice(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get modelsRemote() {
        const ret = wasm.engine_modelsRemote(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Promise<Model[]>}
     */
    get models() {
        const ret = wasm.engine_models(this.__wbg_ptr);
        return ret;
    }
    /**
     * Refresh registries and wait until the catalog has been published to storage.
     * Registries that fail to refresh retain their last known entries.
     * @returns {Promise<Model[]>}
     */
    refreshModels() {
        const ret = wasm.engine_refreshModels(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {EngineCallback} callback
     * @returns {Promise<void>}
     */
    registerCallback(callback) {
        _assertClass(callback, EngineCallback);
        const ret = wasm.engine_registerCallback(this.__wbg_ptr, callback.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} identifier
     * @returns {Promise<void>}
     */
    removeBackend(identifier) {
        const ptr0 = passStringToWasm0(identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_removeBackend(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {string} registry_identifier
     * @returns {Promise<void>}
     */
    removeRegistry(registry_identifier) {
        const ptr0 = passStringToWasm0(registry_identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engine_removeRegistry(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
}
if (Symbol.dispose) Engine.prototype[Symbol.dispose] = Engine.prototype.free;

export class EngineCallback {
    static __wrap(ptr) {
        const obj = Object.create(EngineCallback.prototype);
        obj.__wbg_ptr = ptr;
        EngineCallbackFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineCallbackFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_enginecallback_free(ptr, 0);
    }
    /**
     * @param {Function} callback
     * @returns {EngineCallback}
     */
    static create(callback) {
        const ret = wasm.enginecallback_create(callback);
        return EngineCallback.__wrap(ret);
    }
}
if (Symbol.dispose) EngineCallback.prototype[Symbol.dispose] = EngineCallback.prototype.free;

export class EngineConfig {
    static __wrap(ptr) {
        const obj = Object.create(EngineConfig.prototype);
        obj.__wbg_ptr = ptr;
        EngineConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engineconfig_free(ptr, 0);
    }
    /**
     * @returns {EngineConfig}
     */
    static create() {
        const ret = wasm.engineconfig_create();
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {boolean} allow_lmstudio_usage
     * @returns {EngineConfig}
     */
    withAllowLmstudioUsage(allow_lmstudio_usage) {
        const ret = wasm.engineconfig_withAllowLmstudioUsage(this.__wbg_ptr, allow_lmstudio_usage);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {boolean} allow_ollama_usage
     * @returns {EngineConfig}
     */
    withAllowOllamaUsage(allow_ollama_usage) {
        const ret = wasm.engineconfig_withAllowOllamaUsage(this.__wbg_ptr, allow_ollama_usage);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} anthropic_api_key
     * @returns {EngineConfig}
     */
    withAnthropicApiKey(anthropic_api_key) {
        const ptr0 = passStringToWasm0(anthropic_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withAnthropicApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} baseten_api_key
     * @returns {EngineConfig}
     */
    withBasetenApiKey(baseten_api_key) {
        const ptr0 = passStringToWasm0(baseten_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withBasetenApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {DownloadManagerType} download_manager_type
     * @returns {EngineConfig}
     */
    withDownloadManagerType(download_manager_type) {
        const ret = wasm.engineconfig_withDownloadManagerType(this.__wbg_ptr, download_manager_type);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} gemini_api_key
     * @returns {EngineConfig}
     */
    withGeminiApiKey(gemini_api_key) {
        const ptr0 = passStringToWasm0(gemini_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withGeminiApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} huggingface_api_key
     * @returns {EngineConfig}
     */
    withHuggingfaceApiKey(huggingface_api_key) {
        const ptr0 = passStringToWasm0(huggingface_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withHuggingfaceApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} local_path
     * @returns {EngineConfig}
     */
    withLocalPath(local_path) {
        const ptr0 = passStringToWasm0(local_path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withLocalPath(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} mirai_api_key
     * @returns {EngineConfig}
     */
    withMiraiApiKey(mirai_api_key) {
        const ptr0 = passStringToWasm0(mirai_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withMiraiApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} openai_api_key
     * @returns {EngineConfig}
     */
    withOpenaiApiKey(openai_api_key) {
        const ptr0 = passStringToWasm0(openai_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withOpenaiApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} openrouter_api_key
     * @returns {EngineConfig}
     */
    withOpenrouterApiKey(openrouter_api_key) {
        const ptr0 = passStringToWasm0(openrouter_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withOpenrouterApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @param {string} xai_api_key
     * @returns {EngineConfig}
     */
    withXaiApiKey(xai_api_key) {
        const ptr0 = passStringToWasm0(xai_api_key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.engineconfig_withXaiApiKey(this.__wbg_ptr, ptr0, len0);
        return EngineConfig.__wrap(ret);
    }
    /**
     * @returns {boolean}
     */
    get allow_lmstudio_usage() {
        const ret = wasm.__wbg_get_engineconfig_allow_lmstudio_usage(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get allow_ollama_usage() {
        const ret = wasm.__wbg_get_engineconfig_allow_ollama_usage(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {string | undefined}
     */
    get anthropic_api_key() {
        const ret = wasm.__wbg_get_engineconfig_anthropic_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get baseten_api_key() {
        const ret = wasm.__wbg_get_engineconfig_baseten_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {DownloadManagerType}
     */
    get download_manager_type() {
        const ret = wasm.__wbg_get_engineconfig_download_manager_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string | undefined}
     */
    get gemini_api_key() {
        const ret = wasm.__wbg_get_engineconfig_gemini_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get huggingface_api_key() {
        const ret = wasm.__wbg_get_engineconfig_huggingface_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get local_path() {
        const ret = wasm.__wbg_get_engineconfig_local_path(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get mirai_api_key() {
        const ret = wasm.__wbg_get_engineconfig_mirai_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get openai_api_key() {
        const ret = wasm.__wbg_get_engineconfig_openai_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get openrouter_api_key() {
        const ret = wasm.__wbg_get_engineconfig_openrouter_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string | undefined}
     */
    get xai_api_key() {
        const ret = wasm.__wbg_get_engineconfig_xai_api_key(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @param {boolean} arg0
     */
    set allow_lmstudio_usage(arg0) {
        wasm.__wbg_set_engineconfig_allow_lmstudio_usage(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} arg0
     */
    set allow_ollama_usage(arg0) {
        wasm.__wbg_set_engineconfig_allow_ollama_usage(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set anthropic_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_anthropic_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set baseten_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_baseten_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {DownloadManagerType} arg0
     */
    set download_manager_type(arg0) {
        wasm.__wbg_set_engineconfig_download_manager_type(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set gemini_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_gemini_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set huggingface_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_huggingface_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set local_path(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_local_path(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set mirai_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_mirai_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set openai_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_openai_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set openrouter_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_openrouter_api_key(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set xai_api_key(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_engineconfig_xai_api_key(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) EngineConfig.prototype[Symbol.dispose] = EngineConfig.prototype.free;

export class Feature {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FeatureFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_feature_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_feature_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string[]}
     */
    get values() {
        const ret = wasm.__wbg_get_feature_values(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_feature_name(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string[]} arg0
     */
    set values(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_feature_values(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Feature.prototype[Symbol.dispose] = Feature.prototype.free;

export class File {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FileFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_file_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get crc32c() {
        const ret = wasm.file_crc32c(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {Hash[]}
     */
    get hashes() {
        const ret = wasm.__wbg_get_file_hashes(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_file_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {bigint}
     */
    get size() {
        const ret = wasm.__wbg_get_file_size(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get url() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_file_url(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {Hash[]} arg0
     */
    set hashes(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_file_hashes(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_file_name(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {bigint} arg0
     */
    set size(arg0) {
        wasm.__wbg_set_file_size(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set url(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_file_url(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) File.prototype[Symbol.dispose] = File.prototype.free;

export class Hash {
    static __wrap(ptr) {
        const obj = Object.create(Hash.prototype);
        obj.__wbg_ptr = ptr;
        HashFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    static __unwrap(jsValue) {
        if (!(jsValue instanceof Hash)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        HashFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_hash_free(ptr, 0);
    }
    /**
     * @returns {HashMethod}
     */
    get method() {
        const ret = wasm.__wbg_get_hash_method(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get value() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_hash_value(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {HashMethod} arg0
     */
    set method(arg0) {
        wasm.__wbg_set_hash_method(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set value(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_hash_value(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Hash.prototype[Symbol.dispose] = Hash.prototype.free;

export class Image {
    static __wrap(ptr) {
        const obj = Object.create(Image.prototype);
        obj.__wbg_ptr = ptr;
        ImageFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    static __unwrap(jsValue) {
        if (!(jsValue instanceof Image)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ImageFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_image_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get description() {
        const ret = wasm.__wbg_get_image_description(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {ImageFormat}
     */
    get format() {
        const ret = wasm.__wbg_get_image_format(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number | undefined}
     */
    get height() {
        const ret = wasm.__wbg_get_image_height(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @returns {string}
     */
    get id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_image_id(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {ImageTheme}
     */
    get theme() {
        const ret = wasm.__wbg_get_image_theme(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get url() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_image_url(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number | undefined}
     */
    get width() {
        const ret = wasm.__wbg_get_image_width(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @param {string | null} [arg0]
     */
    set description(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_image_description(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ImageFormat} arg0
     */
    set format(arg0) {
        wasm.__wbg_set_image_format(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set height(arg0) {
        wasm.__wbg_set_image_height(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
    /**
     * @param {string} arg0
     */
    set id(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_image_id(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ImageTheme} arg0
     */
    set theme(arg0) {
        wasm.__wbg_set_image_theme(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set url(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_image_url(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set width(arg0) {
        wasm.__wbg_set_image_width(this.__wbg_ptr, isLikeNone(arg0) ? Number.MAX_SAFE_INTEGER : (arg0) >>> 0);
    }
}
if (Symbol.dispose) Image.prototype[Symbol.dispose] = Image.prototype.free;

export class IntoUnderlyingByteSource {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingByteSourceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingbytesource_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get autoAllocateChunkSize() {
        const ret = wasm.intounderlyingbytesource_autoAllocateChunkSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    cancel() {
        const ptr = this.__destroy_into_raw();
        wasm.intounderlyingbytesource_cancel(ptr);
    }
    /**
     * @param {ReadableByteStreamController} controller
     * @returns {Promise<any>}
     */
    pull(controller) {
        const ret = wasm.intounderlyingbytesource_pull(this.__wbg_ptr, controller);
        return ret;
    }
    /**
     * @param {ReadableByteStreamController} controller
     */
    start(controller) {
        wasm.intounderlyingbytesource_start(this.__wbg_ptr, controller);
    }
    /**
     * @returns {ReadableStreamType}
     */
    get type() {
        const ret = wasm.intounderlyingbytesource_type(this.__wbg_ptr);
        return __wbindgen_enum_ReadableStreamType[ret];
    }
}
if (Symbol.dispose) IntoUnderlyingByteSource.prototype[Symbol.dispose] = IntoUnderlyingByteSource.prototype.free;

export class IntoUnderlyingSink {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingSinkFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingsink_free(ptr, 0);
    }
    /**
     * @param {any} reason
     * @returns {Promise<any>}
     */
    abort(reason) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.intounderlyingsink_abort(ptr, reason);
        return ret;
    }
    /**
     * @returns {Promise<any>}
     */
    close() {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.intounderlyingsink_close(ptr);
        return ret;
    }
    /**
     * @param {any} chunk
     * @returns {Promise<any>}
     */
    write(chunk) {
        const ret = wasm.intounderlyingsink_write(this.__wbg_ptr, chunk);
        return ret;
    }
}
if (Symbol.dispose) IntoUnderlyingSink.prototype[Symbol.dispose] = IntoUnderlyingSink.prototype.free;

export class IntoUnderlyingSource {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntoUnderlyingSourceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_intounderlyingsource_free(ptr, 0);
    }
    cancel() {
        const ptr = this.__destroy_into_raw();
        wasm.intounderlyingsource_cancel(ptr);
    }
    /**
     * @param {ReadableStreamDefaultController} controller
     * @returns {Promise<any>}
     */
    pull(controller) {
        const ret = wasm.intounderlyingsource_pull(this.__wbg_ptr, controller);
        return ret;
    }
}
if (Symbol.dispose) IntoUnderlyingSource.prototype[Symbol.dispose] = IntoUnderlyingSource.prototype.free;

export class Metadata {
    static __wrap(ptr) {
        const obj = Object.create(Metadata.prototype);
        obj.__wbg_ptr = ptr;
        MetadataFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MetadataFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_metadata_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get description() {
        const ret = wasm.__wbg_get_metadata_description(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {Image[]}
     */
    get icons() {
        const ret = wasm.__wbg_get_metadata_icons(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_metadata_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_metadata_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} name
     * @returns {Metadata}
     */
    static external(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.metadata_external(ptr0, len0);
        return Metadata.__wrap(ret);
    }
    /**
     * @param {string | null} [arg0]
     */
    set description(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_metadata_description(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Image[]} arg0
     */
    set icons(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_metadata_icons(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_metadata_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_metadata_name(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Metadata.prototype[Symbol.dispose] = Metadata.prototype.free;

export class Model {
    static __wrap(ptr) {
        const obj = Object.create(Model.prototype);
        obj.__wbg_ptr = ptr;
        ModelFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_model_free(ptr, 0);
    }
    /**
     * @returns {ModelAccessibility}
     */
    get accessibility() {
        const ret = wasm.__wbg_get_model_accessibility(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ModelBackend[]}
     */
    get backends() {
        const ret = wasm.__wbg_get_model_backends(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Value | undefined}
     */
    get encoding() {
        const ret = wasm.__wbg_get_model_encoding(this.__wbg_ptr);
        return ret === 0 ? undefined : Value.__wrap(ret);
    }
    /**
     * @returns {ModelFamily | undefined}
     */
    get family() {
        const ret = wasm.__wbg_get_model_family(this.__wbg_ptr);
        return ret === 0 ? undefined : ModelFamily.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_model_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {ModelProperties | undefined}
     */
    get properties() {
        const ret = wasm.__wbg_get_model_properties(this.__wbg_ptr);
        return ret === 0 ? undefined : ModelProperties.__wrap(ret);
    }
    /**
     * @returns {ModelQuantization | undefined}
     */
    get quantization() {
        const ret = wasm.__wbg_get_model_quantization(this.__wbg_ptr);
        return ret === 0 ? undefined : ModelQuantization.__wrap(ret);
    }
    /**
     * @returns {ModelRegistry}
     */
    get registry() {
        const ret = wasm.__wbg_get_model_registry(this.__wbg_ptr);
        return ModelRegistry.__wrap(ret);
    }
    /**
     * @returns {ModelSpecialization[]}
     */
    get specializations() {
        const ret = wasm.__wbg_get_model_specializations(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    get cacheIdentifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.model_cacheIdentifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string | undefined}
     */
    get checkpointVersion() {
        const ret = wasm.model_checkpointVersion(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @param {string} identifier
     * @param {string} registry_identifier
     * @param {string} registry_name
     * @param {string} backend_identifier
     * @param {string} backend_name
     * @param {string} backend_version
     * @param {ModelSpecialization[]} specializations
     * @param {ModelAccessibility} accessibility
     * @param {Value | null} [encoding]
     * @returns {Model}
     */
    static external(identifier, registry_identifier, registry_name, backend_identifier, backend_name, backend_version, specializations, accessibility, encoding) {
        const ptr0 = passStringToWasm0(identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(registry_identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(registry_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(backend_identifier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passStringToWasm0(backend_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len4 = WASM_VECTOR_LEN;
        const ptr5 = passStringToWasm0(backend_version, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len5 = WASM_VECTOR_LEN;
        const ptr6 = passArrayJsValueToWasm0(specializations, wasm.__wbindgen_malloc);
        const len6 = WASM_VECTOR_LEN;
        let ptr7 = 0;
        if (!isLikeNone(encoding)) {
            _assertClass(encoding, Value);
            ptr7 = encoding.__destroy_into_raw();
        }
        const ret = wasm.model_external(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, accessibility, ptr7);
        return Model.__wrap(ret);
    }
    /**
     * @returns {string | undefined}
     */
    get filesystemPath() {
        const ret = wasm.model_filesystemPath(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get isChatCapable() {
        const ret = wasm.model_isChatCapable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isClassificationCapable() {
        const ret = wasm.model_isClassificationCapable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isDownloadable() {
        const ret = wasm.model_isDownloadable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isOnDevice() {
        const ret = wasm.model_isOnDevice(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isQuantized() {
        const ret = wasm.model_isQuantized(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isRemote() {
        const ret = wasm.model_isRemote(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isSpeculationCapable() {
        const ret = wasm.model_isSpeculationCapable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isTextToSpeechCapable() {
        const ret = wasm.model_isTextToSpeechCapable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    get isTranslationCapable() {
        const ret = wasm.model_isTranslationCapable(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.model_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string[]}
     */
    get repoIds() {
        const ret = wasm.model_repoIds(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {ModelAccessibility} arg0
     */
    set accessibility(arg0) {
        wasm.__wbg_set_model_accessibility(this.__wbg_ptr, arg0);
    }
    /**
     * @param {ModelBackend[]} arg0
     */
    set backends(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_model_backends(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Value | null} [arg0]
     */
    set encoding(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, Value);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_model_encoding(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {ModelFamily | null} [arg0]
     */
    set family(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, ModelFamily);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_model_family(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_model_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ModelProperties | null} [arg0]
     */
    set properties(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, ModelProperties);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_model_properties(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {ModelQuantization | null} [arg0]
     */
    set quantization(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, ModelQuantization);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_model_quantization(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {ModelRegistry} arg0
     */
    set registry(arg0) {
        _assertClass(arg0, ModelRegistry);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_model_registry(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {ModelSpecialization[]} arg0
     */
    set specializations(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_model_specializations(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Model.prototype[Symbol.dispose] = Model.prototype.free;

export class ModelBackend {
    static __wrap(ptr) {
        const obj = Object.create(ModelBackend.prototype);
        obj.__wbg_ptr = ptr;
        ModelBackendFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    static __unwrap(jsValue) {
        if (!(jsValue instanceof ModelBackend)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelBackendFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelbackend_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelbackend_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Metadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_modelbackend_metadata(this.__wbg_ptr);
        return Metadata.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get version() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelbackend_version(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.modelbackend_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelbackend_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Metadata} arg0
     */
    set metadata(arg0) {
        _assertClass(arg0, Metadata);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelbackend_metadata(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {string} arg0
     */
    set version(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelbackend_version(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ModelBackend.prototype[Symbol.dispose] = ModelBackend.prototype.free;

export class ModelFamily {
    static __wrap(ptr) {
        const obj = Object.create(ModelFamily.prototype);
        obj.__wbg_ptr = ptr;
        ModelFamilyFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelFamilyFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelfamily_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelfamily_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Metadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_modelfamily_metadata(this.__wbg_ptr);
        return Metadata.__wrap(ret);
    }
    /**
     * @returns {ModelVendor}
     */
    get vendor() {
        const ret = wasm.__wbg_get_modelfamily_vendor(this.__wbg_ptr);
        return ModelVendor.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.modelfamily_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelfamily_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Metadata} arg0
     */
    set metadata(arg0) {
        _assertClass(arg0, Metadata);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelfamily_metadata(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {ModelVendor} arg0
     */
    set vendor(arg0) {
        _assertClass(arg0, ModelVendor);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelfamily_vendor(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ModelFamily.prototype[Symbol.dispose] = ModelFamily.prototype.free;

export class ModelProperties {
    static __wrap(ptr) {
        const obj = Object.create(ModelProperties.prototype);
        obj.__wbg_ptr = ptr;
        ModelPropertiesFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelPropertiesFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelproperties_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelproperties_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Metadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_modelproperties_metadata(this.__wbg_ptr);
        return Metadata.__wrap(ret);
    }
    /**
     * @returns {bigint}
     */
    get size() {
        const ret = wasm.__wbg_get_modelproperties_size(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string | undefined}
     */
    get version() {
        const ret = wasm.__wbg_get_modelproperties_version(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.modelproperties_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelproperties_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Metadata} arg0
     */
    set metadata(arg0) {
        _assertClass(arg0, Metadata);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelproperties_metadata(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {bigint} arg0
     */
    set size(arg0) {
        wasm.__wbg_set_modelproperties_size(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set version(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelproperties_version(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ModelProperties.prototype[Symbol.dispose] = ModelProperties.prototype.free;

export class ModelQuantization {
    static __wrap(ptr) {
        const obj = Object.create(ModelQuantization.prototype);
        obj.__wbg_ptr = ptr;
        ModelQuantizationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelQuantizationFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelquantization_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get bits_per_weight() {
        const ret = wasm.__wbg_get_modelquantization_bits_per_weight(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelquantization_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Metadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_modelquantization_metadata(this.__wbg_ptr);
        return Metadata.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get method() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelquantization_method(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {ModelVendor}
     */
    get vendor() {
        const ret = wasm.__wbg_get_modelquantization_vendor(this.__wbg_ptr);
        return ModelVendor.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.modelquantization_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {number} arg0
     */
    set bits_per_weight(arg0) {
        wasm.__wbg_set_modelquantization_bits_per_weight(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelquantization_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Metadata} arg0
     */
    set metadata(arg0) {
        _assertClass(arg0, Metadata);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelquantization_metadata(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {string} arg0
     */
    set method(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelquantization_method(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ModelVendor} arg0
     */
    set vendor(arg0) {
        _assertClass(arg0, ModelVendor);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelquantization_vendor(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ModelQuantization.prototype[Symbol.dispose] = ModelQuantization.prototype.free;

export class ModelRegistry {
    static __wrap(ptr) {
        const obj = Object.create(ModelRegistry.prototype);
        obj.__wbg_ptr = ptr;
        ModelRegistryFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelRegistryFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelregistry_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelregistry_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Metadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_modelregistry_metadata(this.__wbg_ptr);
        return Metadata.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.modelregistry_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelregistry_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Metadata} arg0
     */
    set metadata(arg0) {
        _assertClass(arg0, Metadata);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelregistry_metadata(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ModelRegistry.prototype[Symbol.dispose] = ModelRegistry.prototype.free;

export class ModelVendor {
    static __wrap(ptr) {
        const obj = Object.create(ModelVendor.prototype);
        obj.__wbg_ptr = ptr;
        ModelVendorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelVendorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelvendor_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_modelvendor_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Metadata}
     */
    get metadata() {
        const ret = wasm.__wbg_get_modelvendor_metadata(this.__wbg_ptr);
        return Metadata.__wrap(ret);
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.modelvendor_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_modelvendor_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Metadata} arg0
     */
    set metadata(arg0) {
        _assertClass(arg0, Metadata);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_modelvendor_metadata(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ModelVendor.prototype[Symbol.dispose] = ModelVendor.prototype.free;

export class NativeTool {
    static __unwrap(jsValue) {
        if (!(jsValue instanceof NativeTool)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        NativeToolFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_nativetool_free(ptr, 0);
    }
    /**
     * The definition uses the serialized ToolFunction shape. The callback receives
     * JSON arguments and an invocation ID, and returns JSON or a promise of JSON.
     * @param {string} definition_json
     * @param {Function} invoke_json
     * @param {Function} cancel
     */
    constructor(definition_json, invoke_json, cancel) {
        const ptr0 = passStringToWasm0(definition_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.nativetool_new(ptr0, len0, invoke_json, cancel);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        NativeToolFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) NativeTool.prototype[Symbol.dispose] = NativeTool.prototype.free;

export class PcmBatch {
    static __wrap(ptr) {
        const obj = Object.create(PcmBatch.prototype);
        obj.__wbg_ptr = ptr;
        PcmBatchFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PcmBatchFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pcmbatch_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get channels() {
        const ret = wasm.__wbg_get_pcmbatch_channels(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Uint32Array}
     */
    get lengths() {
        const ret = wasm.__wbg_get_pcmbatch_lengths(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get sample_rate() {
        const ret = wasm.__wbg_get_pcmbatch_sample_rate(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    get samples() {
        const ret = wasm.__wbg_get_pcmbatch_samples(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    get batchSize() {
        const ret = wasm.pcmbatch_batchSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get duration() {
        const ret = wasm.pcmbatch_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} path
     */
    saveAsWav(path) {
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.pcmbatch_saveAsWav(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    get totalFrames() {
        const ret = wasm.pcmbatch_totalFrames(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set channels(arg0) {
        wasm.__wbg_set_pcmbatch_channels(this.__wbg_ptr, arg0);
    }
    /**
     * @param {Uint32Array} arg0
     */
    set lengths(arg0) {
        const ptr0 = passArray32ToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_pcmbatch_lengths(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} arg0
     */
    set sample_rate(arg0) {
        wasm.__wbg_set_pcmbatch_sample_rate(this.__wbg_ptr, arg0);
    }
    /**
     * @param {Float64Array} arg0
     */
    set samples(arg0) {
        const ptr0 = passArrayF64ToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_pcmbatch_samples(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) PcmBatch.prototype[Symbol.dispose] = PcmBatch.prototype.free;

export class Repository {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RepositoryFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_repository_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get commit_hash() {
        const ret = wasm.__wbg_get_repository_commit_hash(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string}
     */
    get identifier() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_repository_identifier(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string[] | undefined}
     */
    get paths() {
        const ret = wasm.__wbg_get_repository_paths(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * @param {string | null} [arg0]
     */
    set commit_hash(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_repository_commit_hash(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set identifier(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_repository_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string[] | null} [arg0]
     */
    set paths(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_repository_paths(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Repository.prototype[Symbol.dispose] = Repository.prototype.free;

export class SamplingParameters {
    static __wrap(ptr) {
        const obj = Object.create(SamplingParameters.prototype);
        obj.__wbg_ptr = ptr;
        SamplingParametersFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingParametersFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingparameters_free(ptr, 0);
    }
    /**
     * @returns {number | undefined}
     */
    get min_p() {
        const ret = wasm.__wbg_get_samplingparameters_min_p(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    get repetition_penalty() {
        const ret = wasm.__wbg_get_samplingparameters_repetition_penalty(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {bigint | undefined}
     */
    get suffix_repetition_length() {
        const ret = wasm.__wbg_get_samplingparameters_suffix_repetition_length(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    get temperature() {
        const ret = wasm.__wbg_get_samplingparameters_temperature(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {bigint | undefined}
     */
    get top_k() {
        const ret = wasm.__wbg_get_samplingparameters_top_k(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @returns {number | undefined}
     */
    get top_p() {
        const ret = wasm.__wbg_get_samplingparameters_top_p(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1];
    }
    /**
     * @param {number | null} [arg0]
     */
    set min_p(arg0) {
        wasm.__wbg_set_samplingparameters_min_p(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set repetition_penalty(arg0) {
        wasm.__wbg_set_samplingparameters_repetition_penalty(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @param {bigint | null} [arg0]
     */
    set suffix_repetition_length(arg0) {
        wasm.__wbg_set_samplingparameters_suffix_repetition_length(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? BigInt(0) : arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set temperature(arg0) {
        wasm.__wbg_set_samplingparameters_temperature(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @param {bigint | null} [arg0]
     */
    set top_k(arg0) {
        wasm.__wbg_set_samplingparameters_top_k(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? BigInt(0) : arg0);
    }
    /**
     * @param {number | null} [arg0]
     */
    set top_p(arg0) {
        wasm.__wbg_set_samplingparameters_top_p(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
}
if (Symbol.dispose) SamplingParameters.prototype[Symbol.dispose] = SamplingParameters.prototype.free;

export class TextToSpeechOutput {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TextToSpeechOutputFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_texttospeechoutput_free(ptr, 0);
    }
    /**
     * @returns {PcmBatch}
     */
    get pcm_batch() {
        const ret = wasm.__wbg_get_texttospeechoutput_pcm_batch(this.__wbg_ptr);
        return PcmBatch.__wrap(ret);
    }
    /**
     * @returns {TextToSpeechStats}
     */
    get stats() {
        const ret = wasm.__wbg_get_texttospeechoutput_stats(this.__wbg_ptr);
        return TextToSpeechStats.__wrap(ret);
    }
    /**
     * @param {PcmBatch} arg0
     */
    set pcm_batch(arg0) {
        _assertClass(arg0, PcmBatch);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_texttospeechoutput_pcm_batch(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {TextToSpeechStats} arg0
     */
    set stats(arg0) {
        _assertClass(arg0, TextToSpeechStats);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_texttospeechoutput_stats(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) TextToSpeechOutput.prototype[Symbol.dispose] = TextToSpeechOutput.prototype.free;

export class TextToSpeechStats {
    static __wrap(ptr) {
        const obj = Object.create(TextToSpeechStats.prototype);
        obj.__wbg_ptr = ptr;
        TextToSpeechStatsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TextToSpeechStatsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_texttospeechstats_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get audio_duration() {
        const ret = wasm.__wbg_get_texttospeechstats_audio_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get first_chunk_seconds() {
        const ret = wasm.__wbg_get_texttospeechstats_first_chunk_seconds(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get generation_duration() {
        const ret = wasm.__wbg_get_texttospeechstats_generation_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get text_length() {
        const ret = wasm.__wbg_get_texttospeechstats_text_length(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set audio_duration(arg0) {
        wasm.__wbg_set_texttospeechstats_audio_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set first_chunk_seconds(arg0) {
        wasm.__wbg_set_texttospeechstats_first_chunk_seconds(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set generation_duration(arg0) {
        wasm.__wbg_set_texttospeechstats_generation_duration(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set text_length(arg0) {
        wasm.__wbg_set_texttospeechstats_text_length(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) TextToSpeechStats.prototype[Symbol.dispose] = TextToSpeechStats.prototype.free;

export class Token {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TokenFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_token_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get id() {
        const ret = wasm.__wbg_get_token_id(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {boolean}
     */
    get is_special() {
        const ret = wasm.__wbg_get_token_is_special(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {string}
     */
    get value() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_token_value(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {number} arg0
     */
    set id(arg0) {
        wasm.__wbg_set_token_id(this.__wbg_ptr, arg0);
    }
    /**
     * @param {boolean} arg0
     */
    set is_special(arg0) {
        wasm.__wbg_set_token_is_special(this.__wbg_ptr, arg0);
    }
    /**
     * @param {string} arg0
     */
    set value(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_token_value(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Token.prototype[Symbol.dispose] = Token.prototype.free;

export class ToolCall {
    static __wrap(ptr) {
        const obj = Object.create(ToolCall.prototype);
        obj.__wbg_ptr = ptr;
        ToolCallFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolCallFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_toolcall_free(ptr, 0);
    }
    /**
     * @returns {Value}
     */
    get arguments() {
        const ret = wasm.__wbg_get_toolcall_arguments(this.__wbg_ptr);
        return Value.__wrap(ret);
    }
    /**
     * @returns {string | undefined}
     */
    get identifier() {
        const ret = wasm.__wbg_get_toolcall_identifier(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_toolcall_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {Value} arg0
     */
    set arguments(arg0) {
        _assertClass(arg0, Value);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_toolcall_arguments(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {string | null} [arg0]
     */
    set identifier(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolcall_identifier(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolcall_name(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ToolCall.prototype[Symbol.dispose] = ToolCall.prototype.free;

export class ToolCallArgumentDelta {
    static __wrap(ptr) {
        const obj = Object.create(ToolCallArgumentDelta.prototype);
        obj.__wbg_ptr = ptr;
        ToolCallArgumentDeltaFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolCallArgumentDeltaFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_toolcallargumentdelta_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get arguments() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_toolcallargumentdelta_arguments(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_toolcallargumentdelta_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set arguments(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolcallargumentdelta_arguments(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolcallargumentdelta_name(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ToolCallArgumentDelta.prototype[Symbol.dispose] = ToolCallArgumentDelta.prototype.free;

/**
 * Converts model tool-call candidates into append-only JSON argument fragments.
 */
export class ToolCallArgumentStream {
    static __wrap(ptr) {
        const obj = Object.create(ToolCallArgumentStream.prototype);
        obj.__wbg_ptr = ptr;
        ToolCallArgumentStreamFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolCallArgumentStreamFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_toolcallargumentstream_free(ptr, 0);
    }
    /**
     * @param {ToolNamespace[]} tools
     * @returns {ToolCallArgumentStream}
     */
    static create(tools) {
        const ptr0 = passArrayJsValueToWasm0(tools, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.toolcallargumentstream_create(ptr0, len0);
        return ToolCallArgumentStream.__wrap(ret);
    }
    /**
     * @param {ToolCall} call
     * @returns {ToolCallArgumentDelta}
     */
    finish(call) {
        _assertClass(call, ToolCall);
        var ptr0 = call.__destroy_into_raw();
        const ret = wasm.toolcallargumentstream_finish(this.__wbg_ptr, ptr0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ToolCallArgumentDelta.__wrap(ret[0]);
    }
    /**
     * @param {string} raw
     * @returns {ToolCallArgumentDelta}
     */
    update(raw) {
        const ptr0 = passStringToWasm0(raw, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.toolcallargumentstream_update(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ToolCallArgumentDelta.__wrap(ret[0]);
    }
}
if (Symbol.dispose) ToolCallArgumentStream.prototype[Symbol.dispose] = ToolCallArgumentStream.prototype.free;

export class ToolFunction {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolFunctionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_toolfunction_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get description() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_toolfunction_description(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_toolfunction_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {Value | undefined}
     */
    get parameters() {
        const ret = wasm.__wbg_get_toolfunction_parameters(this.__wbg_ptr);
        return ret === 0 ? undefined : Value.__wrap(ret);
    }
    /**
     * @returns {Value | undefined}
     */
    get return_definition() {
        const ret = wasm.__wbg_get_toolfunction_return_definition(this.__wbg_ptr);
        return ret === 0 ? undefined : Value.__wrap(ret);
    }
    /**
     * @param {string} arg0
     */
    set description(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolfunction_description(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolfunction_name(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Value | null} [arg0]
     */
    set parameters(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, Value);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_toolfunction_parameters(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {Value | null} [arg0]
     */
    set return_definition(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, Value);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_toolfunction_return_definition(this.__wbg_ptr, ptr0);
    }
}
if (Symbol.dispose) ToolFunction.prototype[Symbol.dispose] = ToolFunction.prototype.free;

export class ToolNamespace {
    static __wrap(ptr) {
        const obj = Object.create(ToolNamespace.prototype);
        obj.__wbg_ptr = ptr;
        ToolNamespaceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    static __unwrap(jsValue) {
        if (!(jsValue instanceof ToolNamespace)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolNamespaceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_toolnamespace_free(ptr, 0);
    }
    /**
     * @returns {string | undefined}
     */
    get description() {
        const ret = wasm.__wbg_get_toolnamespace_description(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]);
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_toolnamespace_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {ToolDescription[]}
     */
    get tools() {
        const ret = wasm.__wbg_get_toolnamespace_tools(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]);
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {string | null} [arg0]
     */
    set description(arg0) {
        var ptr0 = isLikeNone(arg0) ? 0 : passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolnamespace_description(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} arg0
     */
    set name(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolnamespace_name(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {ToolDescription[]} arg0
     */
    set tools(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_toolnamespace_tools(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) ToolNamespace.prototype[Symbol.dispose] = ToolNamespace.prototype.free;

export class Value {
    static __wrap(ptr) {
        const obj = Object.create(Value.prototype);
        obj.__wbg_ptr = ptr;
        ValueFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ValueFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_value_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    get json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.__wbg_get_value_json(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} arg0
     */
    set json(arg0) {
        const ptr0 = passStringToWasm0(arg0, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_value_json(this.__wbg_ptr, ptr0, len0);
    }
}
if (Symbol.dispose) Value.prototype[Symbol.dispose] = Value.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_30c8987f7c2ed4e2: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_Number_14af1003b8dd5ead: function(arg0) {
            const ret = Number(arg0);
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_Window_a2a6c4d665047b14: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_2664448a7c667d67: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_bigint_get_as_i64_a2383202b9353e4c: function(arg0, arg1) {
            const v = arg1;
            const ret = typeof(v) === 'bigint' ? v : undefined;
            getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_boolean_get_5b446f51afd21013: function(arg0) {
            const v = arg0;
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_debug_string_4687d8d8c2017d52: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_in_92f62ee1427d9e49: function(arg0, arg1) {
            const ret = arg0 in arg1;
            return ret;
        },
        __wbg___wbindgen_is_bigint_b123553bed3bb382: function(arg0) {
            const ret = typeof(arg0) === 'bigint';
            return ret;
        },
        __wbg___wbindgen_is_function_1f9d30630b8b1d3d: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_e343b7d08827ba72: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_object_3c45d4f2dde4e749: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_90b56bc79aad6f6c: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_8865fb403f8fe9d8: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_jsval_eq_02babf21faa37971: function(arg0, arg1) {
            const ret = arg0 === arg1;
            return ret;
        },
        __wbg___wbindgen_jsval_loose_eq_677f21e468d6b461: function(arg0, arg1) {
            const ret = arg0 == arg1;
            return ret;
        },
        __wbg___wbindgen_number_get_2e0e7dee9f701a71: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_0380ccaa2f57f0d9: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_dcc1a90847f04c41: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_abort_334e7f918b7c5047: function(arg0, arg1) {
            arg0.abort(arg1);
        },
        __wbg_abort_b46650b76e524767: function(arg0) {
            const ret = arg0.abort();
            return ret;
        },
        __wbg_abort_f78b99d80248bacb: function(arg0) {
            arg0.abort();
        },
        __wbg_append_1938ed58fb5613b8: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.append(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
        }, arguments); },
        __wbg_apply_a910804df6e1e433: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.apply(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_arrayBuffer_0fe6e1300abcf908: function(arg0) {
            const ret = arg0.arrayBuffer();
            return ret;
        },
        __wbg_arrayBuffer_f07be74819944ea2: function() { return handleError(function (arg0) {
            const ret = arg0.arrayBuffer();
            return ret;
        }, arguments); },
        __wbg_beginComputePass_b9a325184985e6ff: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_body_4e088733babc630d: function(arg0) {
            const ret = arg0.body;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_buffer_56ec2905a66f58b9: function(arg0) {
            const ret = arg0.buffer;
            return ret;
        },
        __wbg_byobRequest_54f7ce41d584549a: function(arg0) {
            const ret = arg0.byobRequest;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_byteLength_9f985b8373f1f3fe: function(arg0) {
            const ret = arg0.byteLength;
            return ret;
        },
        __wbg_byteOffset_0bd5a6a03553639c: function(arg0) {
            const ret = arg0.byteOffset;
            return ret;
        },
        __wbg_call_1875a20c43a36133: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.call(arg1, arg2, arg3);
            return ret;
        }, arguments); },
        __wbg_call_187d372bd5fdd4aa: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_call_6137034ef55c9d0f: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.call(arg1);
            return ret;
        }, arguments); },
        __wbg_call_939a2607c4484b0b: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            const ret = arg0.call(arg1, arg2, arg3, arg4);
            return ret;
        }, arguments); },
        __wbg_cancel_7c2a08c8db4c73f3: function(arg0) {
            const ret = arg0.cancel();
            return ret;
        },
        __wbg_catch_92594c2d17f55f12: function(arg0, arg1) {
            const ret = arg0.catch(arg1);
            return ret;
        },
        __wbg_chatinstance_new: function(arg0) {
            const ret = ChatInstance.__wrap(arg0);
            return ret;
        },
        __wbg_chatmessage_new: function(arg0) {
            const ret = ChatMessage.__wrap(arg0);
            return ret;
        },
        __wbg_chatmessage_unwrap: function(arg0) {
            const ret = ChatMessage.__unwrap(arg0);
            return ret;
        },
        __wbg_chatmodelinfo_new: function(arg0) {
            const ret = ChatModelInfo.__wrap(arg0);
            return ret;
        },
        __wbg_chatreply_new: function(arg0) {
            const ret = ChatReply.__wrap(arg0);
            return ret;
        },
        __wbg_chatsession_new: function(arg0) {
            const ret = ChatSession.__wrap(arg0);
            return ret;
        },
        __wbg_chatsessionstream_new: function(arg0) {
            const ret = ChatSessionStream.__wrap(arg0);
            return ret;
        },
        __wbg_clearBuffer_3d5a3057c3c8a9bf: function(arg0, arg1, arg2, arg3) {
            arg0.clearBuffer(arg1, arg2, arg3);
        },
        __wbg_clearBuffer_82ba999c294f42fb: function(arg0, arg1, arg2) {
            arg0.clearBuffer(arg1, arg2);
        },
        __wbg_clearTimeout_1daad5c6ee4fcd3b: function(arg0) {
            const ret = clearTimeout(arg0);
            return ret;
        },
        __wbg_clearTimeout_c7bc4c3c7a7774af: function(arg0, arg1) {
            arg0.clearTimeout(arg1);
        },
        __wbg_close_185286988f30f252: function() { return handleError(function (arg0) {
            arg0.close();
        }, arguments); },
        __wbg_close_4d4e83804e073061: function(arg0) {
            const ret = arg0.close();
            return ret;
        },
        __wbg_close_748f26698dabe269: function() { return handleError(function (arg0) {
            arg0.close();
        }, arguments); },
        __wbg_configure_1e2c1c9edad07d26: function() { return handleError(function (arg0, arg1) {
            arg0.configure(arg1);
        }, arguments); },
        __wbg_copyBufferToBuffer_01766818654a9868: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4);
        }, arguments); },
        __wbg_copyBufferToBuffer_9c174b96fb08d551: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        }, arguments); },
        __wbg_createBindGroupLayout_b1bd63b4e88459d8: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_f539b26ca341308f: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_d800e9b1d41b2ee5: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createComputePipeline_224fa2618d9948a0: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_6eab52c327118937: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createShaderModule_cefa51336cb288ae: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createWritable_7ed35ba182338bbb: function(arg0, arg1) {
            const ret = arg0.createWritable(arg1);
            return ret;
        },
        __wbg_decisionoutput_new: function(arg0) {
            const ret = DecisionOutput.__wrap(arg0);
            return ret;
        },
        __wbg_decisionresult_new: function(arg0) {
            const ret = DecisionResult.__wrap(arg0);
            return ret;
        },
        __wbg_decisionresult_unwrap: function(arg0) {
            const ret = DecisionResult.__unwrap(arg0);
            return ret;
        },
        __wbg_decisionsession_new: function(arg0) {
            const ret = DecisionSession.__wrap(arg0);
            return ret;
        },
        __wbg_description_83b8a393160021b9: function(arg0, arg1) {
            const ret = arg1.description;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_dispatchWorkgroups_56b943172790add0: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_done_b41a1d26cdb37fb6: function(arg0) {
            const ret = arg0.done;
            return ret;
        },
        __wbg_downloaderstream_new: function(arg0) {
            const ret = DownloaderStream.__wrap(arg0);
            return ret;
        },
        __wbg_downloadstate_new: function(arg0) {
            const ret = DownloadState.__wrap(arg0);
            return ret;
        },
        __wbg_end_82fd0c12ba185009: function(arg0) {
            arg0.end();
        },
        __wbg_engine_new: function(arg0) {
            const ret = Engine.__wrap(arg0);
            return ret;
        },
        __wbg_enqueue_4a01c128f9a0de16: function() { return handleError(function (arg0, arg1) {
            arg0.enqueue(arg1);
        }, arguments); },
        __wbg_entries_b55ec7e0dc443355: function(arg0) {
            const ret = arg0.entries();
            return ret;
        },
        __wbg_entries_d6be391337d66561: function(arg0) {
            const ret = arg0.entries();
            return ret;
        },
        __wbg_entries_fb6397112b1de25f: function(arg0) {
            const ret = Object.entries(arg0);
            return ret;
        },
        __wbg_fetch_7a37768ab67c5bd8: function(arg0) {
            const ret = fetch(arg0);
            return ret;
        },
        __wbg_fetch_efc64baf325b084d: function(arg0, arg1) {
            const ret = arg0.fetch(arg1);
            return ret;
        },
        __wbg_from_296ca31f8d0f1c52: function(arg0) {
            const ret = Array.from(arg0);
            return ret;
        },
        __wbg_getContext_e567868594d2e0c6: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getDirectoryHandle_32ff6719da8ccf70: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getDirectoryHandle(getStringFromWasm0(arg1, arg2), arg3);
            return ret;
        },
        __wbg_getDirectoryHandle_3df089cef76afec9: function(arg0, arg1, arg2) {
            const ret = arg0.getDirectoryHandle(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_getDirectory_8257f053b0d3e746: function(arg0) {
            const ret = arg0.getDirectory();
            return ret;
        },
        __wbg_getFileHandle_0357a6e30a3f80ba: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getFileHandle(getStringFromWasm0(arg1, arg2), arg3);
            return ret;
        },
        __wbg_getFileHandle_41a1c1b19fcfb783: function(arg0, arg1, arg2) {
            const ret = arg0.getFileHandle(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_getFile_75187c3eb15ebeee: function(arg0) {
            const ret = arg0.getFile();
            return ret;
        },
        __wbg_getMappedRange_fb54c6327b2d8d20: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_getRandomValues_973dabfb0ad1769f: function() { return handleError(function (arg0, arg1) {
            globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
        }, arguments); },
        __wbg_getRandomValues_a608c4436c19407a: function() { return handleError(function (arg0, arg1) {
            globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
        }, arguments); },
        __wbg_getReader_9facd4f899beac89: function() { return handleError(function (arg0) {
            const ret = arg0.getReader();
            return ret;
        }, arguments); },
        __wbg_getTime_f5a55efff2585d5d: function(arg0) {
            const ret = arg0.getTime();
            return ret;
        },
        __wbg_getTimezoneOffset_2e851fdaecc2dc92: function(arg0) {
            const ret = arg0.getTimezoneOffset();
            return ret;
        },
        __wbg_get_31af05bd4842a84f: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_get_658f6698067d9515: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_get_6c896e0571ddae51: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_done_7020b91c987365c7: function(arg0) {
            const ret = arg0.done;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg_get_unchecked_288889d017702237: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_value_87fbd6ad1283f2e7: function(arg0) {
            const ret = arg0.value;
            return ret;
        },
        __wbg_get_with_ref_key_6412cf3094599694: function(arg0, arg1) {
            const ret = arg0[arg1];
            return ret;
        },
        __wbg_gpu_afdd4387c7afe5f9: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_has_5d6706e5209576c1: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.has(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_hash_new: function(arg0) {
            const ret = Hash.__wrap(arg0);
            return ret;
        },
        __wbg_hash_unwrap: function(arg0) {
            const ret = Hash.__unwrap(arg0);
            return ret;
        },
        __wbg_headers_eba93595f8944c2f: function(arg0) {
            const ret = arg0.headers;
            return ret;
        },
        __wbg_image_new: function(arg0) {
            const ret = Image.__wrap(arg0);
            return ret;
        },
        __wbg_image_unwrap: function(arg0) {
            const ret = Image.__unwrap(arg0);
            return ret;
        },
        __wbg_info_971d8b9db3dae69f: function(arg0) {
            const ret = arg0.info;
            return ret;
        },
        __wbg_instanceof_ArrayBuffer_a99f175873e5d9b8: function(arg0) {
            let result;
            try {
                result = arg0 instanceof ArrayBuffer;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_DomException_550db3f9465a933f: function(arg0) {
            let result;
            try {
                result = arg0 instanceof DOMException;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Error_80a725f81f2e102d: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Error;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_FileSystemDirectoryHandle_f9d9528a8c896a2b: function(arg0) {
            let result;
            try {
                result = arg0 instanceof FileSystemDirectoryHandle;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_FileSystemFileHandle_f5198fa86c4096bc: function(arg0) {
            let result;
            try {
                result = arg0 instanceof FileSystemFileHandle;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Map_b2611749102d7ba3: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Map;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Response_b8758567269c30b2: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Response;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Uint8Array_828cef2aaacafc31: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Uint8Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_82d71df4eddf88bc: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_isArray_e15a2ff68ffdbef2: function(arg0) {
            const ret = Array.isArray(arg0);
            return ret;
        },
        __wbg_isFallbackAdapter_4c8cc3b18677460a: function(arg0) {
            const ret = arg0.isFallbackAdapter;
            return ret;
        },
        __wbg_isSafeInteger_717808ad6a54bd9e: function(arg0) {
            const ret = Number.isSafeInteger(arg0);
            return ret;
        },
        __wbg_iterator_e3c31c892080e444: function() {
            const ret = Symbol.iterator;
            return ret;
        },
        __wbg_length_7f3c00c40364105e: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_d4bdea10311bd9cf: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_06bcb36c8409843b: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_limits_601ad2e086ef8141: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_maxBindGroupsPlusVertexBuffers_52369f089736ef9d: function(arg0) {
            const ret = arg0.maxBindGroupsPlusVertexBuffers;
            return ret;
        },
        __wbg_maxBindGroups_4e424afe6ce86ca2: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_7d035da36821c44f: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_423f4a084e32a195: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_c4cd9126f6d287c6: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_d924670762b9e250: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_707a3868f7cebb59: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_0a4d99463cbd6e5e: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_85123ea0587f7558: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_a3186b4c5267d44f: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_57b297355cfb6204: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_4158f95e673d54c4: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_226b0b70910aa16c: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_0e835fda711fc7e6: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderVariables_8c4a1d727e2aa35a: function(arg0) {
            const ret = arg0.maxInterStageShaderVariables;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_6675f5e91d9a728a: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_1910fa38a6ed1e1f: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_2e244bded070b18d: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_a285f3ebca51ca0d: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_7aa946f0fc322a2b: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_0e699147ad00502d: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_aabf6add54decfe2: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_dd598b27e9c0c1c4: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_f944266c65dfd1a9: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_59fa6be7cfbeeb53: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_bee5f00a4d706c7f: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_5cf6392c4e9033fe: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_548baa887375d865: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_75d881156591f5da: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_message_5f8387f0c32b90a7: function(arg0) {
            const ret = arg0.message;
            return ret;
        },
        __wbg_message_d988ea596c5f5c9a: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_minStorageBufferOffsetAlignment_5ba9b77792bdadb3: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_ab7d52a5293b22bd: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_model_new: function(arg0) {
            const ret = Model.__wrap(arg0);
            return ret;
        },
        __wbg_modelbackend_new: function(arg0) {
            const ret = ModelBackend.__wrap(arg0);
            return ret;
        },
        __wbg_modelbackend_unwrap: function(arg0) {
            const ret = ModelBackend.__unwrap(arg0);
            return ret;
        },
        __wbg_modelfamily_new: function(arg0) {
            const ret = ModelFamily.__wrap(arg0);
            return ret;
        },
        __wbg_modelregistry_new: function(arg0) {
            const ret = ModelRegistry.__wrap(arg0);
            return ret;
        },
        __wbg_modelvendor_new: function(arg0) {
            const ret = ModelVendor.__wrap(arg0);
            return ret;
        },
        __wbg_name_30c2cf5d3e6226e8: function(arg0, arg1) {
            const ret = arg1.name;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_nativetool_unwrap: function(arg0) {
            const ret = NativeTool.__unwrap(arg0);
            return ret;
        },
        __wbg_navigator_2156486643462a87: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_c9bce48d4b578c9c: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_0_72d020f0c63443d4: function() {
            const ret = new Date();
            return ret;
        },
        __wbg_new_1dbf7428bba60a42: function(arg0) {
            const ret = new Uint8Array(arg0);
            return ret;
        },
        __wbg_new_28744009d011f847: function() {
            const ret = new Map();
            return ret;
        },
        __wbg_new_2e41357094e99294: function() { return handleError(function () {
            const ret = new AbortController();
            return ret;
        }, arguments); },
        __wbg_new_343a093a3c2ffb4e: function(arg0, arg1) {
            const ret = new Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_34e56228080aa9f8: function(arg0) {
            const ret = new Date(arg0);
            return ret;
        },
        __wbg_new_5502aad30c185fc8: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_617a8cdb8bb1130e: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_83ca405eae412a62: function() { return handleError(function () {
            const ret = new Headers();
            return ret;
        }, arguments); },
        __wbg_new_a217df4351db4b9b: function() { return handleError(function (arg0, arg1) {
            const ret = new OffscreenCanvas(arg0 >>> 0, arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_new_ee2291f50781bf1d: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_9a868026ffa4208a: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_689a3a281a8d4da6: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_b01cb72a8af741a3: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_2f5d7fc2a828b74d: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_new_with_str_and_init_4618dee4e950224f: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = new Request(getStringFromWasm0(arg0, arg1), arg2);
            return ret;
        }, arguments); },
        __wbg_new_with_u8_array_sequence_f380e3e783ae5617: function() { return handleError(function (arg0) {
            const ret = new Blob(arg0);
            return ret;
        }, arguments); },
        __wbg_next_33784799010f1bbe: function(arg0) {
            const ret = arg0.next;
            return ret;
        },
        __wbg_next_762aca09b0915bfc: function() { return handleError(function (arg0) {
            const ret = arg0.next();
            return ret;
        }, arguments); },
        __wbg_next_f4aac29c42af995c: function() { return handleError(function (arg0) {
            const ret = arg0.next();
            return ret;
        }, arguments); },
        __wbg_now_e7c6795a7f81e10f: function(arg0) {
            const ret = arg0.now();
            return ret;
        },
        __wbg_performance_3fcf6e32a7e1ed0a: function(arg0) {
            const ret = arg0.performance;
            return ret;
        },
        __wbg_prototypesetcall_bc27214492979395: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_2baf45db356cf468: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queueMicrotask_9833f9a49df95a49: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_a72f977e97f23c5f: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_7b62c28143d44293: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_read_2a943774344c9728: function(arg0) {
            const ret = arg0.read();
            return ret;
        },
        __wbg_releaseLock_eb64cd83aa9fcfe4: function(arg0) {
            arg0.releaseLock();
        },
        __wbg_removeEntry_4d962be4429d4c17: function(arg0, arg1, arg2) {
            const ret = arg0.removeEntry(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_removeEntry_ce5069307f56e429: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.removeEntry(getStringFromWasm0(arg1, arg2), arg3);
            return ret;
        },
        __wbg_requestAdapter_a539af006419f2e9: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_5cb8a582e55d08cb: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_0076e10020304ede: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_respond_c102fabcc79e5ef0: function() { return handleError(function (arg0, arg1) {
            arg0.respond(arg1 >>> 0);
        }, arguments); },
        __wbg_samplingparameters_new: function(arg0) {
            const ret = SamplingParameters.__wrap(arg0);
            return ret;
        },
        __wbg_setBindGroup_0b7c2a055ef1f314: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        }, arguments); },
        __wbg_setBindGroup_c83391351ce68826: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_5146f7b8d2b4af5c: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setTimeout_b4a9096784635514: function(arg0, arg1) {
            const ret = setTimeout(arg0, arg1);
            return ret;
        },
        __wbg_setTimeout_db7bbc18a17e152a: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.setTimeout(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_145a351398b48c65: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_34d08fd992d43d61: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_575d3ddb70fe831d: function(arg0, arg1, arg2) {
            arg0.set(getArrayU8FromWasm0(arg1, arg2));
        },
        __wbg_set_6ae97e73113c4f0b: function(arg0, arg1, arg2) {
            const ret = arg0.set(arg1, arg2);
            return ret;
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_access_a099cfbbeec9b96f: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_bea140a88be9b277: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_beginning_of_pass_write_index_fa9ae10d5d2804ce: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_458c44ba55100b82: function(arg0, arg1, arg2) {
            arg0.bindGroupLayouts = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_binding_81b3fac7f7acaf8d: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_b6cee57f35ac5190: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_body_1fb0f1008bfc7df6: function(arg0, arg1) {
            arg0.body = arg1;
        },
        __wbg_set_buffer_1548ae88a9188037: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_8d0ac64ad20dfc84: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_cache_7507c4cc5938ccc7: function(arg0, arg1) {
            arg0.cache = __wbindgen_enum_RequestCache[arg1];
        },
        __wbg_set_code_5d5b0b9e2fd0dca7: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compute_bc754d9e37f6e4c9: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_create_4f5fb19abc6eb52a: function(arg0, arg1) {
            arg0.create = arg1 !== 0;
        },
        __wbg_set_create_55977c3183b4be6e: function(arg0, arg1) {
            arg0.create = arg1 !== 0;
        },
        __wbg_set_credentials_e0dcce31ca532b7d: function(arg0, arg1) {
            arg0.credentials = __wbindgen_enum_RequestCredentials[arg1];
        },
        __wbg_set_data_a6e012931e09470c: function(arg0, arg1) {
            arg0.data = arg1;
        },
        __wbg_set_device_210484a77b675c9c: function(arg0, arg1) {
            arg0.device = arg1;
        },
        __wbg_set_end_of_pass_write_index_b54a8e3802b9ae0d: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_6f866302103b81e9: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entries_f26b77ab9548e906: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entry_point_ac8db535971761fe: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_7f966c604c4f8098: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_format_92732ea75d3b79f5: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_f009e603f7d4c28e: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_has_dynamic_offset_0c72ffa900c5a269: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_headers_40ad33dcf016613f: function(arg0, arg1) {
            arg0.headers = arg1;
        },
        __wbg_set_keep_existing_data_354b50731143b818: function(arg0, arg1) {
            arg0.keepExistingData = arg1 !== 0;
        },
        __wbg_set_label_09e2da5f8522dbe8: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_17202740051e9722: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_17b4858e1eef23dd: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_2fefb39c0e0dbbe8: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_612add98a4398f92: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_92cd3811e96b487c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_c7987704d29f284b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_cfe64bca8945ee30: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_454e3a091b390cd4: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_4b5d5b12fbb72a76: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_gpu_auto_layout_mode_ee432e515c357fa8: function(arg0, arg1) {
            arg0.layout = __wbindgen_enum_GpuAutoLayoutMode[arg1];
        },
        __wbg_set_mapped_at_creation_3f320fef6761b02c: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_method_dcb32343247ec427: function(arg0, arg1, arg2) {
            arg0.method = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_min_binding_size_f64_897e3cd4496ddec9: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_mode_74e772635acb96ec: function(arg0, arg1) {
            arg0.mode = __wbindgen_enum_RequestMode[arg1];
        },
        __wbg_set_module_c9946af218d23b53: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisampled_039f032dc4b67367: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_f64_127e8a0aa5c5485a: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_position_3cdfecffdd426827: function(arg0, arg1, arg2) {
            arg0.position = arg1 === 0 ? undefined : arg2;
        },
        __wbg_set_power_preference_b42d00a8facfbade: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_query_set_f9a94586851fdba2: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_recursive_d2035f94f392c618: function(arg0, arg1) {
            arg0.recursive = arg1 !== 0;
        },
        __wbg_set_referrer_efd0fb956d6b70b2: function(arg0, arg1, arg2) {
            arg0.referrer = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_referrer_policy_465382f52c1a03d3: function(arg0, arg1) {
            arg0.referrerPolicy = __wbindgen_enum_ReferrerPolicy[arg1];
        },
        __wbg_set_required_features_bbab71414c45e621: function(arg0, arg1, arg2) {
            arg0.requiredFeatures = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_required_limits_837f62d865e7cfac: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_8fd8658b30d86ecf: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_buffer_binding_33099b25da65b610: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_texture_view_4cffe7bc7c8e5cbe: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_sample_type_ebc5fcd029513bda: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_89cb4a7efcfc6005: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_signal_2dbd377604627f26: function(arg0, arg1) {
            arg0.signal = arg1;
        },
        __wbg_set_size_6bd6a74dcc5a9704: function(arg0, arg1, arg2) {
            arg0.size = arg1 === 0 ? undefined : arg2;
        },
        __wbg_set_size_f64_2f591b0654540477: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_f64_e844c985b8f95261: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_storage_texture_786aea7c5773b6c1: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_texture_a33be3fe02ac6264: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_5f240bbaad97fcaf: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_43e0092f16775979: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_type_79cec55caf4cdb6d: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_type_fcd7578ab6849a04: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_WriteCommandType[arg1];
        },
        __wbg_set_usage_f3e34822998d2147: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_dimension_893e2d16561e56e8: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_f2c5fe4bf927c3fe: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_visibility_d8a6821789538c25: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_signal_e5a49a2d9c65b7f6: function(arg0) {
            const ret = arg0.signal;
            return ret;
        },
        __wbg_size_8f1c0b1a1fbb3810: function(arg0) {
            const ret = arg0.size;
            return ret;
        },
        __wbg_slice_afd8274aed4149c0: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.slice(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_static_accessor_GLOBAL_266715b9d96ba635: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_10fb7dc1ae063179: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_0b583911f537483a: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_d7f903d1508cbdc4: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_status_ce0a98d3c57125f3: function(arg0) {
            const ret = arg0.status;
            return ret;
        },
        __wbg_storage_f32f42a5cc35294c: function(arg0) {
            const ret = arg0.storage;
            return ret;
        },
        __wbg_subgroupMaxSize_b43be0aa16182403: function(arg0) {
            const ret = arg0.subgroupMaxSize;
            return ret;
        },
        __wbg_subgroupMinSize_03feb6ee0cda6775: function(arg0) {
            const ret = arg0.subgroupMinSize;
            return ret;
        },
        __wbg_text_ff3f476b3d6b1246: function() { return handleError(function (arg0) {
            const ret = arg0.text();
            return ret;
        }, arguments); },
        __wbg_then_c949d5a25a4e78f8: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_e71170d78fcf8954: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_toolcall_new: function(arg0) {
            const ret = ToolCall.__wrap(arg0);
            return ret;
        },
        __wbg_toolnamespace_new: function(arg0) {
            const ret = ToolNamespace.__wrap(arg0);
            return ret;
        },
        __wbg_toolnamespace_unwrap: function(arg0) {
            const ret = ToolNamespace.__unwrap(arg0);
            return ret;
        },
        __wbg_unconfigure_835307f58dc68d80: function(arg0) {
            arg0.unconfigure();
        },
        __wbg_unmap_6a96b14c9ef5f7f5: function(arg0) {
            arg0.unmap();
        },
        __wbg_url_8e28edeab4cd6bed: function(arg0, arg1) {
            const ret = arg1.url;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_value_f3c585ee8f5ba40c: function(arg0) {
            const ret = arg0.value;
            return ret;
        },
        __wbg_view_9c570f33e8d6ab96: function(arg0) {
            const ret = arg0.view;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_write_2dad0c360cf0dc17: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.write(arg1);
            return ret;
        }, arguments); },
        __wbg_write_edd7efd3f87bfd2f: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.write(arg1);
            return ret;
        }, arguments); },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 1252, ret: Externref, inner_ret: Some(Externref) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__wasm_bindgen_6c51f868a643a6ac___JsValue__true_);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 1593, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue______true_);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 3484, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUDevice")], shim_idx: 3302, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("any")], shim_idx: 3302, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__483);
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [], shim_idx: 1566, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true_);
            return ret;
        },
        __wbindgen_generic_0000000000000007: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [], shim_idx: 1700, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true__1_);
            return ret;
        },
        __wbindgen_generic_0000000000000008: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000009: function(arg0) {
            // Cast intrinsic for `I64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_000000000000000a: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_000000000000000b: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_000000000000000c: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return ret;
        },
        __wbindgen_generic_000000000000000d: function(arg0, arg1) {
            var v0 = getArrayJsValueFromWasm0(arg0, arg1);
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(RustStruct { name: "ChatMessage", unique_crate_identifier: "shoji-33ae217e8f89287f" }) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_generic_000000000000000e: function(arg0, arg1) {
            var v0 = getArrayJsValueFromWasm0(arg0, arg1);
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(RustStruct { name: "ChatReply", unique_crate_identifier: "shoji-33ae217e8f89287f" }) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_generic_000000000000000f: function(arg0, arg1) {
            var v0 = getArrayJsValueFromWasm0(arg0, arg1);
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(RustStruct { name: "Model", unique_crate_identifier: "shoji-33ae217e8f89287f" }) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_generic_0000000000000010: function(arg0, arg1) {
            var v0 = getArrayJsValueFromWasm0(arg0, arg1);
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(RustStruct { name: "ModelFamily", unique_crate_identifier: "shoji-33ae217e8f89287f" }) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_generic_0000000000000011: function(arg0, arg1) {
            var v0 = getArrayJsValueFromWasm0(arg0, arg1);
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(RustStruct { name: "ModelRegistry", unique_crate_identifier: "shoji-33ae217e8f89287f" }) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_generic_0000000000000012: function(arg0, arg1) {
            var v0 = getArrayJsValueFromWasm0(arg0, arg1);
            wasm.__wbindgen_free(arg0, arg1 * 4, 4);
            // Cast intrinsic for `Vector(RustStruct { name: "ModelVendor", unique_crate_identifier: "shoji-33ae217e8f89287f" }) -> Externref`.
            const ret = v0;
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./uzu_bg.js": import0,
    };
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true_(arg0, arg1) {
    wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true_(arg0, arg1);
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true__1_(arg0, arg1) {
    wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true__1_(arg0, arg1);
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue______true_(arg0, arg1, arg2);
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__wasm_bindgen_6c51f868a643a6ac___JsValue__true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__wasm_bindgen_6c51f868a643a6ac___JsValue__true_(arg0, arg1, arg2);
    return ret;
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__483(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_5471360f92b7a73a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__483(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined___js_sys_50975383be31856c___Function_fn_wasm_bindgen_6c51f868a643a6ac___JsValue_____wasm_bindgen_6c51f868a643a6ac___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuAutoLayoutMode = ["auto"];


const __wbindgen_enum_GpuBufferBindingType = ["uniform", "storage", "read-only-storage"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuSamplerBindingType = ["filtering", "non-filtering", "comparison"];


const __wbindgen_enum_GpuStorageTextureAccess = ["write-only", "read-only", "read-write"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureSampleType = ["float", "unfilterable-float", "depth", "sint", "uint"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];


const __wbindgen_enum_ReadableStreamType = ["bytes"];


const __wbindgen_enum_ReferrerPolicy = ["", "no-referrer", "no-referrer-when-downgrade", "origin", "origin-when-cross-origin", "unsafe-url", "same-origin", "strict-origin", "strict-origin-when-cross-origin"];


const __wbindgen_enum_RequestCache = ["default", "no-store", "reload", "no-cache", "force-cache", "only-if-cached"];


const __wbindgen_enum_RequestCredentials = ["omit", "same-origin", "include"];


const __wbindgen_enum_RequestMode = ["same-origin", "no-cors", "cors", "navigate"];


const __wbindgen_enum_WriteCommandType = ["write", "seek", "truncate"];
const CancelTokenFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_canceltoken_free(ptr, 1));
const ChatConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatconfig_free(ptr, 1));
const ChatInstanceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatinstance_free(ptr, 1));
const ChatMessageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatmessage_free(ptr, 1));
const ChatModelCapabilitiesFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatmodelcapabilities_free(ptr, 1));
const ChatModelInfoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatmodelinfo_free(ptr, 1));
const ChatReplyFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreply_free(ptr, 1));
const ChatReplyConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplyconfig_free(ptr, 1));
const ChatReplySpeculatorStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplyspeculatorstats_free(ptr, 1));
const ChatReplyStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplystats_free(ptr, 1));
const ChatSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsession_free(ptr, 1));
const ChatSessionStreamFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsessionstream_free(ptr, 1));
const ClassificationLabelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationlabel_free(ptr, 1));
const ClassificationMessageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationmessage_free(ptr, 1));
const ClassificationOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationoutput_free(ptr, 1));
const ClassificationStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationstats_free(ptr, 1));
const DecisionInputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisioninput_free(ptr, 1));
const DecisionOptionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionoption_free(ptr, 1));
const DecisionOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionoutput_free(ptr, 1));
const DecisionResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionresult_free(ptr, 1));
const DecisionSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionsession_free(ptr, 1));
const DecisionStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionstats_free(ptr, 1));
const DeviceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_device_free(ptr, 1));
const DownloadStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadstate_free(ptr, 1));
const DownloaderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloader_free(ptr, 1));
const DownloaderStreamFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloaderstream_free(ptr, 1));
const EngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engine_free(ptr, 1));
const EngineCallbackFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_enginecallback_free(ptr, 1));
const EngineConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engineconfig_free(ptr, 1));
const FeatureFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_feature_free(ptr, 1));
const FileFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_file_free(ptr, 1));
const HashFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_hash_free(ptr, 1));
const ImageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_image_free(ptr, 1));
const IntoUnderlyingByteSourceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingbytesource_free(ptr, 1));
const IntoUnderlyingSinkFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingsink_free(ptr, 1));
const IntoUnderlyingSourceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_intounderlyingsource_free(ptr, 1));
const MetadataFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_metadata_free(ptr, 1));
const ModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_model_free(ptr, 1));
const ModelBackendFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelbackend_free(ptr, 1));
const ModelFamilyFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelfamily_free(ptr, 1));
const ModelPropertiesFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelproperties_free(ptr, 1));
const ModelQuantizationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelquantization_free(ptr, 1));
const ModelRegistryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelregistry_free(ptr, 1));
const ModelVendorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelvendor_free(ptr, 1));
const NativeToolFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_nativetool_free(ptr, 1));
const PcmBatchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pcmbatch_free(ptr, 1));
const RepositoryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_repository_free(ptr, 1));
const SamplingParametersFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingparameters_free(ptr, 1));
const TextToSpeechOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_texttospeechoutput_free(ptr, 1));
const TextToSpeechStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_texttospeechstats_free(ptr, 1));
const TokenFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_token_free(ptr, 1));
const ToolCallFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolcall_free(ptr, 1));
const ToolCallArgumentDeltaFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolcallargumentdelta_free(ptr, 1));
const ToolCallArgumentStreamFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolcallargumentstream_free(ptr, 1));
const ToolFunctionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolfunction_free(ptr, 1));
const ToolNamespaceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolnamespace_free(ptr, 1));
const ValueFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_value_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

function getArrayJsValueViewFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('uzu_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
