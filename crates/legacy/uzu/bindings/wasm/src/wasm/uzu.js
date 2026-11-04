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
    /**
     * @returns {boolean}
     */
    get isCancelled() {
        const ret = wasm.canceltoken___wasm_is_cancelled(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    cancel() {
        const ret = wasm.canceltoken_cancel(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {CancelToken}
     */
    static create() {
        const ret = wasm.canceltoken_create();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
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
     * @returns {number}
     */
    __uzuCloneChatConfig() {
        const ret = wasm.chatconfig___uzuCloneChatConfig(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get contextLength() {
        const ret = wasm.chatconfig___wasm_get_context_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get samplingSeed() {
        const ret = wasm.chatconfig___wasm_get_sampling_seed(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ContextLength} contextLength
     * @param {SamplingSeed} samplingSeed
     */
    constructor(contextLength, samplingSeed) {
        const ret = wasm.chatconfig___wasm_new(contextLength, samplingSeed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set contextLength(value) {
        wasm.chatconfig___wasm_set_context_length(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set samplingSeed(value) {
        wasm.chatconfig___wasm_set_sampling_seed(this.__wbg_ptr, value);
    }
    /**
     * @returns {ChatConfig}
     */
    static create() {
        const ret = wasm.chatconfig_create();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ContextLength} contextLength
     * @returns {ChatConfig}
     */
    withContextLength(contextLength) {
        const ret = wasm.chatconfig_withContextLength(this.__wbg_ptr, contextLength);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {SamplingSeed} samplingSeed
     * @returns {ChatConfig}
     */
    withSamplingSeed(samplingSeed) {
        const ret = wasm.chatconfig_withSamplingSeed(this.__wbg_ptr, samplingSeed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) ChatConfig.prototype[Symbol.dispose] = ChatConfig.prototype.free;

export class ChatContentBlockAudio {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockAudio.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockAudioFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockAudioFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockaudio_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockaudio___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.chatcontentblockaudio___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} url
     */
    constructor(url) {
        const ret = wasm.chatcontentblockaudio___wasm_new(url);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockAudioFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.chatcontentblockaudio___wasm_set_url(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockAudio.prototype[Symbol.dispose] = ChatContentBlockAudio.prototype.free;

export class ChatContentBlockBuiltinTools {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockBuiltinTools.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockBuiltinToolsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockBuiltinToolsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockbuiltintools_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockbuiltintools___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get names() {
        const ret = wasm.chatcontentblockbuiltintools___wasm_get_names(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<string>} names
     */
    constructor(names) {
        const ret = wasm.chatcontentblockbuiltintools___wasm_new(names);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockBuiltinToolsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set names(value) {
        wasm.chatcontentblockbuiltintools___wasm_set_names(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockBuiltinTools.prototype[Symbol.dispose] = ChatContentBlockBuiltinTools.prototype.free;

export class ChatContentBlockConversationStartDate {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockConversationStartDate.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockConversationStartDateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockConversationStartDateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockconversationstartdate_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockconversationstartdate___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblockconversationstartdate___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblockconversationstartdate___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockConversationStartDateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblockconversationstartdate___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockConversationStartDate.prototype[Symbol.dispose] = ChatContentBlockConversationStartDate.prototype.free;

export class ChatContentBlockCustom {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockCustom.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockCustomFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockCustomFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockcustom_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockcustom___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblockcustom___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Value} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblockcustom___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockCustomFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblockcustom___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockCustom.prototype[Symbol.dispose] = ChatContentBlockCustom.prototype.free;

export class ChatContentBlockFile {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockFile.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockFileFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockFileFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockfile_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockfile___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.chatcontentblockfile___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} url
     */
    constructor(url) {
        const ret = wasm.chatcontentblockfile___wasm_new(url);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockFileFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.chatcontentblockfile___wasm_set_url(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockFile.prototype[Symbol.dispose] = ChatContentBlockFile.prototype.free;

export class ChatContentBlockIdentity {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockIdentity.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockIdentityFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockIdentityFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockidentity_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockidentity___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblockidentity___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblockidentity___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockIdentityFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblockidentity___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockIdentity.prototype[Symbol.dispose] = ChatContentBlockIdentity.prototype.free;

export class ChatContentBlockImage {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockImage.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockImageFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockImageFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockimage_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockimage___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.chatcontentblockimage___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} url
     */
    constructor(url) {
        const ret = wasm.chatcontentblockimage___wasm_new(url);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockImageFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.chatcontentblockimage___wasm_set_url(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockImage.prototype[Symbol.dispose] = ChatContentBlockImage.prototype.free;

export class ChatContentBlockKnowledgeCutoff {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockKnowledgeCutoff.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockKnowledgeCutoffFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockKnowledgeCutoffFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockknowledgecutoff_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockknowledgecutoff___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblockknowledgecutoff___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblockknowledgecutoff___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockKnowledgeCutoffFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblockknowledgecutoff___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockKnowledgeCutoff.prototype[Symbol.dispose] = ChatContentBlockKnowledgeCutoff.prototype.free;

export class ChatContentBlockReasoning {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockReasoning.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockReasoningFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockReasoningFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockreasoning_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockreasoning___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblockreasoning___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblockreasoning___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockReasoningFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblockreasoning___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockReasoning.prototype[Symbol.dispose] = ChatContentBlockReasoning.prototype.free;

export class ChatContentBlockReasoningEffort {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockReasoningEffort.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockReasoningEffortFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockReasoningEffortFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockreasoningeffort_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockreasoningeffort___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblockreasoningeffort___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ReasoningEffort} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblockreasoningeffort___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockReasoningEffortFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblockreasoningeffort___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockReasoningEffort.prototype[Symbol.dispose] = ChatContentBlockReasoningEffort.prototype.free;

export class ChatContentBlockText {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockText.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockTextFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockTextFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktext_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblocktext___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblocktext___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblocktext___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockTextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblocktext___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockText.prototype[Symbol.dispose] = ChatContentBlockText.prototype.free;

export class ChatContentBlockToolCall {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockToolCall.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockToolCallFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockToolCallFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktoolcall_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblocktoolcall___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblocktoolcall___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ToolCall} value
     */
    constructor(value) {
        const ret = wasm.chatcontentblocktoolcall___wasm_new(value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockToolCallFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblocktoolcall___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockToolCall.prototype[Symbol.dispose] = ChatContentBlockToolCall.prototype.free;

export class ChatContentBlockToolCallCandidate {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockToolCallCandidate.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockToolCallCandidateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockToolCallCandidateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktoolcallcandidate_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblocktoolcallcandidate___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.chatcontentblocktoolcallcandidate___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblocktoolcallcandidate___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Value} value
     * @param {string | null} [identifier]
     */
    constructor(value, identifier) {
        const ret = wasm.chatcontentblocktoolcallcandidate___wasm_new(value, identifier);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockToolCallCandidateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.chatcontentblocktoolcallcandidate___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblocktoolcallcandidate___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockToolCallCandidate.prototype[Symbol.dispose] = ChatContentBlockToolCallCandidate.prototype.free;

export class ChatContentBlockToolCallResult {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockToolCallResult.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockToolCallResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockToolCallResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktoolcallresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblocktoolcallresult___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.chatcontentblocktoolcallresult___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.chatcontentblocktoolcallresult___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.chatcontentblocktoolcallresult___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | undefined | null} identifier
     * @param {string | undefined | null} name
     * @param {Value} value
     */
    constructor(identifier, name, value) {
        const ret = wasm.chatcontentblocktoolcallresult___wasm_new(identifier, name, value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockToolCallResultFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.chatcontentblocktoolcallresult___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.chatcontentblocktoolcallresult___wasm_set_name(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.chatcontentblocktoolcallresult___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockToolCallResult.prototype[Symbol.dispose] = ChatContentBlockToolCallResult.prototype.free;

export class ChatContentBlockTools {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockTools.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockToolsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockToolsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktools_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblocktools___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get namespaces() {
        const ret = wasm.chatcontentblocktools___wasm_get_namespaces(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<ToolNamespace>} namespaces
     */
    constructor(namespaces) {
        const ret = wasm.chatcontentblocktools___wasm_new(namespaces);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockToolsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set namespaces(value) {
        wasm.chatcontentblocktools___wasm_set_namespaces(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockTools.prototype[Symbol.dispose] = ChatContentBlockTools.prototype.free;

export class ChatContentBlockTranslation {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockTranslation.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockTranslationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockTranslationFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktranslation_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblocktranslation___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get payload() {
        const ret = wasm.chatcontentblocktranslation___wasm_get_payload(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get sourceLanguageCode() {
        const ret = wasm.chatcontentblocktranslation___wasm_get_source_language_code(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get targetLanguageCode() {
        const ret = wasm.chatcontentblocktranslation___wasm_get_target_language_code(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {TranslationPayload} payload
     * @param {string} sourceLanguageCode
     * @param {string} targetLanguageCode
     */
    constructor(payload, sourceLanguageCode, targetLanguageCode) {
        const ret = wasm.chatcontentblocktranslation___wasm_new(payload, sourceLanguageCode, targetLanguageCode);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockTranslationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set payload(value) {
        wasm.chatcontentblocktranslation___wasm_set_payload(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set sourceLanguageCode(value) {
        wasm.chatcontentblocktranslation___wasm_set_source_language_code(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set targetLanguageCode(value) {
        wasm.chatcontentblocktranslation___wasm_set_target_language_code(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockTranslation.prototype[Symbol.dispose] = ChatContentBlockTranslation.prototype.free;

export class ChatContentBlockType {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockTypeFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblocktype_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Audio() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Audio();
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
    static get BuiltinTools() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_BuiltinTools();
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
    static get ConversationStartDate() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_ConversationStartDate();
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
    static get Custom() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Custom();
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
    static get File() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_File();
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
    static get Identity() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Identity();
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
    static get Image() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Image();
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
    static get KnowledgeCutoff() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_KnowledgeCutoff();
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
    static get ReasoningEffort() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_ReasoningEffort();
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
    static get Reasoning() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Reasoning();
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
    static get Text() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Text();
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
    static get ToolCallCandidate() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_ToolCallCandidate();
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
    static get ToolCallResult() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_ToolCallResult();
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
    static get ToolCall() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_ToolCall();
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
    static get Tools() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Tools();
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
    static get Translation() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Translation();
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
    static get Video() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatcontentblocktype___wasm_variant_Video();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ChatContentBlockType.prototype[Symbol.dispose] = ChatContentBlockType.prototype.free;

export class ChatContentBlockVideo {
    static __wrap(ptr) {
        const obj = Object.create(ChatContentBlockVideo.prototype);
        obj.__wbg_ptr = ptr;
        ChatContentBlockVideoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatContentBlockVideoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatcontentblockvideo_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatContentBlock() {
        const ret = wasm.chatcontentblockvideo___uzuCloneChatContentBlock(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.chatcontentblockvideo___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} url
     */
    constructor(url) {
        const ret = wasm.chatcontentblockvideo___wasm_new(url);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatContentBlockVideoFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.chatcontentblockvideo___wasm_set_url(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatContentBlockVideo.prototype[Symbol.dispose] = ChatContentBlockVideo.prototype.free;

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
        const ret = wasm.chatinstance___wasm_model(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
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
     * @returns {number}
     */
    __uzuCloneChatMessage() {
        const ret = wasm.chatmessage___uzuCloneChatMessage(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get content() {
        const ret = wasm.chatmessage___wasm_get_content(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.chatmessage___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get role() {
        const ret = wasm.chatmessage___wasm_get_role(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatRole} role
     * @param {Array<ChatContentBlock>} content
     * @param {ChatMessageMetadata} metadata
     */
    constructor(role, content, metadata) {
        const ret = wasm.chatmessage___wasm_new(role, content, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatMessageFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {string | null}
     */
    get reasoning() {
        const ret = wasm.chatmessage___wasm_reasoning(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {ReasoningEffort | null}
     */
    get reasoningEffort() {
        const ret = wasm.chatmessage___wasm_reasoning_effort(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {any} value
     */
    set content(value) {
        wasm.chatmessage___wasm_set_content(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.chatmessage___wasm_set_metadata(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set role(value) {
        wasm.chatmessage___wasm_set_role(this.__wbg_ptr, value);
    }
    /**
     * @returns {string | null}
     */
    get text() {
        const ret = wasm.chatmessage___wasm_text(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {Array<ToolCall>}
     */
    get toolCalls() {
        const ret = wasm.chatmessage___wasm_tool_calls(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {Array<ToolNamespace>}
     */
    get toolNamespaces() {
        const ret = wasm.chatmessage___wasm_tool_namespaces(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {ChatMessage}
     */
    static assistant() {
        const ret = wasm.chatmessage_assistant();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {ChatMessage}
     */
    static developer() {
        const ret = wasm.chatmessage_developer();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ChatRole} role
     * @returns {ChatMessage}
     */
    static for_role(role) {
        const ret = wasm.chatmessage_for_role(role);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {ChatMessage}
     */
    static system() {
        const ret = wasm.chatmessage_system();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {ChatMessage}
     */
    static tool() {
        const ret = wasm.chatmessage_tool();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {ChatMessage}
     */
    static user() {
        const ret = wasm.chatmessage_user();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ChatContentBlock} block
     * @returns {ChatMessage}
     */
    withBlock(block) {
        const ret = wasm.chatmessage_withBlock(this.__wbg_ptr, block);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ReasoningEffort} reasoningEffort
     * @returns {ChatMessage}
     */
    withReasoningEffort(reasoningEffort) {
        const ret = wasm.chatmessage_withReasoningEffort(this.__wbg_ptr, reasoningEffort);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} reasoning
     * @returns {ChatMessage}
     */
    withReasoning(reasoning) {
        const ret = wasm.chatmessage_withReasoning(this.__wbg_ptr, reasoning);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} text
     * @returns {ChatMessage}
     */
    withText(text) {
        const ret = wasm.chatmessage_withText(this.__wbg_ptr, text);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {Value} value
     * @returns {ChatMessage}
     */
    withToolCallCandidate(value) {
        const ret = wasm.chatmessage_withToolCallCandidate(this.__wbg_ptr, value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ToolCall} toolCall
     * @returns {ChatMessage}
     */
    withToolCall(toolCall) {
        const ret = wasm.chatmessage_withToolCall(this.__wbg_ptr, toolCall);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {Array<ToolNamespace>} toolNamespaces
     * @returns {ChatMessage}
     */
    withToolNamespaces(toolNamespaces) {
        const ret = wasm.chatmessage_withToolNamespaces(this.__wbg_ptr, toolNamespaces);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
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
     * @returns {number}
     */
    __uzuCloneChatModelCapabilities() {
        const ret = wasm.chatmodelcapabilities___uzuCloneChatModelCapabilities(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get reasoningEfforts() {
        const ret = wasm.chatmodelcapabilities___wasm_get_reasoning_efforts(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get requiresTools() {
        const ret = wasm.chatmodelcapabilities___wasm_get_requires_tools(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get supportsDisableReasoning() {
        const ret = wasm.chatmodelcapabilities___wasm_get_supports_disable_reasoning(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get supportsMultipleToolCalls() {
        const ret = wasm.chatmodelcapabilities___wasm_get_supports_multiple_tool_calls(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get supportsReasoning() {
        const ret = wasm.chatmodelcapabilities___wasm_get_supports_reasoning(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get supportsTools() {
        const ret = wasm.chatmodelcapabilities___wasm_get_supports_tools(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {boolean} supportsReasoning
     * @param {boolean} supportsDisableReasoning
     * @param {Array<ReasoningEffort>} reasoningEfforts
     * @param {boolean} supportsTools
     * @param {boolean} supportsMultipleToolCalls
     * @param {boolean} requiresTools
     */
    constructor(supportsReasoning, supportsDisableReasoning, reasoningEfforts, supportsTools, supportsMultipleToolCalls, requiresTools) {
        const ret = wasm.chatmodelcapabilities___wasm_new(supportsReasoning, supportsDisableReasoning, reasoningEfforts, supportsTools, supportsMultipleToolCalls, requiresTools);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatModelCapabilitiesFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set reasoningEfforts(value) {
        wasm.chatmodelcapabilities___wasm_set_reasoning_efforts(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set requiresTools(value) {
        wasm.chatmodelcapabilities___wasm_set_requires_tools(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set supportsDisableReasoning(value) {
        wasm.chatmodelcapabilities___wasm_set_supports_disable_reasoning(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set supportsMultipleToolCalls(value) {
        wasm.chatmodelcapabilities___wasm_set_supports_multiple_tool_calls(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set supportsReasoning(value) {
        wasm.chatmodelcapabilities___wasm_set_supports_reasoning(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set supportsTools(value) {
        wasm.chatmodelcapabilities___wasm_set_supports_tools(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneChatModelInfo() {
        const ret = wasm.chatmodelinfo___uzuCloneChatModelInfo(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get capabilities() {
        const ret = wasm.chatmodelinfo___wasm_get_capabilities(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get contextLength() {
        const ret = wasm.chatmodelinfo___wasm_get_context_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get defaultReasoningEffort() {
        const ret = wasm.chatmodelinfo___wasm_get_default_reasoning_effort(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number | undefined | null} contextLength
     * @param {ChatModelCapabilities} capabilities
     * @param {ReasoningEffort | null} [defaultReasoningEffort]
     */
    constructor(contextLength, capabilities, defaultReasoningEffort) {
        const ret = wasm.chatmodelinfo___wasm_new(contextLength, capabilities, defaultReasoningEffort);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatModelInfoFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set capabilities(value) {
        wasm.chatmodelinfo___wasm_set_capabilities(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set contextLength(value) {
        wasm.chatmodelinfo___wasm_set_context_length(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set defaultReasoningEffort(value) {
        wasm.chatmodelinfo___wasm_set_default_reasoning_effort(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneChatReply() {
        const ret = wasm.chatreply___uzuCloneChatReply(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get finishReason() {
        const ret = wasm.chatreply___wasm_get_finish_reason(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get message() {
        const ret = wasm.chatreply___wasm_get_message(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get stats() {
        const ret = wasm.chatreply___wasm_get_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatMessage} message
     * @param {ChatReplyStats} stats
     * @param {ChatReplyFinishReason | null} [finishReason]
     */
    constructor(message, stats, finishReason) {
        const ret = wasm.chatreply___wasm_new(message, stats, finishReason);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatReplyFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set finishReason(value) {
        wasm.chatreply___wasm_set_finish_reason(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set message(value) {
        wasm.chatreply___wasm_set_message(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set stats(value) {
        wasm.chatreply___wasm_set_stats(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneChatReplyConfig() {
        const ret = wasm.chatreplyconfig___uzuCloneChatReplyConfig(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get grammar() {
        const ret = wasm.chatreplyconfig___wasm_get_grammar(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get samplingPolicy() {
        const ret = wasm.chatreplyconfig___wasm_get_sampling_policy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokenLimit() {
        const ret = wasm.chatreplyconfig___wasm_get_token_limit(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get toolTurnLimit() {
        const ret = wasm.chatreplyconfig___wasm_get_tool_turn_limit(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number | undefined | null} tokenLimit
     * @param {SamplingPolicy} samplingPolicy
     * @param {Grammar | null} [grammar]
     * @param {number | null} [toolTurnLimit]
     */
    constructor(tokenLimit, samplingPolicy, grammar, toolTurnLimit) {
        const ret = wasm.chatreplyconfig___wasm_new(tokenLimit, samplingPolicy, grammar, toolTurnLimit);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatReplyConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set grammar(value) {
        wasm.chatreplyconfig___wasm_set_grammar(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set samplingPolicy(value) {
        wasm.chatreplyconfig___wasm_set_sampling_policy(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokenLimit(value) {
        wasm.chatreplyconfig___wasm_set_token_limit(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set toolTurnLimit(value) {
        wasm.chatreplyconfig___wasm_set_tool_turn_limit(this.__wbg_ptr, value);
    }
    /**
     * @returns {ChatReplyConfig}
     */
    static create() {
        const ret = wasm.chatreplyconfig_create();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {Grammar | null} [grammar]
     * @returns {ChatReplyConfig}
     */
    withGrammar(grammar) {
        const ret = wasm.chatreplyconfig_withGrammar(this.__wbg_ptr, grammar);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {SamplingMethod} samplingMethod
     * @returns {ChatReplyConfig}
     */
    withSamplingMethod(samplingMethod) {
        const ret = wasm.chatreplyconfig_withSamplingMethod(this.__wbg_ptr, samplingMethod);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {SamplingPolicy} samplingPolicy
     * @returns {ChatReplyConfig}
     */
    withSamplingPolicy(samplingPolicy) {
        const ret = wasm.chatreplyconfig_withSamplingPolicy(this.__wbg_ptr, samplingPolicy);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number | null} [tokenLimit]
     * @returns {ChatReplyConfig}
     */
    withTokenLimit(tokenLimit) {
        const ret = wasm.chatreplyconfig_withTokenLimit(this.__wbg_ptr, tokenLimit);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number | null} [toolTurnLimit]
     * @returns {ChatReplyConfig}
     */
    withToolTurnLimit(toolTurnLimit) {
        const ret = wasm.chatreplyconfig_withToolTurnLimit(this.__wbg_ptr, toolTurnLimit);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) ChatReplyConfig.prototype[Symbol.dispose] = ChatReplyConfig.prototype.free;

export class ChatReplyFinishReason {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplyFinishReasonFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreplyfinishreason_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Cancelled() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatreplyfinishreason___wasm_variant_Cancelled();
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
    static get ContextLimitReached() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatreplyfinishreason___wasm_variant_ContextLimitReached();
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
    static get Length() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatreplyfinishreason___wasm_variant_Length();
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
    static get Rejected() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatreplyfinishreason___wasm_variant_Rejected();
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
    static get Stop() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatreplyfinishreason___wasm_variant_Stop();
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
    static get ToolCalls() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatreplyfinishreason___wasm_variant_ToolCalls();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ChatReplyFinishReason.prototype[Symbol.dispose] = ChatReplyFinishReason.prototype.free;

export class ChatReplyJoulesPerTokenComponents {
    static __wrap(ptr) {
        const obj = Object.create(ChatReplyJoulesPerTokenComponents.prototype);
        obj.__wbg_ptr = ptr;
        ChatReplyJoulesPerTokenComponentsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplyJoulesPerTokenComponentsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreplyjoulespertokencomponents_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatReplyJoulesPerToken() {
        const ret = wasm.chatreplyjoulespertokencomponents___uzuCloneChatReplyJoulesPerToken(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get ane() {
        const ret = wasm.chatreplyjoulespertokencomponents___wasm_get_ane(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get cpu() {
        const ret = wasm.chatreplyjoulespertokencomponents___wasm_get_cpu(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get dram() {
        const ret = wasm.chatreplyjoulespertokencomponents___wasm_get_dram(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get gpu() {
        const ret = wasm.chatreplyjoulespertokencomponents___wasm_get_gpu(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} cpu
     * @param {number} gpu
     * @param {number} ane
     * @param {number} dram
     */
    constructor(cpu, gpu, ane, dram) {
        const ret = wasm.chatreplyjoulespertokencomponents___wasm_new(cpu, gpu, ane, dram);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatReplyJoulesPerTokenComponentsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set ane(value) {
        wasm.chatreplyjoulespertokencomponents___wasm_set_ane(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set cpu(value) {
        wasm.chatreplyjoulespertokencomponents___wasm_set_cpu(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set dram(value) {
        wasm.chatreplyjoulespertokencomponents___wasm_set_dram(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set gpu(value) {
        wasm.chatreplyjoulespertokencomponents___wasm_set_gpu(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatReplyJoulesPerTokenComponents.prototype[Symbol.dispose] = ChatReplyJoulesPerTokenComponents.prototype.free;

export class ChatReplyJoulesPerTokenTotal {
    static __wrap(ptr) {
        const obj = Object.create(ChatReplyJoulesPerTokenTotal.prototype);
        obj.__wbg_ptr = ptr;
        ChatReplyJoulesPerTokenTotalFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatReplyJoulesPerTokenTotalFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatreplyjoulespertokentotal_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatReplyJoulesPerToken() {
        const ret = wasm.chatreplyjoulespertokentotal___uzuCloneChatReplyJoulesPerToken(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get total() {
        const ret = wasm.chatreplyjoulespertokentotal___wasm_get_total(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} total
     */
    constructor(total) {
        const ret = wasm.chatreplyjoulespertokentotal___wasm_new(total);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatReplyJoulesPerTokenTotalFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set total(value) {
        wasm.chatreplyjoulespertokentotal___wasm_set_total(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatReplyJoulesPerTokenTotal.prototype[Symbol.dispose] = ChatReplyJoulesPerTokenTotal.prototype.free;

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
    __uzuCloneChatReplySpeculatorStats() {
        const ret = wasm.chatreplyspeculatorstats___uzuCloneChatReplySpeculatorStats(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get numDecodeForwardPasses() {
        const ret = wasm.chatreplyspeculatorstats___wasm_get_num_decode_forward_passes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensPerForwardPass() {
        const ret = wasm.chatreplyspeculatorstats___wasm_get_tokens_per_forward_pass(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} tokensPerForwardPass
     * @param {number} numDecodeForwardPasses
     */
    constructor(tokensPerForwardPass, numDecodeForwardPasses) {
        const ret = wasm.chatreplyspeculatorstats___wasm_new(tokensPerForwardPass, numDecodeForwardPasses);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatReplySpeculatorStatsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set numDecodeForwardPasses(value) {
        wasm.chatreplyspeculatorstats___wasm_set_num_decode_forward_passes(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensPerForwardPass(value) {
        wasm.chatreplyspeculatorstats___wasm_set_tokens_per_forward_pass(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneChatReplyStats() {
        const ret = wasm.chatreplystats___uzuCloneChatReplyStats(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get duration() {
        const ret = wasm.chatreplystats___wasm_get_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get generateTokensPerSecond() {
        const ret = wasm.chatreplystats___wasm_get_generate_tokens_per_second(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get inputEnergy() {
        const ret = wasm.chatreplystats___wasm_get_input_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get memoryUsedBytes() {
        const ret = wasm.chatreplystats___wasm_get_memory_used_bytes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get outputEnergy() {
        const ret = wasm.chatreplystats___wasm_get_output_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get prefillTokensPerSecond() {
        const ret = wasm.chatreplystats___wasm_get_prefill_tokens_per_second(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get speculatorStats() {
        const ret = wasm.chatreplystats___wasm_get_speculator_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get timeToFirstToken() {
        const ret = wasm.chatreplystats___wasm_get_time_to_first_token(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensCountInput() {
        const ret = wasm.chatreplystats___wasm_get_tokens_count_input(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensCountInputCached() {
        const ret = wasm.chatreplystats___wasm_get_tokens_count_input_cached(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensCountOutput() {
        const ret = wasm.chatreplystats___wasm_get_tokens_count_output(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ChatReplyJoulesPerToken | null}
     */
    get inputJoulesPerToken() {
        const ret = wasm.chatreplystats___wasm_input_joules_per_token(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} duration
     * @param {number | null} [timeToFirstToken]
     * @param {number | null} [prefillTokensPerSecond]
     * @param {number | null} [generateTokensPerSecond]
     * @param {number | null} [tokensCountInput]
     * @param {number | null} [tokensCountInputCached]
     * @param {number | null} [tokensCountOutput]
     * @param {number | null} [memoryUsedBytes]
     * @param {ChatReplySpeculatorStats | null} [speculatorStats]
     * @param {ReplyEnergy | null} [inputEnergy]
     * @param {ReplyEnergy | null} [outputEnergy]
     */
    constructor(duration, timeToFirstToken, prefillTokensPerSecond, generateTokensPerSecond, tokensCountInput, tokensCountInputCached, tokensCountOutput, memoryUsedBytes, speculatorStats, inputEnergy, outputEnergy) {
        const ret = wasm.chatreplystats___wasm_new(duration, timeToFirstToken, prefillTokensPerSecond, generateTokensPerSecond, tokensCountInput, tokensCountInputCached, tokensCountOutput, memoryUsedBytes, speculatorStats, inputEnergy, outputEnergy);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatReplyStatsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {ChatReplyJoulesPerToken | null}
     */
    get outputJoulesPerToken() {
        const ret = wasm.chatreplystats___wasm_output_joules_per_token(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {any} value
     */
    set duration(value) {
        wasm.chatreplystats___wasm_set_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set generateTokensPerSecond(value) {
        wasm.chatreplystats___wasm_set_generate_tokens_per_second(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set inputEnergy(value) {
        wasm.chatreplystats___wasm_set_input_energy(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set memoryUsedBytes(value) {
        wasm.chatreplystats___wasm_set_memory_used_bytes(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set outputEnergy(value) {
        wasm.chatreplystats___wasm_set_output_energy(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set prefillTokensPerSecond(value) {
        wasm.chatreplystats___wasm_set_prefill_tokens_per_second(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set speculatorStats(value) {
        wasm.chatreplystats___wasm_set_speculator_stats(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set timeToFirstToken(value) {
        wasm.chatreplystats___wasm_set_time_to_first_token(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensCountInput(value) {
        wasm.chatreplystats___wasm_set_tokens_count_input(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensCountInputCached(value) {
        wasm.chatreplystats___wasm_set_tokens_count_input_cached(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensCountOutput(value) {
        wasm.chatreplystats___wasm_set_tokens_count_output(this.__wbg_ptr, value);
    }
    /**
     * @returns {number | null}
     */
    get tokensCount() {
        const ret = wasm.chatreplystats___wasm_tokens_count(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {number | null}
     */
    get totalJoules() {
        const ret = wasm.chatreplystats___wasm_total_joules(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) ChatReplyStats.prototype[Symbol.dispose] = ChatReplyStats.prototype.free;

export class ChatRoleAssistant {
    static __wrap(ptr) {
        const obj = Object.create(ChatRoleAssistant.prototype);
        obj.__wbg_ptr = ptr;
        ChatRoleAssistantFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatRoleAssistantFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatroleassistant_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatRole() {
        const ret = wasm.chatroleassistant___uzuCloneChatRole(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.chatroleassistant___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatRoleAssistantFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ChatRoleAssistant.prototype[Symbol.dispose] = ChatRoleAssistant.prototype.free;

export class ChatRoleCustom {
    static __wrap(ptr) {
        const obj = Object.create(ChatRoleCustom.prototype);
        obj.__wbg_ptr = ptr;
        ChatRoleCustomFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatRoleCustomFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatrolecustom_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatRole() {
        const ret = wasm.chatrolecustom___uzuCloneChatRole(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.chatrolecustom___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} name
     */
    constructor(name) {
        const ret = wasm.chatrolecustom___wasm_new(name);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatRoleCustomFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.chatrolecustom___wasm_set_name(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatRoleCustom.prototype[Symbol.dispose] = ChatRoleCustom.prototype.free;

export class ChatRoleDeveloper {
    static __wrap(ptr) {
        const obj = Object.create(ChatRoleDeveloper.prototype);
        obj.__wbg_ptr = ptr;
        ChatRoleDeveloperFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatRoleDeveloperFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatroledeveloper_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatRole() {
        const ret = wasm.chatroledeveloper___uzuCloneChatRole(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.chatroledeveloper___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatRoleDeveloperFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ChatRoleDeveloper.prototype[Symbol.dispose] = ChatRoleDeveloper.prototype.free;

export class ChatRoleSystem {
    static __wrap(ptr) {
        const obj = Object.create(ChatRoleSystem.prototype);
        obj.__wbg_ptr = ptr;
        ChatRoleSystemFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatRoleSystemFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatrolesystem_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatRole() {
        const ret = wasm.chatrolesystem___uzuCloneChatRole(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.chatrolesystem___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatRoleSystemFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ChatRoleSystem.prototype[Symbol.dispose] = ChatRoleSystem.prototype.free;

export class ChatRoleTool {
    static __wrap(ptr) {
        const obj = Object.create(ChatRoleTool.prototype);
        obj.__wbg_ptr = ptr;
        ChatRoleToolFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatRoleToolFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatroletool_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatRole() {
        const ret = wasm.chatroletool___uzuCloneChatRole(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.chatroletool___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatRoleToolFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ChatRoleTool.prototype[Symbol.dispose] = ChatRoleTool.prototype.free;

export class ChatRoleUser {
    static __wrap(ptr) {
        const obj = Object.create(ChatRoleUser.prototype);
        obj.__wbg_ptr = ptr;
        ChatRoleUserFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatRoleUserFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatroleuser_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatRole() {
        const ret = wasm.chatroleuser___uzuCloneChatRole(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.chatroleuser___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatRoleUserFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ChatRoleUser.prototype[Symbol.dispose] = ChatRoleUser.prototype.free;

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
     * @returns {Array<ChatMessage>}
     */
    get messages() {
        const ret = wasm.chatsession___wasm_messages(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {SamplingParameters | null}
     */
    get samplingDefaults() {
        const ret = wasm.chatsession___wasm_sampling_defaults(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {ChatSessionState}
     */
    get state() {
        const ret = wasm.chatsession___wasm_state(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get supportsToolCalls() {
        const ret = wasm.chatsession___wasm_supports_tool_calls(this.__wbg_ptr);
        return ret;
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
     * @param {Array<NativeTool>} tools
     * @returns {Promise<void>}
     */
    addTools(tools) {
        const ret = wasm.chatsession_addTools(this.__wbg_ptr, tools);
        return ret;
    }
    /**
     * @param {Array<ChatMessage>} input
     * @param {ChatReplyConfig} config
     * @returns {ChatSessionStream}
     */
    replyWithStream(input, config) {
        const ret = wasm.chatsession_replyWithStream(this.__wbg_ptr, input, config);
        return ret;
    }
    /**
     * @param {Array<ChatMessage>} input
     * @param {ChatReplyConfig} config
     * @returns {Array<ChatReply>}
     */
    reply(input, config) {
        const ret = wasm.chatsession_reply(this.__wbg_ptr, input, config);
        return ret;
    }
    /**
     * @returns {Promise<void>}
     */
    reset() {
        const ret = wasm.chatsession_reset(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) ChatSession.prototype[Symbol.dispose] = ChatSession.prototype.free;

export class ChatSessionState {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatSessionStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatsessionstate_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Generation() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatsessionstate___wasm_variant_Generation();
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
    static get Idle() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatsessionstate___wasm_variant_Idle();
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
    static get Resetting() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatsessionstate___wasm_variant_Resetting();
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
    static get ToolCalling() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.chatsessionstate___wasm_variant_ToolCalling();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ChatSessionState.prototype[Symbol.dispose] = ChatSessionState.prototype.free;

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
        const ret = wasm.chatsessionstream___wasm_cancel_token(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    static get __uzuStream() {
        const ret = wasm.chatsessionstream___wasm_stream_marker();
        return ret !== 0;
    }
    /**
     * @returns {ChatSessionStreamChunk | null}
     */
    next() {
        const ret = wasm.chatsessionstream_next(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) ChatSessionStream.prototype[Symbol.dispose] = ChatSessionStream.prototype.free;

export class ChatSessionStreamChunkError {
    static __wrap(ptr) {
        const obj = Object.create(ChatSessionStreamChunkError.prototype);
        obj.__wbg_ptr = ptr;
        ChatSessionStreamChunkErrorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatSessionStreamChunkErrorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatsessionstreamchunkerror_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatSessionStreamChunk() {
        const ret = wasm.chatsessionstreamchunkerror___uzuCloneChatSessionStreamChunk(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get error() {
        const ret = wasm.chatsessionstreamchunkerror___wasm_get_error(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatSessionError} error
     */
    constructor(error) {
        const ret = wasm.chatsessionstreamchunkerror___wasm_new(error);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatSessionStreamChunkErrorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set error(value) {
        wasm.chatsessionstreamchunkerror___wasm_set_error(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatSessionStreamChunkError.prototype[Symbol.dispose] = ChatSessionStreamChunkError.prototype.free;

export class ChatSessionStreamChunkReplies {
    static __wrap(ptr) {
        const obj = Object.create(ChatSessionStreamChunkReplies.prototype);
        obj.__wbg_ptr = ptr;
        ChatSessionStreamChunkRepliesFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatSessionStreamChunkRepliesFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatsessionstreamchunkreplies_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatSessionStreamChunk() {
        const ret = wasm.chatsessionstreamchunkreplies___uzuCloneChatSessionStreamChunk(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get replies() {
        const ret = wasm.chatsessionstreamchunkreplies___wasm_get_replies(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<ChatReply>} replies
     */
    constructor(replies) {
        const ret = wasm.chatsessionstreamchunkreplies___wasm_new(replies);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatSessionStreamChunkRepliesFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set replies(value) {
        wasm.chatsessionstreamchunkreplies___wasm_set_replies(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatSessionStreamChunkReplies.prototype[Symbol.dispose] = ChatSessionStreamChunkReplies.prototype.free;

export class ChatSessionStreamChunkToolResults {
    static __wrap(ptr) {
        const obj = Object.create(ChatSessionStreamChunkToolResults.prototype);
        obj.__wbg_ptr = ptr;
        ChatSessionStreamChunkToolResultsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChatSessionStreamChunkToolResultsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_chatsessionstreamchunktoolresults_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneChatSessionStreamChunk() {
        const ret = wasm.chatsessionstreamchunktoolresults___uzuCloneChatSessionStreamChunk(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get messages() {
        const ret = wasm.chatsessionstreamchunktoolresults___wasm_get_messages(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<ChatMessage>} messages
     */
    constructor(messages) {
        const ret = wasm.chatsessionstreamchunktoolresults___wasm_new(messages);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ChatSessionStreamChunkToolResultsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set messages(value) {
        wasm.chatsessionstreamchunktoolresults___wasm_set_messages(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ChatSessionStreamChunkToolResults.prototype[Symbol.dispose] = ChatSessionStreamChunkToolResults.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneClassificationLabel() {
        const ret = wasm.classificationlabel___uzuCloneClassificationLabel(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get description() {
        const ret = wasm.classificationlabel___wasm_get_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get index() {
        const ret = wasm.classificationlabel___wasm_get_index(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get label() {
        const ret = wasm.classificationlabel___wasm_get_label(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get recommendedThreshold() {
        const ret = wasm.classificationlabel___wasm_get_recommended_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get title() {
        const ret = wasm.classificationlabel___wasm_get_title(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} index
     * @param {string} label
     * @param {string} title
     * @param {string} description
     * @param {number} recommendedThreshold
     */
    constructor(index, label, title, description, recommendedThreshold) {
        const ret = wasm.classificationlabel___wasm_new(index, label, title, description, recommendedThreshold);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ClassificationLabelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set description(value) {
        wasm.classificationlabel___wasm_set_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set index(value) {
        wasm.classificationlabel___wasm_set_index(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set label(value) {
        wasm.classificationlabel___wasm_set_label(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set recommendedThreshold(value) {
        wasm.classificationlabel___wasm_set_recommended_threshold(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set title(value) {
        wasm.classificationlabel___wasm_set_title(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneClassificationMessage() {
        const ret = wasm.classificationmessage___uzuCloneClassificationMessage(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get content() {
        const ret = wasm.classificationmessage___wasm_get_content(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get role() {
        const ret = wasm.classificationmessage___wasm_get_role(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ClassificationRole} role
     * @param {string} content
     */
    constructor(role, content) {
        const ret = wasm.classificationmessage___wasm_new(role, content);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ClassificationMessageFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set content(value) {
        wasm.classificationmessage___wasm_set_content(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set role(value) {
        wasm.classificationmessage___wasm_set_role(this.__wbg_ptr, value);
    }
    /**
     * @param {string} content
     * @returns {ClassificationMessage}
     */
    static assistant(content) {
        const ret = wasm.classificationmessage_assistant(content);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} content
     * @returns {ClassificationMessage}
     */
    static user(content) {
        const ret = wasm.classificationmessage_user(content);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
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
     * @returns {number}
     */
    __uzuCloneClassificationOutput() {
        const ret = wasm.classificationoutput___uzuCloneClassificationOutput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get logits() {
        const ret = wasm.classificationoutput___wasm_get_logits(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get probabilities() {
        const ret = wasm.classificationoutput___wasm_get_probabilities(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get stats() {
        const ret = wasm.classificationoutput___wasm_get_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<number>} logits
     * @param {ClassificationOutputProbabilities} probabilities
     * @param {ClassificationStats} stats
     */
    constructor(logits, probabilities, stats) {
        const ret = wasm.classificationoutput___wasm_new(logits, probabilities, stats);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ClassificationOutputFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set logits(value) {
        wasm.classificationoutput___wasm_set_logits(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set probabilities(value) {
        wasm.classificationoutput___wasm_set_probabilities(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set stats(value) {
        wasm.classificationoutput___wasm_set_stats(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ClassificationOutput.prototype[Symbol.dispose] = ClassificationOutput.prototype.free;

export class ClassificationRole {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClassificationRoleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_classificationrole_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Assistant() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.classificationrole___wasm_variant_Assistant();
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
    static get User() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.classificationrole___wasm_variant_User();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ClassificationRole.prototype[Symbol.dispose] = ClassificationRole.prototype.free;

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
    __uzuCloneClassificationStats() {
        const ret = wasm.classificationstats___uzuCloneClassificationStats(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get confidence() {
        const ret = wasm.classificationstats___wasm_get_confidence(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get forwardPassDuration() {
        const ret = wasm.classificationstats___wasm_get_forward_pass_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get postprocessingDuration() {
        const ret = wasm.classificationstats___wasm_get_postprocessing_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get predictedLabel() {
        const ret = wasm.classificationstats___wasm_get_predicted_label(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get preprocessingDuration() {
        const ret = wasm.classificationstats___wasm_get_preprocessing_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensCount() {
        const ret = wasm.classificationstats___wasm_get_tokens_count(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensPerSecond() {
        const ret = wasm.classificationstats___wasm_get_tokens_per_second(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get totalDuration() {
        const ret = wasm.classificationstats___wasm_get_total_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} preprocessingDuration
     * @param {number} forwardPassDuration
     * @param {number} postprocessingDuration
     * @param {number} totalDuration
     * @param {number} tokensCount
     * @param {number} tokensPerSecond
     * @param {string} predictedLabel
     * @param {number} confidence
     */
    constructor(preprocessingDuration, forwardPassDuration, postprocessingDuration, totalDuration, tokensCount, tokensPerSecond, predictedLabel, confidence) {
        const ret = wasm.classificationstats___wasm_new(preprocessingDuration, forwardPassDuration, postprocessingDuration, totalDuration, tokensCount, tokensPerSecond, predictedLabel, confidence);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ClassificationStatsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set confidence(value) {
        wasm.classificationstats___wasm_set_confidence(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set forwardPassDuration(value) {
        wasm.classificationstats___wasm_set_forward_pass_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set postprocessingDuration(value) {
        wasm.classificationstats___wasm_set_postprocessing_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set predictedLabel(value) {
        wasm.classificationstats___wasm_set_predicted_label(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set preprocessingDuration(value) {
        wasm.classificationstats___wasm_set_preprocessing_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensCount(value) {
        wasm.classificationstats___wasm_set_tokens_count(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensPerSecond(value) {
        wasm.classificationstats___wasm_set_tokens_per_second(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set totalDuration(value) {
        wasm.classificationstats___wasm_set_total_duration(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ClassificationStats.prototype[Symbol.dispose] = ClassificationStats.prototype.free;

export class ContextLengthCustom {
    static __wrap(ptr) {
        const obj = Object.create(ContextLengthCustom.prototype);
        obj.__wbg_ptr = ptr;
        ContextLengthCustomFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ContextLengthCustomFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_contextlengthcustom_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneContextLength() {
        const ret = wasm.contextlengthcustom___uzuCloneContextLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get length() {
        const ret = wasm.contextlengthcustom___wasm_get_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} length
     */
    constructor(length) {
        const ret = wasm.contextlengthcustom___wasm_new(length);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ContextLengthCustomFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set length(value) {
        wasm.contextlengthcustom___wasm_set_length(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ContextLengthCustom.prototype[Symbol.dispose] = ContextLengthCustom.prototype.free;

export class ContextLengthDefault {
    static __wrap(ptr) {
        const obj = Object.create(ContextLengthDefault.prototype);
        obj.__wbg_ptr = ptr;
        ContextLengthDefaultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ContextLengthDefaultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_contextlengthdefault_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneContextLength() {
        const ret = wasm.contextlengthdefault___uzuCloneContextLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.contextlengthdefault___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ContextLengthDefaultFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ContextLengthDefault.prototype[Symbol.dispose] = ContextLengthDefault.prototype.free;

export class ContextLengthMaximal {
    static __wrap(ptr) {
        const obj = Object.create(ContextLengthMaximal.prototype);
        obj.__wbg_ptr = ptr;
        ContextLengthMaximalFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ContextLengthMaximalFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_contextlengthmaximal_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneContextLength() {
        const ret = wasm.contextlengthmaximal___uzuCloneContextLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.contextlengthmaximal___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ContextLengthMaximalFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ContextLengthMaximal.prototype[Symbol.dispose] = ContextLengthMaximal.prototype.free;

export class DecisionAnswerChoice {
    static __wrap(ptr) {
        const obj = Object.create(DecisionAnswerChoice.prototype);
        obj.__wbg_ptr = ptr;
        DecisionAnswerChoiceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionAnswerChoiceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionanswerchoice_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDecisionAnswer() {
        const ret = wasm.decisionanswerchoice___uzuCloneDecisionAnswer(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get index() {
        const ret = wasm.decisionanswerchoice___wasm_get_index(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get key() {
        const ret = wasm.decisionanswerchoice___wasm_get_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get probabilities() {
        const ret = wasm.decisionanswerchoice___wasm_get_probabilities(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} index
     * @param {string} key
     * @param {Array<number>} probabilities
     */
    constructor(index, key, probabilities) {
        const ret = wasm.decisionanswerchoice___wasm_new(index, key, probabilities);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionAnswerChoiceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set index(value) {
        wasm.decisionanswerchoice___wasm_set_index(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set key(value) {
        wasm.decisionanswerchoice___wasm_set_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set probabilities(value) {
        wasm.decisionanswerchoice___wasm_set_probabilities(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionAnswerChoice.prototype[Symbol.dispose] = DecisionAnswerChoice.prototype.free;

export class DecisionAnswerScore {
    static __wrap(ptr) {
        const obj = Object.create(DecisionAnswerScore.prototype);
        obj.__wbg_ptr = ptr;
        DecisionAnswerScoreFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionAnswerScoreFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionanswerscore_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDecisionAnswer() {
        const ret = wasm.decisionanswerscore___uzuCloneDecisionAnswer(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get probabilities() {
        const ret = wasm.decisionanswerscore___wasm_get_probabilities(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get score() {
        const ret = wasm.decisionanswerscore___wasm_get_score(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<number>} probabilities
     * @param {number} score
     */
    constructor(probabilities, score) {
        const ret = wasm.decisionanswerscore___wasm_new(probabilities, score);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionAnswerScoreFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set probabilities(value) {
        wasm.decisionanswerscore___wasm_set_probabilities(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set score(value) {
        wasm.decisionanswerscore___wasm_set_score(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionAnswerScore.prototype[Symbol.dispose] = DecisionAnswerScore.prototype.free;

export class DecisionAnswerYesNo {
    static __wrap(ptr) {
        const obj = Object.create(DecisionAnswerYesNo.prototype);
        obj.__wbg_ptr = ptr;
        DecisionAnswerYesNoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionAnswerYesNoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionansweryesno_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDecisionAnswer() {
        const ret = wasm.decisionansweryesno___uzuCloneDecisionAnswer(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get yesProbability() {
        const ret = wasm.decisionansweryesno___wasm_get_yes_probability(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} yesProbability
     */
    constructor(yesProbability) {
        const ret = wasm.decisionansweryesno___wasm_new(yesProbability);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionAnswerYesNoFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set yesProbability(value) {
        wasm.decisionansweryesno___wasm_set_yes_probability(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionAnswerYesNo.prototype[Symbol.dispose] = DecisionAnswerYesNo.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneDecisionInput() {
        const ret = wasm.decisioninput___uzuCloneDecisionInput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get questions() {
        const ret = wasm.decisioninput___wasm_get_questions(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get state() {
        const ret = wasm.decisioninput___wasm_get_state(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} state
     * @param {Array<DecisionQuestion>} questions
     */
    constructor(state, questions) {
        const ret = wasm.decisioninput___wasm_new(state, questions);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionInputFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set questions(value) {
        wasm.decisioninput___wasm_set_questions(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set state(value) {
        wasm.decisioninput___wasm_set_state(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionInput.prototype[Symbol.dispose] = DecisionInput.prototype.free;

export class DecisionOption {
    static __wrap(ptr) {
        const obj = Object.create(DecisionOption.prototype);
        obj.__wbg_ptr = ptr;
        DecisionOptionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
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
     * @returns {number}
     */
    __uzuCloneDecisionOption() {
        const ret = wasm.decisionoption___uzuCloneDecisionOption(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get description() {
        const ret = wasm.decisionoption___wasm_get_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get key() {
        const ret = wasm.decisionoption___wasm_get_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} key
     * @param {string | null} [description]
     */
    constructor(key, description) {
        const ret = wasm.decisionoption___wasm_new(key, description);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionOptionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set description(value) {
        wasm.decisionoption___wasm_set_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set key(value) {
        wasm.decisionoption___wasm_set_key(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneDecisionOutput() {
        const ret = wasm.decisionoutput___uzuCloneDecisionOutput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get results() {
        const ret = wasm.decisionoutput___wasm_get_results(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<DecisionResult>} results
     */
    constructor(results) {
        const ret = wasm.decisionoutput___wasm_new(results);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionOutputFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set results(value) {
        wasm.decisionoutput___wasm_set_results(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionOutput.prototype[Symbol.dispose] = DecisionOutput.prototype.free;

export class DecisionQuestionChoice {
    static __wrap(ptr) {
        const obj = Object.create(DecisionQuestionChoice.prototype);
        obj.__wbg_ptr = ptr;
        DecisionQuestionChoiceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionQuestionChoiceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionquestionchoice_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDecisionQuestion() {
        const ret = wasm.decisionquestionchoice___uzuCloneDecisionQuestion(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get instructions() {
        const ret = wasm.decisionquestionchoice___wasm_get_instructions(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get options() {
        const ret = wasm.decisionquestionchoice___wasm_get_options(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | undefined | null} instructions
     * @param {Array<DecisionOption>} options
     */
    constructor(instructions, options) {
        const ret = wasm.decisionquestionchoice___wasm_new(instructions, options);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionQuestionChoiceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set instructions(value) {
        wasm.decisionquestionchoice___wasm_set_instructions(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set options(value) {
        wasm.decisionquestionchoice___wasm_set_options(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionQuestionChoice.prototype[Symbol.dispose] = DecisionQuestionChoice.prototype.free;

export class DecisionQuestionScore {
    static __wrap(ptr) {
        const obj = Object.create(DecisionQuestionScore.prototype);
        obj.__wbg_ptr = ptr;
        DecisionQuestionScoreFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionQuestionScoreFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionquestionscore_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDecisionQuestion() {
        const ret = wasm.decisionquestionscore___uzuCloneDecisionQuestion(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get instructions() {
        const ret = wasm.decisionquestionscore___wasm_get_instructions(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get levels() {
        const ret = wasm.decisionquestionscore___wasm_get_levels(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | undefined | null} instructions
     * @param {Array<string>} levels
     */
    constructor(instructions, levels) {
        const ret = wasm.decisionquestionscore___wasm_new(instructions, levels);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionQuestionScoreFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set instructions(value) {
        wasm.decisionquestionscore___wasm_set_instructions(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set levels(value) {
        wasm.decisionquestionscore___wasm_set_levels(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionQuestionScore.prototype[Symbol.dispose] = DecisionQuestionScore.prototype.free;

export class DecisionQuestionYesNo {
    static __wrap(ptr) {
        const obj = Object.create(DecisionQuestionYesNo.prototype);
        obj.__wbg_ptr = ptr;
        DecisionQuestionYesNoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionQuestionYesNoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionquestionyesno_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDecisionQuestion() {
        const ret = wasm.decisionquestionyesno___uzuCloneDecisionQuestion(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get instructions() {
        const ret = wasm.decisionquestionyesno___wasm_get_instructions(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get noDescription() {
        const ret = wasm.decisionquestionyesno___wasm_get_no_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get yesDescription() {
        const ret = wasm.decisionquestionyesno___wasm_get_yes_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | null} [instructions]
     * @param {string | null} [noDescription]
     * @param {string | null} [yesDescription]
     */
    constructor(instructions, noDescription, yesDescription) {
        const ret = wasm.decisionquestionyesno___wasm_new(instructions, noDescription, yesDescription);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionQuestionYesNoFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set instructions(value) {
        wasm.decisionquestionyesno___wasm_set_instructions(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set noDescription(value) {
        wasm.decisionquestionyesno___wasm_set_no_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set yesDescription(value) {
        wasm.decisionquestionyesno___wasm_set_yes_description(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DecisionQuestionYesNo.prototype[Symbol.dispose] = DecisionQuestionYesNo.prototype.free;

export class DecisionResult {
    static __wrap(ptr) {
        const obj = Object.create(DecisionResult.prototype);
        obj.__wbg_ptr = ptr;
        DecisionResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
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
     * @returns {number}
     */
    __uzuCloneDecisionResult() {
        const ret = wasm.decisionresult___uzuCloneDecisionResult(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get answer() {
        const ret = wasm.decisionresult___wasm_get_answer(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get stats() {
        const ret = wasm.decisionresult___wasm_get_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {DecisionAnswer} answer
     * @param {DecisionStats} stats
     */
    constructor(answer, stats) {
        const ret = wasm.decisionresult___wasm_new(answer, stats);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionResultFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set answer(value) {
        wasm.decisionresult___wasm_set_answer(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set stats(value) {
        wasm.decisionresult___wasm_set_stats(this.__wbg_ptr, value);
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
     * @returns {DecisionSessionState}
     */
    get state() {
        const ret = wasm.decisionsession___wasm_state(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {DecisionInput} input
     * @returns {DecisionOutput}
     */
    decide(input) {
        const ret = wasm.decisionsession_decide(this.__wbg_ptr, input);
        return ret;
    }
}
if (Symbol.dispose) DecisionSession.prototype[Symbol.dispose] = DecisionSession.prototype.free;

export class DecisionSessionState {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecisionSessionStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decisionsessionstate_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Deciding() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.decisionsessionstate___wasm_variant_Deciding();
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
    static get Idle() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.decisionsessionstate___wasm_variant_Idle();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) DecisionSessionState.prototype[Symbol.dispose] = DecisionSessionState.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneDecisionStats() {
        const ret = wasm.decisionstats___uzuCloneDecisionStats(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get energy() {
        const ret = wasm.decisionstats___wasm_get_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get prefillDuration() {
        const ret = wasm.decisionstats___wasm_get_prefill_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tokensCount() {
        const ret = wasm.decisionstats___wasm_get_tokens_count(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get totalDuration() {
        const ret = wasm.decisionstats___wasm_get_total_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} tokensCount
     * @param {number} prefillDuration
     * @param {number} totalDuration
     * @param {ReplyEnergy | null} [energy]
     */
    constructor(tokensCount, prefillDuration, totalDuration, energy) {
        const ret = wasm.decisionstats___wasm_new(tokensCount, prefillDuration, totalDuration, energy);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecisionStatsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set energy(value) {
        wasm.decisionstats___wasm_set_energy(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set prefillDuration(value) {
        wasm.decisionstats___wasm_set_prefill_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tokensCount(value) {
        wasm.decisionstats___wasm_set_tokens_count(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set totalDuration(value) {
        wasm.decisionstats___wasm_set_total_duration(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneDevice() {
        const ret = wasm.device___uzuCloneDevice(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get cpuName() {
        const ret = wasm.device___wasm_get_cpu_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get homePath() {
        const ret = wasm.device___wasm_get_home_path(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get memoryTotal() {
        const ret = wasm.device___wasm_get_memory_total(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get osName() {
        const ret = wasm.device___wasm_get_os_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | undefined | null} osName
     * @param {string | undefined | null} cpuName
     * @param {number | undefined | null} memoryTotal
     * @param {string} homePath
     */
    constructor(osName, cpuName, memoryTotal, homePath) {
        const ret = wasm.device___wasm_new(osName, cpuName, memoryTotal, homePath);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DeviceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set cpuName(value) {
        wasm.device___wasm_set_cpu_name(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set homePath(value) {
        wasm.device___wasm_set_home_path(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set memoryTotal(value) {
        wasm.device___wasm_set_memory_total(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set osName(value) {
        wasm.device___wasm_set_os_name(this.__wbg_ptr, value);
    }
    /**
     * @returns {Device}
     */
    static create() {
        const ret = wasm.device_create();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) Device.prototype[Symbol.dispose] = Device.prototype.free;

export class DownloadManagerType {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadManagerTypeFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadmanagertype_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Universal() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.downloadmanagertype___wasm_variant_Universal();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) DownloadManagerType.prototype[Symbol.dispose] = DownloadManagerType.prototype.free;

export class DownloadPhaseDownloaded {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhaseDownloaded.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhaseDownloadedFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhaseDownloadedFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphasedownloaded_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphasedownloaded___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.downloadphasedownloaded___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhaseDownloadedFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) DownloadPhaseDownloaded.prototype[Symbol.dispose] = DownloadPhaseDownloaded.prototype.free;

export class DownloadPhaseDownloading {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhaseDownloading.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhaseDownloadingFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhaseDownloadingFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphasedownloading_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphasedownloading___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.downloadphasedownloading___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhaseDownloadingFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) DownloadPhaseDownloading.prototype[Symbol.dispose] = DownloadPhaseDownloading.prototype.free;

export class DownloadPhaseError {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhaseError.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhaseErrorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhaseErrorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphaseerror_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphaseerror___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get message() {
        const ret = wasm.downloadphaseerror___wasm_get_message(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} message
     */
    constructor(message) {
        const ret = wasm.downloadphaseerror___wasm_new(message);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhaseErrorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set message(value) {
        wasm.downloadphaseerror___wasm_set_message(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DownloadPhaseError.prototype[Symbol.dispose] = DownloadPhaseError.prototype.free;

export class DownloadPhaseInitializing {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhaseInitializing.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhaseInitializingFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhaseInitializingFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphaseinitializing_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphaseinitializing___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.downloadphaseinitializing___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhaseInitializingFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) DownloadPhaseInitializing.prototype[Symbol.dispose] = DownloadPhaseInitializing.prototype.free;

export class DownloadPhaseLocked {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhaseLocked.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhaseLockedFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhaseLockedFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphaselocked_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphaselocked___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get managerId() {
        const ret = wasm.downloadphaselocked___wasm_get_manager_id(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} managerId
     */
    constructor(managerId) {
        const ret = wasm.downloadphaselocked___wasm_new(managerId);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhaseLockedFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set managerId(value) {
        wasm.downloadphaselocked___wasm_set_manager_id(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) DownloadPhaseLocked.prototype[Symbol.dispose] = DownloadPhaseLocked.prototype.free;

export class DownloadPhaseNotDownloaded {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhaseNotDownloaded.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhaseNotDownloadedFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhaseNotDownloadedFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphasenotdownloaded_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphasenotdownloaded___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.downloadphasenotdownloaded___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhaseNotDownloadedFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) DownloadPhaseNotDownloaded.prototype[Symbol.dispose] = DownloadPhaseNotDownloaded.prototype.free;

export class DownloadPhasePaused {
    static __wrap(ptr) {
        const obj = Object.create(DownloadPhasePaused.prototype);
        obj.__wbg_ptr = ptr;
        DownloadPhasePausedFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DownloadPhasePausedFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_downloadphasepaused_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneDownloadPhase() {
        const ret = wasm.downloadphasepaused___uzuCloneDownloadPhase(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.downloadphasepaused___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadPhasePausedFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) DownloadPhasePaused.prototype[Symbol.dispose] = DownloadPhasePaused.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneDownloadState() {
        const ret = wasm.downloadstate___uzuCloneDownloadState(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {boolean}
     */
    get canDelete() {
        const ret = wasm.downloadstate___wasm_can_delete(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get canPause() {
        const ret = wasm.downloadstate___wasm_can_pause(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {any}
     */
    get downloadedBytes() {
        const ret = wasm.downloadstate___wasm_get_downloaded_bytes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get phase() {
        const ret = wasm.downloadstate___wasm_get_phase(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get totalBytes() {
        const ret = wasm.downloadstate___wasm_get_total_bytes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get isInProgress() {
        const ret = wasm.downloadstate___wasm_is_in_progress(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.downloadstate___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} totalBytes
     * @param {number} downloadedBytes
     * @param {DownloadPhase} phase
     */
    constructor(totalBytes, downloadedBytes, phase) {
        const ret = wasm.downloadstate___wasm_new(totalBytes, downloadedBytes, phase);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DownloadStateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get progress() {
        const ret = wasm.downloadstate___wasm_progress(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {any} value
     */
    set downloadedBytes(value) {
        wasm.downloadstate___wasm_set_downloaded_bytes(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set phase(value) {
        wasm.downloadstate___wasm_set_phase(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set totalBytes(value) {
        wasm.downloadstate___wasm_set_total_bytes(this.__wbg_ptr, value);
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
     * @returns {DownloadState | null}
     */
    get state() {
        const ret = wasm.downloader___wasm_state(this.__wbg_ptr);
        return ret;
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
     * @returns {DownloaderStream}
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
     * @returns {boolean}
     */
    static get __uzuStream() {
        const ret = wasm.downloaderstream___wasm_stream_marker();
        return ret !== 0;
    }
    /**
     * @returns {DownloadState | null}
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
     * @returns {Record<ModelIdentifier, DownloadState>}
     */
    get downloadStates() {
        const ret = wasm.engine___wasm_download_states(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<ModelFamily>}
     */
    get modelFamilies() {
        const ret = wasm.engine___wasm_model_families(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<ModelRegistry>}
     */
    get modelRegistries() {
        const ret = wasm.engine___wasm_model_registries(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<ModelVendor>}
     */
    get modelVendors() {
        const ret = wasm.engine___wasm_model_vendors(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get models() {
        const ret = wasm.engine___wasm_models(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsDownloadable() {
        const ret = wasm.engine___wasm_models_downloadable(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsForChat() {
        const ret = wasm.engine___wasm_models_for_chat(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsForClassification() {
        const ret = wasm.engine___wasm_models_for_classification(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsForSpeculation() {
        const ret = wasm.engine___wasm_models_for_speculation(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsForTextToSpeech() {
        const ret = wasm.engine___wasm_models_for_text_to_speech(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsForTranslation() {
        const ret = wasm.engine___wasm_models_for_translation(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsOnDevice() {
        const ret = wasm.engine___wasm_models_on_device(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Array<Model>}
     */
    get modelsRemote() {
        const ret = wasm.engine___wasm_models_remote(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @param {ChatConfig} config
     * @returns {ChatInstance}
     */
    chatInstance(model, config) {
        const ret = wasm.engine_chatInstance(this.__wbg_ptr, model, config);
        return ret;
    }
    /**
     * Returns metadata for a Uzu chat model, including models not yet downloaded.
     * Encoding capabilities come from the catalog; the context limit needs local configuration.
     * Remote and non-chat models return `None`.
     * @param {Model} model
     * @returns {ChatModelInfo | null}
     */
    chatModelInfo(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_chatModelInfo(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ChatInstance} instance
     * @returns {ChatSession}
     */
    chatWithInstance(instance) {
        _assertClass(instance, ChatInstance);
        const ret = wasm.engine_chatWithInstance(this.__wbg_ptr, instance.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @param {ChatConfig} config
     * @returns {ChatSession}
     */
    chat(model, config) {
        const ret = wasm.engine_chat(this.__wbg_ptr, model, config);
        return ret;
    }
    /**
     * @param {EngineConfig} config
     * @returns {Engine}
     */
    static create(config) {
        const ret = wasm.engine_create(config);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {DecisionSession}
     */
    decision(model) {
        const ret = wasm.engine_decision(this.__wbg_ptr, model);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {DownloadState | null}
     */
    downloadState(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_downloadState(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {DownloaderStream}
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
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ModelIdentifier} identifier
     * @returns {Model | null}
     */
    modelByIdentifier(identifier) {
        const ret = wasm.engine_modelByIdentifier(this.__wbg_ptr, identifier);
        return ret;
    }
    /**
     * @param {string} path
     * @returns {Model | null}
     */
    modelByPath(path) {
        const ret = wasm.engine_modelByPath(this.__wbg_ptr, path);
        return ret;
    }
    /**
     * @param {string} repoId
     * @returns {Model | null}
     */
    modelByRepoId(repoId) {
        const ret = wasm.engine_modelByRepoId(this.__wbg_ptr, repoId);
        return ret;
    }
    /**
     * @param {Model} model
     * @returns {string | null}
     */
    modelPath(model) {
        _assertClass(model, Model);
        const ret = wasm.engine_modelPath(this.__wbg_ptr, model.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} identifier
     * @returns {Model | null}
     */
    model(identifier) {
        const ret = wasm.engine_model(this.__wbg_ptr, identifier);
        return ret;
    }
    /**
     * @param {string} familyIdentifier
     * @returns {Array<Model>}
     */
    modelsByFamily(familyIdentifier) {
        const ret = wasm.engine_modelsByFamily(this.__wbg_ptr, familyIdentifier);
        return ret;
    }
    /**
     * @param {string} vendorIdentifier
     * @returns {Array<Model>}
     */
    modelsByVendor(vendorIdentifier) {
        const ret = wasm.engine_modelsByVendor(this.__wbg_ptr, vendorIdentifier);
        return ret;
    }
    /**
     * Refresh registries and wait until the catalog has been published to storage.
     * Registries that fail to refresh retain their last known entries.
     * @returns {Array<Model>}
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
        const ret = wasm.engine_removeBackend(this.__wbg_ptr, identifier);
        return ret;
    }
    /**
     * @param {string} registryIdentifier
     * @returns {Promise<void>}
     */
    removeRegistry(registryIdentifier) {
        const ret = wasm.engine_removeRegistry(this.__wbg_ptr, registryIdentifier);
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
     * @param {() => void} callback
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
     * @returns {number}
     */
    __uzuCloneEngineConfig() {
        const ret = wasm.engineconfig___uzuCloneEngineConfig(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get allowLmstudioUsage() {
        const ret = wasm.engineconfig___wasm_get_allow_lmstudio_usage(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get allowOllamaUsage() {
        const ret = wasm.engineconfig___wasm_get_allow_ollama_usage(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get anthropicApiKey() {
        const ret = wasm.engineconfig___wasm_get_anthropic_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get basetenApiKey() {
        const ret = wasm.engineconfig___wasm_get_baseten_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get downloadManagerType() {
        const ret = wasm.engineconfig___wasm_get_download_manager_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get geminiApiKey() {
        const ret = wasm.engineconfig___wasm_get_gemini_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get huggingfaceApiKey() {
        const ret = wasm.engineconfig___wasm_get_huggingface_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get localPath() {
        const ret = wasm.engineconfig___wasm_get_local_path(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get miraiApiKey() {
        const ret = wasm.engineconfig___wasm_get_mirai_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get openaiApiKey() {
        const ret = wasm.engineconfig___wasm_get_openai_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get openrouterApiKey() {
        const ret = wasm.engineconfig___wasm_get_openrouter_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get xaiApiKey() {
        const ret = wasm.engineconfig___wasm_get_xai_api_key(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | undefined | null} miraiApiKey
     * @param {string | undefined | null} localPath
     * @param {string | undefined | null} huggingfaceApiKey
     * @param {string | undefined | null} openaiApiKey
     * @param {string | undefined | null} anthropicApiKey
     * @param {string | undefined | null} geminiApiKey
     * @param {string | undefined | null} xaiApiKey
     * @param {string | undefined | null} basetenApiKey
     * @param {string | undefined | null} openrouterApiKey
     * @param {boolean} allowOllamaUsage
     * @param {boolean} allowLmstudioUsage
     * @param {DownloadManagerType} downloadManagerType
     */
    constructor(miraiApiKey, localPath, huggingfaceApiKey, openaiApiKey, anthropicApiKey, geminiApiKey, xaiApiKey, basetenApiKey, openrouterApiKey, allowOllamaUsage, allowLmstudioUsage, downloadManagerType) {
        const ret = wasm.engineconfig___wasm_new(miraiApiKey, localPath, huggingfaceApiKey, openaiApiKey, anthropicApiKey, geminiApiKey, xaiApiKey, basetenApiKey, openrouterApiKey, allowOllamaUsage, allowLmstudioUsage, downloadManagerType);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        EngineConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set allowLmstudioUsage(value) {
        wasm.engineconfig___wasm_set_allow_lmstudio_usage(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set allowOllamaUsage(value) {
        wasm.engineconfig___wasm_set_allow_ollama_usage(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set anthropicApiKey(value) {
        wasm.engineconfig___wasm_set_anthropic_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set basetenApiKey(value) {
        wasm.engineconfig___wasm_set_baseten_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set downloadManagerType(value) {
        wasm.engineconfig___wasm_set_download_manager_type(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set geminiApiKey(value) {
        wasm.engineconfig___wasm_set_gemini_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set huggingfaceApiKey(value) {
        wasm.engineconfig___wasm_set_huggingface_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set localPath(value) {
        wasm.engineconfig___wasm_set_local_path(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set miraiApiKey(value) {
        wasm.engineconfig___wasm_set_mirai_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set openaiApiKey(value) {
        wasm.engineconfig___wasm_set_openai_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set openrouterApiKey(value) {
        wasm.engineconfig___wasm_set_openrouter_api_key(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set xaiApiKey(value) {
        wasm.engineconfig___wasm_set_xai_api_key(this.__wbg_ptr, value);
    }
    /**
     * @returns {EngineConfig}
     */
    static create() {
        const ret = wasm.engineconfig_create();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {boolean} allowLmstudioUsage
     * @returns {EngineConfig}
     */
    withAllowLmstudioUsage(allowLmstudioUsage) {
        const ret = wasm.engineconfig_withAllowLmstudioUsage(this.__wbg_ptr, allowLmstudioUsage);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {boolean} allowOllamaUsage
     * @returns {EngineConfig}
     */
    withAllowOllamaUsage(allowOllamaUsage) {
        const ret = wasm.engineconfig_withAllowOllamaUsage(this.__wbg_ptr, allowOllamaUsage);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} anthropicApiKey
     * @returns {EngineConfig}
     */
    withAnthropicApiKey(anthropicApiKey) {
        const ret = wasm.engineconfig_withAnthropicApiKey(this.__wbg_ptr, anthropicApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} basetenApiKey
     * @returns {EngineConfig}
     */
    withBasetenApiKey(basetenApiKey) {
        const ret = wasm.engineconfig_withBasetenApiKey(this.__wbg_ptr, basetenApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {DownloadManagerType} downloadManagerType
     * @returns {EngineConfig}
     */
    withDownloadManagerType(downloadManagerType) {
        const ret = wasm.engineconfig_withDownloadManagerType(this.__wbg_ptr, downloadManagerType);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} geminiApiKey
     * @returns {EngineConfig}
     */
    withGeminiApiKey(geminiApiKey) {
        const ret = wasm.engineconfig_withGeminiApiKey(this.__wbg_ptr, geminiApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} huggingfaceApiKey
     * @returns {EngineConfig}
     */
    withHuggingfaceApiKey(huggingfaceApiKey) {
        const ret = wasm.engineconfig_withHuggingfaceApiKey(this.__wbg_ptr, huggingfaceApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} localPath
     * @returns {EngineConfig}
     */
    withLocalPath(localPath) {
        const ret = wasm.engineconfig_withLocalPath(this.__wbg_ptr, localPath);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} miraiApiKey
     * @returns {EngineConfig}
     */
    withMiraiApiKey(miraiApiKey) {
        const ret = wasm.engineconfig_withMiraiApiKey(this.__wbg_ptr, miraiApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} openaiApiKey
     * @returns {EngineConfig}
     */
    withOpenaiApiKey(openaiApiKey) {
        const ret = wasm.engineconfig_withOpenaiApiKey(this.__wbg_ptr, openaiApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} openrouterApiKey
     * @returns {EngineConfig}
     */
    withOpenrouterApiKey(openrouterApiKey) {
        const ret = wasm.engineconfig_withOpenrouterApiKey(this.__wbg_ptr, openrouterApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} xaiApiKey
     * @returns {EngineConfig}
     */
    withXaiApiKey(xaiApiKey) {
        const ret = wasm.engineconfig_withXaiApiKey(this.__wbg_ptr, xaiApiKey);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
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
     * @returns {number}
     */
    __uzuCloneFeature() {
        const ret = wasm.feature___uzuCloneFeature(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.feature___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get values() {
        const ret = wasm.feature___wasm_get_values(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} name
     * @param {Array<string>} values
     */
    constructor(name, values) {
        const ret = wasm.feature___wasm_new(name, values);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        FeatureFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.feature___wasm_set_name(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set values(value) {
        wasm.feature___wasm_set_values(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) Feature.prototype[Symbol.dispose] = Feature.prototype.free;

export class File {
    static __wrap(ptr) {
        const obj = Object.create(File.prototype);
        obj.__wbg_ptr = ptr;
        FileFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
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
     * @returns {number}
     */
    __uzuCloneFile() {
        const ret = wasm.file___uzuCloneFile(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string | null}
     */
    get crc32C() {
        const ret = wasm.file___wasm_crc32c(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {any}
     */
    get hashes() {
        const ret = wasm.file___wasm_get_hashes(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.file___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get size() {
        const ret = wasm.file___wasm_get_size(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.file___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} url
     * @param {string} name
     * @param {number} size
     * @param {Array<Hash>} hashes
     */
    constructor(url, name, size, hashes) {
        const ret = wasm.file___wasm_new(url, name, size, hashes);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        FileFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set hashes(value) {
        wasm.file___wasm_set_hashes(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.file___wasm_set_name(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set size(value) {
        wasm.file___wasm_set_size(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.file___wasm_set_url(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) File.prototype[Symbol.dispose] = File.prototype.free;

export class GrammarJsonAny {
    static __wrap(ptr) {
        const obj = Object.create(GrammarJsonAny.prototype);
        obj.__wbg_ptr = ptr;
        GrammarJsonAnyFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GrammarJsonAnyFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_grammarjsonany_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneGrammar() {
        const ret = wasm.grammarjsonany___uzuCloneGrammar(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.grammarjsonany___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        GrammarJsonAnyFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) GrammarJsonAny.prototype[Symbol.dispose] = GrammarJsonAny.prototype.free;

export class GrammarJsonSchema {
    static __wrap(ptr) {
        const obj = Object.create(GrammarJsonSchema.prototype);
        obj.__wbg_ptr = ptr;
        GrammarJsonSchemaFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GrammarJsonSchemaFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_grammarjsonschema_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneGrammar() {
        const ret = wasm.grammarjsonschema___uzuCloneGrammar(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get schema() {
        const ret = wasm.grammarjsonschema___wasm_get_schema(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} schema
     */
    constructor(schema) {
        const ret = wasm.grammarjsonschema___wasm_new(schema);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        GrammarJsonSchemaFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set schema(value) {
        wasm.grammarjsonschema___wasm_set_schema(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) GrammarJsonSchema.prototype[Symbol.dispose] = GrammarJsonSchema.prototype.free;

export class GrammarRegex {
    static __wrap(ptr) {
        const obj = Object.create(GrammarRegex.prototype);
        obj.__wbg_ptr = ptr;
        GrammarRegexFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GrammarRegexFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_grammarregex_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneGrammar() {
        const ret = wasm.grammarregex___uzuCloneGrammar(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get pattern() {
        const ret = wasm.grammarregex___wasm_get_pattern(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} pattern
     */
    constructor(pattern) {
        const ret = wasm.grammarregex___wasm_new(pattern);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        GrammarRegexFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set pattern(value) {
        wasm.grammarregex___wasm_set_pattern(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) GrammarRegex.prototype[Symbol.dispose] = GrammarRegex.prototype.free;

export class Hash {
    static __wrap(ptr) {
        const obj = Object.create(Hash.prototype);
        obj.__wbg_ptr = ptr;
        HashFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
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
     * @returns {number}
     */
    __uzuCloneHash() {
        const ret = wasm.hash___uzuCloneHash(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get method() {
        const ret = wasm.hash___wasm_get_method(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.hash___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {HashMethod} method
     * @param {string} value
     */
    constructor(method, value) {
        const ret = wasm.hash___wasm_new(method, value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        HashFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set method(value) {
        wasm.hash___wasm_set_method(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.hash___wasm_set_value(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) Hash.prototype[Symbol.dispose] = Hash.prototype.free;

export class HashMethod {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        HashMethodFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_hashmethod_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get CRC32C() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.hashmethod___wasm_variant_CRC32C();
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
    static get GitBlobSha1() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.hashmethod___wasm_variant_GitBlobSha1();
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
    static get Sha256() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.hashmethod___wasm_variant_Sha256();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) HashMethod.prototype[Symbol.dispose] = HashMethod.prototype.free;

export class Image {
    static __wrap(ptr) {
        const obj = Object.create(Image.prototype);
        obj.__wbg_ptr = ptr;
        ImageFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
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
     * @returns {number}
     */
    __uzuCloneImage() {
        const ret = wasm.image___uzuCloneImage(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get description() {
        const ret = wasm.image___wasm_get_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get format() {
        const ret = wasm.image___wasm_get_format(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get height() {
        const ret = wasm.image___wasm_get_height(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get id() {
        const ret = wasm.image___wasm_get_id(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get theme() {
        const ret = wasm.image___wasm_get_theme(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.image___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get width() {
        const ret = wasm.image___wasm_get_width(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} id
     * @param {string} url
     * @param {ImageTheme} theme
     * @param {ImageFormat} format
     * @param {string | null} [description]
     * @param {number | null} [width]
     * @param {number | null} [height]
     */
    constructor(id, url, theme, format, description, width, height) {
        const ret = wasm.image___wasm_new(id, url, theme, format, description, width, height);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ImageFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set description(value) {
        wasm.image___wasm_set_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set format(value) {
        wasm.image___wasm_set_format(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set height(value) {
        wasm.image___wasm_set_height(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set id(value) {
        wasm.image___wasm_set_id(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set theme(value) {
        wasm.image___wasm_set_theme(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.image___wasm_set_url(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set width(value) {
        wasm.image___wasm_set_width(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) Image.prototype[Symbol.dispose] = Image.prototype.free;

export class ImageFormat {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ImageFormatFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_imageformat_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Png() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.imageformat___wasm_variant_Png();
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
    static get Svg() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.imageformat___wasm_variant_Svg();
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
    static get Webp() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.imageformat___wasm_variant_Webp();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ImageFormat.prototype[Symbol.dispose] = ImageFormat.prototype.free;

export class ImageTheme {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ImageThemeFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_imagetheme_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Dark() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.imagetheme___wasm_variant_Dark();
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
    static get Light() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.imagetheme___wasm_variant_Light();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ImageTheme.prototype[Symbol.dispose] = ImageTheme.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneMetadata() {
        const ret = wasm.metadata___uzuCloneMetadata(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get description() {
        const ret = wasm.metadata___wasm_get_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get icons() {
        const ret = wasm.metadata___wasm_get_icons(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.metadata___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.metadata___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} identifier
     * @param {string} name
     * @param {string | undefined | null} description
     * @param {Array<Image>} icons
     */
    constructor(identifier, name, description, icons) {
        const ret = wasm.metadata___wasm_new(identifier, name, description, icons);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        MetadataFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set description(value) {
        wasm.metadata___wasm_set_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set icons(value) {
        wasm.metadata___wasm_set_icons(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.metadata___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.metadata___wasm_set_name(this.__wbg_ptr, value);
    }
    /**
     * @param {string} name
     * @returns {Metadata}
     */
    static external(name) {
        const ret = wasm.metadata_external(name);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
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
     * @returns {number}
     */
    __uzuCloneModel() {
        const ret = wasm.model___uzuCloneModel(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string}
     */
    get cacheIdentifier() {
        const ret = wasm.model___wasm_cache_identifier(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {string | null}
     */
    get checkpointVersion() {
        const ret = wasm.model___wasm_checkpoint_version(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {string | null}
     */
    get filesystemPath() {
        const ret = wasm.model___wasm_filesystem_path(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {any}
     */
    get accessibility() {
        const ret = wasm.model___wasm_get_accessibility(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get backends() {
        const ret = wasm.model___wasm_get_backends(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get encoding() {
        const ret = wasm.model___wasm_get_encoding(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get family() {
        const ret = wasm.model___wasm_get_family(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.model___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get properties() {
        const ret = wasm.model___wasm_get_properties(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get quantization() {
        const ret = wasm.model___wasm_get_quantization(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get registry() {
        const ret = wasm.model___wasm_get_registry(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get specializations() {
        const ret = wasm.model___wasm_get_specializations(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {boolean}
     */
    get isChatCapable() {
        const ret = wasm.model___wasm_is_chat_capable(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isClassificationCapable() {
        const ret = wasm.model___wasm_is_classification_capable(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isDownloadable() {
        const ret = wasm.model___wasm_is_downloadable(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isOnDevice() {
        const ret = wasm.model___wasm_is_on_device(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isQuantized() {
        const ret = wasm.model___wasm_is_quantized(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isRemote() {
        const ret = wasm.model___wasm_is_remote(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isSpeculationCapable() {
        const ret = wasm.model___wasm_is_speculation_capable(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isTextToSpeechCapable() {
        const ret = wasm.model___wasm_is_text_to_speech_capable(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    get isTranslationCapable() {
        const ret = wasm.model___wasm_is_translation_capable(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.model___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ModelIdentifier} identifier
     * @param {ModelRegistry} registry
     * @param {Array<ModelBackend>} backends
     * @param {ModelFamily | undefined | null} family
     * @param {ModelProperties | undefined | null} properties
     * @param {ModelQuantization | undefined | null} quantization
     * @param {Array<ModelSpecialization>} specializations
     * @param {ModelAccessibility} accessibility
     * @param {Value | null} [encoding]
     */
    constructor(identifier, registry, backends, family, properties, quantization, specializations, accessibility, encoding) {
        const ret = wasm.model___wasm_new(identifier, registry, backends, family, properties, quantization, specializations, accessibility, encoding);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Array<string>}
     */
    get repoIds() {
        const ret = wasm.model___wasm_repo_ids(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {any} value
     */
    set accessibility(value) {
        wasm.model___wasm_set_accessibility(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set backends(value) {
        wasm.model___wasm_set_backends(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set encoding(value) {
        wasm.model___wasm_set_encoding(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set family(value) {
        wasm.model___wasm_set_family(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.model___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set properties(value) {
        wasm.model___wasm_set_properties(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set quantization(value) {
        wasm.model___wasm_set_quantization(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set registry(value) {
        wasm.model___wasm_set_registry(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set specializations(value) {
        wasm.model___wasm_set_specializations(this.__wbg_ptr, value);
    }
    /**
     * @param {ModelIdentifier} identifier
     * @param {string} registryIdentifier
     * @param {string} registryName
     * @param {string} backendIdentifier
     * @param {string} backendName
     * @param {string} backendVersion
     * @param {Array<ModelSpecialization>} specializations
     * @param {ModelAccessibility} accessibility
     * @param {Value | null} [encoding]
     * @returns {Model}
     */
    static external(identifier, registryIdentifier, registryName, backendIdentifier, backendName, backendVersion, specializations, accessibility, encoding) {
        const ret = wasm.model_external(identifier, registryIdentifier, registryName, backendIdentifier, backendName, backendVersion, specializations, accessibility, encoding);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) Model.prototype[Symbol.dispose] = Model.prototype.free;

export class ModelAccessibilityOnDevice {
    static __wrap(ptr) {
        const obj = Object.create(ModelAccessibilityOnDevice.prototype);
        obj.__wbg_ptr = ptr;
        ModelAccessibilityOnDeviceFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelAccessibilityOnDeviceFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelaccessibilityondevice_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelAccessibility() {
        const ret = wasm.modelaccessibilityondevice___uzuCloneModelAccessibility(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get source() {
        const ret = wasm.modelaccessibilityondevice___wasm_get_source(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ModelSource} source
     */
    constructor(source) {
        const ret = wasm.modelaccessibilityondevice___wasm_new(source);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelAccessibilityOnDeviceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set source(value) {
        wasm.modelaccessibilityondevice___wasm_set_source(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ModelAccessibilityOnDevice.prototype[Symbol.dispose] = ModelAccessibilityOnDevice.prototype.free;

export class ModelAccessibilityRemote {
    static __wrap(ptr) {
        const obj = Object.create(ModelAccessibilityRemote.prototype);
        obj.__wbg_ptr = ptr;
        ModelAccessibilityRemoteFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelAccessibilityRemoteFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelaccessibilityremote_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelAccessibility() {
        const ret = wasm.modelaccessibilityremote___uzuCloneModelAccessibility(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get repository() {
        const ret = wasm.modelaccessibilityremote___wasm_get_repository(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Repository | null} [repository]
     */
    constructor(repository) {
        const ret = wasm.modelaccessibilityremote___wasm_new(repository);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelAccessibilityRemoteFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set repository(value) {
        wasm.modelaccessibilityremote___wasm_set_repository(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ModelAccessibilityRemote.prototype[Symbol.dispose] = ModelAccessibilityRemote.prototype.free;

export class ModelBackend {
    static __wrap(ptr) {
        const obj = Object.create(ModelBackend.prototype);
        obj.__wbg_ptr = ptr;
        ModelBackendFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
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
     * @returns {number}
     */
    __uzuCloneModelBackend() {
        const ret = wasm.modelbackend___uzuCloneModelBackend(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.modelbackend___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.modelbackend___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get version() {
        const ret = wasm.modelbackend___wasm_get_version(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.modelbackend___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} identifier
     * @param {string} version
     * @param {Metadata} metadata
     */
    constructor(identifier, version, metadata) {
        const ret = wasm.modelbackend___wasm_new(identifier, version, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelBackendFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.modelbackend___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.modelbackend___wasm_set_metadata(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set version(value) {
        wasm.modelbackend___wasm_set_version(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneModelFamily() {
        const ret = wasm.modelfamily___uzuCloneModelFamily(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.modelfamily___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.modelfamily___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get vendor() {
        const ret = wasm.modelfamily___wasm_get_vendor(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.modelfamily___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} identifier
     * @param {ModelVendor} vendor
     * @param {Metadata} metadata
     */
    constructor(identifier, vendor, metadata) {
        const ret = wasm.modelfamily___wasm_new(identifier, vendor, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelFamilyFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.modelfamily___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.modelfamily___wasm_set_metadata(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set vendor(value) {
        wasm.modelfamily___wasm_set_vendor(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneModelProperties() {
        const ret = wasm.modelproperties___uzuCloneModelProperties(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.modelproperties___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.modelproperties___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get size() {
        const ret = wasm.modelproperties___wasm_get_size(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get version() {
        const ret = wasm.modelproperties___wasm_get_version(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.modelproperties___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} identifier
     * @param {number} size
     * @param {string | undefined | null} version
     * @param {Metadata} metadata
     */
    constructor(identifier, size, version, metadata) {
        const ret = wasm.modelproperties___wasm_new(identifier, size, version, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelPropertiesFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.modelproperties___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.modelproperties___wasm_set_metadata(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set size(value) {
        wasm.modelproperties___wasm_set_size(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set version(value) {
        wasm.modelproperties___wasm_set_version(this.__wbg_ptr, value);
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
    __uzuCloneModelQuantization() {
        const ret = wasm.modelquantization___uzuCloneModelQuantization(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get bitsPerWeight() {
        const ret = wasm.modelquantization___wasm_get_bits_per_weight(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.modelquantization___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.modelquantization___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get method() {
        const ret = wasm.modelquantization___wasm_get_method(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get vendor() {
        const ret = wasm.modelquantization___wasm_get_vendor(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.modelquantization___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} identifier
     * @param {string} method
     * @param {number} bitsPerWeight
     * @param {ModelVendor} vendor
     * @param {Metadata} metadata
     */
    constructor(identifier, method, bitsPerWeight, vendor, metadata) {
        const ret = wasm.modelquantization___wasm_new(identifier, method, bitsPerWeight, vendor, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelQuantizationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set bitsPerWeight(value) {
        wasm.modelquantization___wasm_set_bits_per_weight(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.modelquantization___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.modelquantization___wasm_set_metadata(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set method(value) {
        wasm.modelquantization___wasm_set_method(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set vendor(value) {
        wasm.modelquantization___wasm_set_vendor(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneModelRegistry() {
        const ret = wasm.modelregistry___uzuCloneModelRegistry(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.modelregistry___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.modelregistry___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.modelregistry___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} identifier
     * @param {Metadata} metadata
     */
    constructor(identifier, metadata) {
        const ret = wasm.modelregistry___wasm_new(identifier, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelRegistryFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.modelregistry___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.modelregistry___wasm_set_metadata(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ModelRegistry.prototype[Symbol.dispose] = ModelRegistry.prototype.free;

export class ModelSourceFilesystem {
    static __wrap(ptr) {
        const obj = Object.create(ModelSourceFilesystem.prototype);
        obj.__wbg_ptr = ptr;
        ModelSourceFilesystemFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSourceFilesystemFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelsourcefilesystem_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSource() {
        const ret = wasm.modelsourcefilesystem___uzuCloneModelSource(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get path() {
        const ret = wasm.modelsourcefilesystem___wasm_get_path(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} path
     */
    constructor(path) {
        const ret = wasm.modelsourcefilesystem___wasm_new(path);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSourceFilesystemFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set path(value) {
        wasm.modelsourcefilesystem___wasm_set_path(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ModelSourceFilesystem.prototype[Symbol.dispose] = ModelSourceFilesystem.prototype.free;

export class ModelSourceRegistry {
    static __wrap(ptr) {
        const obj = Object.create(ModelSourceRegistry.prototype);
        obj.__wbg_ptr = ptr;
        ModelSourceRegistryFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSourceRegistryFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelsourceregistry_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSource() {
        const ret = wasm.modelsourceregistry___uzuCloneModelSource(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get files() {
        const ret = wasm.modelsourceregistry___wasm_get_files(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get repository() {
        const ret = wasm.modelsourceregistry___wasm_get_repository(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get sourceRepository() {
        const ret = wasm.modelsourceregistry___wasm_get_source_repository(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get toolchainVersion() {
        const ret = wasm.modelsourceregistry___wasm_get_toolchain_version(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} toolchainVersion
     * @param {Repository | undefined | null} repository
     * @param {Repository | undefined | null} sourceRepository
     * @param {Array<File>} files
     */
    constructor(toolchainVersion, repository, sourceRepository, files) {
        const ret = wasm.modelsourceregistry___wasm_new(toolchainVersion, repository, sourceRepository, files);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSourceRegistryFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set files(value) {
        wasm.modelsourceregistry___wasm_set_files(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set repository(value) {
        wasm.modelsourceregistry___wasm_set_repository(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set sourceRepository(value) {
        wasm.modelsourceregistry___wasm_set_source_repository(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set toolchainVersion(value) {
        wasm.modelsourceregistry___wasm_set_toolchain_version(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ModelSourceRegistry.prototype[Symbol.dispose] = ModelSourceRegistry.prototype.free;

export class ModelSpecializationChat {
    static __wrap(ptr) {
        const obj = Object.create(ModelSpecializationChat.prototype);
        obj.__wbg_ptr = ptr;
        ModelSpecializationChatFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSpecializationChatFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelspecializationchat_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSpecialization() {
        const ret = wasm.modelspecializationchat___uzuCloneModelSpecialization(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.modelspecializationchat___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSpecializationChatFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ModelSpecializationChat.prototype[Symbol.dispose] = ModelSpecializationChat.prototype.free;

export class ModelSpecializationClassification {
    static __wrap(ptr) {
        const obj = Object.create(ModelSpecializationClassification.prototype);
        obj.__wbg_ptr = ptr;
        ModelSpecializationClassificationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSpecializationClassificationFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelspecializationclassification_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSpecialization() {
        const ret = wasm.modelspecializationclassification___uzuCloneModelSpecialization(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.modelspecializationclassification___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSpecializationClassificationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ModelSpecializationClassification.prototype[Symbol.dispose] = ModelSpecializationClassification.prototype.free;

export class ModelSpecializationDecision {
    static __wrap(ptr) {
        const obj = Object.create(ModelSpecializationDecision.prototype);
        obj.__wbg_ptr = ptr;
        ModelSpecializationDecisionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSpecializationDecisionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelspecializationdecision_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSpecialization() {
        const ret = wasm.modelspecializationdecision___uzuCloneModelSpecialization(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.modelspecializationdecision___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSpecializationDecisionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ModelSpecializationDecision.prototype[Symbol.dispose] = ModelSpecializationDecision.prototype.free;

export class ModelSpecializationSpeculation {
    static __wrap(ptr) {
        const obj = Object.create(ModelSpecializationSpeculation.prototype);
        obj.__wbg_ptr = ptr;
        ModelSpecializationSpeculationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSpecializationSpeculationFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelspecializationspeculation_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSpecialization() {
        const ret = wasm.modelspecializationspeculation___uzuCloneModelSpecialization(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.modelspecializationspeculation___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSpecializationSpeculationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ModelSpecializationSpeculation.prototype[Symbol.dispose] = ModelSpecializationSpeculation.prototype.free;

export class ModelSpecializationTextToSpeech {
    static __wrap(ptr) {
        const obj = Object.create(ModelSpecializationTextToSpeech.prototype);
        obj.__wbg_ptr = ptr;
        ModelSpecializationTextToSpeechFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSpecializationTextToSpeechFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelspecializationtexttospeech_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSpecialization() {
        const ret = wasm.modelspecializationtexttospeech___uzuCloneModelSpecialization(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.modelspecializationtexttospeech___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSpecializationTextToSpeechFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ModelSpecializationTextToSpeech.prototype[Symbol.dispose] = ModelSpecializationTextToSpeech.prototype.free;

export class ModelSpecializationTranslation {
    static __wrap(ptr) {
        const obj = Object.create(ModelSpecializationTranslation.prototype);
        obj.__wbg_ptr = ptr;
        ModelSpecializationTranslationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelSpecializationTranslationFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelspecializationtranslation_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneModelSpecialization() {
        const ret = wasm.modelspecializationtranslation___uzuCloneModelSpecialization(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.modelspecializationtranslation___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelSpecializationTranslationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) ModelSpecializationTranslation.prototype[Symbol.dispose] = ModelSpecializationTranslation.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneModelVendor() {
        const ret = wasm.modelvendor___uzuCloneModelVendor(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.modelvendor___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get metadata() {
        const ret = wasm.modelvendor___wasm_get_metadata(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {string}
     */
    get name() {
        const ret = wasm.modelvendor___wasm_name(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} identifier
     * @param {Metadata} metadata
     */
    constructor(identifier, metadata) {
        const ret = wasm.modelvendor___wasm_new(identifier, metadata);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelVendorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.modelvendor___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set metadata(value) {
        wasm.modelvendor___wasm_set_metadata(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ModelVendor.prototype[Symbol.dispose] = ModelVendor.prototype.free;

export class NativeTool {
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
     * @returns {number}
     */
    __uzuCloneNativeTool() {
        const ret = wasm.nativetool___uzuCloneNativeTool(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The callback receives the JSON arguments and an invocation ID, and returns JSON or a promise
     * of JSON. `cancel` receives the invocation ID when the call is cancelled.
     * @param {ToolFunction} definition
     * @param {(arg0: string, arg1: string) => Promise<string>} invoke_json
     * @param {(arg: string) => void} cancel
     */
    constructor(definition, invoke_json, cancel) {
        const ret = wasm.nativetool_new(definition, invoke_json, cancel);
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
    __uzuClonePcmBatch() {
        const ret = wasm.pcmbatch___uzuClonePcmBatch(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get batchSize() {
        const ret = wasm.pcmbatch___wasm_batch_size(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {number}
     */
    get duration() {
        const ret = wasm.pcmbatch___wasm_duration(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {any}
     */
    get channels() {
        const ret = wasm.pcmbatch___wasm_get_channels(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get lengths() {
        const ret = wasm.pcmbatch___wasm_get_lengths(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get sampleRate() {
        const ret = wasm.pcmbatch___wasm_get_sample_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get samples() {
        const ret = wasm.pcmbatch___wasm_get_samples(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Array<number>} samples
     * @param {number} sampleRate
     * @param {number} channels
     * @param {Array<number>} lengths
     */
    constructor(samples, sampleRate, channels, lengths) {
        const ret = wasm.pcmbatch___wasm_new(samples, sampleRate, channels, lengths);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        PcmBatchFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set channels(value) {
        wasm.pcmbatch___wasm_set_channels(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set lengths(value) {
        wasm.pcmbatch___wasm_set_lengths(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set sampleRate(value) {
        wasm.pcmbatch___wasm_set_sample_rate(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set samples(value) {
        wasm.pcmbatch___wasm_set_samples(this.__wbg_ptr, value);
    }
    /**
     * @returns {number}
     */
    get totalFrames() {
        const ret = wasm.pcmbatch___wasm_total_frames(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} path
     */
    saveAsWav(path) {
        const ret = wasm.pcmbatch_saveAsWav(this.__wbg_ptr, path);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) PcmBatch.prototype[Symbol.dispose] = PcmBatch.prototype.free;

export class ReasoningEffort {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ReasoningEffortFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_reasoningeffort_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    static get Default() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.reasoningeffort___wasm_variant_Default();
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
    static get Disabled() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.reasoningeffort___wasm_variant_Disabled();
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
    static get High() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.reasoningeffort___wasm_variant_High();
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
    static get Low() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.reasoningeffort___wasm_variant_Low();
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
    static get Medium() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.reasoningeffort___wasm_variant_Medium();
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
    static get XHigh() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.reasoningeffort___wasm_variant_XHigh();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) ReasoningEffort.prototype[Symbol.dispose] = ReasoningEffort.prototype.free;

export class ReplyEnergyComponents {
    static __wrap(ptr) {
        const obj = Object.create(ReplyEnergyComponents.prototype);
        obj.__wbg_ptr = ptr;
        ReplyEnergyComponentsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ReplyEnergyComponentsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_replyenergycomponents_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneReplyEnergy() {
        const ret = wasm.replyenergycomponents___uzuCloneReplyEnergy(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get ane() {
        const ret = wasm.replyenergycomponents___wasm_get_ane(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get cpu() {
        const ret = wasm.replyenergycomponents___wasm_get_cpu(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get dram() {
        const ret = wasm.replyenergycomponents___wasm_get_dram(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get gpu() {
        const ret = wasm.replyenergycomponents___wasm_get_gpu(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} cpu
     * @param {number} gpu
     * @param {number} ane
     * @param {number} dram
     */
    constructor(cpu, gpu, ane, dram) {
        const ret = wasm.replyenergycomponents___wasm_new(cpu, gpu, ane, dram);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ReplyEnergyComponentsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set ane(value) {
        wasm.replyenergycomponents___wasm_set_ane(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set cpu(value) {
        wasm.replyenergycomponents___wasm_set_cpu(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set dram(value) {
        wasm.replyenergycomponents___wasm_set_dram(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set gpu(value) {
        wasm.replyenergycomponents___wasm_set_gpu(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ReplyEnergyComponents.prototype[Symbol.dispose] = ReplyEnergyComponents.prototype.free;

export class ReplyEnergyTotal {
    static __wrap(ptr) {
        const obj = Object.create(ReplyEnergyTotal.prototype);
        obj.__wbg_ptr = ptr;
        ReplyEnergyTotalFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ReplyEnergyTotalFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_replyenergytotal_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneReplyEnergy() {
        const ret = wasm.replyenergytotal___uzuCloneReplyEnergy(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get total() {
        const ret = wasm.replyenergytotal___wasm_get_total(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} total
     */
    constructor(total) {
        const ret = wasm.replyenergytotal___wasm_new(total);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ReplyEnergyTotalFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set total(value) {
        wasm.replyenergytotal___wasm_set_total(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ReplyEnergyTotal.prototype[Symbol.dispose] = ReplyEnergyTotal.prototype.free;

export class Repository {
    static __wrap(ptr) {
        const obj = Object.create(Repository.prototype);
        obj.__wbg_ptr = ptr;
        RepositoryFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
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
     * @returns {number}
     */
    __uzuCloneRepository() {
        const ret = wasm.repository___uzuCloneRepository(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get commitHash() {
        const ret = wasm.repository___wasm_get_commit_hash(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.repository___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get paths() {
        const ret = wasm.repository___wasm_get_paths(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} identifier
     * @param {string | null} [commitHash]
     * @param {Array<string> | null} [paths]
     */
    constructor(identifier, commitHash, paths) {
        const ret = wasm.repository___wasm_new(identifier, commitHash, paths);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        RepositoryFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set commitHash(value) {
        wasm.repository___wasm_set_commit_hash(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.repository___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set paths(value) {
        wasm.repository___wasm_set_paths(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) Repository.prototype[Symbol.dispose] = Repository.prototype.free;

export class SamplingMethodGreedy {
    static __wrap(ptr) {
        const obj = Object.create(SamplingMethodGreedy.prototype);
        obj.__wbg_ptr = ptr;
        SamplingMethodGreedyFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingMethodGreedyFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingmethodgreedy_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneSamplingMethod() {
        const ret = wasm.samplingmethodgreedy___uzuCloneSamplingMethod(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.samplingmethodgreedy___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingMethodGreedyFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) SamplingMethodGreedy.prototype[Symbol.dispose] = SamplingMethodGreedy.prototype.free;

export class SamplingMethodStochastic {
    static __wrap(ptr) {
        const obj = Object.create(SamplingMethodStochastic.prototype);
        obj.__wbg_ptr = ptr;
        SamplingMethodStochasticFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingMethodStochasticFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingmethodstochastic_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneSamplingMethod() {
        const ret = wasm.samplingmethodstochastic___uzuCloneSamplingMethod(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get minP() {
        const ret = wasm.samplingmethodstochastic___wasm_get_min_p(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get repetitionPenalty() {
        const ret = wasm.samplingmethodstochastic___wasm_get_repetition_penalty(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get suffixRepetitionLength() {
        const ret = wasm.samplingmethodstochastic___wasm_get_suffix_repetition_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get temperature() {
        const ret = wasm.samplingmethodstochastic___wasm_get_temperature(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get topK() {
        const ret = wasm.samplingmethodstochastic___wasm_get_top_k(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get topP() {
        const ret = wasm.samplingmethodstochastic___wasm_get_top_p(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number | null} [temperature]
     * @param {number | null} [topK]
     * @param {number | null} [topP]
     * @param {number | null} [minP]
     * @param {number | null} [repetitionPenalty]
     * @param {number | null} [suffixRepetitionLength]
     */
    constructor(temperature, topK, topP, minP, repetitionPenalty, suffixRepetitionLength) {
        const ret = wasm.samplingmethodstochastic___wasm_new(temperature, topK, topP, minP, repetitionPenalty, suffixRepetitionLength);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingMethodStochasticFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set minP(value) {
        wasm.samplingmethodstochastic___wasm_set_min_p(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set repetitionPenalty(value) {
        wasm.samplingmethodstochastic___wasm_set_repetition_penalty(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set suffixRepetitionLength(value) {
        wasm.samplingmethodstochastic___wasm_set_suffix_repetition_length(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set temperature(value) {
        wasm.samplingmethodstochastic___wasm_set_temperature(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set topK(value) {
        wasm.samplingmethodstochastic___wasm_set_top_k(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set topP(value) {
        wasm.samplingmethodstochastic___wasm_set_top_p(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) SamplingMethodStochastic.prototype[Symbol.dispose] = SamplingMethodStochastic.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneSamplingParameters() {
        const ret = wasm.samplingparameters___uzuCloneSamplingParameters(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get minP() {
        const ret = wasm.samplingparameters___wasm_get_min_p(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get repetitionPenalty() {
        const ret = wasm.samplingparameters___wasm_get_repetition_penalty(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get suffixRepetitionLength() {
        const ret = wasm.samplingparameters___wasm_get_suffix_repetition_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get temperature() {
        const ret = wasm.samplingparameters___wasm_get_temperature(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get topK() {
        const ret = wasm.samplingparameters___wasm_get_top_k(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get topP() {
        const ret = wasm.samplingparameters___wasm_get_top_p(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number | null} [temperature]
     * @param {number | null} [topK]
     * @param {number | null} [topP]
     * @param {number | null} [minP]
     * @param {number | null} [repetitionPenalty]
     * @param {number | null} [suffixRepetitionLength]
     */
    constructor(temperature, topK, topP, minP, repetitionPenalty, suffixRepetitionLength) {
        const ret = wasm.samplingparameters___wasm_new(temperature, topK, topP, minP, repetitionPenalty, suffixRepetitionLength);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingParametersFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set minP(value) {
        wasm.samplingparameters___wasm_set_min_p(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set repetitionPenalty(value) {
        wasm.samplingparameters___wasm_set_repetition_penalty(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set suffixRepetitionLength(value) {
        wasm.samplingparameters___wasm_set_suffix_repetition_length(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set temperature(value) {
        wasm.samplingparameters___wasm_set_temperature(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set topK(value) {
        wasm.samplingparameters___wasm_set_top_k(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set topP(value) {
        wasm.samplingparameters___wasm_set_top_p(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) SamplingParameters.prototype[Symbol.dispose] = SamplingParameters.prototype.free;

export class SamplingPolicyCustom {
    static __wrap(ptr) {
        const obj = Object.create(SamplingPolicyCustom.prototype);
        obj.__wbg_ptr = ptr;
        SamplingPolicyCustomFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingPolicyCustomFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingpolicycustom_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneSamplingPolicy() {
        const ret = wasm.samplingpolicycustom___uzuCloneSamplingPolicy(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get method() {
        const ret = wasm.samplingpolicycustom___wasm_get_method(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {SamplingMethod} method
     */
    constructor(method) {
        const ret = wasm.samplingpolicycustom___wasm_new(method);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingPolicyCustomFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set method(value) {
        wasm.samplingpolicycustom___wasm_set_method(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) SamplingPolicyCustom.prototype[Symbol.dispose] = SamplingPolicyCustom.prototype.free;

export class SamplingPolicyDefault {
    static __wrap(ptr) {
        const obj = Object.create(SamplingPolicyDefault.prototype);
        obj.__wbg_ptr = ptr;
        SamplingPolicyDefaultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingPolicyDefaultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingpolicydefault_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneSamplingPolicy() {
        const ret = wasm.samplingpolicydefault___uzuCloneSamplingPolicy(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.samplingpolicydefault___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingPolicyDefaultFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) SamplingPolicyDefault.prototype[Symbol.dispose] = SamplingPolicyDefault.prototype.free;

export class SamplingSeedCustom {
    static __wrap(ptr) {
        const obj = Object.create(SamplingSeedCustom.prototype);
        obj.__wbg_ptr = ptr;
        SamplingSeedCustomFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingSeedCustomFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingseedcustom_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneSamplingSeed() {
        const ret = wasm.samplingseedcustom___uzuCloneSamplingSeed(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get seed() {
        const ret = wasm.samplingseedcustom___wasm_get_seed(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} seed
     */
    constructor(seed) {
        const ret = wasm.samplingseedcustom___wasm_new(seed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingSeedCustomFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set seed(value) {
        wasm.samplingseedcustom___wasm_set_seed(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) SamplingSeedCustom.prototype[Symbol.dispose] = SamplingSeedCustom.prototype.free;

export class SamplingSeedDefault {
    static __wrap(ptr) {
        const obj = Object.create(SamplingSeedDefault.prototype);
        obj.__wbg_ptr = ptr;
        SamplingSeedDefaultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SamplingSeedDefaultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_samplingseeddefault_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneSamplingSeed() {
        const ret = wasm.samplingseeddefault___uzuCloneSamplingSeed(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.samplingseeddefault___wasm_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        SamplingSeedDefaultFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) SamplingSeedDefault.prototype[Symbol.dispose] = SamplingSeedDefault.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneTextToSpeechOutput() {
        const ret = wasm.texttospeechoutput___uzuCloneTextToSpeechOutput(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get pcmBatch() {
        const ret = wasm.texttospeechoutput___wasm_get_pcm_batch(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get stats() {
        const ret = wasm.texttospeechoutput___wasm_get_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {PcmBatch} pcmBatch
     * @param {TextToSpeechStats} stats
     */
    constructor(pcmBatch, stats) {
        const ret = wasm.texttospeechoutput___wasm_new(pcmBatch, stats);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TextToSpeechOutputFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set pcmBatch(value) {
        wasm.texttospeechoutput___wasm_set_pcm_batch(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set stats(value) {
        wasm.texttospeechoutput___wasm_set_stats(this.__wbg_ptr, value);
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
    __uzuCloneTextToSpeechStats() {
        const ret = wasm.texttospeechstats___uzuCloneTextToSpeechStats(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get audioDuration() {
        const ret = wasm.texttospeechstats___wasm_get_audio_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get firstChunkSeconds() {
        const ret = wasm.texttospeechstats___wasm_get_first_chunk_seconds(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get generationDuration() {
        const ret = wasm.texttospeechstats___wasm_get_generation_duration(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get textLength() {
        const ret = wasm.texttospeechstats___wasm_get_text_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} textLength
     * @param {number} firstChunkSeconds
     * @param {number} generationDuration
     * @param {number} audioDuration
     */
    constructor(textLength, firstChunkSeconds, generationDuration, audioDuration) {
        const ret = wasm.texttospeechstats___wasm_new(textLength, firstChunkSeconds, generationDuration, audioDuration);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TextToSpeechStatsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set audioDuration(value) {
        wasm.texttospeechstats___wasm_set_audio_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set firstChunkSeconds(value) {
        wasm.texttospeechstats___wasm_set_first_chunk_seconds(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set generationDuration(value) {
        wasm.texttospeechstats___wasm_set_generation_duration(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set textLength(value) {
        wasm.texttospeechstats___wasm_set_text_length(this.__wbg_ptr, value);
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
    __uzuCloneToken() {
        const ret = wasm.token___uzuCloneToken(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get id() {
        const ret = wasm.token___wasm_get_id(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get isSpecial() {
        const ret = wasm.token___wasm_get_is_special(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get value() {
        const ret = wasm.token___wasm_get_value(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {TokenId} id
     * @param {TokenValue} value
     * @param {boolean} isSpecial
     */
    constructor(id, value, isSpecial) {
        const ret = wasm.token___wasm_new(id, value, isSpecial);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TokenFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set id(value) {
        wasm.token___wasm_set_id(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set isSpecial(value) {
        wasm.token___wasm_set_is_special(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set value(value) {
        wasm.token___wasm_set_value(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneToolCall() {
        const ret = wasm.toolcall___uzuCloneToolCall(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get arguments() {
        const ret = wasm.toolcall___wasm_get_arguments(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get identifier() {
        const ret = wasm.toolcall___wasm_get_identifier(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.toolcall___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string | undefined | null} identifier
     * @param {string} name
     * @param {Value} _arguments
     */
    constructor(identifier, name, _arguments) {
        const ret = wasm.toolcall___wasm_new(identifier, name, _arguments);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ToolCallFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set arguments(value) {
        wasm.toolcall___wasm_set_arguments(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set identifier(value) {
        wasm.toolcall___wasm_set_identifier(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.toolcall___wasm_set_name(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneToolCallArgumentDelta() {
        const ret = wasm.toolcallargumentdelta___uzuCloneToolCallArgumentDelta(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get arguments() {
        const ret = wasm.toolcallargumentdelta___wasm_get_arguments(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.toolcallargumentdelta___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} name
     * @param {string} _arguments
     */
    constructor(name, _arguments) {
        const ret = wasm.toolcallargumentdelta___wasm_new(name, _arguments);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ToolCallArgumentDeltaFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set arguments(value) {
        wasm.toolcallargumentdelta___wasm_set_arguments(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.toolcallargumentdelta___wasm_set_name(this.__wbg_ptr, value);
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
     * @param {Array<ToolNamespace>} tools
     * @returns {ToolCallArgumentStream}
     */
    static create(tools) {
        const ret = wasm.toolcallargumentstream_create(tools);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {ToolCall} call
     * @returns {ToolCallArgumentDelta}
     */
    finish(call) {
        const ret = wasm.toolcallargumentstream_finish(this.__wbg_ptr, call);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {string} raw
     * @returns {ToolCallArgumentDelta}
     */
    update(raw) {
        const ret = wasm.toolcallargumentstream_update(this.__wbg_ptr, raw);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) ToolCallArgumentStream.prototype[Symbol.dispose] = ToolCallArgumentStream.prototype.free;

export class ToolDescriptionFunction {
    static __wrap(ptr) {
        const obj = Object.create(ToolDescriptionFunction.prototype);
        obj.__wbg_ptr = ptr;
        ToolDescriptionFunctionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ToolDescriptionFunctionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tooldescriptionfunction_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneToolDescription() {
        const ret = wasm.tooldescriptionfunction___uzuCloneToolDescription(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get toolFunction() {
        const ret = wasm.tooldescriptionfunction___wasm_get_tool_function(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {ToolFunction} toolFunction
     */
    constructor(toolFunction) {
        const ret = wasm.tooldescriptionfunction___wasm_new(toolFunction);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ToolDescriptionFunctionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set toolFunction(value) {
        wasm.tooldescriptionfunction___wasm_set_tool_function(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ToolDescriptionFunction.prototype[Symbol.dispose] = ToolDescriptionFunction.prototype.free;

export class ToolFunction {
    static __wrap(ptr) {
        const obj = Object.create(ToolFunction.prototype);
        obj.__wbg_ptr = ptr;
        ToolFunctionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
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
     * @returns {number}
     */
    __uzuCloneToolFunction() {
        const ret = wasm.toolfunction___uzuCloneToolFunction(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get description() {
        const ret = wasm.toolfunction___wasm_get_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.toolfunction___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get parameters() {
        const ret = wasm.toolfunction___wasm_get_parameters(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get returnDefinition() {
        const ret = wasm.toolfunction___wasm_get_return_definition(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} name
     * @param {string} description
     * @param {Value | null} [parameters]
     * @param {Value | null} [returnDefinition]
     */
    constructor(name, description, parameters, returnDefinition) {
        const ret = wasm.toolfunction___wasm_new(name, description, parameters, returnDefinition);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ToolFunctionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set description(value) {
        wasm.toolfunction___wasm_set_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.toolfunction___wasm_set_name(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set parameters(value) {
        wasm.toolfunction___wasm_set_parameters(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set returnDefinition(value) {
        wasm.toolfunction___wasm_set_return_definition(this.__wbg_ptr, value);
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
     * @returns {number}
     */
    __uzuCloneToolNamespace() {
        const ret = wasm.toolnamespace___uzuCloneToolNamespace(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get description() {
        const ret = wasm.toolnamespace___wasm_get_description(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get name() {
        const ret = wasm.toolnamespace___wasm_get_name(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get tools() {
        const ret = wasm.toolnamespace___wasm_get_tools(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} name
     * @param {string | undefined | null} description
     * @param {Array<ToolDescription>} tools
     */
    constructor(name, description, tools) {
        const ret = wasm.toolnamespace___wasm_new(name, description, tools);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ToolNamespaceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set description(value) {
        wasm.toolnamespace___wasm_set_description(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set name(value) {
        wasm.toolnamespace___wasm_set_name(this.__wbg_ptr, value);
    }
    /**
     * @param {any} value
     */
    set tools(value) {
        wasm.toolnamespace___wasm_set_tools(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) ToolNamespace.prototype[Symbol.dispose] = ToolNamespace.prototype.free;

export class TranslationPayloadImage {
    static __wrap(ptr) {
        const obj = Object.create(TranslationPayloadImage.prototype);
        obj.__wbg_ptr = ptr;
        TranslationPayloadImageFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TranslationPayloadImageFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_translationpayloadimage_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneTranslationPayload() {
        const ret = wasm.translationpayloadimage___uzuCloneTranslationPayload(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get url() {
        const ret = wasm.translationpayloadimage___wasm_get_url(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} url
     */
    constructor(url) {
        const ret = wasm.translationpayloadimage___wasm_new(url);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TranslationPayloadImageFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set url(value) {
        wasm.translationpayloadimage___wasm_set_url(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) TranslationPayloadImage.prototype[Symbol.dispose] = TranslationPayloadImage.prototype.free;

export class TranslationPayloadText {
    static __wrap(ptr) {
        const obj = Object.create(TranslationPayloadText.prototype);
        obj.__wbg_ptr = ptr;
        TranslationPayloadTextFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TranslationPayloadTextFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_translationpayloadtext_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    __uzuCloneTranslationPayload() {
        const ret = wasm.translationpayloadtext___uzuCloneTranslationPayload(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get text() {
        const ret = wasm.translationpayloadtext___wasm_get_text(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} text
     */
    constructor(text) {
        const ret = wasm.translationpayloadtext___wasm_new(text);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TranslationPayloadTextFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set text(value) {
        wasm.translationpayloadtext___wasm_set_text(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) TranslationPayloadText.prototype[Symbol.dispose] = TranslationPayloadText.prototype.free;

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
     * @returns {number}
     */
    __uzuCloneValue() {
        const ret = wasm.value___uzuCloneValue(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {any}
     */
    get json() {
        const ret = wasm.value___wasm_get_json(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {string} json
     */
    constructor(json) {
        const ret = wasm.value___wasm_new(json);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ValueFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {any} value
     */
    set json(value) {
        wasm.value___wasm_set_json(this.__wbg_ptr, value);
    }
}
if (Symbol.dispose) Value.prototype[Symbol.dispose] = Value.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Window_a2a6c4d665047b14: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_2664448a7c667d67: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
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
        __wbg___wbindgen_number_get_2e0e7dee9f701a71: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_rethrow_cb2e88c6b2a16733: function(arg0) {
            throw arg0;
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
        __wbg_canceltoken_new: function(arg0) {
            const ret = CancelToken.__wrap(arg0);
            return ret;
        },
        __wbg_catch_92594c2d17f55f12: function(arg0, arg1) {
            const ret = arg0.catch(arg1);
            return ret;
        },
        __wbg_chatconfig_new: function(arg0) {
            const ret = ChatConfig.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockaudio_new: function(arg0) {
            const ret = ChatContentBlockAudio.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockbuiltintools_new: function(arg0) {
            const ret = ChatContentBlockBuiltinTools.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockconversationstartdate_new: function(arg0) {
            const ret = ChatContentBlockConversationStartDate.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockcustom_new: function(arg0) {
            const ret = ChatContentBlockCustom.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockfile_new: function(arg0) {
            const ret = ChatContentBlockFile.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockidentity_new: function(arg0) {
            const ret = ChatContentBlockIdentity.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockimage_new: function(arg0) {
            const ret = ChatContentBlockImage.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockknowledgecutoff_new: function(arg0) {
            const ret = ChatContentBlockKnowledgeCutoff.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockreasoning_new: function(arg0) {
            const ret = ChatContentBlockReasoning.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockreasoningeffort_new: function(arg0) {
            const ret = ChatContentBlockReasoningEffort.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblocktext_new: function(arg0) {
            const ret = ChatContentBlockText.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblocktoolcall_new: function(arg0) {
            const ret = ChatContentBlockToolCall.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblocktoolcallcandidate_new: function(arg0) {
            const ret = ChatContentBlockToolCallCandidate.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblocktoolcallresult_new: function(arg0) {
            const ret = ChatContentBlockToolCallResult.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblocktools_new: function(arg0) {
            const ret = ChatContentBlockTools.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblocktranslation_new: function(arg0) {
            const ret = ChatContentBlockTranslation.__wrap(arg0);
            return ret;
        },
        __wbg_chatcontentblockvideo_new: function(arg0) {
            const ret = ChatContentBlockVideo.__wrap(arg0);
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
        __wbg_chatmodelcapabilities_new: function(arg0) {
            const ret = ChatModelCapabilities.__wrap(arg0);
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
        __wbg_chatreplyconfig_new: function(arg0) {
            const ret = ChatReplyConfig.__wrap(arg0);
            return ret;
        },
        __wbg_chatreplyjoulespertokencomponents_new: function(arg0) {
            const ret = ChatReplyJoulesPerTokenComponents.__wrap(arg0);
            return ret;
        },
        __wbg_chatreplyjoulespertokentotal_new: function(arg0) {
            const ret = ChatReplyJoulesPerTokenTotal.__wrap(arg0);
            return ret;
        },
        __wbg_chatreplyspeculatorstats_new: function(arg0) {
            const ret = ChatReplySpeculatorStats.__wrap(arg0);
            return ret;
        },
        __wbg_chatreplystats_new: function(arg0) {
            const ret = ChatReplyStats.__wrap(arg0);
            return ret;
        },
        __wbg_chatroleassistant_new: function(arg0) {
            const ret = ChatRoleAssistant.__wrap(arg0);
            return ret;
        },
        __wbg_chatrolecustom_new: function(arg0) {
            const ret = ChatRoleCustom.__wrap(arg0);
            return ret;
        },
        __wbg_chatroledeveloper_new: function(arg0) {
            const ret = ChatRoleDeveloper.__wrap(arg0);
            return ret;
        },
        __wbg_chatrolesystem_new: function(arg0) {
            const ret = ChatRoleSystem.__wrap(arg0);
            return ret;
        },
        __wbg_chatroletool_new: function(arg0) {
            const ret = ChatRoleTool.__wrap(arg0);
            return ret;
        },
        __wbg_chatroleuser_new: function(arg0) {
            const ret = ChatRoleUser.__wrap(arg0);
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
        __wbg_chatsessionstreamchunkerror_new: function(arg0) {
            const ret = ChatSessionStreamChunkError.__wrap(arg0);
            return ret;
        },
        __wbg_chatsessionstreamchunkreplies_new: function(arg0) {
            const ret = ChatSessionStreamChunkReplies.__wrap(arg0);
            return ret;
        },
        __wbg_chatsessionstreamchunktoolresults_new: function(arg0) {
            const ret = ChatSessionStreamChunkToolResults.__wrap(arg0);
            return ret;
        },
        __wbg_classificationmessage_new: function(arg0) {
            const ret = ClassificationMessage.__wrap(arg0);
            return ret;
        },
        __wbg_classificationstats_new: function(arg0) {
            const ret = ClassificationStats.__wrap(arg0);
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
        __wbg_contextlengthcustom_new: function(arg0) {
            const ret = ContextLengthCustom.__wrap(arg0);
            return ret;
        },
        __wbg_contextlengthdefault_new: function(arg0) {
            const ret = ContextLengthDefault.__wrap(arg0);
            return ret;
        },
        __wbg_contextlengthmaximal_new: function(arg0) {
            const ret = ContextLengthMaximal.__wrap(arg0);
            return ret;
        },
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
        __wbg_decisionanswerchoice_new: function(arg0) {
            const ret = DecisionAnswerChoice.__wrap(arg0);
            return ret;
        },
        __wbg_decisionanswerscore_new: function(arg0) {
            const ret = DecisionAnswerScore.__wrap(arg0);
            return ret;
        },
        __wbg_decisionansweryesno_new: function(arg0) {
            const ret = DecisionAnswerYesNo.__wrap(arg0);
            return ret;
        },
        __wbg_decisionoption_new: function(arg0) {
            const ret = DecisionOption.__wrap(arg0);
            return ret;
        },
        __wbg_decisionoutput_new: function(arg0) {
            const ret = DecisionOutput.__wrap(arg0);
            return ret;
        },
        __wbg_decisionquestionchoice_new: function(arg0) {
            const ret = DecisionQuestionChoice.__wrap(arg0);
            return ret;
        },
        __wbg_decisionquestionscore_new: function(arg0) {
            const ret = DecisionQuestionScore.__wrap(arg0);
            return ret;
        },
        __wbg_decisionquestionyesno_new: function(arg0) {
            const ret = DecisionQuestionYesNo.__wrap(arg0);
            return ret;
        },
        __wbg_decisionresult_new: function(arg0) {
            const ret = DecisionResult.__wrap(arg0);
            return ret;
        },
        __wbg_decisionsession_new: function(arg0) {
            const ret = DecisionSession.__wrap(arg0);
            return ret;
        },
        __wbg_decisionstats_new: function(arg0) {
            const ret = DecisionStats.__wrap(arg0);
            return ret;
        },
        __wbg_description_83b8a393160021b9: function(arg0, arg1) {
            const ret = arg1.description;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_device_new: function(arg0) {
            const ret = Device.__wrap(arg0);
            return ret;
        },
        __wbg_dispatchWorkgroups_56b943172790add0: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_done_b41a1d26cdb37fb6: function(arg0) {
            const ret = arg0.done;
            return ret;
        },
        __wbg_downloader_new: function(arg0) {
            const ret = Downloader.__wrap(arg0);
            return ret;
        },
        __wbg_downloaderstream_new: function(arg0) {
            const ret = DownloaderStream.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphasedownloaded_new: function(arg0) {
            const ret = DownloadPhaseDownloaded.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphasedownloading_new: function(arg0) {
            const ret = DownloadPhaseDownloading.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphaseerror_new: function(arg0) {
            const ret = DownloadPhaseError.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphaseinitializing_new: function(arg0) {
            const ret = DownloadPhaseInitializing.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphaselocked_new: function(arg0) {
            const ret = DownloadPhaseLocked.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphasenotdownloaded_new: function(arg0) {
            const ret = DownloadPhaseNotDownloaded.__wrap(arg0);
            return ret;
        },
        __wbg_downloadphasepaused_new: function(arg0) {
            const ret = DownloadPhasePaused.__wrap(arg0);
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
        __wbg_engineconfig_new: function(arg0) {
            const ret = EngineConfig.__wrap(arg0);
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
        __wbg_file_new: function(arg0) {
            const ret = File.__wrap(arg0);
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
        __wbg_gpu_afdd4387c7afe5f9: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_grammarjsonany_new: function(arg0) {
            const ret = GrammarJsonAny.__wrap(arg0);
            return ret;
        },
        __wbg_grammarjsonschema_new: function(arg0) {
            const ret = GrammarJsonSchema.__wrap(arg0);
            return ret;
        },
        __wbg_grammarregex_new: function(arg0) {
            const ret = GrammarRegex.__wrap(arg0);
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
        __wbg_headers_eba93595f8944c2f: function(arg0) {
            const ret = arg0.headers;
            return ret;
        },
        __wbg_image_new: function(arg0) {
            const ret = Image.__wrap(arg0);
            return ret;
        },
        __wbg_info_971d8b9db3dae69f: function(arg0) {
            const ret = arg0.info;
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
        __wbg_metadata_new: function(arg0) {
            const ret = Metadata.__wrap(arg0);
            return ret;
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
        __wbg_modelaccessibilityondevice_new: function(arg0) {
            const ret = ModelAccessibilityOnDevice.__wrap(arg0);
            return ret;
        },
        __wbg_modelaccessibilityremote_new: function(arg0) {
            const ret = ModelAccessibilityRemote.__wrap(arg0);
            return ret;
        },
        __wbg_modelbackend_new: function(arg0) {
            const ret = ModelBackend.__wrap(arg0);
            return ret;
        },
        __wbg_modelfamily_new: function(arg0) {
            const ret = ModelFamily.__wrap(arg0);
            return ret;
        },
        __wbg_modelproperties_new: function(arg0) {
            const ret = ModelProperties.__wrap(arg0);
            return ret;
        },
        __wbg_modelquantization_new: function(arg0) {
            const ret = ModelQuantization.__wrap(arg0);
            return ret;
        },
        __wbg_modelregistry_new: function(arg0) {
            const ret = ModelRegistry.__wrap(arg0);
            return ret;
        },
        __wbg_modelsourcefilesystem_new: function(arg0) {
            const ret = ModelSourceFilesystem.__wrap(arg0);
            return ret;
        },
        __wbg_modelsourceregistry_new: function(arg0) {
            const ret = ModelSourceRegistry.__wrap(arg0);
            return ret;
        },
        __wbg_modelspecializationchat_new: function(arg0) {
            const ret = ModelSpecializationChat.__wrap(arg0);
            return ret;
        },
        __wbg_modelspecializationclassification_new: function(arg0) {
            const ret = ModelSpecializationClassification.__wrap(arg0);
            return ret;
        },
        __wbg_modelspecializationdecision_new: function(arg0) {
            const ret = ModelSpecializationDecision.__wrap(arg0);
            return ret;
        },
        __wbg_modelspecializationspeculation_new: function(arg0) {
            const ret = ModelSpecializationSpeculation.__wrap(arg0);
            return ret;
        },
        __wbg_modelspecializationtexttospeech_new: function(arg0) {
            const ret = ModelSpecializationTextToSpeech.__wrap(arg0);
            return ret;
        },
        __wbg_modelspecializationtranslation_new: function(arg0) {
            const ret = ModelSpecializationTranslation.__wrap(arg0);
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
        __wbg_new_2bdcdfcebdfc34d2: function(arg0, arg1) {
            const ret = new TypeError(getStringFromWasm0(arg0, arg1));
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
        __wbg_pcmbatch_new: function(arg0) {
            const ret = PcmBatch.__wrap(arg0);
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
        __wbg_replyenergycomponents_new: function(arg0) {
            const ret = ReplyEnergyComponents.__wrap(arg0);
            return ret;
        },
        __wbg_replyenergytotal_new: function(arg0) {
            const ret = ReplyEnergyTotal.__wrap(arg0);
            return ret;
        },
        __wbg_repository_new: function(arg0) {
            const ret = Repository.__wrap(arg0);
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
        __wbg_samplingmethodgreedy_new: function(arg0) {
            const ret = SamplingMethodGreedy.__wrap(arg0);
            return ret;
        },
        __wbg_samplingmethodstochastic_new: function(arg0) {
            const ret = SamplingMethodStochastic.__wrap(arg0);
            return ret;
        },
        __wbg_samplingparameters_new: function(arg0) {
            const ret = SamplingParameters.__wrap(arg0);
            return ret;
        },
        __wbg_samplingpolicycustom_new: function(arg0) {
            const ret = SamplingPolicyCustom.__wrap(arg0);
            return ret;
        },
        __wbg_samplingpolicydefault_new: function(arg0) {
            const ret = SamplingPolicyDefault.__wrap(arg0);
            return ret;
        },
        __wbg_samplingseedcustom_new: function(arg0) {
            const ret = SamplingSeedCustom.__wrap(arg0);
            return ret;
        },
        __wbg_samplingseeddefault_new: function(arg0) {
            const ret = SamplingSeedDefault.__wrap(arg0);
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
        __wbg_set_access_a099cfbbeec9b96f: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
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
        __wbg_texttospeechstats_new: function(arg0) {
            const ret = TextToSpeechStats.__wrap(arg0);
            return ret;
        },
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
        __wbg_toolcallargumentdelta_new: function(arg0) {
            const ret = ToolCallArgumentDelta.__wrap(arg0);
            return ret;
        },
        __wbg_toolcallargumentstream_new: function(arg0) {
            const ret = ToolCallArgumentStream.__wrap(arg0);
            return ret;
        },
        __wbg_tooldescriptionfunction_new: function(arg0) {
            const ret = ToolDescriptionFunction.__wrap(arg0);
            return ret;
        },
        __wbg_toolfunction_new: function(arg0) {
            const ret = ToolFunction.__wrap(arg0);
            return ret;
        },
        __wbg_toolnamespace_new: function(arg0) {
            const ret = ToolNamespace.__wrap(arg0);
            return ret;
        },
        __wbg_translationpayloadimage_new: function(arg0) {
            const ret = TranslationPayloadImage.__wrap(arg0);
            return ret;
        },
        __wbg_translationpayloadtext_new: function(arg0) {
            const ret = TranslationPayloadText.__wrap(arg0);
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
        __wbg_value_new: function(arg0) {
            const ret = Value.__wrap(arg0);
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
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 1233, ret: Externref, inner_ret: Some(Externref) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__wasm_bindgen_6c51f868a643a6ac___JsValue__true_);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 1555, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue______true_);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 3313, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___JsValue__core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUDevice")], shim_idx: 3251, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("any")], shim_idx: 3251, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__898);
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [], shim_idx: 1528, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true_);
            return ret;
        },
        __wbindgen_generic_0000000000000007: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [], shim_idx: 1658, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke_______true__1_);
            return ret;
        },
        __wbindgen_generic_0000000000000008: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000009: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_000000000000000a: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
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

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__898(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_6c51f868a643a6ac___convert__closures_____invoke___wasm_bindgen_6c51f868a643a6ac___sys__JsNullable_wgpu_379461efccdc673a___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_b4dac1b1fa325dc6___result__Result_____wasm_bindgen_6c51f868a643a6ac___JsError___true__898(arg0, arg1, arg2);
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
const ChatContentBlockAudioFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockaudio_free(ptr, 1));
const ChatContentBlockBuiltinToolsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockbuiltintools_free(ptr, 1));
const ChatContentBlockConversationStartDateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockconversationstartdate_free(ptr, 1));
const ChatContentBlockCustomFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockcustom_free(ptr, 1));
const ChatContentBlockFileFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockfile_free(ptr, 1));
const ChatContentBlockIdentityFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockidentity_free(ptr, 1));
const ChatContentBlockImageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockimage_free(ptr, 1));
const ChatContentBlockKnowledgeCutoffFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockknowledgecutoff_free(ptr, 1));
const ChatContentBlockReasoningFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockreasoning_free(ptr, 1));
const ChatContentBlockReasoningEffortFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockreasoningeffort_free(ptr, 1));
const ChatContentBlockTextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktext_free(ptr, 1));
const ChatContentBlockToolCallFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktoolcall_free(ptr, 1));
const ChatContentBlockToolCallCandidateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktoolcallcandidate_free(ptr, 1));
const ChatContentBlockToolCallResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktoolcallresult_free(ptr, 1));
const ChatContentBlockToolsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktools_free(ptr, 1));
const ChatContentBlockTranslationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktranslation_free(ptr, 1));
const ChatContentBlockTypeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblocktype_free(ptr, 1));
const ChatContentBlockVideoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatcontentblockvideo_free(ptr, 1));
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
const ChatReplyFinishReasonFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplyfinishreason_free(ptr, 1));
const ChatReplyJoulesPerTokenComponentsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplyjoulespertokencomponents_free(ptr, 1));
const ChatReplyJoulesPerTokenTotalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplyjoulespertokentotal_free(ptr, 1));
const ChatReplySpeculatorStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplyspeculatorstats_free(ptr, 1));
const ChatReplyStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatreplystats_free(ptr, 1));
const ChatRoleAssistantFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatroleassistant_free(ptr, 1));
const ChatRoleCustomFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatrolecustom_free(ptr, 1));
const ChatRoleDeveloperFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatroledeveloper_free(ptr, 1));
const ChatRoleSystemFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatrolesystem_free(ptr, 1));
const ChatRoleToolFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatroletool_free(ptr, 1));
const ChatRoleUserFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatroleuser_free(ptr, 1));
const ChatSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsession_free(ptr, 1));
const ChatSessionStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsessionstate_free(ptr, 1));
const ChatSessionStreamFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsessionstream_free(ptr, 1));
const ChatSessionStreamChunkErrorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsessionstreamchunkerror_free(ptr, 1));
const ChatSessionStreamChunkRepliesFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsessionstreamchunkreplies_free(ptr, 1));
const ChatSessionStreamChunkToolResultsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_chatsessionstreamchunktoolresults_free(ptr, 1));
const ClassificationLabelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationlabel_free(ptr, 1));
const ClassificationMessageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationmessage_free(ptr, 1));
const ClassificationOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationoutput_free(ptr, 1));
const ClassificationRoleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationrole_free(ptr, 1));
const ClassificationStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_classificationstats_free(ptr, 1));
const ContextLengthCustomFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_contextlengthcustom_free(ptr, 1));
const ContextLengthDefaultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_contextlengthdefault_free(ptr, 1));
const ContextLengthMaximalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_contextlengthmaximal_free(ptr, 1));
const DecisionAnswerChoiceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionanswerchoice_free(ptr, 1));
const DecisionAnswerScoreFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionanswerscore_free(ptr, 1));
const DecisionAnswerYesNoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionansweryesno_free(ptr, 1));
const DecisionInputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisioninput_free(ptr, 1));
const DecisionOptionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionoption_free(ptr, 1));
const DecisionOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionoutput_free(ptr, 1));
const DecisionQuestionChoiceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionquestionchoice_free(ptr, 1));
const DecisionQuestionScoreFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionquestionscore_free(ptr, 1));
const DecisionQuestionYesNoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionquestionyesno_free(ptr, 1));
const DecisionResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionresult_free(ptr, 1));
const DecisionSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionsession_free(ptr, 1));
const DecisionSessionStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionsessionstate_free(ptr, 1));
const DecisionStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decisionstats_free(ptr, 1));
const DeviceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_device_free(ptr, 1));
const DownloadManagerTypeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadmanagertype_free(ptr, 1));
const DownloadPhaseDownloadedFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphasedownloaded_free(ptr, 1));
const DownloadPhaseDownloadingFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphasedownloading_free(ptr, 1));
const DownloadPhaseErrorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphaseerror_free(ptr, 1));
const DownloadPhaseInitializingFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphaseinitializing_free(ptr, 1));
const DownloadPhaseLockedFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphaselocked_free(ptr, 1));
const DownloadPhaseNotDownloadedFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphasenotdownloaded_free(ptr, 1));
const DownloadPhasePausedFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_downloadphasepaused_free(ptr, 1));
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
const GrammarJsonAnyFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_grammarjsonany_free(ptr, 1));
const GrammarJsonSchemaFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_grammarjsonschema_free(ptr, 1));
const GrammarRegexFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_grammarregex_free(ptr, 1));
const HashFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_hash_free(ptr, 1));
const HashMethodFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_hashmethod_free(ptr, 1));
const ImageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_image_free(ptr, 1));
const ImageFormatFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_imageformat_free(ptr, 1));
const ImageThemeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_imagetheme_free(ptr, 1));
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
const ModelAccessibilityOnDeviceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelaccessibilityondevice_free(ptr, 1));
const ModelAccessibilityRemoteFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelaccessibilityremote_free(ptr, 1));
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
const ModelSourceFilesystemFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelsourcefilesystem_free(ptr, 1));
const ModelSourceRegistryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelsourceregistry_free(ptr, 1));
const ModelSpecializationChatFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelspecializationchat_free(ptr, 1));
const ModelSpecializationClassificationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelspecializationclassification_free(ptr, 1));
const ModelSpecializationDecisionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelspecializationdecision_free(ptr, 1));
const ModelSpecializationSpeculationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelspecializationspeculation_free(ptr, 1));
const ModelSpecializationTextToSpeechFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelspecializationtexttospeech_free(ptr, 1));
const ModelSpecializationTranslationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelspecializationtranslation_free(ptr, 1));
const ModelVendorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelvendor_free(ptr, 1));
const NativeToolFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_nativetool_free(ptr, 1));
const PcmBatchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pcmbatch_free(ptr, 1));
const ReasoningEffortFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_reasoningeffort_free(ptr, 1));
const ReplyEnergyComponentsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_replyenergycomponents_free(ptr, 1));
const ReplyEnergyTotalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_replyenergytotal_free(ptr, 1));
const RepositoryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_repository_free(ptr, 1));
const SamplingMethodGreedyFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingmethodgreedy_free(ptr, 1));
const SamplingMethodStochasticFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingmethodstochastic_free(ptr, 1));
const SamplingParametersFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingparameters_free(ptr, 1));
const SamplingPolicyCustomFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingpolicycustom_free(ptr, 1));
const SamplingPolicyDefaultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingpolicydefault_free(ptr, 1));
const SamplingSeedCustomFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingseedcustom_free(ptr, 1));
const SamplingSeedDefaultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_samplingseeddefault_free(ptr, 1));
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
const ToolDescriptionFunctionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tooldescriptionfunction_free(ptr, 1));
const ToolFunctionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolfunction_free(ptr, 1));
const ToolNamespaceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_toolnamespace_free(ptr, 1));
const TranslationPayloadImageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_translationpayloadimage_free(ptr, 1));
const TranslationPayloadTextFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_translationpayloadtext_free(ptr, 1));
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
