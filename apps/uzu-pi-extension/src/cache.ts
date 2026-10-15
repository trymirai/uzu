import {
  ChatConfig,
  ContextLengthMaximal,
  type ChatInstance,
  type ChatSession,
  type Engine,
  type Model,
} from "@trymirai/uzu";
import { abortable } from "./abort.js";
import {
  assistantInput,
  nativeMessage,
  nativePreamble,
  type Input,
} from "./messages.js";
import type { AssistantMessage } from "@earendil-works/pi-ai";

// A single resident model and KV cache bound memory use. Requests serialize, including
// cancellation cleanup, so another turn can never race a native session still generating.
export class SessionCache {
  private tail: Promise<void> = Promise.resolve();
  private instance?: ChatInstance;
  private session?: ChatSession;
  private modelKey?: string;
  private ownerSessionId?: string;
  private sessionId?: string;
  private settings?: string;
  private history: string[] = [];

  constructor(
    private readonly engine: Pick<Engine, "chatInstance" | "chatWithInstance">,
    private readonly getOwnerSessionId?: () => string | undefined,
  ) {}

  async exclusive<T>(
    signal: AbortSignal | undefined,
    run: () => Promise<T>,
  ): Promise<T> {
    const previous = this.tail;
    let release!: () => void;
    const finished = new Promise<void>((resolve) => {
      release = resolve;
    });
    // A cancelled waiter can finish immediately, but its successor must still
    // wait for the active request to finish using native state.
    this.tail = previous.then(() => finished);
    try {
      await abortable(previous, signal);
      signal?.throwIfAborted();
      return await run();
    } finally {
      release();
    }
  }

  async prepare(
    model: Model,
    sessionId: string | undefined,
    input: Input,
    retain: boolean,
  ) {
    const key = model.cacheIdentifier;
    if (key !== this.modelKey) {
      this.clear();
      this.instance = await this.engine.chatInstance(
        model,
        ChatConfig.create().withContextLength(new ContextLengthMaximal()),
      );
      this.modelKey = key;
    }
    // Summaries use temporary request IDs; weights still belong to the Pi session.
    this.ownerSessionId =
      this.getOwnerSessionId?.() ?? sessionId ?? this.ownerSessionId;
    const settings = JSON.stringify([
      input.system,
      input.tools,
      input.reasoning,
    ]);
    const history = input.messages.map((message) => JSON.stringify(message));
    const continues =
      retain &&
      sessionId !== undefined &&
      sessionId === this.sessionId &&
      settings === this.settings &&
      this.history.length <= history.length &&
      this.history.every((message, index) => message === history[index]);
    if (!this.session)
      this.session = await this.engine.chatWithInstance(this.instance!);
    else if (!continues) await this.session.reset();
    const offset = continues ? this.history.length : 0;
    const messages = input.messages.slice(offset).map(nativeMessage);
    if (!continues) messages.unshift(...nativePreamble(input));
    this.sessionId = retain ? sessionId : undefined;
    this.settings = settings;
    this.history = history;
    return { session: this.session, messages };
  }

  complete(message: AssistantMessage) {
    this.history.push(JSON.stringify(assistantInput(message)));
  }

  invalidate() {
    this.session = undefined;
    this.sessionId = undefined;
    this.settings = undefined;
    this.history = [];
  }

  clear() {
    this.invalidate();
    this.instance = undefined;
    this.modelKey = undefined;
    this.ownerSessionId = undefined;
  }

  release(sessionId?: string) {
    return this.exclusive(undefined, async () => {
      if (sessionId === undefined || sessionId === this.ownerSessionId)
        this.clear();
    });
  }
}
