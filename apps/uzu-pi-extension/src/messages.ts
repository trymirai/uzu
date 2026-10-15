import {
  getCurrentSystemPrompt,
  getCurrentTools,
  type AssistantMessage,
  type JsonObject,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import {
  ChatContentBlockToolCallResult,
  ChatMessage,
  ReasoningEffort,
  ToolCall,
  ToolDescriptionFunction,
  ToolFunction,
  ToolNamespace,
  Value,
  type ChatModelInfo,
} from "@trymirai/uzu";

export type InputBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | { type: "toolCall"; id: string; name: string; arguments: JsonObject }
  | { type: "toolResult"; id: string; name: string; text: string };
export interface InputMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: InputBlock[];
}
export interface Input {
  system: string;
  tools: { name: string; description: string; parameters: unknown }[];
  reasoning?: ReasoningEffort;
  messages: InputMessage[];
}

export function assistantInput(message: AssistantMessage): InputMessage {
  return {
    role: "assistant",
    content: message.content.flatMap((block): InputBlock[] => {
      switch (block.type) {
        case "text":
          return [{ type: "text", text: block.text }];
        case "thinking":
          return block.redacted
            ? []
            : [{ type: "thinking", thinking: block.thinking }];
        case "toolCall":
          return [
            {
              type: "toolCall",
              id: block.id,
              name: block.name,
              arguments: block.arguments,
            },
          ];
      }
    }),
  };
}

export function prepareInput(
  context: TranscriptContext,
  info: ChatModelInfo,
  options: SimpleStreamOptions,
): Input {
  const tools =
    options.toolChoice === "none" ? [] : getCurrentTools(context.messages);
  if (tools.length && !info.capabilities.supportsTools)
    throw new Error("This Uzu model does not support tools");
  if (!tools.length && info.capabilities.requiresTools)
    throw new Error("This Uzu model requires tool declarations");
  const levels: Record<string, ReasoningEffort> = {
    minimal: ReasoningEffort.Low,
    low: ReasoningEffort.Low,
    medium: ReasoningEffort.Medium,
    high: ReasoningEffort.High,
    xhigh: ReasoningEffort.XHigh,
    max: ReasoningEffort.XHigh,
  };
  const requested = options.reasoning ? levels[options.reasoning] : undefined;
  const reasoning = !info.capabilities.supportsReasoning
    ? undefined
    : !requested && info.capabilities.supportsDisableReasoning
      ? ReasoningEffort.Disabled
      : requested && info.capabilities.reasoningEfforts.includes(requested)
        ? requested
        : (info.defaultReasoningEffort ?? ReasoningEffort.Default);
  const messages: InputMessage[] = [];
  const ignoredToolCalls = new Set<string>();
  for (const message of context.messages) {
    if (message.role === "system") continue;
    if (message.role === "assistant") {
      // Failed requests may contain incomplete tool calls that never ran.
      if (message.stopReason === "error" || message.stopReason === "aborted") {
        for (const block of message.content) {
          if (block.type === "toolCall") ignoredToolCalls.add(block.id);
        }
      } else messages.push(assistantInput(message));
    } else if (message.role === "user") {
      const content =
        typeof message.content === "string"
          ? [{ type: "text" as const, text: message.content }]
          : message.content;
      messages.push({
        role: "user",
        content: content.map((block) => {
          if (block.type !== "text")
            throw new Error(
              "Uzu currently supports text input only; remove image attachments",
            );
          return { type: "text", text: block.text };
        }),
      });
    } else {
      if (ignoredToolCalls.has(message.toolCallId)) continue;
      const text = message.content
        .map((block) => {
          if (block.type !== "text")
            throw new Error("Uzu currently supports text tool results only");
          return block.text;
        })
        .join("\n");
      messages.push({
        role: "tool",
        content: [
          {
            type: "toolResult",
            id: message.toolCallId,
            name: message.toolName,
            text,
          },
        ],
      });
    }
  }
  return {
    system: getCurrentSystemPrompt(context.messages),
    tools,
    reasoning,
    messages,
  };
}

export function nativePreamble(input: Input): ChatMessage[] {
  let message = ChatMessage.system().withText(input.system);
  if (input.reasoning !== undefined)
    message = message.withReasoningEffort(input.reasoning);
  const messages = [message];
  if (input.tools.length)
    messages.push(
      ChatMessage.developer().withToolNamespaces(nativeTools(input)),
    );
  return messages;
}

export function nativeTools(input: Input): ToolNamespace[] {
  return [
    new ToolNamespace(
      "functions",
      undefined,
      input.tools.map(
        (tool) =>
          new ToolDescriptionFunction(
            new ToolFunction(
              tool.name,
              tool.description,
              new Value(JSON.stringify(tool.parameters)),
            ),
          ),
      ),
    ),
  ];
}

export function nativeMessage(message: InputMessage): ChatMessage {
  let native = ChatMessage[message.role]();
  for (const block of message.content) {
    switch (block.type) {
      case "text":
        native = native.withText(block.text);
        break;
      case "thinking":
        native = native.withReasoning(block.thinking);
        break;
      case "toolCall":
        native = native.withToolCall(
          new ToolCall(
            block.id,
            block.name,
            new Value(JSON.stringify(block.arguments)),
          ),
        );
        break;
      case "toolResult":
        native = native.withBlock(
          new ChatContentBlockToolCallResult(
            block.id,
            block.name,
            new Value(JSON.stringify(block.text)),
          ),
        );
        break;
    }
  }
  return native;
}
