import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useChatSessionStore } from "@/stores/use-chat-session-store";
import { useChatStore } from "@/stores/use-chat-store";
import type { Message as StoredMessage } from "@/types/message";
import type { TranscriptItem } from "@/types/llm-stream";
import { Message } from "./message";

const clipboard = vi.hoisted(() => ({ write: vi.fn<(items: ClipboardItem[]) => Promise<void>>(async () => {}) }));
vi.mock("@/utils/clipboard", () => ({ writeItemsWithFocus: clipboard.write }));
vi.mock("chart.js/auto", () => ({
  default: class {
    destroy() {}
  },
}));

vi.mock("./markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}));
vi.mock("./model-selector", () => ({ ModelSelector: () => null }));
vi.mock("./performance-dropdown", () => ({ PerformanceDropdown: () => null }));

const sessionDefaults = useChatSessionStore.getState();
const chatDefaults = useChatStore.getState();
const message: StoredMessage = {
  id: "answer",
  sender: "assistant",
  timestamp: 1,
  text: "Response",
  output: { text: { parsed: { chainOfThought: "Reasoning", response: "Response" } } },
};

beforeEach(() => {
  useChatSessionStore.setState(sessionDefaults, true);
  useChatStore.setState({ ...chatDefaults, messages: [message] }, true);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("copies chart-only replies as readable data instead of an empty canvas", async () => {
  class ClipboardEntry {
    constructor(public items: Record<string, Blob>) {}
  }
  vi.stubGlobal("ClipboardItem", ClipboardEntry);
  clipboard.write.mockClear();
  const chartMessage: StoredMessage = {
    ...message,
    text: "",
    output: {
      transcript: [
        {
          type: "chart",
          chart: {
            type: "bar",
            title: "Budget",
            labels: ["Rent", "Food"],
            datasets: [{ label: "Monthly spending", data: [900, 350] }],
          },
        },
      ],
    },
  };
  useChatStore.setState({ messages: [chartMessage] });
  render(<Message {...chartMessage} chatId="chat" />);
  await screen.findByRole("img", { name: "Budget (bar chart)" });
  fireEvent.click(screen.getByRole("button", { name: "Copy" }));
  await waitFor(() => expect(clipboard.write).toHaveBeenCalledOnce());
  const copied = clipboard.write.mock.calls[0]![0][0] as unknown as ClipboardEntry;
  const read = (kind: string) =>
    new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(copied.items[kind]!);
    });
  const html = await read("text/html");
  expect(html).toContain("<table>");
  expect(html).toContain("Monthly spending");
  expect(html).toContain("900");
  expect(html).not.toContain("<canvas");
  expect(await read("text/plain")).toContain("Category\tMonthly spending\nRent\t900\nFood\t350");
});

it("changes Thinking... to Thinking when this response finishes", () => {
  useChatSessionStore.setState({
    isGenerating: true,
    activeGeneratingChatId: "chat",
    activeAssistantMessageId: message.id,
  });
  render(<Message {...message} chatId="chat" isLast isUiStreaming />);
  expect(screen.getByText("Thinking...")).toBeTruthy();

  act(() => useChatSessionStore.getState().setGenerating(false));
  expect(screen.getByText("Thinking")).toBeTruthy();
});

it("tracks the actual message when regenerating an earlier response", () => {
  const later = { ...message, id: "later-answer" };
  useChatStore.setState({ messages: [message, later] });
  useChatSessionStore.setState({
    isGenerating: true,
    activeGeneratingChatId: "chat",
    activeAssistantMessageId: message.id,
  });
  render(
    <>
      <Message {...message} chatId="chat" />
      <Message {...later} chatId="chat" isLast isUiStreaming />
    </>,
  );
  expect(screen.getAllByText(/^Thinking/).map((element) => element.textContent)).toEqual(["Thinking...", "Thinking"]);
});

it("keeps an older version and other chats marked finished", () => {
  const versions = ["old", "new"].map((id) => ({ ...message, id, modelId: "model", modelName: "Model" }));
  useChatStore.setState({ messages: [{ ...message, versions, currentVersionIndex: 0 }] });
  useChatSessionStore.setState({
    isGenerating: true,
    activeGeneratingChatId: "chat",
    activeAssistantMessageId: message.id,
  });
  const view = render(<Message {...message} chatId="chat" isLast isUiStreaming />);
  expect(screen.getByText("Thinking")).toBeTruthy();

  act(() => useChatStore.setState({ messages: [message] }));
  view.rerender(<Message {...message} chatId="other-chat" isLast isUiStreaming />);
  expect(screen.getByText("Thinking")).toBeTruthy();
});

const renderThinkingMessage = () => {
  const thinkingMessage: StoredMessage = {
    ...message,
    text: "",
    output: { text: { parsed: { chainOfThought: "Reasoning", response: "" } } },
  };
  useChatStore.setState({ messages: [thinkingMessage] });
  useChatSessionStore.setState({
    isGenerating: true,
    activeGeneratingChatId: "chat",
    activeAssistantMessageId: message.id,
  });
  const view = render(<Message {...thinkingMessage} chatId="chat" isLast isUiStreaming />);
  return { view, thinkingMessage, toggle: screen.getByRole("button", { name: /^Thinking/ }) };
};

it("opens during reasoning, collapses when the answer starts, and allows reopening", () => {
  const { toggle } = renderThinkingMessage();
  expect(toggle.getAttribute("aria-expanded")).toBe("true");

  act(() => useChatStore.getState().updateMessage(message.id, message));
  expect(toggle.getAttribute("aria-expanded")).toBe("false");

  fireEvent.click(toggle);
  expect(toggle.getAttribute("aria-expanded")).toBe("true");

  act(() => useChatSessionStore.getState().setGenerating(false));
  expect(toggle.getAttribute("aria-expanded")).toBe("true");

  act(() =>
    useChatStore.setState((state) => ({ messages: [...state.messages, { ...message, id: "another-answer" }] })),
  );
  expect(toggle.getAttribute("aria-expanded")).toBe("true");
});

it.each(["finish", "cancel", "error"] as const)(
  "collapses when reasoning ends through %s, then stays open if reopened",
  (outcome) => {
    const { view, thinkingMessage, toggle } = renderThinkingMessage();
    expect(toggle.getAttribute("aria-expanded")).toBe("true");

    if (outcome === "finish") {
      act(() => useChatSessionStore.getState().setGenerating(false));
    } else if (outcome === "cancel") {
      view.rerender(<Message {...thinkingMessage} chatId="chat" isLast isUiStreaming isCanceled />);
    } else {
      act(() => useChatStore.getState().updateMessage(message.id, { error: "Generation failed" }));
    }
    expect(toggle.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");

    act(() => useChatSessionStore.getState().clearActiveGenerating());
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
  },
);

it("keeps reasoning closed when the user closes it while more reasoning arrives", () => {
  const { toggle } = renderThinkingMessage();
  expect(toggle.getAttribute("aria-expanded")).toBe("true");

  fireEvent.click(toggle);
  expect(toggle.getAttribute("aria-expanded")).toBe("false");

  act(() =>
    useChatStore.getState().updateMessage(message.id, {
      output: { text: { parsed: { chainOfThought: "Reasoning continues", response: "" } } },
    }),
  );
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
});

it("renders each reasoning, text and tool turn in order and collapses reasoning independently", () => {
  const first: TranscriptItem = { type: "thinking", text: "First reasoning", completed: false };
  const update = (transcript: TranscriptItem[]) =>
    act(() => useChatStore.getState().updateMessage(message.id, { output: { transcript } }));
  update([first]);
  useChatSessionStore.setState({
    isGenerating: true,
    activeGeneratingChatId: "chat",
    activeAssistantMessageId: message.id,
  });
  const view = render(<Message {...message} chatId="chat" isLast isUiStreaming />);
  const firstToggle = screen.getByRole("button", { name: /^Thinking/ });
  expect(firstToggle.getAttribute("aria-expanded")).toBe("true");

  const firstTurn: TranscriptItem[] = [
    { ...first, completed: true },
    { type: "text", text: "Checking the time." },
    { type: "toolCall", name: "get_current_date_time", called: false },
  ];
  update(firstTurn);
  expect(firstToggle.getAttribute("aria-expanded")).toBe("false");
  expect(view.container.textContent).toContain("Checking the date and time...");
  expect(view.container.textContent).not.toContain("Renaming the chat");
  expect(view.container.textContent).not.toContain("Response");
  const answerNode = screen.getByText("Checking the time.");

  const nextTurn: TranscriptItem[] = [
    ...firstTurn.slice(0, -1),
    { type: "toolCall", name: "get_current_date_time", called: true },
    { type: "thinking", text: "Second reasoning", completed: false },
  ];
  update(nextTurn);
  expect(screen.getByText("Checking the time.")).toBe(answerNode);
  expect(view.container.textContent).toMatch(/Checking the time\..*Checked the date and time.*Thinking\.\.\./s);
  const secondToggle = screen.getAllByRole("button", { name: /^Thinking/ })[1]!;
  expect(secondToggle.getAttribute("aria-expanded")).toBe("true");
  fireEvent.click(firstToggle);
  expect(firstToggle.getAttribute("aria-expanded")).toBe("true");

  update([
    ...nextTurn.slice(0, -1),
    { type: "thinking", text: "Second reasoning", completed: true },
    { type: "text", text: "It is noon." },
  ]);
  expect(firstToggle.getAttribute("aria-expanded")).toBe("true");
  expect(secondToggle.getAttribute("aria-expanded")).toBe("false");
  fireEvent.click(secondToggle);
  expect(secondToggle.getAttribute("aria-expanded")).toBe("true");
  act(() => useChatSessionStore.getState().setGenerating(false));
  expect(secondToggle.getAttribute("aria-expanded")).toBe("true");
});

it("keeps one reasoning block across hidden housekeeping and preserves manual toggles", () => {
  useChatStore.getState().updateMessage(message.id, {
    output: { transcript: [{ type: "thinking", text: "Choose a title", completed: false }] },
  });
  useChatSessionStore.setState({
    isGenerating: true,
    activeGeneratingChatId: "chat",
    activeAssistantMessageId: message.id,
  });
  render(<Message {...message} chatId="chat" isLast isUiStreaming />);
  const toggle = screen.getByRole("button", { name: /^Thinking/ });
  expect(toggle.getAttribute("aria-expanded")).toBe("true");
  fireEvent.click(toggle);
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  act(() =>
    useChatStore.getState().updateMessage(message.id, {
      output: {
        transcript: [{ type: "thinking", text: "Choose a title\n\nAnswer the question", completed: false }],
      },
    }),
  );
  expect(screen.getAllByRole("button", { name: /^Thinking/ })).toEqual([toggle]);
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  fireEvent.click(toggle);
  expect(toggle.getAttribute("aria-expanded")).toBe("true");
  act(() =>
    useChatStore.getState().updateMessage(message.id, {
      output: {
        transcript: [
          { type: "thinking", text: "Choose a title\n\nAnswer the question", completed: true },
          { type: "text", text: "The answer" },
        ],
      },
    }),
  );
  expect(toggle.getAttribute("aria-expanded")).toBe("false");
  expect(screen.getByText("Thinking")).toBeTruthy();
});

it("shows a chat naming tool requested explicitly by the user", () => {
  useChatStore.getState().updateMessage(message.id, {
    output: { transcript: [{ type: "toolCall", name: "set_chat_name", called: true }] },
  });
  const view = render(<Message {...message} chatId="chat" />);
  expect(view.container.textContent).toContain("Renamed the chat");
  expect(view.container.textContent).not.toContain("set_chat_name");
});

it("keeps interrupted tool calls visible without claiming they are still running", () => {
  useChatStore.getState().updateMessage(message.id, {
    output: { transcript: [{ type: "toolCall", name: "get_current_date_time", called: false }] },
  });
  const view = render(<Message {...message} chatId="chat" isCanceled />);
  expect(view.container.textContent).toContain("Stopped checking the date and time");
  expect(view.container.textContent).not.toContain("Checking the date and time...");
});

it("does not claim a failed tool call succeeded", () => {
  useChatStore.getState().updateMessage(message.id, {
    output: { transcript: [{ type: "toolCall", name: "set_chat_name", called: true, failed: true }] },
  });
  render(<Message {...message} chatId="chat" />);
  expect(screen.getByText("Couldn't rename the chat")).toBeTruthy();
  expect(screen.queryByText("Renamed the chat")).toBeNull();
});
