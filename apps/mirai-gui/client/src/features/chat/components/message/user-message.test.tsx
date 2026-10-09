import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { AttachedFile } from "@/types/files";
import { UserMessage, type UserMessageProps } from "./user-message";

const mocks = vi.hoisted(() => ({
  writeItemsWithFocus: vi.fn<(items: ClipboardItem[]) => Promise<void>>(async () => {}),
  getFiles: vi.fn<() => AttachedFile[]>(() => []),
}));
vi.mock("@/utils/clipboard", () => ({ writeItemsWithFocus: mocks.writeItemsWithFocus }));
vi.mock("../../services/attachment-storage", () => ({ attachmentStorage: { getFiles: mocks.getFiles } }));

class TestClipboardItem {
  constructor(public items: Record<string, Blob>) {}
}

const originalText = "  **original**\n\n[link](https://example.com)  ";
const onEdit = vi.fn<NonNullable<UserMessageProps["onEdit"]>>(async (_id, _text, onSaved) => onSaved());
const props = { id: "user-message", text: originalText, canEdit: true, onEdit };
const edit = () => fireEvent.click(screen.getByRole("button", { name: "Edit message" }));
const draft = () => screen.getByRole("textbox", { name: "Edit message" }) as HTMLTextAreaElement;
const save = () => fireEvent.click(screen.getByRole("button", { name: "Save & send" }));

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("ClipboardItem", TestClipboardItem);
  mocks.getFiles.mockReturnValue([]);
  onEdit.mockImplementation(async (_id, _text, onSaved) => onSaved());
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("renders pasted tables, spacing, and Markdown syntax as unchanged plain text", () => {
  const text =
    "  Model       Prefill    Decode\n  Qwen        123.4      56.7\n\n\t**Raw** | `text`\n  <b>literal</b>  \n";
  render(<UserMessage {...props} text={text} />);

  const message = screen.getByText(text, { normalizer: (value) => value });
  expect(message.textContent).toBe(text);
  expect(message.childNodes).toHaveLength(1);
  expect(message.firstChild?.nodeType).toBe(Node.TEXT_NODE);
});

it("copies the exact original text, including Markdown and whitespace", async () => {
  render(<UserMessage {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Copy" }));
  await waitFor(() => expect(mocks.writeItemsWithFocus).toHaveBeenCalledOnce());
  const item = mocks.writeItemsWithFocus.mock.calls[0]?.[0][0] as unknown as TestClipboardItem;
  expect(Object.keys(item.items)).toEqual(["text/plain"]);
  const copiedText = await new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.readAsText(item.items["text/plain"]!);
  });
  expect(copiedText).toBe(originalText);
});

it("edits raw text, preserves attachments, and discards the draft on Cancel", () => {
  mocks.getFiles.mockReturnValue([
    { id: "file", name: "notes.txt", extension: "txt", content: "notes", mimeType: "text/plain", size: 5 },
  ]);
  render(<UserMessage {...props} attachmentIds={["file"]} />);
  expect(screen.getByText("notes")).toBeTruthy();
  edit();
  expect(draft().value).toBe(originalText);
  expect(document.activeElement).toBe(draft());
  expect(screen.getByText("notes")).toBeTruthy();
  expect(screen.getByText("Saving removes all later messages.")).toBeTruthy();
  fireEvent.change(draft(), { target: { value: "Discard this" } });
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("textbox")).toBeNull();
  expect(onEdit).not.toHaveBeenCalled();
  edit();
  expect(draft().value).toBe(originalText);
});

it("cancels on Escape without saving", () => {
  render(<UserMessage {...props} />);
  edit();
  fireEvent.keyDown(draft(), { key: "Escape" });
  expect(screen.queryByRole("textbox")).toBeNull();
  expect(onEdit).not.toHaveBeenCalled();
});

it.each(["metaKey", "ctrlKey"])("saves exact edited text with %s + Enter", async (modifier) => {
  render(<UserMessage {...props} />);
  edit();
  fireEvent.change(draft(), { target: { value: "  revised\nquestion  " } });
  fireEvent.keyDown(draft(), { key: "Enter" });
  expect(onEdit).not.toHaveBeenCalled();
  fireEvent.keyDown(draft(), { key: "Enter", [modifier]: true });
  await waitFor(() => expect(screen.queryByRole("textbox")).toBeNull());
  expect(onEdit).toHaveBeenCalledWith(props.id, "  revised\nquestion  ", expect.any(Function));
});

it("prevents duplicate submissions and closes after storage succeeds while generation continues", async () => {
  let finishGeneration!: () => void;
  let markSaved!: () => void;
  onEdit.mockImplementation((_id, _text, onSaved) => {
    markSaved = onSaved;
    return new Promise((resolve) => {
      finishGeneration = resolve;
    });
  });
  render(<UserMessage {...props} />);
  edit();
  save();
  expect((screen.getByRole("button", { name: "Saving..." }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("button", { name: "Cancel" }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.keyDown(draft(), { key: "Enter", metaKey: true });
  expect(onEdit).toHaveBeenCalledOnce();
  act(markSaved);
  expect(screen.queryByRole("textbox")).toBeNull();
  await act(async () => finishGeneration());
});

it("retains the draft after a save failure and allows retrying", async () => {
  onEdit.mockRejectedValueOnce(new Error("Disk full"));
  render(<UserMessage {...props} />);
  edit();
  fireEvent.change(draft(), { target: { value: "Keep this draft" } });
  save();
  await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("Disk full"));
  expect(draft().value).toBe("Keep this draft");
  save();
  await waitFor(() => expect(screen.queryByRole("textbox")).toBeNull());
  expect(onEdit).toHaveBeenCalledTimes(2);
});

it("rejects blank text unless the message retains attachments", async () => {
  const view = render(<UserMessage {...props} />);
  edit();
  fireEvent.change(draft(), { target: { value: " \n " } });
  expect((screen.getByRole("button", { name: "Save & send" }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.keyDown(draft(), { key: "Enter", ctrlKey: true });
  expect(onEdit).not.toHaveBeenCalled();
  view.rerender(<UserMessage {...props} attachmentIds={["file"]} />);
  expect((screen.getByRole("button", { name: "Save & send" }) as HTMLButtonElement).disabled).toBe(false);
  save();
  await waitFor(() => expect(onEdit).toHaveBeenCalledWith(props.id, " \n ", expect.any(Function)));
});

it("disables editing and saving when another operation prevents sending", () => {
  const view = render(<UserMessage {...props} canEdit={false} />);
  expect((screen.getByRole("button", { name: "Edit message" }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("button", { name: "Copy" }) as HTMLButtonElement).disabled).toBe(false);
  view.rerender(<UserMessage {...props} />);
  edit();
  view.rerender(<UserMessage {...props} canEdit={false} />);
  expect((screen.getByRole("button", { name: "Save & send" }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.keyDown(draft(), { key: "Enter", metaKey: true });
  expect(onEdit).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("textbox")).toBeNull();
});
