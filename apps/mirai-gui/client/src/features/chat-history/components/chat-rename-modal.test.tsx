import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ChatRenameModal } from "./chat-rename-modal";

afterEach(cleanup);

it("accepts a title longer than the former hard limit without cutting it", async () => {
  const onConfirm = vi.fn(async () => {});
  render(<ChatRenameModal isOpen onClose={vi.fn()} onConfirm={onConfirm} currentName="Original title" />);
  const title = "Detailed comparison of model throughput, response latency, and energy usage";
  const input = screen.getByRole("textbox", { name: "Chat name" });
  expect(input.getAttribute("maxlength")).toBeNull();
  fireEvent.change(input, { target: { value: title } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(onConfirm).toHaveBeenCalledWith(title));
});

it("preserves edits when the generated title changes, and refreshes it when reopened", async () => {
  const onConfirm = vi.fn(async () => {});
  const props = {
    isOpen: true,
    onClose: vi.fn(),
    onConfirm,
    currentName: "Original title",
  };
  const view = render(<ChatRenameModal {...props} />);
  fireEvent.change(screen.getByRole("textbox", { name: "Chat name" }), { target: { value: "  Mine  " } });
  const currentName = "Model rename";
  view.rerender(<ChatRenameModal {...props} currentName={currentName} />);
  expect((screen.getByRole("textbox", { name: "Chat name" }) as HTMLInputElement).value).toBe("  Mine  ");
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(onConfirm).toHaveBeenCalledWith("Mine"));
  view.rerender(<ChatRenameModal {...props} isOpen={false} currentName={currentName} />);
  view.rerender(<ChatRenameModal {...props} currentName={currentName} />);
  expect((screen.getByRole("textbox", { name: "Chat name" }) as HTMLInputElement).value).toBe("Model rename");
});

it("requires a nonempty title", async () => {
  const onConfirm = vi.fn(async () => {});
  render(<ChatRenameModal isOpen onClose={vi.fn()} onConfirm={onConfirm} currentName="Original title" />);
  fireEvent.change(screen.getByRole("textbox", { name: "Chat name" }), { target: { value: "  " } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  expect(await screen.findByText("Name cannot be empty")).toBeDefined();
  expect(onConfirm).not.toHaveBeenCalled();
});
