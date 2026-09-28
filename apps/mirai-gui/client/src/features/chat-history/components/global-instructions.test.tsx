import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import GlobalInstructions from "./global-instructions";

afterEach(cleanup);

const textarea = () => screen.getByPlaceholderText("Add instructions to all chats") as HTMLTextAreaElement;

it("keeps text typed while an earlier save is still confirming", () => {
  const onSave = vi.fn();
  const view = render(<GlobalInstructions instructions="" onSave={onSave} />);

  fireEvent.change(textarea(), { target: { value: "first draft" } });
  fireEvent.blur(textarea());
  expect(onSave).toHaveBeenCalledWith("first draft");

  fireEvent.change(textarea(), { target: { value: "first draft, extended" } });
  view.rerender(<GlobalInstructions instructions="first draft" onSave={onSave} />);

  expect(textarea().value).toBe("first draft, extended");
});

it("takes the stored text while nothing is unsaved", () => {
  const view = render(<GlobalInstructions instructions="" onSave={vi.fn()} />);

  view.rerender(<GlobalInstructions instructions="loaded from disk" onSave={vi.fn()} />);

  expect(textarea().value).toBe("loaded from disk");
});
