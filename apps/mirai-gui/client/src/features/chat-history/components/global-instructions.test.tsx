import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import GlobalInstructions from "./global-instructions";

vi.mock("@/components/ui/toast/use-toast", () => ({ useToast: () => ({ error: vi.fn() }) }));

afterEach(cleanup);

const textarea = () => screen.getByPlaceholderText("Add instructions to all chats") as HTMLTextAreaElement;

it("takes the stored text while nothing is unsaved", () => {
  const view = render(<GlobalInstructions instructions="" onSave={vi.fn(async () => true)} />);

  view.rerender(<GlobalInstructions instructions="loaded from disk" onSave={vi.fn(async () => true)} />);

  expect(textarea().value).toBe("loaded from disk");
});

it("does not revive a save still in flight after the text was cleared", () => {
  const onSave = vi.fn(() => new Promise<boolean>(() => {}));
  const view = render(<GlobalInstructions instructions="" onSave={onSave} />);

  fireEvent.change(textarea(), { target: { value: "abc" } });
  fireEvent.blur(textarea());
  fireEvent.change(textarea(), { target: { value: "" } });
  fireEvent.blur(textarea());
  view.rerender(<GlobalInstructions instructions="abc" onSave={onSave} />);

  expect(textarea().value).toBe("");
});

it("still writes a clear typed while the previous save is in flight", async () => {
  let finishFirst!: (ok: boolean) => void;
  const onSave = vi
    .fn<(value: string) => Promise<boolean>>()
    .mockImplementationOnce(() => new Promise((resolve) => (finishFirst = resolve)))
    .mockResolvedValue(true);
  const view = render(<GlobalInstructions instructions="" onSave={onSave} />);

  fireEvent.change(textarea(), { target: { value: "abc" } });
  fireEvent.blur(textarea());
  fireEvent.change(textarea(), { target: { value: "" } });
  view.unmount();

  expect(onSave.mock.calls.map(([v]) => v)).toEqual(["abc", ""]);
  finishFirst(true);
  await Promise.resolve();
  expect(onSave).toHaveBeenCalledTimes(2);
});
