import { expect, it, vi } from "vitest";
import { withFileQueue } from "./file-queue";

it("runs writes to one key in call order even when an earlier one is slower", async () => {
  let finishFirst!: () => void;
  const order: string[] = [];
  const first = withFileQueue("k", () =>
    new Promise<void>((resolve) => (finishFirst = () => resolve())).then(() => order.push("first")),
  );
  const second = withFileQueue("k", async () => void order.push("second"));
  await Promise.resolve();
  expect(order).toEqual([]);

  finishFirst();
  await Promise.all([first, second]);
  expect(order).toEqual(["first", "second"]);
});

it("lets the next write run after a failed one", async () => {
  const failing = withFileQueue("k", () => Promise.reject(new Error("disk full")));
  const next = vi.fn(async () => "ok");

  await expect(failing).rejects.toThrow("disk full");
  await expect(withFileQueue("k", next)).resolves.toBe("ok");
});
