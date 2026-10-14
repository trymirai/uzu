import "fake-indexeddb/auto";
import { expect, it } from "vitest";
import { browserDb } from "./browser-db";

it("merges concurrent metadata updates atomically without losing either writer", async () => {
  const key = "test:concurrent-history";
  await browserDb.delete("modelStates", key);
  await Promise.all([
    browserDb.update<string[]>("modelStates", key, (value = []) => [...value, "first"]),
    browserDb.update<string[]>("modelStates", key, (value = []) => [...value, "second"]),
  ]);
  expect(await browserDb.get("modelStates", key)).toEqual(["first", "second"]);
  const failure = new Error("Do not replace the committed metadata");
  await expect(
    browserDb.update("modelStates", key, () => {
      throw failure;
    }),
  ).rejects.toBe(failure);
  expect(await browserDb.get("modelStates", key)).toEqual(["first", "second"]);
  await browserDb.delete("modelStates", key);
});
