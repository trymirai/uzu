import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { useCleanupStore } from "@/stores/use-cleanup-store";
import { ClearDataDialog } from "./clear-data-dialog";

const preview = {
  dialogs: { count: 3, sizeBytes: 3_000 },
  models: { count: 2, sizeBytes: 5_000_000_000 },
  logs: { sizeBytes: 100 },
};
const cleanupDefaults = useCleanupStore.getState();

beforeEach(() => {
  useCleanupStore.setState(
    { ...cleanupDefaults, fetchPreview: async () => useCleanupStore.setState({ preview }) },
    true,
  );
});
afterEach(cleanup);

it("preselects chats and logs, and leaves downloaded models to opt into", async () => {
  render(<ClearDataDialog open onClose={() => {}} />);

  await screen.findByText("2 models · 5.0 GB");
  expect(screen.getAllByRole("checkbox").map((box) => (box as HTMLInputElement).checked)).toEqual([true, false, true]);
});
