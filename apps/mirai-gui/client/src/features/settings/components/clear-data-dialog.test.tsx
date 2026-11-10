import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ClearDataDialog } from "./clear-data-dialog";

vi.mock("@/platform/platform-info", () => ({ platformInfo: { features: { logExport: false } } }));
vi.mock("@/components/ui/modal", () => ({
  Modal: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));
vi.mock("@/stores/use-cleanup-store", () => {
  const state = {
    preview: { dialogs: { count: 1, sizeBytes: 10 }, models: { count: 1, sizeBytes: 100 }, logs: { sizeBytes: 500 } },
    previewLoading: false,
    result: null,
    executing: false,
    fetchPreview: vi.fn(),
    execute: vi.fn(),
    reset: vi.fn(),
  };
  return { useCleanupStore: (selector: (value: typeof state) => unknown) => selector(state) };
});
afterEach(cleanup);

it("offers only implemented cleanup categories in the browser", () => {
  render(<ClearDataDialog open onClose={() => {}} />);
  expect(screen.getByText("Dialogs")).toBeTruthy();
  expect(screen.getByText("Downloaded models")).toBeTruthy();
  expect(screen.queryByText("Logs")).toBeNull();
  expect(screen.getAllByRole("checkbox")).toHaveLength(2);
});
