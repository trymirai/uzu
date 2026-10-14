import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip/provider";
import { ModelCardRow } from "./model-card-row";

afterEach(cleanup);

it("shows a preflight failure and lets the user remove its partial download", () => {
  const cancel = vi.fn();
  render(
    <TooltipProvider>
      <ModelCardRow
        name="Model"
        logo={null}
        state={{ status: "error" }}
        error="This model architecture is not supported by WebGPU"
        onRetry={vi.fn()}
        onCancel={cancel}
      />
    </TooltipProvider>,
  );
  expect(screen.getByRole("alert").textContent).toContain("not supported by WebGPU");
  expect(screen.getByRole("button", { name: "Retry download" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Delete partial download" }));
  expect(cancel).toHaveBeenCalledOnce();
});

it("keeps partial-file deletion available when model capabilities are unavailable", () => {
  render(
    <TooltipProvider>
      <ModelCardRow
        name="Model"
        logo={null}
        state={{ status: "error" }}
        unavailableReason="Could not check this model's chat capabilities"
        onRetry={vi.fn()}
        onCancel={vi.fn()}
      />
    </TooltipProvider>,
  );
  expect(screen.queryByRole("button", { name: "Retry download" })).toBeNull();
  expect(screen.getByRole("button", { name: "Delete partial download" })).toBeTruthy();
});
