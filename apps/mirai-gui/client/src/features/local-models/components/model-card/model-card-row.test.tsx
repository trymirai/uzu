import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip/provider";
import { ModelCardRow } from "./model-card-row";

afterEach(cleanup);

it("shows a download failure and lets the user remove its partial download", () => {
  const cancel = vi.fn();
  render(
    <TooltipProvider>
      <ModelCardRow
        name="Model"
        logo={null}
        state={{ status: "error" }}
        error="Storage is full"
        onRetry={vi.fn()}
        onCancel={cancel}
      />
    </TooltipProvider>,
  );
  expect(screen.getByRole("alert").textContent).toContain("Storage is full");
  expect(screen.getByRole("button", { name: "Retry download" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Delete partial download" }));
  expect(cancel).toHaveBeenCalledOnce();
});
