import { cleanup, render, waitFor } from "@testing-library/react";
import { memo, Profiler, type ComponentProps } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { MarkdownRenderer } from "./markdown-renderer";

const renderBlock = vi.hoisted(() => vi.fn());
vi.mock("streamdown", async (importOriginal) => {
  const actual = await importOriginal<typeof import("streamdown")>();
  const MeasuredBlock = memo(function MeasuredBlock(props: ComponentProps<typeof actual.Block>) {
    return (
      <Profiler id={props.content} onRender={renderBlock}>
        <actual.Block {...props} />
      </Profiler>
    );
  });
  return {
    ...actual,
    Streamdown: (props: ComponentProps<typeof actual.Streamdown>) => (
      <actual.Streamdown {...props} BlockComponent={MeasuredBlock} />
    ),
  };
});

afterEach(() => {
  cleanup();
  renderBlock.mockClear();
});

it("does not revisit completed code blocks when more prose streams into the response", async () => {
  const code = "```typescript\nconst answer = 42;\n```\n\n";
  const view = render(<MarkdownRenderer content={`${code}Answer`} streaming />);
  await waitFor(() => expect(view.container.querySelector("pre code span[style]")).toBeTruthy());
  renderBlock.mockClear();

  for (let i = 1; i <= 20; i++) {
    view.rerender(<MarkdownRenderer content={`${code}Answer${" more".repeat(i)}`} streaming />);
  }

  expect(view.container.textContent).toContain(`Answer${" more".repeat(20)}`);
  // The lazy highlighter may finish once; appending prose must not repeatedly
  // revisit the completed code block through a changing plugin context.
  expect(renderBlock.mock.calls.filter(([content]) => content.includes("const answer")).length).toBeLessThanOrEqual(1);
});
