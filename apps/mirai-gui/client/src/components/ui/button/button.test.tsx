import { cleanup, render } from "@testing-library/react";
import { Download } from "lucide-react";
import { afterEach, describe, expect, it } from "vitest";
import { Button } from ".";

const FunctionIcon = () => <svg data-testid="function-icon" />;

describe("Button icon", () => {
  afterEach(cleanup);

  it("renders a forwardRef component such as a lucide icon", () => {
    const { container } = render(<Button icon={Download}>Export</Button>);
    const icon = container.querySelector("svg");
    expect(icon).not.toBeNull();
    expect(icon?.getAttribute("aria-hidden")).toBe("true");
    expect(icon?.style.width).toBe("16px");
  });

  it("renders a function component and a ready element", () => {
    expect(render(<Button icon={FunctionIcon}>A</Button>).queryByTestId("function-icon")).not.toBeNull();
    cleanup();
    expect(render(<Button icon={<FunctionIcon />}>B</Button>).queryByTestId("function-icon")).not.toBeNull();
  });

  it("hides the icon while loading", () => {
    const { queryByTestId } = render(
      <Button icon={FunctionIcon} loading>
        C
      </Button>,
    );
    expect(queryByTestId("function-icon")).toBeNull();
  });
});
