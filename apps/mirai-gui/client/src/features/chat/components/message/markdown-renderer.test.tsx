import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MarkdownRenderer } from "./markdown-renderer";

afterEach(cleanup);

const links = (container: HTMLElement) =>
  Array.from(container.querySelectorAll("a")).map((a) => ({ href: a.getAttribute("href"), text: a.textContent }));

describe("MarkdownRenderer links", () => {
  it("renders markdown links, bare URLs and autolinks as anchors that open externally", () => {
    const { container } = render(
      <MarkdownRenderer
        content={"See [Mirai](https://trymirai.com) or https://example.com/p?x=1 or <https://angle.test>."}
      />,
    );
    expect(links(container)).toEqual([
      { href: "https://trymirai.com/", text: "Mirai" },
      { href: "https://example.com/p?x=1", text: "https://example.com/p?x=1" },
      { href: "https://angle.test/", text: "https://angle.test" },
    ]);
    for (const a of container.querySelectorAll("a")) {
      expect(a.getAttribute("target")).toBe("_blank");
      expect(a.getAttribute("rel")).toContain("noreferrer");
    }
  });

  it("leaves scheme-less domains as plain text, like GitHub does", () => {
    const { container } = render(<MarkdownRenderer content={"Try **Spotify.com** or Reddit.com today."} />);
    expect(links(container)).toEqual([]);
    expect(container.textContent).toContain("Spotify.com");
  });

  it("keeps links inside tables and lists", () => {
    const md = "| a | b |\n|---|---|\n| [t](https://t.test/x) | y |\n\n- [l](https://l.test/y)";
    const { container } = render(<MarkdownRenderer content={md} />);
    expect(links(container).map((l) => l.href)).toEqual(["https://t.test/x", "https://l.test/y"]);
  });
});

describe("MarkdownRenderer incomplete markdown", () => {
  const unterminated = "Sources: [1](https://a.test) and [2\n\nLet me know! ✨";

  it("renders an unterminated link as text, never as a placeholder href", () => {
    const { container } = render(<MarkdownRenderer content={unterminated} />);
    expect(links(container)).toEqual([{ href: "https://a.test/", text: "1" }]);
    expect(container.textContent).toContain("Let me know! ✨");
    expect(container.textContent).not.toContain("streamdown:incomplete-link");
  });

  it("renders the same whether or not the text is still streaming", () => {
    const done = render(<MarkdownRenderer content={unterminated} />).container.innerHTML;
    const live = render(<MarkdownRenderer content={unterminated} streaming />).container.innerHTML;
    expect(live).toBe(done);
  });
});

const streaming = (content: string) => render(<MarkdownRenderer content={content} streaming />);

describe("MarkdownRenderer unterminated formatting while streaming", () => {
  it("closes bold, italic and inline code", () => {
    expect(streaming("Some **bold text").container.querySelector("strong")?.textContent).toBe("bold text");
    expect(streaming("Some *italic text").container.querySelector("em")?.textContent).toBe("italic text");
    expect(streaming("Run `pnpm check").container.querySelector("code")?.textContent).toBe("pnpm check");
  });

  it("does not leak marker characters as text", () => {
    expect(streaming("Some **bold").container.textContent).not.toContain("**");
    expect(streaming("Run `pnpm check").container.textContent).not.toContain("`");
    expect(streaming("Some ~~gone").container.textContent).not.toContain("~~");
  });

  it("renders an open code fence as a code block", async () => {
    const { container } = streaming("```ts\nconst a = 1;\nconst b = ");
    await waitFor(() => expect(container.querySelector("pre, [data-streamdown='code-block']")).not.toBeNull());
    expect(container.textContent).toContain("const a = 1;");
    expect(container.textContent).not.toContain("```");
  });

  it("renders a partial table row without breaking the table", () => {
    const { container } = streaming("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 |");
    expect(container.querySelectorAll("table").length).toBe(1);
    expect(container.querySelectorAll("tbody tr").length).toBe(2);
  });
});

describe("MarkdownRenderer streaming parity", () => {
  const full = [
    "# Title",
    "Text with **bold**, *em*, `code`, [link](https://x.test/) and ~~gone~~.",
    "- one\n- two [l](https://l.test/)",
    "| a | b |\n|---|---|\n| 1 | 2 |",
    "> quote",
  ].join("\n\n");

  it("renders a finished message the same way with and without the streaming flag", () => {
    const stat = render(<MarkdownRenderer content={full} />).container;
    const live = render(<MarkdownRenderer content={full} streaming />).container;
    const text = (el: HTMLElement) => (el.textContent ?? "").replace(/\s+/g, "");
    expect(text(live)).toBe(text(stat));
    for (const sel of ["h1", "strong", "em", "code", "a", "li", "table", "tr", "blockquote", "del"]) {
      expect(live.querySelectorAll(sel).length, sel).toBe(stat.querySelectorAll(sel).length);
    }
  });
});

describe("MarkdownRenderer sanitization", () => {
  it("drops script tags and inline event handlers from raw HTML", () => {
    const { container } = render(
      <MarkdownRenderer
        content={'Hi <script>window.pwned = 1</script><img src="x" onerror="window.pwned = 1"> there'}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("[onerror]")).toBeNull();
    expect(container.textContent).toContain("Hi");
  });

  it("neutralises javascript: links", () => {
    const { container } = render(<MarkdownRenderer content={"[click](javascript:alert(1))"} />);
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(container.textContent).toContain("click");
  });

  it("drops iframes and forms", () => {
    const { container } = render(
      <MarkdownRenderer
        content={'<iframe src="https://x.test"></iframe><form action="https://x.test"><input name="a"></form>'}
      />,
    );
    expect(container.querySelector("iframe")).toBeNull();
    expect(container.querySelector("form")).toBeNull();
  });
});

describe("MarkdownRenderer math and CJK", () => {
  it("renders block and inline math with KaTeX, leaving single-dollar prices alone", () => {
    const { container } = render(
      <MarkdownRenderer content={"Energy: $$E = mc^2$$ costs $5 and $10.\n\n$$\n\\int_0^1 x\\,dx\n$$"} />,
    );
    expect(container.querySelectorAll(".katex").length).toBe(2);
    expect(container.textContent).toContain("$5 and $10");
    expect(container.textContent).not.toContain("$$");
  });

  it("applies emphasis next to CJK punctuation", () => {
    const { container } = render(<MarkdownRenderer content={"これは**「重要」**です。**中文测试**，好的。"} />);
    const strong = Array.from(container.querySelectorAll("strong")).map((s) => s.textContent);
    expect(strong).toEqual(["「重要」", "中文测试"]);
  });
});
