import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { memo, type ComponentProps } from "react";
import { Block, useIsCodeFenceIncomplete, type BlockProps } from "streamdown";
import { MarkdownBlock } from "./markdown-block";
import { MarkdownRenderer } from "./markdown-renderer";

const control = vi.hoisted(() => ({ stock: false }));
const parsedBlocks = vi.hoisted(() => vi.fn());
vi.mock("streamdown", async (importOriginal) => {
  const actual = await importOriginal<typeof import("streamdown")>();
  const MeasuredBlock = memo(function MeasuredBlock(props: ComponentProps<typeof actual.Block>) {
    parsedBlocks(props.content);
    return <actual.Block {...props} />;
  });
  return {
    ...actual,
    Block: MeasuredBlock,
    Streamdown: (props: ComponentProps<typeof actual.Streamdown>) => (
      <actual.Streamdown {...props} BlockComponent={control.stock ? actual.Block : props.BlockComponent} />
    ),
  };
});

afterEach(cleanup);

const html = (content: string, stock: boolean, small: boolean) => {
  control.stock = stock;
  const view = render(<MarkdownRenderer content={content} useOneFontSize={small} />);
  const output = view.container.innerHTML.replace(/ node="\[object Object\]"/g, "").replace(/>\n+</g, "><");
  cleanup();
  return output;
};

const fixtures = {
  "one loose list item with many paragraphs": "- First paragraph.\n\n  Second paragraph.\n\n  Third paragraph.",
  "multiple loose items": "- First.\n\n  Another paragraph.\n\n- Second.\n\n  More.",
  "ordered list starting at five": "5. First.\n\n   More.\n\n6. Second.\n\n   More.",
  "ordered list starting at zero": "0. First.\n\n   More.\n\n1. Second.\n\n   More.",
  "tight lists": "- One\n- Two\n- Three",
  "nested loose lists": "- Parent.\n\n  - Child.\n\n    Another paragraph.\n\n  - Other child.\n\n- Other parent.",
  "nested ordered and unordered lists": "3. Parent.\n\n   - Child.\n\n     Paragraph.\n\n   - Other.\n\n4. Next.",
  "tight list inside loose list": "- Parent.\n\n  - One\n  - Two\n\n  Final paragraph.",
  "task lists": "- [x] Done.\n\n  Explanation.\n\n- [ ] Pending.",
  "code fence": "- Explanation.\n\n  ```text\n  a < b\n  next\n  ```\n\n  Conclusion.",
  "headings and inline formatting": "- ## Heading\n\n  Paragraph with **bold**, *italic*, and `code`.\n\n  More.",
  math: "- Formula.\n\n  $$\n  x^2 + y^2 = 1\n  $$\n\n  Inline $$a + b$$.",
  links: "- [Mirai](https://trymirai.com).\n\n  More at https://example.com.",
  "reference definitions": "- [Mirai][m].\n\n  More.\n\n[m]: https://trymirai.com",
  "definition inside list": "- [Mirai][m].\n\n  [m]: https://trymirai.com\n\n  More.",
  footnotes: "- Some text[^note].\n\n  More.\n\n[^note]: Footnote.",
  "footnotes with multiple paragraphs":
    "- Intro.\n\n  [^a]: first paragraph\n\n    second paragraph\n\n  This is a footnote[^a].",
  "footnotes shared across items": "- Intro[^a].\n\n  More.\n\n- Second[^a].\n\n  [^a]: Shared footnote.",
  "unfinished display math": "- Intro.\n\n  $$\n  a + b\n\n  c + d\n\n  End.",
  "raw HTML": "- First.\n\n  <div>\n  **Literal**\n  </div>\n\n  More.",
  "inline HTML": "- First.\n\n  <span>Inline</span> content.\n\n  More.",
  "nested quote": "- Parent.\n\n  > Quoted **text**.\n  >\n  > Another quoted paragraph.\n\n  Final.",
  "lazy continuation": "- First line.\nLazy continuation.\n\n  Another paragraph.\nStill inside the item.",
  "indented code": "- First.\n\n      indented code\n      second line\n\n  More.",
  "multiple digit ordered list": "12. First.\n\n    Paragraph.\n\n13. Next.",
  "tab-indented paragraph": "- First.\n\n\t  Another paragraph.\n\n  Final.",
  "tab after list marker": "-\tFirst.\n\n\tSecond paragraph.",
  "Arabic paragraphs": "- الفقرة الأولى.\n\n  الفقرة الثانية مع **نص غامق**.\n\n  النص الأخير.",
  "Hebrew paragraphs": "- הפסקה הראשונה.\n\n  הפסקה השנייה עם **טקסט מודגש**.\n\n  טקסט אחרון.",
  "nested task list": "- Parent.\n\n  - [x] Child.\n\n  Final.",
  table: "- Parent.\n\n  | Name | Value |\n  | --- | --- |\n  | First | 1 |\n\n  Final.",
};

it.each(Object.entries(fixtures))("preserves existing markup and styles for %s", (_name, markdown) => {
  for (const small of [false, true]) {
    expect(html(markdown, false, small)).toBe(html(markdown, true, small));
  }
});

it("only reparses the active paragraph inside a growing long list item", () => {
  control.stock = false;
  const beginning = "- Opening paragraph.\n\n";
  const completed = Array.from({ length: 80 }, (_, i) => `  Paragraph ${i}: ${"completed words ".repeat(12)}\n\n`).join(
    "",
  );
  const content = `${beginning}${completed}  Still thinking`;
  const view = render(<MarkdownRenderer content={content} useOneFontSize />);
  const firstParagraph = view.container.querySelector("p");
  parsedBlocks.mockClear();

  view.rerender(<MarkdownRenderer content={`${content} about the answer.`} useOneFontSize />);

  expect(view.container.querySelectorAll("ul").length).toBe(1);
  expect(view.container.querySelectorAll("li").length).toBe(1);
  expect(view.container.querySelector("p")).toBe(firstParagraph);
  expect(view.container.textContent).toContain("Still thinking about the answer.");
  expect(parsedBlocks).toHaveBeenCalledTimes(1);
  expect(parsedBlocks.mock.calls[0]?.[0]).toContain("Still thinking about the answer.");
  expect(parsedBlocks.mock.calls[0]?.[0].length).toBeLessThan(100);
});

it("keeps the same output while a list changes between tight and loose during streaming", () => {
  const stages = [
    "- Start",
    "- Start\n\n  More",
    "- Start\n\n  More\n\n- Next",
    "- Start\n\n  More\n\n- Next\n\n  Final",
  ];
  for (const content of stages) expect(html(content, false, true)).toBe(html(content, true, true));
});

it("updates code-fence context when the final list block completes", () => {
  const Code = (props: object) => (
    <code data-incomplete={useIsCodeFenceIncomplete()}>
      {"children" in props && typeof props.children === "string" ? props.children : null}
    </code>
  );
  const props: BlockProps = {
    content: "- First.\n\n  ```text\n  completed\n  ```\n\n  ```text\n  arriving\n  ```",
    index: 0,
    isIncomplete: true,
    shouldParseIncompleteMarkdown: true,
    shouldNormalizeHtmlIndentation: true,
    components: { code: Code },
  };
  const view = render(<MarkdownBlock {...props} />);
  expect(view.container.querySelectorAll('code[data-incomplete="true"]')).toHaveLength(2);

  view.rerender(<MarkdownBlock {...props} isIncomplete={false} />);
  expect(view.container.querySelectorAll('code[data-incomplete="false"]')).toHaveLength(2);
  expect(view.container.textContent).toContain("arriving");
});

it("keeps explicit direction on the entire list including its markers", () => {
  const props: BlockProps = {
    content: "- الفقرة الأولى.\n\n  الفقرة الثانية.",
    index: 0,
    isIncomplete: false,
    shouldParseIncompleteMarkdown: true,
    shouldNormalizeHtmlIndentation: true,
    dir: "rtl",
  };
  const optimized = render(<MarkdownBlock {...props} />);
  const actual = optimized.container.innerHTML.replace(/>\n+</g, "><");
  cleanup();
  const stock = render(<Block {...props} />);
  expect(actual).toBe(stock.container.innerHTML.replace(/>\n+</g, "><"));
});
