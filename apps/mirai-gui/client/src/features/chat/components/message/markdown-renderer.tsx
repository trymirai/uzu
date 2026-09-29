import { CopyButton } from "@/components/ui/copy-button";
import { Checkbox } from "@/components/ui/checkbox";
import { useAppStore } from "@/stores/use-app-store";
import { writeItemsWithFocus } from "@/utils/clipboard";
import { cjk } from "@streamdown/cjk";
import { createCodePlugin } from "@streamdown/code";
import { math } from "@streamdown/math";
import type { Element } from "hast";
import "katex/dist/katex.min.css";
import React, { useMemo, useRef } from "react";
import type { BundledTheme } from "shiki";
import { Streamdown } from "streamdown";

type MarkdownRendererProps = {
  content: string;
  className?: string;
  useOneFontSize?: boolean;
  /** Animates arriving text; incomplete markdown is completed either way. */
  streaming?: boolean;
};

// Safety mode opens links via window.open, which WKWebView drops; anchors go
// through the platform's click interception instead. This also drops
// Streamdown's own leave-site confirmation.
const LINK_SAFETY = { enabled: false } as const;
// The default placeholder href gets sanitized into "[blocked]" mid-stream.
const REMEND = { linkMode: "text-only" } as const;

type MarkdownComponentProps<K extends keyof JSX.IntrinsicElements> = JSX.IntrinsicElements[K] & { node?: Element };

const cloneTableElement = (table: HTMLTableElement): HTMLTableElement => {
  const clone = table.cloneNode(true);
  if (clone instanceof HTMLTableElement) {
    return clone;
  }
  throw new Error("Table element not found");
};

const MarkdownTable = ({ children, ...props }: MarkdownComponentProps<"table">) => {
  const tableRef = useRef<HTMLTableElement>(null);

  const handleTableCopy = async () => {
    if (!tableRef.current) {
      throw new Error("Table element not found");
    }

    const tableClone = cloneTableElement(tableRef.current);

    const buttons = tableClone.querySelectorAll('button, .copy-button, [class*="copy"]');
    buttons.forEach((button) => button.remove());

    await writeItemsWithFocus([
      new ClipboardItem({
        "text/html": new Blob([tableClone.outerHTML], { type: "text/html" }),
        "text/plain": new Blob([tableClone.textContent || ""], {
          type: "text/plain",
        }),
      }),
    ]);
  };

  return (
    <div className="relative mt-3 mb-5">
      <div className="absolute right-1 top-[10px] z-10">
        <CopyButton className="!min-w-6 !min-h-6" onCopy={handleTableCopy} />
      </div>
      <div className="overflow-x-auto thin-scrollbar">
        <table ref={tableRef} className="min-w-full" {...props}>
          {children}
        </table>
      </div>
    </div>
  );
};

const CustomComponents = {
  h1: ({ children, ...props }: MarkdownComponentProps<"h1">) => (
    <h1 className="text-[24px] font-medium leading-[130%] mb-4 text-label-title" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }: MarkdownComponentProps<"h2">) => (
    <h2 className="text-[20px] font-medium leading-[130%] mb-3 text-label-title" {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }: MarkdownComponentProps<"h3">) => (
    <h3 className="text-[18px] font-medium leading-[130%] mb-2 text-label-title" {...props}>
      {children}
    </h3>
  ),
  h4: ({ children, ...props }: MarkdownComponentProps<"h4">) => (
    <h4 className="text-[16px] font-medium leading-[130%] mb-2 text-label-title" {...props}>
      {children}
    </h4>
  ),
  h5: ({ children, ...props }: MarkdownComponentProps<"h5">) => (
    <h5 className="text-[14px] font-medium leading-[130%] mb-1 text-label-title" {...props}>
      {children}
    </h5>
  ),
  h6: ({ children, ...props }: MarkdownComponentProps<"h6">) => (
    <h6 className="text-xs font-medium leading-[130%] mb-1 text-label-title" {...props}>
      {children}
    </h6>
  ),
  p: ({ children, ...props }: MarkdownComponentProps<"p">) => (
    <p className="mb-3 mt-2 text-[15px] font-[350] leading-[150%]" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }: MarkdownComponentProps<"ul">) => (
    <ul
      className="mb-3 ml-[9px] list-disc text-[15px] font-[350] leading-[150%] text-label-title space-y-3 mt-3 pl-[10px]"
      {...props}
    >
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: MarkdownComponentProps<"ol">) => (
    <ol
      className="mb-3 list-decimal list-inside text-[15px] font-[350] leading-[150%] text-label-title space-y-3 mt-3 tabular-nums marker:[font-variant-numeric:tabular-nums]"
      {...props}
    >
      {children}
    </ol>
  ),
  li: ({ children, ...props }: MarkdownComponentProps<"li">) => (
    <li
      className="text-[15px] font-[350] leading-[150%] text-label-title mb-3 last:mb-0 tabular-nums [&>p:first-child]:inline [&>p:first-child]:m-0"
      {...props}
    >
      {children}
    </li>
  ),
  blockquote: ({ children, ...props }: MarkdownComponentProps<"blockquote">) => (
    <blockquote
      className="border-l-[4px] border-button-border pl-3 text-[15px] font-[350] leading-[150%] text-label-title my-3"
      {...props}
    >
      {children}
    </blockquote>
  ),
  table: MarkdownTable,
  thead: ({ children, ...props }: MarkdownComponentProps<"thead">) => (
    <thead className="bg-background border-b border-button-border" {...props}>
      {children}
    </thead>
  ),
  tbody: ({ children, ...props }: MarkdownComponentProps<"tbody">) => (
    <tbody className="bg-background" {...props}>
      {children}
    </tbody>
  ),
  tr: ({ children, ...props }: MarkdownComponentProps<"tr">) => (
    <tr className="border-b border-cell-border [&:has(th)]:border-button-border" {...props}>
      {children}
    </tr>
  ),
  th: ({ children, ...props }: MarkdownComponentProps<"th">) => (
    <th className="py-3 text-left text-[15px] font-[350] leading-[150%] text-label-title" {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }: MarkdownComponentProps<"td">) => (
    <td className="py-3 text-[15px] font-[350] leading-[150%] text-label-title" {...props}>
      {children}
    </td>
  ),
  hr: ({ ...props }: MarkdownComponentProps<"hr">) => (
    <hr className="my-5 border-gray-300 dark:border-gray-600" {...props} />
  ),
  img: ({ ...props }: MarkdownComponentProps<"img">) => <img className="max-w-full h-auto rounded my-3" {...props} />,
  input: ({ type, checked, ...props }: MarkdownComponentProps<"input">) => {
    if (type === "checkbox") {
      return (
        <div className="inline-flex items-center align-text-bottom">
          <Checkbox checked={!!checked} onChange={() => {}} disabled size="sm" className="mr-1" />
        </div>
      );
    }
    return <input type={type} {...props} />;
  },
  strong: ({ children, ...props }: MarkdownComponentProps<"strong">) => (
    <strong className="text-[15px] font-medium leading-[150%] tracking-[0.2px] text-label-title" {...props}>
      {children}
    </strong>
  ),
  em: ({ children, ...props }: MarkdownComponentProps<"em">) => (
    <em className="text-[15px] italic font-normal leading-[150%] text-label-title" {...props}>
      {children}
    </em>
  ),
};

const OneFontSizeComponents = {
  h1: ({ children, ...props }: MarkdownComponentProps<"h1">) => (
    <h1 className="text-xs font-medium mb-1" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }: MarkdownComponentProps<"h2">) => (
    <h2 className="text-xs font-medium mb-1" {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }: MarkdownComponentProps<"h3">) => (
    <h3 className="text-xs font-medium mb-1" {...props}>
      {children}
    </h3>
  ),
  h4: ({ children, ...props }: MarkdownComponentProps<"h4">) => (
    <h4 className="text-xs font-medium mb-1" {...props}>
      {children}
    </h4>
  ),
  h5: ({ children, ...props }: MarkdownComponentProps<"h5">) => (
    <h5 className="text-xs font-medium mb-1" {...props}>
      {children}
    </h5>
  ),
  h6: ({ children, ...props }: MarkdownComponentProps<"h6">) => (
    <h6 className="text-xs font-medium mb-1" {...props}>
      {children}
    </h6>
  ),
  p: ({ children, ...props }: MarkdownComponentProps<"p">) => (
    <p className="mb-1 mt-1 text-xs font-[350]" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }: MarkdownComponentProps<"ul">) => (
    <ul className="mb-2 ml-[9px] list-disc text-xs font-[350] space-y-2 mt-2 pl-[10px]" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: MarkdownComponentProps<"ol">) => (
    <ol
      className="mb-2 list-decimal list-inside text-xs font-[350] space-y-2 mt-2 tabular-nums marker:[font-variant-numeric:tabular-nums]"
      {...props}
    >
      {children}
    </ol>
  ),
  li: ({ children, ...props }: MarkdownComponentProps<"li">) => (
    <li className="font-[350] mb-2 last:mb-0 tabular-nums [&>p:first-child]:inline [&>p:first-child]:m-0" {...props}>
      {children}
    </li>
  ),
  blockquote: ({ children, ...props }: MarkdownComponentProps<"blockquote">) => (
    <blockquote className="border-l-[4px] border-button-border pl-2 text-xs font-[350] my-3" {...props}>
      {children}
    </blockquote>
  ),
  table: MarkdownTable,
  thead: ({ children, ...props }: MarkdownComponentProps<"thead">) => (
    <thead className="bg-background border-b border-button-border" {...props}>
      {children}
    </thead>
  ),
  tbody: ({ children, ...props }: MarkdownComponentProps<"tbody">) => (
    <tbody className="bg-background" {...props}>
      {children}
    </tbody>
  ),
  tr: ({ children, ...props }: MarkdownComponentProps<"tr">) => (
    <tr className="border-b border-cell-border [&:has(th)]:border-button-border" {...props}>
      {children}
    </tr>
  ),
  th: ({ children, ...props }: MarkdownComponentProps<"th">) => (
    <th className="py-2 text-left text-xs font-[350]" {...props}>
      {children}
    </th>
  ),
  td: ({ children, ...props }: MarkdownComponentProps<"td">) => (
    <td className="py-2 text-xs font-[350]" {...props}>
      {children}
    </td>
  ),
  hr: ({ ...props }: MarkdownComponentProps<"hr">) => (
    <hr className="my-2 border-gray-300 dark:border-gray-600" {...props} />
  ),
  img: ({ ...props }: MarkdownComponentProps<"img">) => <img className="max-w-full h-auto rounded my-3" {...props} />,
  input: ({ type, checked, ...props }: MarkdownComponentProps<"input">) => {
    if (type === "checkbox") {
      return (
        <div className="inline-flex items-center align-text-bottom">
          <Checkbox checked={!!checked} onChange={() => {}} disabled size="sm" className="mr-1" />
        </div>
      );
    }
    return <input type={type} {...props} />;
  },
  strong: ({ children, ...props }: MarkdownComponentProps<"strong">) => (
    <strong className="text-xs font-medium tracking-[0.2px]" {...props}>
      {children}
    </strong>
  ),
  em: ({ children, ...props }: MarkdownComponentProps<"em">) => (
    <em className="text-xs italic font-normal" {...props}>
      {children}
    </em>
  ),
};

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  useOneFontSize = false,
  className = "max-w-none text-label-title markdown-body",
  streaming = false,
}) => {
  const isDarkMode = useAppStore((s) => s.isDarkMode);

  const codePlugin = useMemo(() => {
    const themes: [BundledTheme, BundledTheme] = isDarkMode
      ? ["github-dark", "github-dark"]
      : ["github-light", "github-light"];
    return createCodePlugin({ themes });
  }, [isDarkMode]);
  return (
    <div className={className}>
      <Streamdown
        mode="streaming"
        isAnimating={streaming}
        parseIncompleteMarkdown
        linkSafety={LINK_SAFETY}
        remend={REMEND}
        components={useOneFontSize ? OneFontSizeComponents : CustomComponents}
        plugins={{ code: codePlugin, math, cjk }}
      >
        {content}
      </Streamdown>
    </div>
  );
};
