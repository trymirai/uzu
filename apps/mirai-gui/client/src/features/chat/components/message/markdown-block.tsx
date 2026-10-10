import { Lexer, walkTokens, type Tokens } from "marked";
import { createElement, memo } from "react";
import { Block, defaultComponents, parseMarkdownIntoBlocks, type BlockProps } from "streamdown";

const looseList = (content: string): Tokens.List | undefined => {
  // Dedenting an item changes the column that determines tab indentation.
  if (content.includes("\t")) return;
  // Marked does not track footnote scope or unfinished display-math blocks.
  if (content.includes("[^") || content.includes("$$")) return;
  const lexer = new Lexer({ gfm: true });
  const tokens = lexer.lex(content);
  const blocks = tokens.filter((token) => token.type !== "space");
  const list = blocks[0];
  if (blocks.length !== 1 || list?.type !== "list" || !list.loose) return;
  if (Object.keys(tokens.links).length > 0) return;
  let supported = true;
  walkTokens(tokens, (token) => {
    if (token.type === "html" || token.type === "def" || (token.type === "list_item" && token.task)) supported = false;
  });
  return supported ? (list as Tokens.List) : undefined;
};

// Streamdown normally treats a whole list as one Markdown block. A single
// list item can contain an entire long reply, so keep its container while
// letting Streamdown cache the completed paragraphs and fences inside it.
export const MarkdownBlock = memo(function MarkdownBlock(props: BlockProps) {
  // Direction belongs to the whole block, including its list markers.
  if (props.dir) return <Block {...props} />;
  const list = looseList(props.content);
  if (!list) return <Block {...props} />;

  const Item = props.components?.li ?? defaultComponents.li;
  const children = list.items.map((item, itemIndex) =>
    createElement(
      Item,
      { key: itemIndex },
      parseMarkdownIntoBlocks(item.text).map((content, index) => (
        <MarkdownBlock key={index} {...props} content={content} index={index} />
      )),
    ),
  );
  if (list.ordered) {
    const OrderedList = props.components?.ol ?? defaultComponents.ol;
    return createElement(
      OrderedList,
      { start: typeof list.start === "number" && list.start !== 1 ? list.start : undefined },
      children,
    );
  }
  const UnorderedList = props.components?.ul ?? defaultComponents.ul;
  return createElement(UnorderedList, {}, children);
});
