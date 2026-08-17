#[cfg(grammar)]
use itertools::Itertools;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{backends::common::gpu_types::trie::TrieNode as GpuTrieNode, encodable_block::sampling::PRng};
#[cfg(grammar)]
use crate::{
    data_type::DataType,
    engine::language_model::grammar::{Grammar, GrammarError},
};

#[derive(Debug, Error)]
pub enum TrieError {
    #[error("child with the same token id is already present")]
    DuplicateTokenId,
}

#[derive(Debug, Error)]
pub enum TrieAcceptError {
    #[cfg(grammar)]
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
}

#[derive(Debug)]
pub struct TrieNode {
    token: u64,
    seed: u64,
    logprob: f32,
    next: Vec<TrieNode>,
}

#[derive(Debug)]
struct FlatTrieNode<'a> {
    node: &'a TrieNode,
    subtrie_range: (usize, usize),
    height: usize,
}

#[derive(Debug)]
pub struct FlatTrie<'a> {
    tokens: Box<[FlatTrieNode<'a>]>,
}

impl TrieNode {
    pub fn new(
        token: u64,
        seed: u64,
        logprob: f32,
    ) -> Self {
        Self {
            token,
            seed,
            logprob,
            next: Vec::new(),
        }
    }

    pub fn add(
        &mut self,
        next: TrieNode,
    ) -> Result<usize, TrieError> {
        if self.next.iter().any(|n| n.token == next.token) {
            return Err(TrieError::DuplicateTokenId);
        }

        self.next.push(next);
        Ok(self.next.len() - 1)
    }

    pub fn get(
        &self,
        token: u64,
    ) -> Option<&TrieNode> {
        self.next.iter().find(|n| n.token == token)
    }

    #[cfg(test)]
    pub fn token(&self) -> u64 {
        self.token
    }

    #[cfg(test)]
    pub fn logprob(&self) -> f32 {
        self.logprob
    }

    #[cfg(test)]
    pub fn node_count(&self) -> usize {
        1 + self.next.iter().map(TrieNode::node_count).sum::<usize>()
    }

    pub fn prune_to_budget(
        &mut self,
        budget: usize,
    ) {
        assert!(budget > 0, "budget must keep at least the root");

        fn collect_logprobs(
            node: &TrieNode,
            parent_logprob: f32,
            logprobs: &mut Vec<f32>,
        ) {
            let logprob = parent_logprob + node.logprob;
            logprobs.push(logprob);
            for child in &node.next {
                collect_logprobs(child, logprob, logprobs);
            }
        }
        let mut logprobs = Vec::new();
        collect_logprobs(self, 0.0, &mut logprobs);
        if budget >= logprobs.len() {
            return;
        }

        let mut order: Box<[usize]> = (0..logprobs.len()).collect();
        order.sort_by(|&a, &b| logprobs[b].total_cmp(&logprobs[a]));
        let mut kept = vec![false; logprobs.len()];
        for &index in order.iter().take(budget) {
            kept[index] = true;
        }

        fn prune(
            node: &mut TrieNode,
            kept: &[bool],
            cursor: &mut usize,
        ) {
            *cursor += 1;
            let mut children = std::mem::take(&mut node.next);
            children.retain_mut(|child| {
                let child_index = *cursor;
                prune(child, kept, cursor);
                kept[child_index]
            });
            node.next = children;
        }
        prune(self, &kept, &mut 0);
    }

    pub fn flat(
        prefix_length: usize,
        tokens: &[u64],
        prng: &PRng,
    ) -> Self {
        assert!(!tokens.is_empty(), "need seed node");

        let mut root = TrieNode::new(tokens[0], prng.derive(prefix_length as u64), 0.0);
        let mut leaf = &mut root;

        for (index, token) in tokens.iter().copied().enumerate().skip(1) {
            leaf.add(TrieNode::new(token, prng.derive((prefix_length + index) as u64), 0.0)).unwrap();
            leaf = &mut leaf.next[0];
        }

        root
    }

    pub fn pretty_print(
        &self,
        tokenizer: &Tokenizer,
        highlight: &[usize],
    ) {
        const GUTTER: usize = 4;
        const HIGHLIGHT_START: &str = "\x1b[1;32m";
        const HIGHLIGHT_END: &str = "\x1b[0m";

        struct DisplayNode {
            index: usize,
            token_line: String,
            prob_line: String,
            width: usize,
            height: usize,
            children: Vec<DisplayNode>,
        }

        fn build(
            node: &TrieNode,
            tokenizer: &Tokenizer,
            next_index: &mut usize,
        ) -> DisplayNode {
            let index = *next_index;
            *next_index += 1;
            let token = tokenizer
                .decode(&[node.token as u32], false)
                .ok()
                .filter(|text| !text.is_empty())
                .or_else(|| tokenizer.id_to_token(node.token as u32))
                .unwrap_or_else(|| node.token.to_string());
            let mut token_line = String::new();
            for ch in token.chars() {
                if ch.is_control() || (ch.is_whitespace() && ch != ' ') {
                    token_line.extend(ch.escape_debug());
                } else {
                    token_line.push(ch);
                }
            }
            let prob_line = format!("{:.2}%", node.logprob.exp() * 100.0);
            let width = token_line.chars().count().max(prob_line.chars().count());
            let children: Vec<DisplayNode> =
                node.next.iter().map(|child| build(child, tokenizer, next_index)).collect();
            let height = if children.is_empty() {
                2
            } else {
                children.iter().map(|child| child.height).sum::<usize>() + children.len() - 1
            };
            DisplayNode {
                index,
                token_line,
                prob_line,
                width,
                height,
                children,
            }
        }

        fn depth_widths(
            node: &DisplayNode,
            depth: usize,
            widths: &mut Vec<usize>,
        ) {
            if widths.len() == depth {
                widths.push(0);
            }
            widths[depth] = widths[depth].max(node.width);
            for child in &node.children {
                depth_widths(child, depth + 1, widths);
            }
        }

        fn write_centered(
            grid: &mut [Vec<char>],
            row: usize,
            col: usize,
            text: &str,
            width: usize,
        ) {
            let padding = width.saturating_sub(text.chars().count());
            for (offset, ch) in text.chars().enumerate() {
                grid[row][col + padding / 2 + offset] = ch;
            }
        }

        fn render(
            node: &DisplayNode,
            grid: &mut Vec<Vec<char>>,
            highlight: &[usize],
            highlights: &mut Vec<(usize, usize, usize)>,
            start: usize,
            col: usize,
            depth: usize,
            widths: &[usize],
        ) -> usize {
            let anchor = start + (node.height - 2) / 2;
            write_centered(grid, anchor, col, &node.token_line, node.width);
            write_centered(grid, anchor + 1, col, &node.prob_line, node.width);
            if highlight.contains(&node.index) {
                highlights.push((anchor, col, col + node.width));
            }

            if node.children.is_empty() {
                return anchor;
            }

            let child_col = col + widths[depth] + GUTTER;
            let bar_col = col + widths[depth] + 1;
            let line_start = col + node.width;

            if node.children.len() == 1 {
                for c in line_start..(child_col - 1) {
                    grid[anchor][c] = '─';
                }
                grid[anchor][child_col - 1] = '▶';
                render(&node.children[0], grid, highlight, highlights, start, child_col, depth + 1, widths);
                return anchor;
            }

            let mut child_anchors = Vec::with_capacity(node.children.len());
            let mut child_start = start;
            for child in &node.children {
                child_anchors.push(render(
                    child,
                    grid,
                    highlight,
                    highlights,
                    child_start,
                    child_col,
                    depth + 1,
                    widths,
                ));
                child_start += child.height + 1;
            }

            let top = child_anchors[0];
            let bottom = *child_anchors.last().unwrap();
            for row in top..=bottom {
                let is_parent = row == anchor;
                let is_child = child_anchors.contains(&row);
                grid[row][bar_col] = match (is_parent, is_child) {
                    (true, true) => '┼',
                    (true, false) => '┤',
                    (false, true) if row == top => '┌',
                    (false, true) if row == bottom => '└',
                    (false, true) => '├',
                    (false, false) => '│',
                };
                if is_child {
                    grid[row][bar_col + 1] = '─';
                    grid[row][bar_col + 2] = '▶';
                }
            }
            for c in line_start..bar_col {
                grid[anchor][c] = '─';
            }

            anchor
        }

        let root = build(self, tokenizer, &mut 0);
        let mut widths = Vec::new();
        depth_widths(&root, 0, &mut widths);

        let total_width = widths.iter().sum::<usize>() + GUTTER * widths.len().saturating_sub(1);
        let mut grid = vec![vec![' '; total_width]; root.height];
        let mut highlights = Vec::new();
        render(&root, &mut grid, highlight, &mut highlights, 0, 0, 0, &widths);

        let mut output = String::new();
        for (row_index, row) in grid.iter().enumerate() {
            let end = row.iter().rposition(|&ch| ch != ' ').map_or(0, |position| position + 1);
            let mut spans: Vec<(usize, usize)> = highlights
                .iter()
                .filter(|&&(span_row, _, _)| span_row == row_index)
                .map(|&(_, span_start, span_end)| (span_start, span_end.min(end)))
                .collect();
            spans.sort_unstable();

            let mut cursor = 0;
            for (start, span_end) in spans {
                output.extend(row[cursor..start].iter());
                output.push_str(HIGHLIGHT_START);
                output.extend(row[start..span_end].iter());
                output.push_str(HIGHLIGHT_END);
                cursor = span_end;
            }
            output.extend(row[cursor..end].iter());
            output.push('\n');
        }
        eprint!("{output}");
    }

    pub fn linearize(&self) -> FlatTrie<'_> {
        let mut tokens = vec![FlatTrieNode::new(self, (0, 0), 0)];

        let mut stack = vec![(0, 0)];
        while let Some((cur_node_idx, next_child_idx)) = stack.last_mut() {
            let Some(next_node) = tokens[*cur_node_idx].node.next.get(*next_child_idx) else {
                tokens[*cur_node_idx].subtrie_range.1 = tokens.len() - 1;
                stack.pop();
                continue;
            };
            *next_child_idx += 1;

            tokens.push(FlatTrieNode::new(next_node, (tokens.len(), tokens.len()), stack.len()));

            if !next_node.next.is_empty() {
                stack.push((tokens.len() - 1, 0));
            }
        }

        FlatTrie::new(tokens.into_boxed_slice())
    }
}

impl<'a> FlatTrieNode<'a> {
    fn new(
        node: &'a TrieNode,
        subtrie_range: (usize, usize),
        height: usize,
    ) -> Self {
        Self {
            node,
            subtrie_range,
            height,
        }
    }
}

impl<'a> FlatTrie<'a> {
    fn new(tokens: Box<[FlatTrieNode<'a>]>) -> Self {
        Self {
            tokens,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_ids(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.token)
    }

    pub fn token_subtrie_ranges(&self) -> impl Iterator<Item = GpuTrieNode> {
        self.tokens.iter().map(|n| {
            let (start, end) = n.subtrie_range;

            GpuTrieNode {
                trie_start: start as u32,
                trie_end: end as u32,
                height: n.height as u32,
            }
        })
    }

    pub fn token_seeds(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.seed)
    }

    #[cfg(grammar)]
    pub fn fill_bitmasks(
        &self,
        bitmasks: &mut [u32],
        vocab_size: usize,
        grammar: &mut Grammar,
    ) -> bool {
        let vocab_size_in_u32s = vocab_size.div_ceil(DataType::U32.size_in_bits());
        assert!(bitmasks.len() == self.tokens.len() * vocab_size_in_u32s);

        let mut any_non_full = false;
        let mut last_token_height = 0;
        for ((token_index, token), bitmask) in
            self.tokens.iter().enumerate().zip_eq(bitmasks.chunks_exact_mut(vocab_size_in_u32s))
        {
            if token_index > 0 {
                if token.height <= last_token_height {
                    grammar.rollback(last_token_height - token.height + 1);
                }
                grammar.accept_token(token.node.token).expect("flat trie doesn't match grammar");
            }

            any_non_full |= grammar.next_bitmask(bitmask);

            last_token_height = token.height;
        }

        if last_token_height > 0 {
            grammar.rollback(last_token_height);
        }

        any_non_full
    }

    pub fn root(&self) -> Option<&TrieNode> {
        self.tokens.first().map(|n| n.node)
    }

    pub fn index(
        &self,
        node: &'a TrieNode,
    ) -> Option<usize> {
        self.tokens.iter().position(|n| std::ptr::eq(n.node, node))
    }

    pub fn accept(
        &self,
        sampled_tokens: &[u64],
        #[cfg(grammar)] mut grammar: Option<&mut Grammar>,
    ) -> Result<Box<[(usize, u64, u64)]>, TrieAcceptError> {
        let mut current_token = self.root().unwrap();
        let mut accepted = Vec::new();
        loop {
            let current_token_index = self.index(current_token).unwrap();
            let current_token_id = sampled_tokens[current_token_index];

            accepted.push((current_token_index, current_token.token, current_token_id));
            #[cfg(grammar)]
            if let Some(grammar) = grammar.as_mut()
                && !grammar.is_terminated()
            {
                grammar.accept_token(current_token_id)?;
            }

            let Some(next_token) = current_token.get(current_token_id) else {
                break;
            };

            #[cfg(grammar)]
            if let Some(grammar) = grammar.as_mut() {
                assert!(!grammar.is_terminated(), "Grammar has terminated but llm continued generation");
            }

            current_token = next_token;
        }

        Ok(accepted.into_boxed_slice())
    }
}

#[cfg(test)]
#[path = "../tests/unit/trie_test.rs"]
mod tests;
