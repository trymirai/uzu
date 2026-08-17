use half::bf16;
#[cfg(grammar)]
use itertools::Itertools;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
    encodable_block::sampling::{PRng, gumbel_float, revidx},
};
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
    gumbel: f32,
    pruned: bool,
    pub(crate) dflash_logprob: Option<f32>,
    target_logprob: Option<f32>,
    sampled_logprob: Option<f32>,
    next: Vec<TrieNode>,
}

// Trace encoding, all integers little-endian: token u32, logprob f32, gumbel f32, a flags
// byte selecting which of dflash/target/sampled logprobs follow (in that order), then the
// child count u32 and the children. `seed` is not recorded (the recorded `gumbel` is what
// pruning experiments rescale) and `pruned` is not recorded (replay re-prunes from scratch).
impl TrieNode {
    pub fn write_trace(
        &self,
        writer: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        writer.write_all(&(self.token as u32).to_le_bytes())?;
        writer.write_all(&self.logprob.to_le_bytes())?;
        writer.write_all(&self.gumbel.to_le_bytes())?;
        let flags = u8::from(self.dflash_logprob.is_some())
            | u8::from(self.target_logprob.is_some()) << 1
            | u8::from(self.sampled_logprob.is_some()) << 2;
        writer.write_all(&[flags])?;
        for value in [self.dflash_logprob, self.target_logprob, self.sampled_logprob].into_iter().flatten() {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&(self.next.len() as u32).to_le_bytes())?;
        for child in &self.next {
            child.write_trace(writer)?;
        }
        Ok(())
    }

    pub fn read_trace(
        reader: &mut impl std::io::Read,
    ) -> std::io::Result<Self> {
        fn read_u32(
            reader: &mut impl std::io::Read,
        ) -> std::io::Result<u32> {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            Ok(u32::from_le_bytes(bytes))
        }
        fn read_f32(
            reader: &mut impl std::io::Read,
        ) -> std::io::Result<f32> {
            Ok(f32::from_bits(read_u32(reader)?))
        }
        fn read_optional_f32(
            reader: &mut impl std::io::Read,
            present: bool,
        ) -> std::io::Result<Option<f32>> {
            present.then(|| read_f32(reader)).transpose()
        }

        let token = u64::from(read_u32(reader)?);
        let logprob = read_f32(reader)?;
        let gumbel = read_f32(reader)?;
        let mut flags = [0];
        reader.read_exact(&mut flags)?;
        let dflash_logprob = read_optional_f32(reader, flags[0] & 1 != 0)?;
        let target_logprob = read_optional_f32(reader, flags[0] & 2 != 0)?;
        let sampled_logprob = read_optional_f32(reader, flags[0] & 4 != 0)?;
        let num_children = read_u32(reader)?;
        let mut next = Vec::with_capacity(num_children as usize);
        for _ in 0..num_children {
            next.push(Self::read_trace(reader)?);
        }
        Ok(Self {
            token,
            seed: 0,
            logprob,
            gumbel,
            pruned: false,
            dflash_logprob,
            target_logprob,
            sampled_logprob,
            next,
        })
    }
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

#[derive(Debug, Clone, Copy)]
pub struct MainStats {
    pub vocab_size: u32,
}

impl TrieNode {
    pub fn new(
        token: u64,
        seed: u64,
        logprob: f32,
        gumbel: f32,
    ) -> Self {
        Self {
            token,
            seed,
            logprob,
            gumbel,
            pruned: false,
            dflash_logprob: None,
            target_logprob: None,
            sampled_logprob: None,
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
        if self.pruned {
            return 0;
        }
        1 + self.next.iter().map(TrieNode::node_count).sum::<usize>()
    }

    pub fn prune_to_budget(
        &mut self,
        budget: usize,
    ) {
        self.prune_to_budget_with_gumbel(budget, 0.0);
    }

    pub fn prune_to_budget_with_gumbel(
        &mut self,
        budget: usize,
        gumbel_scale: f32,
    ) {
        assert!(budget > 0, "budget must keep at least the root");

        // Rank each node by its cumulative path logprob plus its scaled gumbel noise. The
        // noise perturbs only the node's own score and is not accumulated into the path
        // logprob passed to children.
        fn collect_scores(
            node: &TrieNode,
            parent_index: usize,
            parent_logprob: f32,
            gumbel_scale: f32,
            scores: &mut Vec<f32>,
            parents: &mut Vec<usize>,
        ) {
            let index = scores.len();
            let logprob = parent_logprob + node.logprob;
            scores.push(logprob + gumbel_scale * node.gumbel);
            parents.push(parent_index);
            for child in &node.next {
                collect_scores(child, index, logprob, gumbel_scale, scores, parents);
            }
        }
        let mut scores = Vec::new();
        let mut parents = Vec::new();
        collect_scores(self, 0, 0.0, gumbel_scale, &mut scores, &mut parents);

        let mut order: Box<[usize]> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        // Perturbed scores are not monotone along paths, so gate on the parent being kept to
        // keep the surviving set a valid trie. Scan repeatedly: a child can outrank its parent
        // and only become keepable on a later pass.
        let mut kept = vec![false; scores.len()];
        kept[0] = true;
        let mut remaining = budget - 1;
        while remaining > 0 {
            let mut progressed = false;
            for &index in order.iter() {
                if remaining == 0 {
                    break;
                }
                if index != 0 && !kept[index] && kept[parents[index]] {
                    kept[index] = true;
                    remaining -= 1;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }

        fn mark_pruned(
            node: &mut TrieNode,
            kept: &[bool],
            cursor: &mut usize,
        ) {
            *cursor += 1;
            for child in &mut node.next {
                let child_index = *cursor;
                mark_pruned(child, kept, cursor);
                child.pruned = !kept[child_index];
            }
        }
        mark_pruned(self, &kept, &mut 0);
    }

    pub fn flat(
        prefix_length: usize,
        tokens: &[u64],
        prng: &PRng,
    ) -> Self {
        assert!(!tokens.is_empty(), "need seed node");

        let mut root = TrieNode::new(tokens[0], prng.derive(prefix_length as u64), 0.0, 0.0);
        let mut leaf = &mut root;

        for (index, token) in tokens.iter().copied().enumerate().skip(1) {
            leaf.add(TrieNode::new(token, prng.derive((prefix_length + index) as u64), 0.0, 0.0)).unwrap();
            leaf = &mut leaf.next[0];
        }

        root
    }

    /// Annotates each node with the target model's logprobs from the verification forward
    /// pass: `target_logprob` is the target's logprob of the node's token given its path
    /// prefix, `sampled_logprob` is the target's logprob of the token it sampled at the
    /// node. `lse` holds per-row log-sum-exp values for the flat-trie rows. Only nodes whose
    /// parent was in the flat batch get a `target_logprob`; deeper pruned nodes keep `None`.
    pub fn annotate_target_logprobs(
        &mut self,
        logits: &[bf16],
        lse: &[f32],
        sampled_tokens: &[u64],
        vocab_size: usize,
    ) {
        self.target_logprob = Some(0.0);
        self.sampled_logprob = Some(logits[sampled_tokens[0] as usize].to_f32() - lse[0]);
        let mut next_row = 1;
        self.annotate_children(logits, lse, sampled_tokens, vocab_size, Some(0), &mut next_row);
    }

    fn annotate_children(
        &mut self,
        logits: &[bf16],
        lse: &[f32],
        sampled_tokens: &[u64],
        vocab_size: usize,
        node_row: Option<usize>,
        next_row: &mut usize,
    ) {
        for child in &mut self.next {
            let child_row = if child.pruned {
                None
            } else {
                let row = *next_row;
                *next_row += 1;
                Some(row)
            };
            if let Some(parent_row) = node_row {
                child.target_logprob =
                    Some(logits[parent_row * vocab_size + child.token as usize].to_f32() - lse[parent_row]);
            }
            if let Some(row) = child_row {
                child.sampled_logprob =
                    Some(logits[row * vocab_size + sampled_tokens[row] as usize].to_f32() - lse[row]);
            }
            child.annotate_children(logits, lse, sampled_tokens, vocab_size, child_row, next_row);
        }
    }

    pub fn pretty_print(
        &self,
        tokenizer: &Tokenizer,
        highlight: &[usize],
        bonus_token: Option<u64>,
        show_pruned: bool,
        main_stats: Option<MainStats>,
    ) {
        const GUTTER: usize = 4;
        const UNACCEPTED_BG: &str = "\x1b[48;5;236m";
        const PRUNED_BG: &str = "\x1b[48;5;124m";
        const ACCEPTED_BG: &str = "\x1b[48;5;28m";
        const BONUS_BG: &str = "\x1b[48;5;67m";
        const COLOR_END: &str = "\x1b[0m";

        struct DisplayNode {
            index: Option<usize>,
            token_line: String,
            prob_line: String,
            gumbel_line: String,
            main_line: String,
            width: usize,
            height: usize,
            pruned: bool,
            bonus: bool,
            has_main_stats: bool,
            children: Vec<DisplayNode>,
        }

        impl DisplayNode {
            fn lines(&self) -> usize {
                if self.bonus {
                    if self.has_main_stats {
                        // Token, blank speculator stats, main-model score
                        NODE_LINES + 1
                    } else {
                        1
                    }
                } else {
                    NODE_LINES + self.has_main_stats as usize
                }
            }
        }

        const NODE_LINES: usize = 3;

        fn score_line(
            logprob: f32,
            gumbel: f32,
        ) -> String {
            format!("{:.2}{:+.2}={:+.2}", logprob, gumbel, logprob + gumbel)
        }

        fn token_text(
            tokenizer: &Tokenizer,
            token: u64,
        ) -> String {
            // Renderable chars print as themselves. Non-renderable ones are escaped as
            // \u{...}: that is escape_debug's non-printable set (control, zero-width, format
            // chars, ...) plus U+FFFD, which is not a style choice but UTF-8's designated
            // marker for "invalid data was substituted here" - i.e. the decode of a partial
            // byte token lost information.
            fn is_renderable(ch: char) -> bool {
                if ch == '\u{fffd}' {
                    return false;
                }
                let mut escaped = ch.escape_debug();
                matches!((escaped.next(), escaped.next()), (Some(same), None) if same == ch)
            }

            let token = tokenizer
                .decode(&[token as u32], false)
                .ok()
                .filter(|text| !text.is_empty())
                .or_else(|| tokenizer.id_to_token(token as u32))
                .unwrap_or_else(|| token.to_string());
            let mut text = String::new();
            for ch in token.chars() {
                if is_renderable(ch) {
                    text.push(ch);
                } else {
                    text.extend(ch.escape_debug());
                }
            }
            text
        }

        fn build(
            node: &TrieNode,
            tokenizer: &Tokenizer,
            next_index: &mut usize,
            bonus: &Option<(usize, u64, String)>,
            show_pruned: bool,
            main_stats: &Option<MainStats>,
        ) -> DisplayNode {
            let index = if node.pruned {
                None
            } else {
                let index = *next_index;
                *next_index += 1;
                Some(index)
            };
            let token_line = token_text(tokenizer, node.token);
            let prob_line = format!("{:.2}%", node.logprob.exp() * 100.0);
            let gumbel_line = if let Some(dflash_logprob) = node.dflash_logprob {
                format!(
                    "{:.2}{:+.2}{:+.2}={:+.2}",
                    dflash_logprob,
                    node.logprob - dflash_logprob,
                    node.gumbel,
                    node.logprob + node.gumbel
                )
            } else {
                score_line(node.logprob, node.gumbel)
            };
            let main_line = if main_stats.is_some() {
                node.target_logprob.map(|logprob| score_line(logprob, node.gumbel)).unwrap_or_default()
            } else {
                String::new()
            };
            let width = token_line
                .chars()
                .count()
                .max(prob_line.chars().count())
                .max(gumbel_line.chars().count())
                .max(main_line.chars().count());
            let mut children: Vec<DisplayNode> = node
                .next
                .iter()
                .filter(|child| show_pruned || !child.pruned)
                .map(|child| build(child, tokenizer, next_index, bonus, show_pruned, main_stats))
                .collect();
            if let Some((bonus_index, bonus_token, bonus_text)) = bonus
                && index == Some(*bonus_index)
            {
                let bonus_main_line = if let Some(main_stats) = main_stats {
                    node.sampled_logprob
                        .map(|logprob| {
                            score_line(
                                logprob,
                                gumbel_float(node.seed, revidx(*bonus_token as u32, main_stats.vocab_size)),
                            )
                        })
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                children.push(DisplayNode {
                    index: None,
                    width: bonus_text.chars().count().max(bonus_main_line.chars().count()),
                    height: if main_stats.is_some() {
                        NODE_LINES + 1
                    } else {
                        1
                    },
                    token_line: bonus_text.clone(),
                    prob_line: String::new(),
                    gumbel_line: String::new(),
                    main_line: bonus_main_line,
                    pruned: false,
                    bonus: true,
                    has_main_stats: main_stats.is_some(),
                    children: Vec::new(),
                });
            }
            let height = if children.is_empty() {
                3 + main_stats.is_some() as usize
            } else {
                (3 + main_stats.is_some() as usize)
                    .max(children.iter().map(|child| child.height).sum::<usize>() + children.len() - 1)
            };
            DisplayNode {
                index,
                token_line,
                prob_line,
                gumbel_line,
                main_line,
                width,
                height,
                pruned: node.pruned,
                bonus: false,
                has_main_stats: main_stats.is_some(),
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
            spans: &mut Vec<(usize, usize, usize, &'static str)>,
            start: usize,
            col: usize,
            depth: usize,
            widths: &[usize],
        ) -> usize {
            let anchor = start + (node.height - node.lines()) / 2;
            // The bonus pseudo-node gets centered within its depth column, like the real
            // nodes around it, instead of sitting flush-left in a text-sized cell.
            let cell_width = if node.bonus { widths[depth].max(node.width) } else { node.width };
            write_centered(grid, anchor, col, &node.token_line, cell_width);
            if !node.bonus {
                write_centered(grid, anchor + 1, col, &node.prob_line, node.width);
                write_centered(grid, anchor + 2, col, &node.gumbel_line, node.width);
            }
            if node.has_main_stats {
                write_centered(grid, anchor + 3, col, &node.main_line, node.width);
            }
            // Classification background over the token text (not the centering padding):
            // gray for unaccepted, red for pruned, light green for accepted, light blue for
            // the bonus. Pruned nodes get it on their stat lines as well.
            let text_len = node.token_line.chars().count();
            let text_start = col + cell_width.saturating_sub(text_len) / 2;
            let text_end = text_start + text_len;
            if node.bonus {
                spans.push((anchor, text_start, text_end, BONUS_BG));
            } else if node.pruned {
                spans.push((anchor, text_start, text_end, PRUNED_BG));
                for (line_offset, line) in [(1usize, &node.prob_line), (2, &node.gumbel_line), (3, &node.main_line)] {
                    let line_len = line.chars().count();
                    if line_len > 0 {
                        let line_start = col + node.width.saturating_sub(line_len) / 2;
                        spans.push((anchor + line_offset, line_start, line_start + line_len, PRUNED_BG));
                    }
                }
            } else if let Some(index) = node.index
                && highlight.contains(&index)
            {
                spans.push((anchor, text_start, text_end, ACCEPTED_BG));
            } else {
                spans.push((anchor, text_start, text_end, UNACCEPTED_BG));
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
                render(&node.children[0], grid, highlight, spans, start, child_col, depth + 1, widths);
                return anchor;
            }

            let mut child_anchors = Vec::with_capacity(node.children.len());
            let mut child_start = start;
            for child in &node.children {
                child_anchors.push(render(child, grid, highlight, spans, child_start, child_col, depth + 1, widths));
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

        let bonus = bonus_token
            .map(|token| (highlight.last().copied().unwrap_or(0), token, token_text(tokenizer, token)));
        let root = build(self, tokenizer, &mut 0, &bonus, show_pruned, &main_stats);
        let mut widths = Vec::new();
        depth_widths(&root, 0, &mut widths);

        let total_width = widths.iter().sum::<usize>() + GUTTER * widths.len().saturating_sub(1);
        let mut grid = vec![vec![' '; total_width]; root.height];
        let mut spans = Vec::new();
        render(&root, &mut grid, highlight, &mut spans, 0, 0, 0, &widths);

        let mut output = String::new();
        for (row_index, row) in grid.iter().enumerate() {
            // A styled trailing space is still content: extend the row past the last
            // non-space char to cover any span that reaches beyond it.
            let end = row
                .iter()
                .rposition(|&ch| ch != ' ')
                .map_or(0, |position| position + 1)
                .max(spans.iter().filter(|&&(r, _, _, _)| r == row_index).map(|&(_, _, e, _)| e).max().unwrap_or(0));

            let mut cuts: Vec<usize> = vec![0, end];
            for &(span_row, span_start, span_end, _) in &spans {
                if span_row == row_index {
                    cuts.push(span_start.min(end));
                    cuts.push(span_end.min(end));
                }
            }
            cuts.sort_unstable();
            cuts.dedup();

            for segment in cuts.windows(2) {
                let (start, stop) = (segment[0], segment[1]);
                if start == stop {
                    continue;
                }
                let color = spans
                    .iter()
                    .find(|&&(span_row, span_start, span_end, _)| {
                        span_row == row_index && span_start <= start && start < span_end
                    })
                    .map(|&(_, _, _, color)| color);
                if let Some(color) = color {
                    output.push_str(color);
                }
                output.extend(row[start..stop].iter());
                if color.is_some() {
                    output.push_str(COLOR_END);
                }
            }
            output.push('\n');
        }
        eprint!("{output}");
    }

    pub fn linearize(&self) -> FlatTrie<'_> {
        let mut tokens = vec![FlatTrieNode::new(self, (0, 0), 0)];

        let mut stack = vec![(0, 0)];
        while let Some((cur_node_idx, next_child_idx)) = stack.last_mut() {
            let cur_node = tokens[*cur_node_idx].node;
            let Some((child_index, next_node)) =
                cur_node.next.iter().enumerate().skip(*next_child_idx).find(|(_, node)| !node.pruned)
            else {
                tokens[*cur_node_idx].subtrie_range.1 = tokens.len() - 1;
                stack.pop();
                continue;
            };
            *next_child_idx = child_index + 1;

            tokens.push(FlatTrieNode::new(next_node, (tokens.len(), tokens.len()), stack.len()));

            if next_node.next.iter().any(|node| !node.pruned) {
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

    pub fn nodes(&self) -> impl Iterator<Item = &'a TrieNode> {
        self.tokens.iter().map(|n| n.node)
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

    /// Replays verification against the true continuation of the sequence. The token the
    /// target model samples at a node depends only on the path prefix and the node's seed,
    /// so it equals `continuation[node height]` regardless of how the trie was pruned.
    ///
    /// Returns the accepted path, an optional "tip" node, and whether the walk stopped
    /// because it could not descend (no kept child matching the sampled token). When the
    /// continuation runs out mid-walk, the walk has just descended into the tip node: that
    /// node was verified and its token emitted, so it should be marked accepted even though
    /// it has no accepted entry (the sample at it is missing). In that case the last accepted
    /// entry's output token is in the tree (it's the tip), so it is not a bonus token.
    pub fn accept_continuation(
        &self,
        continuation: &[u64],
    ) -> (Box<[(usize, u64, u64)]>, Option<usize>, bool) {
        let mut current_token = self.root().unwrap();
        let mut accepted = Vec::new();
        loop {
            let current_token_index = self.index(current_token).unwrap();
            let Some(&current_token_id) = continuation.get(self.tokens[current_token_index].height) else {
                return (accepted.into_boxed_slice(), Some(current_token_index), false);
            };

            accepted.push((current_token_index, current_token.token, current_token_id));

            let Some(next_token) = current_token.get(current_token_id) else {
                return (accepted.into_boxed_slice(), None, true);
            };
            if next_token.pruned {
                return (accepted.into_boxed_slice(), None, true);
            }

            current_token = next_token;
        }
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
            if next_token.pruned {
                break;
            }

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
#[path = "../unit/trie_test.rs"]
mod tests;
