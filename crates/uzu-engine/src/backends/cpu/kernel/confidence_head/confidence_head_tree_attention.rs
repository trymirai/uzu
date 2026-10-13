use uzu_engine_macros::kernel;

const NONE: u32 = u32::MAX;

/// Attention of the head's tree layers: every token attends to every present token with a learned bias per head
/// and pair relation (self, parent, child, ancestor two or more edges up, descendant likewise, sibling); absent keys
/// are masked out, and an absent query reads the present keys without relation biases (its row is never used).
/// qkv is [tokens, 3 * width] (queries, keys, values; heads of head_dim inside each).
#[kernel(ConfidenceHeadTreeAttention)]
pub fn confidence_head_tree_attention(
    qkv: *const f32,
    relation_weights: *const f32,
    token_parents: *const u32,
    token_depths: *const u32,
    token_present: *const u32,
    token_ancestors: *const u32,
    output: *mut f32,
    token_count: u32,
    #[specialize] heads: u32,
    #[specialize] head_dim: u32,
    #[specialize] relations: u32,
    #[specialize] max_depth: u32,
) {
    let (count, heads, head_dim, relations) =
        (token_count as usize, heads as usize, head_dim as usize, relations as usize);
    let width = heads * head_dim;
    let levels = max_depth as usize + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let parent = |token: usize| unsafe { *token_parents.add(token) };
    let depth = |token: usize| unsafe { *token_depths.add(token) } as usize;
    let present = |token: usize| unsafe { *token_present.add(token) } != 0;
    // Whether `upper` is an ancestor of `lower` at least two edges up.
    let far_ancestor = |lower: usize, upper: usize| {
        depth(upper) + 2 <= depth(lower)
            && unsafe { *token_ancestors.add(lower * levels + depth(upper).min(levels - 1)) } == upper as u32
    };
    for query in 0..count {
        for head in 0..heads {
            let weights = unsafe { std::slice::from_raw_parts(relation_weights.add(head * relations), relations) };
            let q = unsafe { std::slice::from_raw_parts(qkv.add(query * 3 * width + head * head_dim), head_dim) };
            let scores = (0..count)
                .map(|key| {
                    if !present(key) {
                        return f32::NEG_INFINITY;
                    }
                    let k = unsafe {
                        std::slice::from_raw_parts(qkv.add(key * 3 * width + width + head * head_dim), head_dim)
                    };
                    let mut score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                    if present(query) {
                        let siblings = parent(query) != NONE && parent(query) == parent(key) && query != key;
                        let pairs = [
                            query == key,
                            parent(query) == key as u32,
                            parent(key) == query as u32,
                            far_ancestor(query, key),
                            far_ancestor(key, query),
                            siblings,
                        ];
                        for (relation, holds) in pairs.into_iter().enumerate().take(relations) {
                            if holds {
                                score += weights[relation];
                            }
                        }
                    }
                    score
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps = scores.iter().map(|score| (score - max).exp()).collect::<Vec<_>>();
            let sum = exps.iter().sum::<f32>();
            for dimension in 0..head_dim {
                let value = (0..count)
                    .filter(|&key| exps[key] > 0.0)
                    .map(|key| {
                        exps[key] * unsafe { *qkv.add(key * 3 * width + 2 * width + head * head_dim + dimension) }
                    })
                    .sum::<f32>();
                unsafe { *output.add(query * width + head * head_dim + dimension) = value / sum };
            }
        }
    }
}
