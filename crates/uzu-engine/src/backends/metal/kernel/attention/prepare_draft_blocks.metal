#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/trie.h"

#define THREADS_PER_THREADGROUP 128

KERNEL(PrepareDraftBlocks)(
    const device uzu::trie::TrieNode* trie OPTIONAL(has_trie),
    device int* block_token_rows,
    const constant uint& node_count,
    const constant uint& prefix_tokens,
    const constant uint& compression_ratio,
    const bool has_trie SPECIALIZE,
    const uint group_index GROUPS(node_count.div_ceil(THREADS_PER_THREADGROUP)),
    const uint thread_index THREADS(THREADS_PER_THREADGROUP)
) {
  const uint node_index = group_index * THREADS_PER_THREADGROUP + thread_index;
  if (node_index >= node_count)
    return;

  const uint height = has_trie ? trie[node_index].height : node_index;

  int ancestor_index = int(node_index);
  int token_position = int(prefix_tokens + height);
  for (int block_offset = int(compression_ratio) - 1; block_offset >= 0; --block_offset, --token_position) {
    int token_row = max(token_position, -1);
    if (has_trie && token_position >= int(prefix_tokens)) {
      token_row = int(prefix_tokens) + ancestor_index;
      ancestor_index = trie[ancestor_index].parent;
    }
    block_token_rows[node_index * compression_ratio + uint(block_offset)] = token_row;
  }
}
