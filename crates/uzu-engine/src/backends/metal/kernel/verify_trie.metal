#include <metal_stdlib>
#include "common/defines.h"
#include "common/dsl.h"
#include "generated/trie.h"

using namespace metal;
using namespace uzu::trie;

#define THREADGROUP_SIZE 256

PUBLIC KERNEL(VerifyTrie)(
    const device TrieNode* trie_nodes,
    const device int* parents,
    const device uint* token_ids,
    const device uint* sampled_tokens,
    const device uint8_t* output_norm,
    device uint* accepted_indices,
    device uint* accepted_input_tokens,
    device uint* accepted_output_tokens,
    device uint8_t* accepted_output_norm,
    constant const uint& num_nodes,
    constant const uint& capacity,
    constant const uint& norm_row_bytes,
    const uint thread_index THREADS(THREADGROUP_SIZE)
) {
  if (num_nodes == 0) {
    return;
  }

  uint len = 0;
  uint current = 0;
  while (true) {
    len += 1;
    const uint sampled = sampled_tokens[current];
    const uint end = trie_nodes[current].trie_end;
    uint child = current + 1;
    bool found = false;
    while (child <= end) {
      if (token_ids[child] == sampled) {
        current = child;
        found = true;
        break;
      }
      child = trie_nodes[child].trie_end + 1;
    }
    if (!found) {
      break;
    }
  }
  const uint last = current;

  if (thread_index == 0) {
    accepted_indices[0] = len;
    accepted_input_tokens[0] = len;
    accepted_output_tokens[0] = len;

    uint entry = 0;
    uint replay = 0;
    while (true) {
      const uint position = len - entry;
      accepted_indices[position] = replay;
      accepted_input_tokens[position] = token_ids[replay];
      accepted_output_tokens[position] = sampled_tokens[replay];
      entry += 1;
      if (entry == len) {
        break;
      }
      const uint sampled = sampled_tokens[replay];
      const uint end = trie_nodes[replay].trie_end;
      uint child = replay + 1;
      while (child <= end) {
        if (token_ids[child] == sampled) {
          replay = child;
          break;
        }
        child = trie_nodes[child].trie_end + 1;
      }
    }
  }

  for (uint offset = thread_index; offset < norm_row_bytes; offset += THREADGROUP_SIZE) {
    accepted_output_norm[offset] = output_norm[last * norm_row_bytes + offset];
  }
}
