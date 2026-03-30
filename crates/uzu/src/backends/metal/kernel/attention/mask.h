#pragma once

#include "../generated/ring.h"
#include "../generated/trie.h"

using namespace uzu::ring;
using namespace uzu::trie;

bool should_use_key(
    const constant RingParams& ring_params,
    const device TrieNode* trie,
    const constant uint& sliding_window_size,
    const uint q_seq_idx,
    const uint prefix_length,
    const uint suffix_position,
    const uint query_position,
    const uint i,
    const bool is_kv_cache_ring,
    const bool is_causal,
    const bool is_trie,
    const bool is_sliding_window
);
