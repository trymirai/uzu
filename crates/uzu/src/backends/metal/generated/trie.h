// Auto-generated from gpu_types/trie.rs - do not edit manually
#pragma once

#ifndef UZU_TRIE_H
#define UZU_TRIE_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace trie {
#else
#include <stdint.h>
#endif

typedef struct {
  uint32_t trie_start;
  uint32_t trie_end;
  uint32_t height;
} TrieNode;

#ifdef __METAL_VERSION__
} // namespace trie
} // namespace uzu
#endif

#endif // UZU_TRIE_H
