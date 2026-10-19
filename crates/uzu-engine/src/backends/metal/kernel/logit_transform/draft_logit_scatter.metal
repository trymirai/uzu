#include <metal_stdlib>
#include "../common/dsl.h"

// Logits over the whole vocabulary from logits over a draft vocabulary (a subset of it): a token takes the logit of
// its slot in the draft vocabulary, a token outside it gets -inf.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(DraftLogitScatter)(
    // [rows, draft_vocab_size]
    device const T* draft_logits,
    // [vocab_size]: the token's slot in the draft vocabulary, or -1
    device const int* vocab_slots,
    // [rows, vocab_size]
    device T* logits,
    constant uint& rows,
    constant uint& draft_vocab_size,
    constant uint& vocab_size,
    const uint row AXIS(rows, 1),
    const uint token AXIS(vocab_size, 256)
) {
  const int slot = vocab_slots[token];
  logits[row * vocab_size + token] =
      slot >= 0 ? draft_logits[row * draft_vocab_size + uint(slot)] : static_cast<T>(-INFINITY);
}
