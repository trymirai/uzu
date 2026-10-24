import definitions;
import generated.delta_net;
import generated.trie;

// A tree of up to 16 NB nodes per (batch, value head) and 32-value tile, key head hk = hv / (num_v_heads / num_k_heads),
// over 128-value heads, before DeltaNetNormGate. The columns holding a row in their trie interval are its ancestors and
// itself:
//   prefix[row] = sum of their log decays, decay(row, col) = e^(prefix[row] - prefix[col]);
//   A[row, col] = beta[row] decay (k[row] . k[col]) off the diagonal, qkd[row, col] = scale decay (q[row] . k[col]);
//   per value column, u = (I + A)^-1 rhs by forward substitution, rhs[row] = beta[row] (v[row] - e^prefix[row] k[row] . h0);
//   o[row] = T(e^prefix[row] scale q[row] . h0 + qkd[row] . u).
// Every tile of a head builds the grams itself. Invocation i, in subgroup_invocation order, accumulates k h0 and q h0
// for value i % 32 of the tile over the R rows from R (i / 32): keys, queries and the tile's h0 rows stream through
// shared memory TREE_VERIFY_DEPTH values at a time, with the next TREE_VERIFY_PREFETCH - 1 slices in flight. Then the
// first 32 invocations solve a value column each.

static const uint TREE_VERIFY_THREADS = 256;
static const uint TREE_VERIFY_DIM = 128;
static const uint TREE_VERIFY_TILE = 32;
static const uint TREE_VERIFY_TOKENS = 2 * TREE_VERIFY_TOKEN_BLOCK;
static const uint TREE_VERIFY_DEPTH = 32;
static const uint TREE_VERIFY_PREFETCH = 4;
// A staged token or h0 row: its slice of depths as float4s, padded to an odd count against bank conflicts.
static const uint TREE_VERIFY_QUADS = TREE_VERIFY_DEPTH / 4;
static const uint TREE_VERIFY_STRIDE = TREE_VERIFY_QUADS + 1;
static const uint TREE_VERIFY_GRAM_STRIDE = TREE_VERIFY_TOKENS + 1;

// The staged slice: keys, then queries, then the tile's h0 rows.
groupshared float4 tree_verify_stage[(2 * TREE_VERIFY_TOKENS + TREE_VERIFY_TILE) * TREE_VERIFY_STRIDE];
static const uint TREE_VERIFY_Q = TREE_VERIFY_TOKENS * TREE_VERIFY_STRIDE;
static const uint TREE_VERIFY_H0 = 2 * TREE_VERIFY_Q;
// rhs and then u, [token][value of the tile].
groupshared float tree_verify_u[TREE_VERIFY_TOKENS * TREE_VERIFY_TILE];
groupshared float tree_verify_a[TREE_VERIFY_TOKENS * TREE_VERIFY_GRAM_STRIDE];
groupshared float tree_verify_qkd[TREE_VERIFY_TOKENS * TREE_VERIFY_GRAM_STRIDE];
groupshared float tree_verify_prefix[TREE_VERIFY_TOKENS];
groupshared float tree_verify_log_decay[TREE_VERIFY_TOKENS];
groupshared float tree_verify_beta[TREE_VERIFY_TOKENS];
groupshared uint tree_verify_start[TREE_VERIFY_TOKENS];
groupshared uint tree_verify_end[TREE_VERIFY_TOKENS];

struct TreeVerifyTile<T : IModelFloat, let NB : uint> {
  static const uint TOKENS = TREE_VERIFY_TOKEN_BLOCK * NB;
  // Rows per invocation: eight groups of 32 cover the tokens.
  static const uint R = TOKENS / 8;
  // Gram entries per invocation, row i / GRAM_SPAN and columns i % GRAM_SPAN + GRAM_SPAN j.
  static const uint GRAM_COLUMNS = TOKENS * TOKENS / TREE_VERIFY_THREADS;
  static const uint GRAM_SPAN = TOKENS / GRAM_COLUMNS;
  static const uint SLICES = TREE_VERIFY_DIM / TREE_VERIFY_DEPTH;
  // The float4s of a slice: keys and queries loaded by invocations below TOKEN_QUADS, h0 by those below H0_QUADS.
  static const uint TOKEN_QUADS = TOKENS * TREE_VERIFY_QUADS;
  static const uint H0_QUADS = TREE_VERIFY_TILE * TREE_VERIFY_QUADS;

  static void run(
    Ptr<T, Access.Read> q,
    Ptr<T, Access.Read> k,
    Ptr<T, Access.Read> v,
    Ptr<TrieNode, Access.Read> trie,
    Ptr<float, Access.Read> log_decay,
    Ptr<float, Access.Read> beta,
    Ptr<float, Access.Read> h0,
    Ptr<T, Access.ReadWrite> o,
    float scale,
    uint tree_size,
    uint num_k_heads,
    uint num_v_heads,
    uint batch,
    uint hv,
    uint tile,
    uint i,
  ) {
    static_assert(R >= 2 && GRAM_COLUMNS >= 1 && TOKEN_QUADS <= TREE_VERIFY_THREADS, "one load each");
    let lane = i % TREE_VERIFY_TILE;
    let value = tile * TREE_VERIFY_TILE + lane;
    let first = i / TREE_VERIFY_TILE * R;
    let qk_stride = num_k_heads * TREE_VERIFY_DIM;
    let qk_base = (batch * tree_size * num_k_heads + hv / (num_v_heads / num_k_heads)) * TREE_VERIFY_DIM;
    let scalar_base = batch * tree_size * num_v_heads + hv;
    let value_stride = num_v_heads * TREE_VERIFY_DIM;
    let value_base = batch * tree_size * value_stride + hv * TREE_VERIFY_DIM + value;
    let h0_quads = bit_cast<Ptr<float4, Access.Read>>(h0);
    let h0_base = ((batch * num_v_heads + hv) * TREE_VERIFY_DIM + tile * TREE_VERIFY_TILE) * TREE_VERIFY_DIM / 4;

    // The slices, each invocation's share TREE_VERIFY_PREFETCH slices ahead of the one it stages.
    float4 next_k[TREE_VERIFY_PREFETCH];
    float4 next_q[TREE_VERIFY_PREFETCH];
    float4 next_h0[TREE_VERIFY_PREFETCH];
    [unroll]
    for (uint slice = 0; slice < TREE_VERIFY_PREFETCH - 1; slice++) {
      load(q, k, h0_quads, tree_size, qk_base, qk_stride, h0_base, slice, i, next_k[slice], next_q[slice], next_h0[slice]);
    }

    // The interval, log decay and beta of node i, in flight during the accumulation.
    TrieNode node = { 1, 0, 0 };
    var node_log_decay = 0.0;
    var node_beta = 0.0;
    if (i < tree_size) {
      node = trie[batch * tree_size + i];
      node_log_decay = log_decay[scalar_base + i * num_v_heads];
      node_beta = beta[scalar_base + i * num_v_heads];
    }

    let gram_row = i / GRAM_SPAN;
    let gram_col = i % GRAM_SPAN;
    float kh0[R];
    float qh0[R];
    [unroll]
    for (uint row = 0; row < R; row++) {
      kh0[row] = 0.0;
      qh0[row] = 0.0;
    }
    float kk[GRAM_COLUMNS];
    float qk[GRAM_COLUMNS];
    [unroll]
    for (uint j = 0; j < GRAM_COLUMNS; j++) {
      kk[j] = 0.0;
      qk[j] = 0.0;
    }
    [unroll]
    for (uint slice = 0; slice < SLICES; slice++) {
      let ahead = slice + TREE_VERIFY_PREFETCH - 1;
      if (ahead < SLICES) {
        let slot = ahead % TREE_VERIFY_PREFETCH;
        load(q, k, h0_quads, tree_size, qk_base, qk_stride, h0_base, ahead, i, next_k[slot], next_q[slot], next_h0[slot]);
      }
      let slot = slice % TREE_VERIFY_PREFETCH;
      GroupMemoryBarrierWithGroupSync();
      let at = i / TREE_VERIFY_QUADS * TREE_VERIFY_STRIDE + i % TREE_VERIFY_QUADS;
      if (i < TOKEN_QUADS) {
        tree_verify_stage[at] = next_k[slot];
        tree_verify_stage[TREE_VERIFY_Q + at] = next_q[slot];
      }
      if (i < H0_QUADS) {
        tree_verify_stage[TREE_VERIFY_H0 + at] = next_h0[slot];
      }
      GroupMemoryBarrierWithGroupSync();
      [unroll]
      for (uint quad = 0; quad < TREE_VERIFY_QUADS; quad++) {
        let h = tree_verify_stage[TREE_VERIFY_H0 + lane * TREE_VERIFY_STRIDE + quad];
        [unroll]
        for (uint row = 0; row < R; row++) {
          kh0[row] += dot(tree_verify_stage[(first + row) * TREE_VERIFY_STRIDE + quad], h);
          qh0[row] += dot(tree_verify_stage[TREE_VERIFY_Q + (first + row) * TREE_VERIFY_STRIDE + quad], h);
        }
        let keys = tree_verify_stage[gram_row * TREE_VERIFY_STRIDE + quad];
        let queries = tree_verify_stage[TREE_VERIFY_Q + gram_row * TREE_VERIFY_STRIDE + quad];
        [unroll]
        for (uint j = 0; j < GRAM_COLUMNS; j++) {
          let columns = tree_verify_stage[(gram_col + GRAM_SPAN * j) * TREE_VERIFY_STRIDE + quad];
          kk[j] += dot(keys, columns);
          qk[j] += dot(queries, columns);
        }
      }
    }

    if (i < TOKENS) {
      tree_verify_start[i] = node.trie_start;
      tree_verify_end[i] = node.trie_end;
      tree_verify_log_decay[i] = node_log_decay;
      tree_verify_beta[i] = node_beta;
    }
    GroupMemoryBarrierWithGroupSync();
    if (i < TOKENS) {
      // Rows past the tree lie in no interval, as do the empty ones of columns past it.
      var prefix = 0.0;
      for (uint col = 0; col < TOKENS; col++) {
        if (i >= tree_verify_start[col] && i <= tree_verify_end[col]) {
          prefix += tree_verify_log_decay[col];
        }
      }
      tree_verify_prefix[i] = prefix;
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint j = 0; j < GRAM_COLUMNS; j++) {
      let col = gram_col + GRAM_SPAN * j;
      let inside = gram_row < tree_size && gram_row >= tree_verify_start[col] && gram_row <= tree_verify_end[col];
      let decay = exp(tree_verify_prefix[gram_row] - tree_verify_prefix[col]);
      let at = gram_row * TREE_VERIFY_GRAM_STRIDE + col;
      tree_verify_a[at] = inside && gram_row != col ? tree_verify_beta[gram_row] * decay * kk[j] : 0.0;
      tree_verify_qkd[at] = inside ? scale * decay * qk[j] : 0.0;
    }
    [unroll]
    for (uint row = 0; row < R; row++) {
      let token = first + row;
      var rhs = 0.0;
      if (token < tree_size) {
        let v_value = v[value_base + token * value_stride].toFloat();
        rhs = tree_verify_beta[token] * (v_value - exp(tree_verify_prefix[token]) * kh0[row]);
      }
      tree_verify_u[token * TREE_VERIFY_TILE + lane] = rhs;
    }
    GroupMemoryBarrierWithGroupSync();

    // Forward substitution down value column i of the tile, in place over rhs.
    if (i < TREE_VERIFY_TILE) {
      float u[TOKENS] = {};
      [unroll]
      for (uint row = 0; row < TOKENS; row++) {
        var sum = 0.0;
        [unroll]
        for (uint col = 0; col < row; col++) {
          sum += tree_verify_a[row * TREE_VERIFY_GRAM_STRIDE + col] * u[col];
        }
        u[row] = tree_verify_u[row * TREE_VERIFY_TILE + i] - sum;
        tree_verify_u[row * TREE_VERIFY_TILE + i] = u[row];
      }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint row = 0; row < R; row++) {
      let token = first + row;
      if (token < tree_size) {
        var sum = 0.0;
        [unroll]
        for (uint col = 0; col < TOKENS; col++) {
          sum += tree_verify_qkd[token * TREE_VERIFY_GRAM_STRIDE + col] * tree_verify_u[col * TREE_VERIFY_TILE + lane];
        }
        o[value_base + token * value_stride] = T.fromFloat(exp(tree_verify_prefix[token]) * scale * qh0[row] + sum);
      }
    }
  }

  // This invocation's share of slice `slice`: below TOKEN_QUADS, four consecutive depths of a key and a query; below
  // H0_QUADS, four of an h0 row of the tile.
  static void load(
    Ptr<T, Access.Read> q,
    Ptr<T, Access.Read> k,
    Ptr<float4, Access.Read> h0_quads,
    uint tree_size,
    uint qk_base,
    uint qk_stride,
    uint h0_base,
    uint slice,
    uint i,
    out float4 next_k,
    out float4 next_q,
    out float4 next_h0,
  ) {
    let depth = slice * TREE_VERIFY_DEPTH;
    next_k = float4(0.0);
    next_q = float4(0.0);
    next_h0 = float4(0.0);
    let row = i / TREE_VERIFY_QUADS;
    let quad = i % TREE_VERIFY_QUADS;
    if (i < TOKEN_QUADS && row < tree_size) {
      let at = qk_base + row * qk_stride + depth + quad * 4;
      next_k = load_float4(k, at);
      next_q = load_float4(q, at);
    }
    if (i < H0_QUADS) {
      next_h0 = h0_quads[h0_base + (row * TREE_VERIFY_DIM + depth) / 4 + quad];
    }
  }
}

[[Kernel]]
[[Precondition("new", "NB == 1 || NB == 2")]]
[[Precondition("encode", "tree_size <= crate::backends::vulkan::gpu_types::delta_net::TREE_VERIFY_TOKEN_BLOCK * NB && head_k_dim == 128 && head_v_dim == 128")]]
[[Precondition("encode", "num_k_heads != 0 && num_v_heads.is_multiple_of(num_k_heads)")]]
func TreeVerifyValues<T : IModelFloat>(
  Ptr<T, Access.Read> q,
  Ptr<T, Access.Read> k,
  Ptr<T, Access.Read> v,
  Ptr<TrieNode, Access.Read> trie,
  Ptr<float, Access.Read> log_decay,
  Ptr<float, Access.Read> beta,
  Ptr<float, Access.Read> h0,
  Ptr<T, Access.ReadWrite> o,
  float scale,
  uint batch_size,
  uint tree_size,
  uint num_k_heads,
  uint num_v_heads,
  uint head_k_dim,
  uint head_v_dim,
  [[Groups("batch_size")]] uint batch_idx,
  [[Groups("num_v_heads")]] uint hv_idx,
  [[Groups("head_v_dim / 32")]] uint tile,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint NB,
) {
  if (NB == 1) {
    TreeVerifyTile<T, 1>.run(
      q, k, v, trie, log_decay, beta, h0, o, scale, tree_size, num_k_heads, num_v_heads, batch_idx, hv_idx, tile,
      subgroup_invocation()
    );
  } else {
    TreeVerifyTile<T, 2>.run(
      q, k, v, trie, log_decay, beta, h0, o, scale, tree_size, num_k_heads, num_v_heads, batch_idx, hv_idx, tile,
      subgroup_invocation()
    );
  }
}
