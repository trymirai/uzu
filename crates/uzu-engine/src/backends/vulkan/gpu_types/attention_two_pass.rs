/// AttentionTwoPass's key blocks: pass 1 runs this many workgroups a query, block b taking keys b, b + blocks, ...
pub const ATTENTION_TWO_PASS_BLOCKS: u32 = 32;
/// AttentionTwoPass1's simdgroups, one a query head, in a workgroup of 256 invocations.
pub const ATTENTION_TWO_PASS_SIMDGROUPS: u32 = 8;
