/// RadixTopKSmall's 8-bit digit passes: two over the score bits BF16's widening can set, four over the inverted column.
pub const RADIX_TOP_K_PASSES: u32 = 6;
/// The buckets of an 8-bit digit's histogram, one per workgroup invocation.
pub const RADIX_TOP_K_BUCKETS: u32 = 256;
