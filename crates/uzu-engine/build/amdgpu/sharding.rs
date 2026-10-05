/// Shard of a kernel variant: FNV-1a 64 of the entry name modulo the shard count.
///
/// The runtime computes the same function (`backends::amdgpu::kernel::shard_index`) to find the code
/// object that holds a variant, so both sides must stay identical.
pub fn shard_index(
    entry_name: &str,
    num_shards: usize,
) -> usize {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in entry_name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash % num_shards as u64) as usize
}
