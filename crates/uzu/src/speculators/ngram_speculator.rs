use std::{collections::HashMap, ops::Deref};

use bytemuck::cast_slice;
use xxhash_rust::xxh3::xxh3_64;
use zerocopy::{FromBytes, Immutable, KnownLayout};

use crate::speculators::speculator::Speculator;

#[cfg(not(target_endian = "little"))]
compile_error!("Only little endian is supported");

const MAX_CTX: usize = 15;

#[inline]
fn full_hash(seq: &[u32]) -> u64 {
    xxh3_64(cast_slice(seq))
}

fn apply_temperature(
    probs: &mut HashMap<u64, f32>,
    inv_tau: f32,
) {
    let mut sum = 0.0f32;
    for v in probs.values_mut() {
        *v = v.powf(inv_tau);
        sum += *v;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in probs.values_mut() {
            *v *= inv_sum;
        }
    }
}

#[repr(C)]
#[derive(FromBytes, KnownLayout, Immutable)]
struct TaggedTableHeader {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
    ngram_pad: u32,
}

const HEADER_SIZE: usize = size_of::<TaggedTableHeader>();

/// Layout of a single tagged n-gram table within the mmap'd buffer.
/// Offsets (`keys_start`, `values_start`) are absolute byte positions.
/// Tags are copied to an aligned buffer at load time to avoid alignment
/// issues with u64 reads from mmap.
struct TaggedTableLayout {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
    /// Token ID used for left-padding when the prefix is shorter than the context window.
    ngram_pad: u32,
    tags: Box<[u64]>,
    keys_start: usize,
    values_start: usize,
}

impl TaggedTableLayout {
    fn parse(
        bytes: &[u8],
        offset: usize,
    ) -> (Self, usize) {
        let header = TaggedTableHeader::ref_from_bytes(&bytes[offset..offset + HEADER_SIZE]).unwrap();

        assert!(header.hashtable_size > 0, "hashtable_size must be > 0");
        assert!(header.top_k > 0, "top_k must be > 0");
        assert!(header.ngram_n >= 1, "ngram_n must be >= 1, got {}", header.ngram_n);
        assert!(
            (header.ngram_n - 1) as usize <= MAX_CTX,
            "ngram_n {} exceeds max supported {}",
            header.ngram_n,
            MAX_CTX + 1
        );

        let hs = header.hashtable_size as usize;
        let k = header.top_k as usize;
        let mut off = offset + HEADER_SIZE;

        let tags_start = off;
        off += 8 * hs;
        let mut tags = vec![0u64; hs].into_boxed_slice();
        for i in 0..hs {
            let t = tags_start + i * 8;
            tags[i] = u64::from_le_bytes(bytes[t..t + 8].try_into().unwrap());
        }

        let keys_start = off;
        off += 4 * k * hs;

        let values_start = off;
        off += 4 * k * hs;

        // cast_slice requires 4-byte alignment for u32/f32.
        // Check the actual byte pointer alignment, not just the numeric offset.
        assert!(
            bytes[keys_start..].as_ptr() as usize % 4 == 0,
            "keys buffer at offset {keys_start} is not 4-byte aligned"
        );
        assert!(
            bytes[values_start..].as_ptr() as usize % 4 == 0,
            "values buffer at offset {values_start} is not 4-byte aligned"
        );

        // Counts: used during training, skipped at inference time
        off += 4 * hs;

        (
            Self {
                hashtable_size: header.hashtable_size,
                top_k: header.top_k,
                ngram_n: header.ngram_n,
                ngram_pad: header.ngram_pad,
                tags,
                keys_start,
                values_start,
            },
            off - offset,
        )
    }

    /// Hash the context, check tag for collision detection, return top-k if tag matches.
    /// For unigram tables (ngram_n=1), context is empty and hash is deterministic —
    /// this matches the Python serializer which stores the hash of an empty context.
    #[inline]
    fn lookup(
        &self,
        bytes: &[u8],
        prefix: &[u64],
    ) -> Option<HashMap<u64, f32>> {
        let ngram_ctx = (self.ngram_n - 1) as usize;

        let mut ctx_buf = [0u32; MAX_CTX];
        if ngram_ctx > 0 {
            let prefix_len = prefix.len();
            if prefix_len >= ngram_ctx {
                for i in 0..ngram_ctx {
                    ctx_buf[i] = prefix[prefix_len - ngram_ctx + i] as u32;
                }
            } else {
                let pad_count = ngram_ctx - prefix_len;
                for i in 0..pad_count {
                    ctx_buf[i] = self.ngram_pad;
                }
                for i in 0..prefix_len {
                    ctx_buf[pad_count + i] = prefix[i] as u32;
                }
            }
        }

        let hash = full_hash(&ctx_buf[..ngram_ctx]);
        let idx = (hash % self.hashtable_size as u64) as usize;
        let tag = hash / self.hashtable_size as u64;

        // Tag mismatch means a hash collision — this context was not stored
        // in this bucket. Back off to a lower-order table.
        if self.tags[idx] != tag {
            return None;
        }

        let k = self.top_k as usize;
        let row_byte_off = idx * k * 4;
        let keys: &[u32] = cast_slice(&bytes[self.keys_start + row_byte_off..self.keys_start + row_byte_off + k * 4]);
        let values: &[f32] =
            cast_slice(&bytes[self.values_start + row_byte_off..self.values_start + row_byte_off + k * 4]);

        let mut result = HashMap::with_capacity(k);
        for i in 0..k {
            result.insert(keys[i] as u64, values[i]);
        }
        Some(result)
    }
}

/// Multi-table n-gram speculator with tag-based collision detection and cascading backoff.
///
/// Binary format: `[max_order: u32, discount: f32]` followed by per-level tagged tables.
/// Each level stores a saturated hash table with tag-based collision detection:
/// `tag = xxh3_64(context) / table_size`, stored alongside each bucket. On tag mismatch,
/// the lookup backs off to the next lower-order table (no interpolation — probabilities
/// are pre-computed with KN discounting at training time).
///
/// Optional temperature scaling sharpens (τ<1) or flattens (τ>1) the draft distribution.
pub struct NGramSpeculator<B: Deref<Target = [u8]> + Send + Sync> {
    bytes: B,
    tables: Vec<TaggedTableLayout>,
    inv_tau: f32,
}

impl<B: Deref<Target = [u8]> + Send + Sync> NGramSpeculator<B> {
    pub fn new(bytes: B) -> Self {
        Self::new_with_temperature(bytes, None)
    }

    pub fn new_with_temperature(
        bytes: B,
        temperature: Option<f32>,
    ) -> Self {
        let mut off = 0;

        let max_order = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        // Discount parameter is baked into the exported probabilities at training time.
        // Stored in the binary for metadata but not used at inference.
        let _discount = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        off += 8;

        let mut tables: Vec<TaggedTableLayout> = Vec::with_capacity(max_order as usize);
        for _ in 0..max_order {
            let table_len = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize;
            off += 8;

            let (layout, parsed_size) = TaggedTableLayout::parse(&bytes, off);
            assert_eq!(
                parsed_size,
                table_len,
                "table {}: parsed size {parsed_size} != declared size {table_len}",
                tables.len()
            );
            // Validate ascending ngram_n order (lookup relies on .iter().rev())
            if let Some(prev) = tables.last() {
                assert!(
                    layout.ngram_n > prev.ngram_n,
                    "tables must be in ascending ngram_n order: got {} after {}",
                    layout.ngram_n,
                    prev.ngram_n
                );
            }
            tables.push(layout);
            off += table_len;
        }

        assert_eq!(off, bytes.len(), "speculator file size mismatch: expected {off} bytes, got {}", bytes.len());

        let inv_tau = match temperature {
            Some(tau) => {
                assert!(tau > 0.0, "temperature must be positive, got {tau}");
                if tau != 1.0 {
                    1.0 / tau
                } else {
                    0.0
                }
            },
            None => 0.0, // no temperature scaling
        };

        Self {
            bytes,
            tables,
            inv_tau,
        }
    }

    /// Cascading backoff: try highest-order table first, fall back to lower orders.
    fn lookup(
        &self,
        prefix: &[u64],
    ) -> Option<HashMap<u64, f32>> {
        for table in self.tables.iter().rev() {
            if let Some(result) = table.lookup(&self.bytes, prefix) {
                return Some(result);
            }
        }
        None
    }
}

impl NGramSpeculator<memmap2::Mmap> {
    pub fn load(path: &str) -> Self {
        Self::load_with_temperature(path, None)
    }

    pub fn load_with_temperature(
        path: &str,
        temperature: Option<f32>,
    ) -> Self {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open speculator file '{path}': {e}"));
        let mmap = unsafe { memmap2::MmapOptions::default().map(&file) }
            .unwrap_or_else(|e| panic!("failed to mmap speculator file '{path}': {e}"));
        Self::new_with_temperature(mmap, temperature)
    }
}

impl<B: Deref<Target = [u8]> + Send + Sync> Speculator for NGramSpeculator<B> {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        match self.lookup(prefix) {
            Some(mut probs) => {
                if self.inv_tau > 0.0 {
                    apply_temperature(&mut probs, self.inv_tau);
                }
                probs
            },
            None => HashMap::new(),
        }
    }
}
