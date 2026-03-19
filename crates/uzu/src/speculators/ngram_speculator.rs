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

struct TaggedTableLayout {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
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

        assert!(keys_start % 4 == 0, "keys_start {keys_start} is not 4-byte aligned");
        assert!(values_start % 4 == 0, "values_start {values_start} is not 4-byte aligned");

        off += 4 * hs; // counts

        let cont_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4 + 8 * cont_len; // cont_keys + cont_vals

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
/// On tag mismatch, backs off to the next lower-order table.
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
        let _discount = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        off += 8;

        let mut tables = Vec::with_capacity(max_order as usize);
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
            tables.push(layout);
            off += table_len;
        }

        assert_eq!(off, bytes.len(), "speculator file size mismatch: expected {off} bytes, got {}", bytes.len());

        let tau = temperature.unwrap_or(1.0);
        let inv_tau = if tau > 0.0 && tau != 1.0 {
            1.0 / tau
        } else {
            0.0
        };

        Self {
            bytes,
            tables,
            inv_tau,
        }
    }

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
