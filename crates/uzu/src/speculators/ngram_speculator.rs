use std::{
    collections::HashMap,
    iter::repeat_n,
    ops::{Deref, Range},
};

use bytemuck::cast_slice;
use xxhash_rust::xxh3::xxh3_64;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::speculators::speculator::Speculator;

#[cfg(not(target_endian = "little"))]
compile_error!("Only little endian is supported");

fn seqhash(
    seq: &[u32],
    size: u32,
) -> u32 {
    xxh3_64(cast_slice(seq)) as u32 % size
}

#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
struct NGramSpeculatorHeader {
    hashtable_size: u32,
    top_k: u32,
    ngram_n: u32,
    ngram_pad: u32,
}

pub struct NGramSpeculator<B: Deref<Target = [u8]> + Send + Sync> {
    bytes: B,
    ngram_keys_range: Range<usize>,
    ngram_values_range: Range<usize>,
    ngram_counts_range: Range<usize>,
}

impl<B: Deref<Target = [u8]> + Send + Sync> NGramSpeculator<B> {
    const HEADER_SIZE: usize = size_of::<NGramSpeculatorHeader>();

    pub fn new(bytes: B) -> Self {
        let mut off = 0;

        let header = NGramSpeculatorHeader::ref_from_bytes(&bytes[..Self::HEADER_SIZE]).unwrap();
        off += Self::HEADER_SIZE;

        let ngram_kv_size = 4 * header.top_k as usize * header.hashtable_size as usize;
        let ngram_c_size = 4 * header.hashtable_size as usize;

        let ngram_keys_range = off..(off + ngram_kv_size);
        off += ngram_kv_size;

        let ngram_values_range = off..(off + ngram_kv_size);
        off += ngram_kv_size;

        let ngram_counts_range = off..(off + ngram_c_size);
        off += ngram_c_size;

        assert_eq!(off, bytes.len());

        Self {
            bytes,
            ngram_keys_range,
            ngram_values_range,
            ngram_counts_range,
        }
    }

    #[inline]
    fn header(&self) -> &NGramSpeculatorHeader {
        NGramSpeculatorHeader::ref_from_bytes(&self.bytes[..Self::HEADER_SIZE]).unwrap()
    }

    #[inline]
    fn ngram_keys(&self) -> &[u32] {
        cast_slice(&self.bytes[self.ngram_keys_range.clone()])
    }

    #[inline]
    fn ngram_values(&self) -> &[f32] {
        cast_slice(&self.bytes[self.ngram_values_range.clone()])
    }

    #[inline]
    fn ngram_counts(&self) -> &[u32] {
        cast_slice(&self.bytes[self.ngram_counts_range.clone()])
    }

    fn _seq_slice<'a>(
        &'a self,
        seq: &[u64],
    ) -> (&'a [u32], &'a [f32], &'a u32) {
        let ngram_ctx = (self.header().ngram_n - 1) as usize;

        let padded_seq: Vec<u32> = if seq.len() >= ngram_ctx {
            seq[(seq.len() - ngram_ctx)..].iter().map(|&x| x as u32).collect()
        } else {
            repeat_n(self.header().ngram_pad, ngram_ctx - seq.len()).chain(seq.iter().map(|&x| x as u32)).collect()
        };

        assert!(padded_seq.len() == ngram_ctx);

        let seq_hash = seqhash(&padded_seq, self.header().hashtable_size);
        let idx_s = (seq_hash * self.header().top_k) as usize;
        let idx_e = (seq_hash * self.header().top_k + self.header().top_k) as usize;

        return (
            &self.ngram_keys()[idx_s..idx_e],
            &self.ngram_values()[idx_s..idx_e],
            &self.ngram_counts()[seq_hash as usize],
        );
    }
}

impl NGramSpeculator<memmap2::Mmap> {
    pub fn load(path: &str) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::default().map(&file).unwrap() };

        Self::new(mmap)
    }
}

impl<B: Deref<Target = [u8]> + Send + Sync> Speculator for NGramSpeculator<B> {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        let (ngram_keys, ngram_values, _ngram_counts) = self._seq_slice(prefix);

        ngram_keys.iter().copied().map(u64::from).zip(ngram_values.iter().copied()).collect()
    }
}
