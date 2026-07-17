use std::mem::size_of;

use super::RadixTopKSmallPartitionedMetalKernel;
use crate::backends::{
    common::{Allocation, Encoder, kernel::radix_top_k_small::RadixTopKSmall},
    metal::{Metal, context::MetalContext, error::MetalError},
};

const PARTITIONS: u32 = 4;
const RADIX_BUCKETS: usize = 1024;
const RADIX_BITS: u32 = 10;

pub struct MetalRadixTopKSmall {
    phases: [RadixTopKSmallPartitionedMetalKernel; 4],
    passes: u32,
}

impl RadixTopKSmall<Metal> for MetalRadixTopKSmall {
    fn new(
        context: &MetalContext,
        columns: u32,
    ) -> Result<Self, MetalError> {
        let phase = |phase| RadixTopKSmallPartitionedMetalKernel::new(context, columns, PARTITIONS, phase);
        let index_bits = if columns <= 1 {
            1
        } else {
            u32::BITS - (columns - 1).leading_zeros()
        };
        Ok(Self {
            phases: [phase(0)?, phase(1)?, phase(2)?, phase(3)?],
            passes: (u32::BITS + index_bits).div_ceil(RADIX_BITS),
        })
    }

    fn encode(
        &self,
        input: &Allocation<Metal>,
        output_ids: &mut Allocation<Metal>,
        output_scores: &mut Allocation<Metal>,
        rows: u32,
        k: u32,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let rows = rows as usize;
        let mut histograms = encoder.allocate_scratch(rows * PARTITIONS as usize * RADIX_BUCKETS * size_of::<u32>())?;
        let mut prefixes = encoder.allocate_scratch(rows * size_of::<u64>())?;
        let mut masks = encoder.allocate_scratch(rows * size_of::<u64>())?;
        let mut ranks = encoder.allocate_scratch(rows * size_of::<u32>())?;
        let mut counts = encoder.allocate_scratch(rows * size_of::<u32>())?;
        let mut keys = encoder.allocate_scratch(rows * k as usize * size_of::<u64>())?;
        encoder.encode_fill(&mut prefixes, 0);
        encoder.encode_fill(&mut masks, 0);
        encoder.encode_fill(&mut counts, 0);

        macro_rules! encode {
            ($phase:expr, $pass:expr) => {
                self.phases[$phase].encode(
                    input,
                    &mut *output_ids,
                    &mut *output_scores,
                    &mut histograms,
                    &mut prefixes,
                    &mut masks,
                    &mut ranks,
                    &mut counts,
                    &mut keys,
                    rows as u32,
                    k,
                    $pass,
                    encoder,
                )
            };
        }
        // Qwen uses 12 dispatches without CPU waits. TODO: replace them with one persistent-threadgroup dispatch.
        for pass in 0..self.passes {
            encode!(0, pass);
            encode!(1, pass);
        }
        encode!(2, 0);
        encode!(3, 0);
        Ok(())
    }
}
