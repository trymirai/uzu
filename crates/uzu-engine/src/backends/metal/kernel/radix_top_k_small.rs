use std::mem::size_of;

use super::{RadixTopKSmallCollectMetalKernel, RadixTopKSmallPassMetalKernel};
use crate::backends::{
    common::{
        Allocation, Encoder,
        kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
    },
    metal::{Metal, context::MetalContext, error::MetalError},
};

const DEFAULT_PARTITIONS: u32 = 4;
const LARGE_COLUMNS_MIN: u32 = 131_072;
const RADIX_BITS: u32 = 10;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

const fn partitions(columns: u32) -> u32 {
    if columns >= LARGE_COLUMNS_MIN {
        8
    } else {
        DEFAULT_PARTITIONS
    }
}

pub struct MetalRadixTopKSmall {
    pass: RadixTopKSmallPassMetalKernel,
    collect: RadixTopKSmallCollectMetalKernel,
    columns: u32,
    partitions: u32,
    passes: u32,
}

impl RadixTopKSmall<Metal> for MetalRadixTopKSmall {
    fn new(
        context: &MetalContext,
        columns: u32,
    ) -> Result<Self, MetalError> {
        assert!(columns > 0);
        let index_bits = if columns <= 1 {
            1
        } else {
            u32::BITS - (columns - 1).leading_zeros()
        };
        let partitions = partitions(columns);
        Ok(Self {
            pass: RadixTopKSmallPassMetalKernel::new(context, columns, partitions)?,
            collect: RadixTopKSmallCollectMetalKernel::new(context, columns, partitions)?,
            columns,
            partitions,
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
        assert!(rows > 0 && k > 0 && k <= MAX_K && k <= self.columns);
        let rows = rows as usize;
        let mut histograms =
            encoder.allocate_scratch(rows * self.partitions as usize * RADIX_BUCKETS * size_of::<u32>())?;
        let mut prefixes = encoder.allocate_scratch(rows * size_of::<u64>())?;
        let mut masks = encoder.allocate_scratch(rows * size_of::<u64>())?;
        let mut ranks = encoder.allocate_scratch(rows * size_of::<u32>())?;
        let mut counts = encoder.allocate_scratch(rows * size_of::<u32>())?;
        let mut done = encoder.allocate_scratch(rows * size_of::<u32>())?;
        let mut keys = encoder.allocate_scratch(rows * k as usize * size_of::<u64>())?;
        encoder.encode_fill(&mut prefixes, 0);
        encoder.encode_fill(&mut masks, 0);
        encoder.encode_fill(&mut counts, 0);
        encoder.encode_fill(&mut done, 0);

        for pass in 0..self.passes {
            self.pass.encode(
                input,
                &mut histograms,
                &mut prefixes,
                &mut masks,
                &mut ranks,
                &mut done,
                rows as u32,
                k,
                pass,
                encoder,
            );
        }
        self.collect.encode(
            input,
            output_ids,
            output_scores,
            &prefixes,
            &mut counts,
            &mut done,
            &mut keys,
            rows as u32,
            k,
            encoder,
        );
        Ok(())
    }
}
