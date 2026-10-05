use std::mem::size_of;

use super::{RadixTopKSmallCollectAmdgpuKernel, RadixTopKSmallPassAmdgpuKernel};
use crate::backends::{
    amdgpu::{Amdgpu, context::AmdgpuContext, error::AmdgpuError},
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
    },
};

// Same policy as the Metal backend (backends/metal/kernel/radix_top_k_small.rs).
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

pub struct AmdgpuRadixTopKSmall {
    pass: RadixTopKSmallPassAmdgpuKernel,
    collect: RadixTopKSmallCollectAmdgpuKernel,
    columns: u32,
    partitions: u32,
    passes: u32,
}

impl RadixTopKSmall for AmdgpuRadixTopKSmall {
    type Backend = Amdgpu;

    fn new(
        context: &AmdgpuContext,
        columns: u32,
    ) -> Result<Self, AmdgpuError> {
        assert!(columns > 0);
        let index_bits = if columns <= 1 {
            1
        } else {
            u32::BITS - (columns - 1).leading_zeros()
        };
        let partitions = partitions(columns);
        Ok(Self {
            pass: RadixTopKSmallPassAmdgpuKernel::new(context, columns, partitions)?,
            collect: RadixTopKSmallCollectAmdgpuKernel::new(context, columns, partitions)?,
            columns,
            partitions,
            passes: (u32::BITS + index_bits).div_ceil(RADIX_BITS),
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = Amdgpu>,
        output_ids: impl BufferMut<Backend = Amdgpu>,
        output_scores: impl BufferMut<Backend = Amdgpu>,
        rows: u32,
        k: u32,
        command_buffer: &mut <<Amdgpu as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), AmdgpuError> {
        assert!(rows > 0 && k > 0 && k <= MAX_K && k <= self.columns);
        let rows = rows as usize;
        let mut histograms =
            command_buffer.allocate_scratch(rows * self.partitions as usize * RADIX_BUCKETS * size_of::<u32>())?;
        let mut prefixes = command_buffer.allocate_scratch(rows * size_of::<u64>())?;
        let mut masks = command_buffer.allocate_scratch(rows * size_of::<u64>())?;
        let mut ranks = command_buffer.allocate_scratch(rows * size_of::<u32>())?;
        let mut counts = command_buffer.allocate_scratch(rows * size_of::<u32>())?;
        let mut done = command_buffer.allocate_scratch(rows * size_of::<u32>())?;
        let mut keys = command_buffer.allocate_scratch(rows * k as usize * size_of::<u64>())?;
        command_buffer.encode_fill(&mut prefixes, 0);
        command_buffer.encode_fill(&mut masks, 0);
        command_buffer.encode_fill(&mut counts, 0);
        command_buffer.encode_fill(&mut done, 0);

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
                command_buffer,
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
            command_buffer,
        );
        Ok(())
    }
}
