use half::bf16;

use crate::{
    backends::{
        common::{
            BufferCpuAccessible, BufferMut, BufferRef,
            kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
        },
        cpu::{
            Cpu, buffer::CpuBufferExt, command_buffer::CpuCommandBufferEncoding, context::CpuContext, error::CpuError,
        },
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct CpuRadixTopKSmall {
    columns: usize,
}

impl RadixTopKSmall for CpuRadixTopKSmall {
    type Backend = Cpu;

    fn new(
        _context: &CpuContext,
        columns: u32,
    ) -> Result<Self, CpuError> {
        assert!(columns > 0);
        Ok(Self {
            columns: columns as usize,
        })
    }

    fn encode(
        &self,
        input: impl BufferRef<Backend = Cpu>,
        output_ids: impl BufferMut<Backend = Cpu>,
        output_scores: impl BufferMut<Backend = Cpu>,
        rows: u32,
        k: u32,
        command_buffer: &mut CpuCommandBufferEncoding,
    ) -> Result<(), CpuError> {
        let rows = rows as usize;
        let columns = self.columns;
        let k = k as usize;
        assert!(rows > 0 && k > 0 && k <= MAX_K as usize && k <= columns);
        let (input, input_range) = input.parts();
        let (output_ids, output_ids_range) = output_ids.parts();
        let (output_scores, output_scores_range) = output_scores.parts();
        let input = SendPtr(
            input.downcast().cpu_ptr().as_ptr().cast::<bf16>().cast_const().wrapping_byte_add(input_range.start),
        );
        let output_ids = SendPtrMut(
            output_ids.downcast().cpu_ptr().as_ptr().cast::<u32>().wrapping_byte_add(output_ids_range.start),
        );
        let output_scores = SendPtrMut(
            output_scores.downcast().cpu_ptr().as_ptr().cast::<f32>().wrapping_byte_add(output_scores_range.start),
        );
        command_buffer.push_command(move || {
            let output_ids = unsafe { std::slice::from_raw_parts_mut(output_ids.as_ptr(), rows * k) };
            let output_scores = unsafe { std::slice::from_raw_parts_mut(output_scores.as_ptr(), rows * k) };
            let values = unsafe { std::slice::from_raw_parts(input.as_ptr(), rows * columns) }
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            for (row, values) in values.chunks_exact(columns).enumerate() {
                let mut indices = (0..columns).collect::<Vec<_>>();
                let compare = |&left: &usize, &right: &usize| {
                    values[right].total_cmp(&values[left]).then_with(|| left.cmp(&right))
                };
                if k < columns {
                    indices.select_nth_unstable_by(k, compare);
                    indices.truncate(k);
                }
                indices.sort_unstable_by(compare);
                for (rank, index) in indices.into_iter().enumerate() {
                    output_ids[row * k + rank] = index as u32;
                    output_scores[row * k + rank] = values[index];
                }
            }
        });
        Ok(())
    }
}
