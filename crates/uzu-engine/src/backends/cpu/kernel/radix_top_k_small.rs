use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeMut, AsBufferRangeRef, Encoder,
            kernel::radix_top_k_small::{MAX_K, RadixTopKSmall},
        },
        cpu::{Cpu, context::CpuContext, error::CpuError},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct CpuRadixTopKSmall {
    columns: usize,
}

impl RadixTopKSmall<Cpu> for CpuRadixTopKSmall {
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
        input: &Allocation<Cpu>,
        output_ids: &mut Allocation<Cpu>,
        output_scores: &mut Allocation<Cpu>,
        rows: u32,
        k: u32,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        let rows = rows as usize;
        let columns = self.columns;
        let k = k as usize;
        assert!(rows > 0 && k > 0 && k <= MAX_K as usize && k <= columns);
        let input = input.as_buffer_range_ref();
        let input = SendPtr(unsafe { (&*input.buffer().get()).as_ptr().add(input.range().start).cast::<f32>() });
        let output_ids = output_ids.as_buffer_range_mut();
        let output_ids = SendPtrMut(unsafe {
            (&mut *output_ids.buffer().get()).as_mut_ptr().add(output_ids.range().start).cast::<u32>()
        });
        let output_scores = output_scores.as_buffer_range_mut();
        let output_scores = SendPtrMut(unsafe {
            (&mut *output_scores.buffer().get()).as_mut_ptr().add(output_scores.range().start).cast::<f32>()
        });
        encoder.as_command_buffer_mut().push_command(move || {
            let values = unsafe { std::slice::from_raw_parts(input.as_ptr(), rows * columns) };
            let output_ids = unsafe { std::slice::from_raw_parts_mut(output_ids.as_ptr(), rows * k) };
            let output_scores = unsafe { std::slice::from_raw_parts_mut(output_scores.as_ptr(), rows * k) };
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
