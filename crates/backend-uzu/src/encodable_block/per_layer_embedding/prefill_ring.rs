use std::{io, mem::size_of, sync::atomic::AtomicBool};

use super::{
    PerLayerEmbeddingError,
    async_stager::{AsyncStager, StageRequest, StageTicket},
};
use crate::{
    backends::common::{Backend, Context, DenseBuffer},
    parameters::ParameterRowSource,
};

pub const PREFILL_CHUNK_SIZE: usize = 1024;
const SLOT_COUNT: usize = 2;

pub(crate) fn prefill_chunk_size() -> usize {
    PREFILL_CHUNK_SIZE
}

pub struct PreparedPrefillBatch<'a, B: Backend> {
    pub allocation: &'a B::DenseBuffer,
    pub indices: &'a B::DenseBuffer,
    pub ready_event: &'a B::SharedEvent,
    pub consumed_event: &'a B::SharedEvent,
    pub value: u64,
    pub batch_size: usize,
}

pub struct PrefillChunkTicket<B: Backend> {
    pub(crate) stage: StageTicket<B>,
    pub(crate) batch_size: usize,
}

pub struct PrefillRing<B: Backend> {
    stager: AsyncStager<B, Box<[u64]>>,
    indices: B::DenseBuffer,
    row_bytes: usize,
    chunk_size: usize,
    next_chunk: usize,
}

impl<B: Backend> Unpin for PrefillRing<B> {}

impl<B: Backend> PrefillRing<B> {
    pub fn new(
        context: &B::Context,
        source: ParameterRowSource,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let row_bytes = source.row_bytes();
        let slot_bytes = PREFILL_CHUNK_SIZE
            .checked_mul(row_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "PLE prefill slot size overflow"))?;
        let stager = AsyncStager::new(
            context,
            SLOT_COUNT,
            slot_bytes,
            move |request: StageRequest<Box<[u64]>>, _cancelled: &AtomicBool, destination| {
                source.read_rows(&request.request, destination)
            },
        )
        .map_err(PerLayerEmbeddingError::BackendError)?;
        let indices = context
            .create_buffer(PREFILL_CHUNK_SIZE * size_of::<u64>())
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let index_slice =
            unsafe { std::slice::from_raw_parts_mut(indices.cpu_ptr().as_ptr().cast::<u64>(), PREFILL_CHUNK_SIZE) };
        for (index, value) in index_slice.iter_mut().enumerate() {
            *value = index as u64;
        }
        Ok(Self {
            stager,
            indices,
            row_bytes,
            chunk_size: prefill_chunk_size(),
            next_chunk: 0,
        })
    }

    pub fn stage(
        &mut self,
        chunk: usize,
        input_chunk: &[u64],
    ) -> io::Result<PrefillChunkTicket<B>> {
        if chunk != self.next_chunk {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected PLE prefill chunk {}, got {chunk}", self.next_chunk),
            ));
        }
        if input_chunk.is_empty() || input_chunk.len() > self.chunk_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("PLE prefill chunk length must be in 1..={}, got {}", self.chunk_size, input_chunk.len()),
            ));
        }
        let stage = self.stager.reserve(true)?;
        let wait_for_consumed = (chunk >= SLOT_COUNT).then_some((chunk - SLOT_COUNT + 1) as u64);
        if let Err(error) = self.stager.enqueue(
            &stage,
            input_chunk.to_vec().into_boxed_slice(),
            input_chunk.len() * self.row_bytes,
            wait_for_consumed,
        ) {
            return Err(error);
        }
        self.next_chunk += 1;
        Ok(PrefillChunkTicket {
            stage,
            batch_size: input_chunk.len(),
        })
    }

    pub fn prepared<'a>(
        &'a self,
        ticket: &'a PrefillChunkTicket<B>,
    ) -> PreparedPrefillBatch<'a, B> {
        let prepared = self.stager.prepared(&ticket.stage);
        PreparedPrefillBatch {
            allocation: prepared.allocation,
            indices: &self.indices,
            ready_event: prepared.ready_event,
            consumed_event: prepared.consumed_event,
            value: prepared.value,
            batch_size: ticket.batch_size,
        }
    }
}
