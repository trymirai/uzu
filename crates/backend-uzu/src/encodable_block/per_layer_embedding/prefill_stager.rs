use std::{
    io,
    mem::size_of,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{
    backends::common::{Backend, Context, DenseBuffer},
    encodable_block::per_layer_embedding::{PerLayerEmbeddingError, StagedRows, row_source::RowSource},
    staging::{AsyncStager, Stage, StageRequest},
};

pub(crate) const PREFILL_CHUNK_SIZE: usize = 1024;
const SLOT_COUNT: usize = 2;

pub struct PrefillBatch<B: Backend> {
    pub(crate) stage: Stage<B>,
    pub(crate) batch_size: usize,
}

pub struct PrefillStager<B: Backend> {
    stager: AsyncStager<B, Box<[u64]>>,
    indices: B::DenseBuffer,
    row_bytes: usize,
}

impl<B: Backend> PrefillStager<B> {
    pub fn new(
        context: &B::Context,
        source: RowSource,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let row_bytes = source.row_bytes();
        let slot_bytes = PREFILL_CHUNK_SIZE
            .checked_mul(row_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "PLE prefill slot size overflow"))?;
        let stager = AsyncStager::new(
            context,
            SLOT_COUNT,
            slot_bytes,
            move |request: StageRequest<Box<[u64]>>, cancelled: &AtomicBool, destination| {
                source.read_rows_while(&request.request, destination, || !cancelled.load(Ordering::Acquire))
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
        })
    }

    pub fn stage(
        &mut self,
        input_chunk: &[u64],
    ) -> io::Result<PrefillBatch<B>> {
        if input_chunk.is_empty() || input_chunk.len() > PREFILL_CHUNK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("PLE prefill chunk length must be in 1..={PREFILL_CHUNK_SIZE}, got {}", input_chunk.len()),
            ));
        }
        let reservation = self.stager.reserve()?;
        let stage = self.stager.submit(
            reservation,
            input_chunk.to_vec().into_boxed_slice(),
            input_chunk.len() * self.row_bytes,
        )?;
        Ok(PrefillBatch {
            stage,
            batch_size: input_chunk.len(),
        })
    }

    pub fn view<'a>(
        &'a self,
        batch: &'a PrefillBatch<B>,
    ) -> StagedRows<'a, B> {
        let prepared = self.stager.view(&batch.stage);
        StagedRows {
            stage: prepared,
            indices: &self.indices,
            batch_size: batch.batch_size,
        }
    }
}
