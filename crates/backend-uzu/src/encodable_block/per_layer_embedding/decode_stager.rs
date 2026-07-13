use std::{io, mem::size_of, sync::atomic::AtomicBool};

use crate::{
    backends::common::{Backend, Context, DenseBuffer, Encoder, SharedEvent},
    encodable_block::per_layer_embedding::{PerLayerEmbeddingError, StagedRows, row_source::RowSource},
    staging::{AsyncStager, Reservation, Stage, StageRequest, wait_until_event_or_cancelled},
};

const SLOT_COUNT: usize = 3;

pub struct DecodeRowStager<B: Backend> {
    stager: AsyncStager<B, ()>,
    sample_id_buffers: Vec<B::DenseBuffer>,
    indices: B::DenseBuffer,
    sample_event: B::SharedEvent,
    row_bytes: usize,
}

impl<B: Backend> DecodeRowStager<B> {
    pub fn new(
        context: &B::Context,
        source: RowSource,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let row_bytes = source.row_bytes();
        let sample_event = context.create_shared_event().map_err(PerLayerEmbeddingError::BackendError)?;
        let sample_id_buffers = (0..SLOT_COUNT)
            .map(|_| context.create_buffer(size_of::<u32>()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let sample_id_ptrs =
            sample_id_buffers.iter().map(|buffer| buffer.cpu_ptr().as_ptr() as usize).collect::<Vec<_>>();
        let worker_event = sample_event.clone();
        let stager = AsyncStager::new(
            context,
            SLOT_COUNT,
            row_bytes,
            move |request: StageRequest<()>, cancelled: &AtomicBool, destination| {
                if !wait_until_event_or_cancelled(&worker_event, request.generation, cancelled) {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "row staging was cancelled"));
                }
                let token_id = unsafe { std::ptr::read_volatile(sample_id_ptrs[request.slot] as *const u32) };
                source.read_rows_while(&[u64::from(token_id)], destination, || true)
            },
        )
        .map_err(PerLayerEmbeddingError::BackendError)?;
        let indices = context.create_buffer(size_of::<u64>()).map_err(PerLayerEmbeddingError::BackendError)?;
        unsafe { indices.cpu_ptr().as_ptr().cast::<u64>().write(0) };
        Ok(Self {
            stager,
            sample_id_buffers,
            indices,
            sample_event,
            row_bytes,
        })
    }

    pub fn reserve_sample(&self) -> io::Result<Reservation> {
        self.stager.try_reserve()
    }

    pub fn sample_readback(
        &mut self,
        reservation: &Reservation,
    ) -> &mut B::DenseBuffer {
        &mut self.sample_id_buffers[reservation.slot()]
    }

    pub fn publish_sample(
        &mut self,
        reservation: Reservation,
        encoder: &mut Encoder<B>,
    ) -> io::Result<Stage<B>> {
        let stage = self.stager.submit(reservation, (), self.row_bytes)?;
        encoder.signal_event(&self.sample_event, stage.generation());
        Ok(stage)
    }

    pub fn stage_token(
        &mut self,
        token_id: u64,
    ) -> io::Result<Stage<B>> {
        let reservation = self.stager.try_reserve()?;
        let sample_id = self.sample_id_buffers[reservation.slot()].cpu_ptr().as_ptr() as *mut u32;
        unsafe { sample_id.write(token_id as u32) };
        let stage = self.stager.submit(reservation, (), self.row_bytes)?;
        self.sample_event.signal(stage.generation());
        Ok(stage)
    }

    pub fn view<'a>(
        &'a self,
        stage: &'a Stage<B>,
    ) -> StagedRows<'a, B> {
        StagedRows {
            stage: self.stager.view(stage),
            indices: &self.indices,
            batch_size: 1,
        }
    }
}
