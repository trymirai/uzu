use std::{io, mem::size_of, sync::atomic::AtomicBool};

use super::{
    PerLayerEmbeddingError,
    async_stager::{AsyncStager, PreparedStage, StageRequest, StageTicket, wait_until_event_or_cancelled},
};
use crate::{
    backends::common::{Allocation, Backend, Context, DenseBuffer, Encoder, SharedEvent},
    parameters::ParameterRowSource,
};

const SLOT_COUNT: usize = 3;

struct RowRequest<B: Backend> {
    sample_event: B::SharedEvent,
    mailbox: usize,
}

pub(crate) type RowTicket<B> = StageTicket<B>;
pub(crate) type PreparedRow<'a, B> = PreparedStage<'a, B>;

pub struct RowRing<B: Backend> {
    mailboxes: Vec<B::DenseBuffer>,
    sample_event: B::SharedEvent,
    stager: AsyncStager<B, RowRequest<B>>,
    row_bytes: usize,
}

impl<B: Backend> Unpin for RowRing<B> {}

impl<B: Backend> RowRing<B> {
    pub fn new(
        context: &B::Context,
        source: ParameterRowSource,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let row_bytes = source.row_bytes();
        let sample_event = context.create_shared_event().map_err(PerLayerEmbeddingError::BackendError)?;
        let stager = AsyncStager::new(
            context,
            SLOT_COUNT,
            row_bytes,
            move |request: StageRequest<RowRequest<B>>, cancelled: &AtomicBool, destination| {
                if !wait_until_event_or_cancelled(&request.request.sample_event, request.generation, cancelled) {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "row staging was cancelled"));
                }
                let token_id = unsafe { std::ptr::read_volatile(request.request.mailbox as *const u32) };
                source.read_rows(&[u64::from(token_id)], destination)
            },
        )
        .map_err(PerLayerEmbeddingError::BackendError)?;
        let mailboxes = (0..SLOT_COUNT)
            .map(|_| context.create_buffer(size_of::<u32>()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PerLayerEmbeddingError::BackendError)?;
        Ok(Self {
            mailboxes,
            sample_event,
            stager,
            row_bytes,
        })
    }

    pub fn publish_sample(
        &mut self,
        sampled_token: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> io::Result<RowTicket<B>> {
        let ticket = self.stager.reserve(false)?;
        encoder.encode_copy(sampled_token, 0..size_of::<u32>(), &mut self.mailboxes[ticket.slot], ..);
        encoder.signal_event(&self.sample_event, ticket.generation);
        let mailbox = self.mailboxes[ticket.slot].cpu_ptr().as_ptr() as usize;
        self.stager.enqueue(
            &ticket,
            RowRequest {
                sample_event: self.sample_event.clone(),
                mailbox,
            },
            self.row_bytes,
            None,
        )?;
        Ok(ticket)
    }

    pub fn publish_known(
        &mut self,
        token_id: u64,
    ) -> io::Result<RowTicket<B>> {
        let ticket = self.stager.reserve(false)?;
        let mailbox = self.mailboxes[ticket.slot].cpu_ptr().as_ptr() as *mut u32;
        unsafe { mailbox.write(token_id as u32) };
        self.sample_event.signal(ticket.generation);
        self.stager.enqueue(
            &ticket,
            RowRequest {
                sample_event: self.sample_event.clone(),
                mailbox: mailbox as usize,
            },
            self.row_bytes,
            None,
        )?;
        Ok(ticket)
    }

    pub fn prepared<'a>(
        &'a self,
        ticket: &'a RowTicket<B>,
    ) -> PreparedRow<'a, B> {
        self.stager.prepared(ticket)
    }
}
