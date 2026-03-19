use std::{
    ptr::NonNull,
    sync::mpsc::{self, Receiver},
    time::Duration,
};

use block2::RcBlock;
use metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback,
    MTL4CommitOptions, MTL4ComputeCommandEncoder, MTL4ComputeCommandEncoderExt, MTL4VisibilityOptions, MTLBuffer,
    MTLDevice, MTLDeviceExt, MTLEvent, MTLRenderStages, MTLSharedEvent,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSError;

use super::Metal;
use crate::backends::{
    common::{
        AccessFlags, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferInitial, CommandBufferPending,
    },
    metal::{MetalContext, error::MetalError},
};

pub struct MetalCommandBuffer;

impl CommandBuffer for MetalCommandBuffer {
    type Backend = Metal;

    type Initial = MetalCommandBufferInitial;
    type Encoding = MetalCommandBufferEncoding;
    type Executable = MetalCommandBufferExecutable;
    type Pending = MetalCommandBufferPending;
    type Completed = MetalCommandBufferCompleted;
}

pub struct MetalCommandBufferInitial {
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    // completion_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl MetalCommandBufferInitial {
    pub fn create(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, MetalError> {
        let command_buffer = device.new_mtl4_command_buffer().ok_or(MetalError::CannotCreateBuffer)?;
        let command_allocator = device.new_command_allocator().ok_or(MetalError::CannotCreateBuffer)?;

        Ok(Self {
            command_buffer,
            command_allocator,
        })
    }
}

impl CommandBufferInitial for MetalCommandBufferInitial {
    type CommandBuffer = MetalCommandBuffer;

    fn start_encoding(self) -> MetalCommandBufferEncoding {
        self.command_buffer.begin_command_buffer_with_allocator(&self.command_allocator);

        let command_encoder = self.command_buffer.compute_command_encoder().unwrap();
        command_encoder.barrier_after_queue_stages_before_stages_visibility_options(
            MTLRenderStages::Dispatch | MTLRenderStages::Blit,
            MTLRenderStages::Dispatch | MTLRenderStages::Blit,
            MTL4VisibilityOptions::Device,
        );

        MetalCommandBufferEncoding {
            command_encoder,
            command_buffer: self.command_buffer,
            command_allocator: self.command_allocator,
        }
    }
}

pub struct MetalCommandBufferEncoding {
    command_encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
}

impl MetalCommandBufferEncoding {
    pub(super) fn command_encoder(&self) -> &ProtocolObject<dyn MTL4ComputeCommandEncoder> {
        &self.command_encoder
    }
}

impl CommandBufferEncoding for MetalCommandBufferEncoding {
    type CommandBuffer = MetalCommandBuffer;

    fn encode_copy(
        &mut self,
        src: &Retained<ProtocolObject<dyn MTLBuffer>>,
        dst: &mut Retained<ProtocolObject<dyn MTLBuffer>>,
        size: usize,
    ) {
        assert!(src.length() >= size && dst.length() >= size);

        self.command_encoder.copy_from_buffer_source_offset_to_buffer_destination_offset_size(src, 0, dst, 0, size);
    }

    fn encode_fill(
        &mut self,
        dst: &mut Retained<ProtocolObject<dyn MTLBuffer>>,
        range: std::ops::Range<usize>,
        value: u8,
    ) {
        assert!(range.end > range.start && range.end <= dst.length());
        assert!(range.start % 4 == 0 && range.end % 4 == 0);

        self.command_encoder.fill_buffer_range_value(dst, range, value);
    }

    fn encode_barrier(
        &mut self,
        after: AccessFlags,
        before: AccessFlags,
    ) {
        let mut after_stages = MTLRenderStages::empty();
        if after.compute_read | after.compute_write {
            after_stages |= MTLRenderStages::Dispatch;
        }
        if after.copy_read | after.copy_write {
            after_stages |= MTLRenderStages::Blit;
        }

        let mut before_stages = MTLRenderStages::empty();
        if before.compute_read | before.compute_write {
            before_stages |= MTLRenderStages::Dispatch;
        }
        if before.copy_read | before.copy_write {
            before_stages |= MTLRenderStages::Blit;
        }

        self.command_encoder.barrier_after_encoder_stages_before_encoder_stages_visibility_options(
            after_stages,
            before_stages,
            MTL4VisibilityOptions::Device,
        );
    }

    fn encode_wait_for_event(
        &mut self,
        _event: &Retained<ProtocolObject<dyn MTLEvent>>,
        _value: u64,
    ) {
        todo!();
    }

    fn encode_signal_event(
        &mut self,
        _event: &Retained<ProtocolObject<dyn MTLEvent>>,
        _value: u64,
    ) {
        todo!();
    }

    fn add_completion_handler(
        &mut self,
        _handler: impl FnOnce(Result<&MetalCommandBufferCompleted, MetalError>) + 'static,
    ) {
        todo!();
        // TODO
        // let cell = Cell::new(Some(handler));
        // self.command_buffer.add_completed_handler(&MTLCommandBufferHandler::new(move |command_buffer| {
        //     let cbuf_ref = MetalCommandBufferCompleted {
        //         command_buffer: command_buffer.retain(),
        //     };
        //     cell.take().expect("completion handler called more than once")(
        //         command_buffer_result(command_buffer).map(|_| &cbuf_ref),
        //     )
        // }));
    }

    fn end_encoding(self) -> <Self::CommandBuffer as CommandBuffer>::Executable {
        self.command_encoder.barrier_after_stages_before_queue_stages_visibility_options(
            MTLRenderStages::Dispatch | MTLRenderStages::Blit,
            MTLRenderStages::Dispatch | MTLRenderStages::Blit,
            MTL4VisibilityOptions::Device,
        );
        self.command_encoder.end_encoding();
        self.command_buffer.end_command_buffer();

        MetalCommandBufferExecutable {
            command_buffer: self.command_buffer,
            command_allocator: self.command_allocator,
        }
    }
}

pub struct MetalCommandBufferExecutable {
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
}

impl CommandBufferExecutable for MetalCommandBufferExecutable {
    type CommandBuffer = MetalCommandBuffer;

    fn submit(
        self,
        context: &MetalContext,
    ) -> MetalCommandBufferPending {
        let mut buf_ptr: NonNull<ProtocolObject<dyn MTL4CommandBuffer>> = NonNull::from(&*self.command_buffer);
        let options = MTL4CommitOptions::new();
        // let (mpsc_sender, mpsc_receiver) = mpsc::channel();
        // let block = RcBlock::new(move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
        //     let feedback = unsafe { feedback.as_ref() };
        //     let r = mpsc_sender.send((feedback.gpu_start_time(), feedback.gpu_end_time(), feedback.error()));
        //     eprintln!(
        //         "\n\n\n\n\n{:?} {:?} {:?} {:?}\n\n\n\n\n\n",
        //         feedback.gpu_start_time(),
        //         feedback.gpu_start_time(),
        //         feedback.error(),
        //         r
        //     );
        // });
        // options.add_feedback_handler(&*block as *const _ as *mut _);
        let mut timeline_counter = context.timeline_counter.borrow_mut();
        context.command_queue.wait_for_event_value(context.timeline_event.as_ref(), *timeline_counter);
        context.command_queue.commit_count_options(NonNull::from(&mut buf_ptr), 1, &options);
        *timeline_counter += 1;
        context.command_queue.signal_event_value(context.timeline_event.as_ref(), *timeline_counter);

        MetalCommandBufferPending {
            command_buffer: self.command_buffer,
            command_allocator: self.command_allocator,
            timeline_event: context.timeline_event.clone(),
            timeline_counter: *timeline_counter,
            // mpsc_receiver,
        }
    }
}

pub struct MetalCommandBufferPending {
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    timeline_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    timeline_counter: u64,
}

impl CommandBufferPending for MetalCommandBufferPending {
    type CommandBuffer = MetalCommandBuffer;

    fn wait_until_completed(self) -> Result<MetalCommandBufferCompleted, MetalError> {
        assert!(self.timeline_event.wait_until_signaled_value_timeout_ms(self.timeline_counter, 10_000));

        Ok(MetalCommandBufferCompleted {
            command_buffer: self.command_buffer,
            command_allocator: self.command_allocator,
        })
    }
}

pub struct MetalCommandBufferCompleted {
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
}

impl CommandBufferCompleted for MetalCommandBufferCompleted {
    type CommandBuffer = MetalCommandBuffer;

    fn gpu_execution_time(&self) -> Option<Duration> {
        None
    }
}
