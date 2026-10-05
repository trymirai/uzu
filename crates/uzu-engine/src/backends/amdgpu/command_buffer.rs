//! Command buffers record dispatches, copies and fills and issue them to the context's stream on
//! `submit`: the host may still write buffers between encoding and submission (as with Metal), and
//! one in-order stream gives the ordering between and within command buffers.

use std::{
    ffi::c_void,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::backends::{
    amdgpu::{
        Amdgpu,
        buffer::{AmdgpuBufferExt, constant::ConstantPage, device::ScratchPage},
        context::AmdgpuContext,
        error::AmdgpuError,
        hip::{
            HIP_EVENT_BLOCKING_SYNC, HIP_EVENT_DEFAULT, HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE,
            HIP_LAUNCH_PARAM_END, HIP_MEMCPY_DEFAULT, HipEvent, hip_call,
        },
        kernel::{AmdgpuFunction, Kernarg},
        profile,
    },
    common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
        CommandBufferExecutable, CommandBufferPending, allocator::bump::BumpAllocator,
    },
};

pub struct AmdgpuCommandBuffer;

impl CommandBuffer for AmdgpuCommandBuffer {
    type Backend = Amdgpu;

    type Encoding = AmdgpuCommandBufferEncoding;
    type Executable = AmdgpuCommandBufferExecutable;
    type Pending = AmdgpuCommandBufferPending;
    type Completed = AmdgpuCommandBufferCompleted;
}

pub(super) enum Command {
    Dispatch {
        function: AmdgpuFunction,
        groups: [u32; 3],
        threads: [u32; 3],
        kernarg: Kernarg,
        name: &'static str,
    },
    Copy {
        source: u64,
        destination: u64,
        size: usize,
    },
    Fill {
        destination: u64,
        value: u8,
        size: usize,
    },
}

/// Copies and fills run as the kernels of `kernel/native/copy_fill.clcpp` (16 bytes per work-item): HIP's
/// hipMemcpyAsync / hipMemsetAsync on pinned host memory cost ~85-150 us each on Windows. Without that code
/// object they fall back to HIP.
const COPY_FILL_THREADS: u32 = 256;

fn copy_fill_groups(bytes: usize) -> [u32; 3] {
    [(bytes.div_ceil(16 * COPY_FILL_THREADS as usize)) as u32, 1, 1]
}

#[derive(Clone, Copy)]
pub(super) enum CopyFill {
    Copy,
    Fill,
    CopyBatch,
}

/// Copies up to this size wait in the encoder and go out as one `uzu_copy_batch` launch (one workgroup per copy):
/// the DFlash accept step copies a hidden-state row per accepted token and captured layer, ~19k copies (each a
/// launch, ~6-7 us of host issue) for a 1355-token prefill.
const BATCHED_COPY_MAX_BYTES: usize = 64 * 1024;
const MAX_BATCHED_COPIES: usize = 8192;

/// Copies waiting to be launched together: none of them writes bytes another one reads or writes, so their
/// order among themselves does not matter; any other command flushes them first.
#[derive(Default)]
struct PendingCopies {
    descriptors: Vec<[u64; 3]>,
    /// destination start -> end of every pending copy
    destinations: std::collections::BTreeMap<u64, u64>,
    /// bounds of the pending sources (a conservative check: a destination inside them flushes)
    sources: Option<(u64, u64)>,
}

impl PendingCopies {
    fn conflicts(
        &self,
        source: u64,
        destination: u64,
        size: u64,
    ) -> bool {
        let overlaps = |start: u64, end: u64| {
            self.destinations.range(..end).next_back().is_some_and(|(_, &previous_end)| previous_end > start)
        };
        let reads_written = overlaps(source, source + size);
        let writes_written = overlaps(destination, destination + size);
        let writes_read = self.sources.is_some_and(|(low, high)| destination < high && low < destination + size);
        reads_written || writes_written || writes_read
    }

    fn push(
        &mut self,
        source: u64,
        destination: u64,
        size: u64,
    ) {
        self.descriptors.push([source, destination, size]);
        self.destinations.insert(destination, destination + size);
        self.sources = Some(match self.sources {
            Some((low, high)) => (low.min(source), high.max(source + size)),
            None => (source, source + size),
        });
    }
}

pub struct AmdgpuCommandBufferEncoding {
    commands: Vec<Command>,
    pending_copies: PendingCopies,
    created: Instant,
    /// For the timeline (`UZU_AMDGPU_PROFILE=timeline`).
    name: Option<String>,
    constant_allocator: BumpAllocator<ConstantPage>,
    allocation_pool: Arc<<Amdgpu as Backend>::AllocationPool>,
    context: Arc<AmdgpuContext>,
}

impl AmdgpuCommandBufferEncoding {
    pub fn new(
        constant_allocator: BumpAllocator<ConstantPage>,
        allocation_pool: Arc<<Amdgpu as Backend>::AllocationPool>,
        context: Arc<AmdgpuContext>,
    ) -> Self {
        Self {
            commands: Vec::new(),
            pending_copies: PendingCopies::default(),
            created: Instant::now(),
            name: None,
            constant_allocator,
            allocation_pool,
            context,
        }
    }

    /// Launches the pending copies: one `uzu_copy_bytes`, or one `uzu_copy_batch` with their descriptors in
    /// constant memory.
    fn flush_copies(&mut self) {
        let descriptors = std::mem::take(&mut self.pending_copies).descriptors;
        let (function, groups, kernarg, name) = match descriptors.as_slice() {
            [] => return,
            &[[source, destination, size]] => {
                let function = self.context.copy_fill_function(CopyFill::Copy).expect("copy kernel");
                let mut kernarg = Kernarg::new();
                kernarg.push_address(source);
                kernarg.push_address(destination);
                kernarg.push_address(size);
                (function, copy_fill_groups(size as usize), kernarg, "copy")
            },
            _ => {
                let function = self.context.copy_fill_function(CopyFill::CopyBatch).expect("copy batch kernel");
                let table = self.allocate_constant_from_slice(&descriptors).expect("copy descriptors");
                let mut kernarg = Kernarg::new();
                kernarg.push_address(table.device_address());
                kernarg.push_u32(descriptors.len() as u32);
                drop(table);
                (function, [descriptors.len() as u32, 1, 1], kernarg, "copy batch")
            },
        };
        self.commands.push(Command::Dispatch {
            function,
            groups,
            threads: [COPY_FILL_THREADS, 1, 1],
            kernarg,
            name,
        });
    }

    pub(super) fn set_name(
        &mut self,
        name: &str,
    ) {
        self.name = Some(name.to_owned());
    }

    /// Records a kernel launch: `groups` workgroups of `threads` work-items each.
    pub fn dispatch(
        &mut self,
        function: &AmdgpuFunction,
        groups: [u32; 3],
        threads: [u32; 3],
        kernarg: Kernarg,
        name: &'static str,
    ) {
        self.flush_copies();
        self.commands.push(Command::Dispatch {
            function: *function,
            groups,
            threads,
            kernarg,
            name,
        });
    }
}

impl CommandBufferEncoding for AmdgpuCommandBufferEncoding {
    type CommandBuffer = AmdgpuCommandBuffer;

    fn context(&self) -> &AmdgpuContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Amdgpu as Backend>::ConstantBuffer, AmdgpuError> {
        let pool = self.context.constant_pages();
        self.constant_allocator.allocate(size, |size| ConstantPage::take(pool, size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Amdgpu as Backend>::ScratchBuffer, AmdgpuError> {
        let cache = self.context.scratch_pages();
        let stream = self.context.stream();
        self.allocation_pool.allocate(size, |size| ScratchPage::take(cache, size, stream))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Amdgpu>,
        dst: impl BufferMut<Backend = Amdgpu>,
    ) {
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        let size = src_range.iter().len();
        assert_eq!(size, dst_range.iter().len());
        if size == 0 {
            return;
        }
        let source = src.device_address() + src_range.start as u64;
        let destination = dst.device_address() + dst_range.start as u64;
        if size <= BATCHED_COPY_MAX_BYTES && self.context.copy_fill_function(CopyFill::CopyBatch).is_some() {
            if self.pending_copies.conflicts(source, destination, size as u64)
                || self.pending_copies.descriptors.len() >= MAX_BATCHED_COPIES
            {
                self.flush_copies();
            }
            self.pending_copies.push(source, destination, size as u64);
            return;
        }
        if let Some(function) = self.context.copy_fill_function(CopyFill::Copy) {
            let mut kernarg = Kernarg::new();
            kernarg.push_address(source);
            kernarg.push_address(destination);
            kernarg.push_address(size as u64);
            self.dispatch(&function, copy_fill_groups(size), [COPY_FILL_THREADS, 1, 1], kernarg, "copy");
            return;
        }
        self.flush_copies();
        self.commands.push(Command::Copy {
            source,
            destination,
            size,
        });
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Amdgpu>,
        value: u8,
    ) {
        let (dst, range) = dst.parts();
        let size = range.iter().len();
        if size == 0 {
            return;
        }
        let destination = dst.device_address() + range.start as u64;
        self.flush_copies();
        if let Some(function) = self.context.copy_fill_function(CopyFill::Fill) {
            let mut kernarg = Kernarg::new();
            kernarg.push_address(destination);
            kernarg.push_u32(u32::from(value));
            kernarg.push_address(size as u64);
            self.dispatch(&function, copy_fill_groups(size), [COPY_FILL_THREADS, 1, 1], kernarg, "fill");
            return;
        }
        self.commands.push(Command::Fill {
            destination,
            value,
            size,
        });
    }

    fn push_debug_group(
        &mut self,
        _name: &str,
    ) {
    }

    fn pop_debug_group(&mut self) {}

    fn end_encoding(mut self) -> AmdgpuCommandBufferExecutable {
        self.flush_copies();
        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");
        AmdgpuCommandBufferExecutable {
            timeline: self.name.map(|name| profile::TimelineEntry::new(name, self.created, &self.commands)),
            commands: self.commands,
            encode_time: self.created.elapsed(),
            constant_allocator: self.constant_allocator,
            allocation_pool: self.allocation_pool,
            context: self.context,
        }
    }
}

pub struct AmdgpuCommandBufferExecutable {
    timeline: Option<profile::TimelineEntry>,
    commands: Vec<Command>,
    /// CPU time from the start of encoding to its end, for the profile.
    encode_time: Duration,
    constant_allocator: BumpAllocator<ConstantPage>,
    allocation_pool: Arc<<Amdgpu as Backend>::AllocationPool>,
    context: Arc<AmdgpuContext>,
}

// Commands hold HIP function handles and device addresses only.
unsafe impl Send for AmdgpuCommandBufferExecutable {}

struct Event(HipEvent);

unsafe impl Send for Event {}

impl AmdgpuCommandBufferExecutable {
    fn issue(
        context: &AmdgpuContext,
        command: Command,
    ) -> Result<(), AmdgpuError> {
        let hip = context.hip;
        let stream = context.stream();
        match command {
            Command::Dispatch {
                function,
                groups,
                threads,
                mut kernarg,
                name,
            } => {
                let mut size = kernarg.len();
                let mut extra = [
                    HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    kernarg.as_mut_ptr().cast::<c_void>(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    (&raw mut size).cast::<c_void>(),
                    HIP_LAUNCH_PARAM_END,
                ];
                hip_call!(
                    hip,
                    hipModuleLaunchKernel(
                        function.0,
                        groups[0],
                        groups[1],
                        groups[2],
                        threads[0],
                        threads[1],
                        threads[2],
                        0,
                        stream,
                        std::ptr::null_mut(),
                        extra.as_mut_ptr(),
                    )
                )
                .map_err(|error| AmdgpuError::KernelDispatchFailed(format!("{name}: {error}").into()))
            },
            Command::Copy {
                source,
                destination,
                size,
            } => hip_call!(
                hip,
                hipMemcpyAsync(destination as *mut c_void, source as *const c_void, size, HIP_MEMCPY_DEFAULT, stream)
            ),
            Command::Fill {
                destination,
                value,
                size,
            } => hip_call!(hip, hipMemsetAsync(destination as *mut c_void, i32::from(value), size, stream)),
        }
    }

    fn record_event(context: &AmdgpuContext) -> Result<Event, AmdgpuError> {
        Self::record_event_with_flags(context, HIP_EVENT_DEFAULT)
    }

    /// The event a command buffer's completion is awaited on. It blocks the waiting thread instead of letting it
    /// poll: on the APU the CPU and the GPU share one power budget, and a CPU core spinning for the whole decode
    /// takes power from the GPU. `UZU_AMDGPU_SPIN_WAIT=1` polls.
    fn record_completion_event(context: &AmdgpuContext) -> Result<Event, AmdgpuError> {
        static SPIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let spin = *SPIN.get_or_init(|| std::env::var("UZU_AMDGPU_SPIN_WAIT").is_ok_and(|value| value != "0"));
        Self::record_event_with_flags(
            context,
            if spin {
                HIP_EVENT_DEFAULT
            } else {
                HIP_EVENT_BLOCKING_SYNC
            },
        )
    }

    fn record_event_with_flags(
        context: &AmdgpuContext,
        flags: std::os::raw::c_uint,
    ) -> Result<Event, AmdgpuError> {
        let mut event = std::ptr::null_mut();
        hip_call!(context.hip, hipEventCreateWithFlags(&mut event, flags))?;
        hip_call!(context.hip, hipEventRecord(event, context.stream()))?;
        Ok(Event(event))
    }
}

/// Debugging switch `UZU_AMDGPU_SYNC_EACH=1`: wait for every command before issuing the next (tells ordering
/// problems between kernels of a stream from other causes).
fn sync_each() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("UZU_AMDGPU_SYNC_EACH").is_ok_and(|value| value == "1"))
}

impl CommandBufferExecutable for AmdgpuCommandBufferExecutable {
    type CommandBuffer = AmdgpuCommandBuffer;

    fn submit(self) -> AmdgpuCommandBufferPending {
        let context = self.context.clone();
        let profiling = profile::per_command();
        let issue_start = Instant::now();
        let mut timeline = self.timeline;
        if let Some(entry) = timeline.as_mut() {
            entry.submitted = Some(issue_start);
        }
        let mut profile_events = Vec::new();
        let result = {
            let _submission = context.submission.lock();
            (|| -> Result<(Event, Event), AmdgpuError> {
                let start = Self::record_event(&context)?;
                for command in self.commands {
                    if profiling {
                        let label = match &command {
                            Command::Dispatch {
                                name,
                                ..
                            } => *name,
                            Command::Copy {
                                ..
                            } => "copy",
                            Command::Fill {
                                ..
                            } => "fill",
                        };
                        let before = Self::record_event(&context)?;
                        Self::issue(&context, command)?;
                        profile_events.push((label, before, Self::record_event(&context)?));
                    } else {
                        Self::issue(&context, command)?;
                    }
                    if sync_each() {
                        hip_call!(context.hip, hipStreamSynchronize(context.stream()))?;
                    }
                }
                let end = Self::record_completion_event(&context)?;
                Ok((start, end))
            })()
        };

        AmdgpuCommandBufferPending {
            events: result,
            host_times: profile::HostTimes {
                encode: self.encode_time,
                issue: issue_start.elapsed(),
                wait: Duration::ZERO,
                in_flight: Duration::ZERO,
            },
            submitted: Instant::now(),
            timeline,
            profile_events,
            constant_allocator: Some(self.constant_allocator),
            allocation_pool: self.allocation_pool,
            context,
        }
    }
}

pub struct AmdgpuCommandBufferPending {
    events: Result<(Event, Event), AmdgpuError>,
    /// CPU time spent on this command buffer, for the profile.
    host_times: profile::HostTimes,
    submitted: Instant,
    timeline: Option<profile::TimelineEntry>,
    /// Events recorded around each command when profiling per command (`profile::per_command`).
    profile_events: Vec<(&'static str, Event, Event)>,
    constant_allocator: Option<BumpAllocator<ConstantPage>>,
    allocation_pool: Arc<<Amdgpu as Backend>::AllocationPool>,
    context: Arc<AmdgpuContext>,
}

impl CommandBufferPending for AmdgpuCommandBufferPending {
    type CommandBuffer = AmdgpuCommandBuffer;

    fn wait_until_completed(mut self) -> Result<AmdgpuCommandBufferCompleted, AmdgpuError> {
        let hip = self.context.hip;
        let events = std::mem::replace(&mut self.events, Err(AmdgpuError::NotSupported));
        let (start, end) = match events {
            Ok(events) => events,
            Err(error) => {
                // a failed submission may have issued part of the commands: drain before releasing memory
                let _ = hip_call!(hip, hipStreamSynchronize(self.context.stream()));
                return Err(error);
            },
        };
        let wait_start = Instant::now();
        let synchronized = hip_call!(hip, hipEventSynchronize(end.0));
        self.host_times.wait = wait_start.elapsed();
        self.host_times.in_flight = self.submitted.elapsed();
        let result = synchronized.and_then(|()| {
            let mut milliseconds = 0.0f32;
            hip_call!(hip, hipEventElapsedTime(&mut milliseconds, start.0, end.0))?;
            Ok(Duration::from_secs_f64(f64::from(milliseconds.max(0.0)) / 1000.0))
        });
        if let Ok(span) = &result
            && profile::enabled()
        {
            let elapsed = |from: &Event, to: &Event| {
                let mut milliseconds = 0.0f32;
                let _ = hip_call!(hip, hipEventElapsedTime(&mut milliseconds, from.0, to.0));
                f64::from(milliseconds.max(0.0))
            };
            let events = std::mem::take(&mut self.profile_events);
            let samples: Vec<_> =
                events.iter().map(|(label, before, after)| (*label, elapsed(before, after))).collect();
            for (_, before, after) in events {
                let _ = hip_call!(hip, hipEventDestroy(before.0));
                let _ = hip_call!(hip, hipEventDestroy(after.0));
            }
            profile::record(samples, span.as_secs_f64() * 1000.0, self.host_times);
            if let Some(entry) = self.timeline.take() {
                entry.print(wait_start, span.as_secs_f64() * 1000.0);
            }
        }
        let _ = hip_call!(hip, hipEventDestroy(start.0));
        let _ = hip_call!(hip, hipEventDestroy(end.0));
        drop(self.constant_allocator.take());
        Ok(AmdgpuCommandBufferCompleted {
            gpu_execution_time: result?,
            _allocation_pool: self.allocation_pool.clone(),
        })
    }
}

impl Drop for AmdgpuCommandBufferPending {
    fn drop(&mut self) {
        for (_, before, after) in self.profile_events.drain(..) {
            let _ = hip_call!(self.context.hip, hipEventDestroy(before.0));
            let _ = hip_call!(self.context.hip, hipEventDestroy(after.0));
        }
        // never release constants while the GPU may still read them
        if self.constant_allocator.is_some()
            && let Ok((_, end)) = &self.events
        {
            let _ = hip_call!(self.context.hip, hipEventSynchronize(end.0));
        }
    }
}

pub struct AmdgpuCommandBufferCompleted {
    gpu_execution_time: Duration,
    _allocation_pool: Arc<<Amdgpu as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for AmdgpuCommandBufferCompleted {
    type CommandBuffer = AmdgpuCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }
}
