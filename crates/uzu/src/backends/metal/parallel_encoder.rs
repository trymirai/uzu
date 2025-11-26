use std::sync::{Arc, Mutex};

use metal::{Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoderRef, Device, Fence};
use rayon::ThreadPool;

/// Wrapper for MTLBuffer that is Send + Sync
/// SAFETY: Metal buffers are thread-safe for reading and encoding
#[derive(Clone)]
pub struct SendableBuffer(pub Buffer);

unsafe impl Send for SendableBuffer {}
unsafe impl Sync for SendableBuffer {}

impl SendableBuffer {
    pub fn new(buffer: Buffer) -> Self {
        Self(buffer)
    }

    pub fn as_ref(&self) -> &Buffer {
        &self.0
    }
}

/// Encoding closure that captures all needed data
/// This is created on the main thread with all buffer pointers extracted
pub type EncodingClosure = Box<dyn FnOnce(&ComputeCommandEncoderRef) + Send>;

/// A single encoding unit with its position in the sequence
pub struct EncodingUnit {
    pub index: usize,
    pub encode_fn: EncodingClosure,
}

/// Pre-encoded forward pass (multiple uncommitted command buffers)
pub struct PreEncodedForwardPass {
    pub command_buffers: Vec<CommandBuffer>,
    pub key: String,
}

impl PreEncodedForwardPass {
    pub fn commit_all(&self) {
        for cb in &self.command_buffers {
            cb.commit();
        }
    }

    pub fn last_command_buffer(&self) -> Option<CommandBuffer> {
        self.command_buffers.last().cloned()
    }
}

/// Context for parallel command buffer encoding
pub struct ParallelEncodingContext {
    queue: CommandQueue,
    fences: Vec<Fence>,
    pool: ThreadPool,
    max_encoders: usize,
}

impl ParallelEncodingContext {
    pub fn new(device: &Device, queue: CommandQueue, max_encoders: usize) -> Self {
        let fences: Vec<_> = (0..max_encoders)
            .map(|_| device.new_fence())
            .collect();

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8)
            .max(2);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("metal-encoder-{}", i))
            .build()
            .expect("Failed to create thread pool");

        eprintln!(
            "[ParallelEncoder] Created with {} threads, {} max encoders",
            num_threads, max_encoders
        );

        Self {
            queue,
            fences,
            pool,
            max_encoders,
        }
    }

    /// Encode forward pass sequentially with multiple command buffers
    /// Each layer gets its own CB but encoding happens on main thread
    ///
    /// `previous_fence`: optional fence that the FIRST unit should wait on (e.g. from embed)
    pub fn encode_and_commit(
        &self,
        units: Vec<EncodingUnit>,
        previous_fence: Option<&Fence>,
    ) -> CommandBuffer {
        assert!(!units.is_empty(), "No encoding units provided");
        assert!(
            units.len() <= self.max_encoders,
            "Too many encoding units: {} > {}",
            units.len(),
            self.max_encoders
        );

        let num_units = units.len();
        let mut last_cb: Option<CommandBuffer> = None;

        // Sequential encoding - one CB per layer, fences between them
        for unit in units {
            let idx = unit.index;
            let cb = self.queue.new_command_buffer();
            cb.set_label(&format!("seq-encoder-{}", idx));
            let encoder = cb.new_compute_command_encoder();

            // GPU wait on previous encoder's fence
            if idx == 0 {
                if let Some(pf) = previous_fence {
                    encoder.wait_for_fence(pf);
                }
            } else {
                encoder.wait_for_fence(&self.fences[idx - 1]);
            }

            // Execute the encoding closure
            (unit.encode_fn)(encoder);

            // Signal fence for next encoder
            encoder.update_fence(&self.fences[idx]);
            encoder.end_encoding();
            cb.commit();

            if idx == num_units - 1 {
                last_cb = Some(cb.to_owned());
            }
        }

        last_cb.expect("Last command buffer not set")
    }

    /// Encode forward pass in parallel WITHOUT committing (for pre-encoding)
    /// Returns PreEncodedForwardPass that can be committed later
    pub fn encode_without_commit(
        &self,
        units: Vec<EncodingUnit>,
        previous_fence: Option<&Fence>,
        key: String,
    ) -> PreEncodedForwardPass {
        assert!(!units.is_empty(), "No encoding units provided");
        assert!(
            units.len() <= self.max_encoders,
            "Too many encoding units: {} > {}",
            units.len(),
            self.max_encoders
        );

        let command_buffers: Arc<Mutex<Vec<(usize, CommandBuffer)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(units.len())));

        let units: Vec<_> = units
            .into_iter()
            .map(|u| Arc::new(Mutex::new(Some(u))))
            .collect();

        let prev_fence_for_first: Option<Fence> =
            previous_fence.map(|f| f.to_owned());

        self.pool.scope(|s| {
            for unit_cell in units.iter() {
                let unit_cell = unit_cell.clone();
                let cbs = command_buffers.clone();
                let queue = &self.queue;
                let fences = &self.fences;
                let prev_fence_first = prev_fence_for_first.clone();

                s.spawn(move |_| {
                    let unit = unit_cell
                        .lock()
                        .unwrap()
                        .take()
                        .expect("Unit already consumed");

                    let idx = unit.index;
                    let cb = queue.new_command_buffer();
                    cb.set_label(&format!("pre-encoded-{}", idx));
                    let encoder = cb.new_compute_command_encoder();

                    // GPU wait on previous encoder's fence
                    if idx == 0 {
                        if let Some(ref pf) = prev_fence_first {
                            encoder.wait_for_fence(pf);
                        }
                    } else {
                        encoder.wait_for_fence(&fences[idx - 1]);
                    }

                    // Execute the encoding closure
                    (unit.encode_fn)(encoder);

                    // Signal fence for next encoder
                    encoder.update_fence(&fences[idx]);
                    encoder.end_encoding();

                    // DON'T commit - store for later
                    cbs.lock().unwrap().push((idx, cb.to_owned()));
                });
            }
        });

        // Sort by index to maintain order
        let mut cbs = command_buffers.lock().unwrap();
        cbs.sort_by_key(|(idx, _)| *idx);
        let command_buffers: Vec<CommandBuffer> =
            cbs.drain(..).map(|(_, cb)| cb).collect();

        PreEncodedForwardPass {
            command_buffers,
            key,
        }
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn fence(&self, index: usize) -> &Fence {
        &self.fences[index]
    }

    pub fn num_workers(&self) -> usize {
        self.pool.current_num_threads()
    }
}

