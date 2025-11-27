use std::cell::RefCell;

use metal::{CommandBuffer, CommandQueue};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use objc2::rc::Retained;

pub struct ExecutionOrchestrator {
    queue: CommandQueue,
    pending_cbs: RefCell<Vec<CommandBuffer>>,
    committed_cbs: RefCell<Vec<CommandBuffer>>,
    // Track committed MPS CB for waiting
    committed_mps_cb: RefCell<Option<CommandBuffer>>,
    // Pending MPS CB (from current encode, before commit)
    pending_mps_cb: RefCell<Option<CommandBuffer>>,
    // Whether pending MPS CB needs commit (pre-encode sets this true)
    pending_mps_needs_commit: RefCell<bool>,
    // Stashed CBs from pre-encode (survive clear)
    stashed_cbs: RefCell<Vec<CommandBuffer>>,
    // Stashed MPS CB from pre-encode
    stashed_mps_cb: RefCell<Option<CommandBuffer>>,
    stashed_mps_needs_commit: RefCell<bool>,
}

impl ExecutionOrchestrator {
    pub fn new(queue: CommandQueue) -> Self {
        Self {
            queue,
            pending_cbs: RefCell::new(Vec::new()),
            committed_cbs: RefCell::new(Vec::new()),
            committed_mps_cb: RefCell::new(None),
            pending_mps_cb: RefCell::new(None),
            pending_mps_needs_commit: RefCell::new(false),
            stashed_cbs: RefCell::new(Vec::new()),
            stashed_mps_cb: RefCell::new(None),
            stashed_mps_needs_commit: RefCell::new(false),
        }
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn new_command_buffer(&self) -> CommandBuffer {
        self.queue.new_command_buffer().to_owned()
    }

    pub fn new_mps_command_buffer(&self) -> Retained<MPSCommandBuffer> {
        MPSCommandBuffer::from_command_queue(&self.queue)
    }

    pub fn add_pending(
        &self,
        cb: CommandBuffer,
    ) {
        self.pending_cbs.borrow_mut().push(cb);
    }

    pub fn set_pending_mps(
        &self,
        mps_cb: &MPSCommandBuffer,
        needs_commit: bool,
    ) {
        *self.pending_mps_cb.borrow_mut() =
            Some(mps_cb.root_command_buffer().to_owned());
        *self.pending_mps_needs_commit.borrow_mut() = needs_commit;
    }

    pub fn commit(&self) {
        let mut pending = self.pending_cbs.borrow_mut();
        let mut committed = self.committed_cbs.borrow_mut();

        // Clear old committed CBs to avoid unbounded growth
        committed.clear();

        for cb in pending.drain(..) {
            cb.commit();
            committed.push(cb);
        }

        // Commit MPS CB if it wasn't already committed (pre-encode case)
        if let Some(mps_cb) = self.pending_mps_cb.borrow_mut().take() {
            if *self.pending_mps_needs_commit.borrow() {
                mps_cb.commit();
            }
            *self.committed_mps_cb.borrow_mut() = Some(mps_cb);
            *self.pending_mps_needs_commit.borrow_mut() = false;
        }
    }

    pub fn wait(&self) {
        // Wait for our CBs
        let committed = self.committed_cbs.borrow();
        if let Some(last) = committed.last() {
            last.wait_until_completed();
        }

        // Also wait for committed MPS CB
        if let Some(ref mps_cb) = *self.committed_mps_cb.borrow() {
            mps_cb.wait_until_completed();
        }
    }

    pub fn commit_and_wait(&self) {
        self.commit();
        self.wait();
    }

    pub fn clear(&self) {
        self.pending_cbs.borrow_mut().clear();
        self.committed_cbs.borrow_mut().clear();
        *self.committed_mps_cb.borrow_mut() = None;
        *self.pending_mps_cb.borrow_mut() = None;
        *self.pending_mps_needs_commit.borrow_mut() = false;
        // Note: stashed NOT cleared - survives for pre-encode
    }

    pub fn has_pending(&self) -> bool {
        !self.pending_cbs.borrow().is_empty()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_cbs.borrow().len()
    }

    /// Stash current pending CBs + MPS for later use (pre-encode)
    pub fn stash_pending(&self) {
        let mut pending = self.pending_cbs.borrow_mut();
        let mut stashed = self.stashed_cbs.borrow_mut();
        stashed.extend(pending.drain(..));

        // Also stash MPS CB and its commit flag
        *self.stashed_mps_cb.borrow_mut() = self.pending_mps_cb.borrow_mut().take();
        *self.stashed_mps_needs_commit.borrow_mut() = *self.pending_mps_needs_commit.borrow();
        *self.pending_mps_needs_commit.borrow_mut() = false;
    }

    /// Restore stashed CBs + MPS to pending (use pre-encoded)
    pub fn restore_stashed(&self) {
        let mut pending = self.pending_cbs.borrow_mut();
        let mut stashed = self.stashed_cbs.borrow_mut();
        pending.extend(stashed.drain(..));

        // Also restore MPS CB and its commit flag
        *self.pending_mps_cb.borrow_mut() = self.stashed_mps_cb.borrow_mut().take();
        *self.pending_mps_needs_commit.borrow_mut() = *self.stashed_mps_needs_commit.borrow();
        *self.stashed_mps_needs_commit.borrow_mut() = false;
    }

    /// Check if we have stashed pre-encoded work
    pub fn has_stashed(&self) -> bool {
        !self.stashed_cbs.borrow().is_empty()
    }

    /// Clear stashed CBs
    pub fn clear_stashed(&self) {
        self.stashed_cbs.borrow_mut().clear();
        *self.stashed_mps_cb.borrow_mut() = None;
        *self.stashed_mps_needs_commit.borrow_mut() = false;
    }
}
