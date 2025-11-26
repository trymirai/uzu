use metal::{CommandBuffer, CommandQueue};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use std::cell::RefCell;

pub struct ExecutionOrchestrator {
    queue: CommandQueue,
    pending_cbs: RefCell<Vec<CommandBuffer>>,
    committed_cbs: RefCell<Vec<CommandBuffer>>,
    mps_cb: RefCell<Option<CommandBuffer>>,
}

impl ExecutionOrchestrator {
    pub fn new(queue: CommandQueue) -> Self {
        Self {
            queue,
            pending_cbs: RefCell::new(Vec::new()),
            committed_cbs: RefCell::new(Vec::new()),
            mps_cb: RefCell::new(None),
        }
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn new_command_buffer(&self) -> CommandBuffer {
        self.queue.new_command_buffer().to_owned()
    }

    pub fn add_pending(&self, cb: CommandBuffer) {
        self.pending_cbs.borrow_mut().push(cb);
    }

    pub fn set_mps_command_buffer(&self, mps_cb: &MPSCommandBuffer) {
        *self.mps_cb.borrow_mut() = Some(mps_cb.root_command_buffer().to_owned());
    }

    pub fn commit(&self) {
        let mut pending = self.pending_cbs.borrow_mut();
        let mut committed = self.committed_cbs.borrow_mut();
        
        for cb in pending.drain(..) {
            cb.commit();
            committed.push(cb);
        }
    }

    pub fn wait(&self) {
        // Wait for our CBs
        let committed = self.committed_cbs.borrow();
        if let Some(last) = committed.last() {
            last.wait_until_completed();
        }
        
        // Also wait for MPS CB if set
        if let Some(ref mps_cb) = *self.mps_cb.borrow() {
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
        *self.mps_cb.borrow_mut() = None;
    }

    pub fn has_pending(&self) -> bool {
        !self.pending_cbs.borrow().is_empty()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_cbs.borrow().len()
    }
}

