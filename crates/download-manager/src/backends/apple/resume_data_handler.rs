use std::ops::Deref;

use block2::{Block, RcBlock};
use objc2_foundation::NSData;

pub struct ResumeDataHandler(RcBlock<dyn Fn(*mut NSData)>);

impl ResumeDataHandler {
    pub fn new_bytes(handler: impl Fn(Box<[u8]>) + Send + Sync + 'static) -> Self {
        Self(RcBlock::new(move |data_pointer: *mut NSData| {
            let resume_data_bytes = if let Some(data) = unsafe { data_pointer.as_ref() } {
                data.to_vec().into_boxed_slice()
            } else {
                Box::new([])
            };
            handler(resume_data_bytes);
        }))
    }
}

// SAFETY: `NSURLSessionDownloadTask::cancelByProducingResumeData` requires a
// sendable completion block. The constructor enforces `Send + Sync + 'static`
// captures, and the block copies `NSData` into owned bytes before invoking the
// Rust closure.
unsafe impl Send for ResumeDataHandler {}
unsafe impl Sync for ResumeDataHandler {}

impl Deref for ResumeDataHandler {
    type Target = Block<dyn Fn(*mut NSData)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
