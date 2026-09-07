use crate::backends::common::Backend;

pub enum InitialTaskAttachment<B: Backend> {
    None,
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    Downloading {
        active_task: B::ActiveTask,
        initial_downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
}
