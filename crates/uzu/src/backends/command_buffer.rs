pub trait CommandBuffer {
    fn commit_and_continue(&mut self);
    fn wait_until_completed(&self);
}
