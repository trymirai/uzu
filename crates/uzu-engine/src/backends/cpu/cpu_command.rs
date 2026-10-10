pub enum CpuCommand {
    Run(Box<dyn FnOnce() + Send>),
    Timestamp,
}
