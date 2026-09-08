#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DownloadGeneration(u64);

impl DownloadGeneration {
    pub fn advance(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}
