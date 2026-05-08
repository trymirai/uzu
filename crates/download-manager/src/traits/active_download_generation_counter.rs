use crate::traits::ActiveDownloadGeneration;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ActiveDownloadGenerationCounter {
    next_value: u64,
}

impl ActiveDownloadGenerationCounter {
    pub fn allocate_next(&mut self) -> ActiveDownloadGeneration {
        let generation = ActiveDownloadGeneration::new(self.next_value);
        self.next_value = self.next_value.saturating_add(1);
        generation
    }
}
