const COLD_WORKING_SET_BYTES: usize = 256 << 20;

pub struct ColdPool<T, F: FnMut() -> T> {
    bytes_per_copy: usize,
    alloc: F,
    copies: Vec<T>,
    next: usize,
}

impl<T, F: FnMut() -> T> ColdPool<T, F> {
    pub fn new(
        bytes_per_copy: usize,
        alloc: F,
    ) -> Self {
        Self {
            bytes_per_copy,
            alloc,
            copies: Vec::new(),
            next: 0,
        }
    }

    pub fn next_mut(&mut self) -> &mut T {
        if self.copies.is_empty() {
            let count = copy_count(COLD_WORKING_SET_BYTES, self.bytes_per_copy);
            self.copies = (0..count).map(|_| (self.alloc)()).collect();
        }
        let index = self.next;
        self.next = (index + 1) % self.copies.len();
        &mut self.copies[index]
    }
}

pub(crate) fn copy_count(
    working_set: usize,
    bytes_per_copy: usize,
) -> usize {
    working_set.div_ceil(bytes_per_copy).max(1)
}
