use std::collections::BTreeMap;

pub(super) const PAGE_SIZE: usize = 16384;

pub(super) fn align_size(size: usize) -> usize {
    if size > PAGE_SIZE {
        (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    } else {
        size
    }
}

struct BufferNode<B> {
    buffer: B,
    size: usize,
    bucket_idx: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

pub(super) struct ScratchPool<B> {
    nodes: Vec<Option<BufferNode<B>>>,
    free_slots: Vec<usize>,
    size_index: BTreeMap<usize, Vec<usize>>,
    lru_head: Option<usize>,
    lru_tail: Option<usize>,
    pool_size: usize,
    pub(super) total_allocated: usize,
}

impl<B> ScratchPool<B> {
    pub(super) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_slots: Vec::new(),
            size_index: BTreeMap::new(),
            lru_head: None,
            lru_tail: None,
            pool_size: 0,
            total_allocated: 0,
        }
    }

    #[inline]
    fn allocate_slot(&mut self) -> usize {
        self.free_slots.pop().unwrap_or_else(|| {
            let slot = self.nodes.len();
            self.nodes.push(None);
            slot
        })
    }

    #[inline]
    fn add_to_lru_head(
        &mut self,
        slot: usize,
    ) {
        if let Some(old_head) = self.lru_head {
            if let Some(node) = self.nodes[old_head].as_mut() {
                node.prev = Some(slot);
            }
            if let Some(node) = self.nodes[slot].as_mut() {
                node.next = Some(old_head);
            }
        }

        self.lru_head = Some(slot);

        if self.lru_tail.is_none() {
            self.lru_tail = Some(slot);
        }
    }

    #[inline]
    fn remove_from_lru(
        &mut self,
        slot: usize,
    ) {
        let (prev, next) = match self.nodes[slot].as_ref() {
            Some(node) => (node.prev, node.next),
            None => return,
        };

        match (prev, next) {
            (Some(p), Some(n)) => {
                if let Some(prev_node) = self.nodes[p].as_mut() {
                    prev_node.next = Some(n);
                }
                if let Some(next_node) = self.nodes[n].as_mut() {
                    next_node.prev = Some(p);
                }
            },
            (Some(p), None) => {
                if let Some(prev_node) = self.nodes[p].as_mut() {
                    prev_node.next = None;
                }
                self.lru_tail = Some(p);
            },
            (None, Some(n)) => {
                if let Some(next_node) = self.nodes[n].as_mut() {
                    next_node.prev = None;
                }
                self.lru_head = Some(n);
            },
            (None, None) => {
                self.lru_head = None;
                self.lru_tail = None;
            },
        }

        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = None;
        }
    }

    #[inline]
    fn remove_slot_from_bucket(
        &mut self,
        size: usize,
        _slot: usize,
        bucket_idx: usize,
    ) {
        if let Some(slots) = self.size_index.get_mut(&size) {
            let last_idx = slots.len() - 1;
            if bucket_idx != last_idx {
                let swapped_slot = slots[last_idx];
                slots[bucket_idx] = swapped_slot;
                if let Some(swapped_node) = self.nodes[swapped_slot].as_mut() {
                    swapped_node.bucket_idx = bucket_idx;
                }
            }
            slots.pop();
            if slots.is_empty() {
                self.size_index.remove(&size);
            }
        }
    }

    #[inline]
    pub(super) fn find_buffer(
        &mut self,
        size: usize,
    ) -> Option<(B, usize)> {
        let max_size = (2 * size).min(size + 2 * PAGE_SIZE);

        let mut found_key = None;
        let mut found_slot = None;
        let mut found_bucket_idx = None;

        for (&key, slots) in self.size_index.range(size..=max_size) {
            if let Some(&slot) = slots.last() {
                found_key = Some(key);
                found_slot = Some(slot);
                found_bucket_idx = Some(slots.len() - 1);
                break;
            }
        }

        let (key, slot, bucket_idx) = match (found_key, found_slot, found_bucket_idx) {
            (Some(k), Some(s), Some(i)) => (k, s, i),
            _ => return None,
        };

        self.remove_slot_from_bucket(key, slot, bucket_idx);
        self.remove_from_lru(slot);

        let node = self.nodes[slot].take()?;
        self.free_slots.push(slot);
        self.pool_size = self.pool_size.saturating_sub(key);

        Some((node.buffer, key))
    }

    #[inline]
    pub(super) fn return_buffer(
        &mut self,
        buffer: B,
        size: usize,
    ) {
        let slot = self.allocate_slot();
        let bucket_idx = self.size_index.get(&size).map_or(0, |v| v.len());

        self.nodes[slot] = Some(BufferNode {
            buffer,
            size,
            bucket_idx,
            prev: None,
            next: None,
        });

        self.add_to_lru_head(slot);
        self.size_index.entry(size).or_default().push(slot);
        self.pool_size += size;
    }

    pub(super) fn release_cached_buffers(
        &mut self,
        min_bytes_to_free: usize,
    ) -> usize {
        if min_bytes_to_free == 0 {
            return 0;
        }

        if min_bytes_to_free >= (self.pool_size * 9) / 10 {
            return self.clear();
        }

        let mut total_freed = 0usize;
        let mut buffers_released = 0usize;

        while total_freed < min_bytes_to_free {
            let tail_slot = match self.lru_tail {
                Some(s) => s,
                None => break,
            };

            let (size, bucket_idx) = match self.nodes[tail_slot].as_ref() {
                Some(n) => (n.size, n.bucket_idx),
                None => break,
            };

            self.remove_slot_from_bucket(size, tail_slot, bucket_idx);
            self.remove_from_lru(tail_slot);
            self.nodes[tail_slot] = None;
            self.free_slots.push(tail_slot);

            total_freed += size;
            buffers_released += 1;
            self.pool_size = self.pool_size.saturating_sub(size);
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }

        buffers_released
    }

    #[inline]
    pub(super) fn available_size(&self) -> usize {
        self.pool_size
    }

    pub(super) fn clear(&mut self) -> usize {
        let count = self.nodes.iter().filter(|n| n.is_some()).count();

        self.nodes.clear();
        self.free_slots.clear();
        self.size_index.clear();
        self.lru_head = None;
        self.lru_tail = None;
        self.total_allocated = self.total_allocated.saturating_sub(self.pool_size);
        self.pool_size = 0;

        count
    }
}
