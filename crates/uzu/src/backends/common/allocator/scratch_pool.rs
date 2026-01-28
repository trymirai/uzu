use std::collections::BTreeMap;

pub(super) const PAGE_SIZE: usize = 16384;
const SMALL_BUFFER_THRESHOLD: usize = 64 * 1024;
const SIZE_TOLERANCE: f64 = 0.1;

pub(super) fn next_power_of_two(size: usize) -> usize {
    if size <= 4096 {
        return 4096;
    }
    1usize << (usize::BITS - (size - 1).leading_zeros())
}

struct CachedBuffer<B> {
    buffer: B,
    last_used_tick: u64,
}

pub(super) struct ScratchPool<B> {
    small_buckets: BTreeMap<usize, Vec<CachedBuffer<B>>>,
    large_exact: BTreeMap<usize, Vec<CachedBuffer<B>>>,
    pub(super) total_allocated: usize,
    current_tick: u64,
}

impl<B> ScratchPool<B> {
    pub(super) fn new() -> Self {
        Self {
            small_buckets: BTreeMap::new(),
            large_exact: BTreeMap::new(),
            total_allocated: 0,
            current_tick: 0,
        }
    }

    pub(super) fn is_small(size: usize) -> bool {
        size < SMALL_BUFFER_THRESHOLD
    }

    pub(super) fn tick(&mut self) -> u64 {
        self.current_tick += 1;
        self.current_tick
    }

    pub(super) fn find_small_buffer(
        &mut self,
        size: usize,
    ) -> Option<B> {
        let bucket_size = next_power_of_two(size);

        let buffers = self.small_buckets.get_mut(&bucket_size)?;
        let cached = buffers.pop()?;

        if buffers.is_empty() {
            self.small_buckets.remove(&bucket_size);
        }

        Some(cached.buffer)
    }

    pub(super) fn find_large_buffer(
        &mut self,
        size: usize,
    ) -> Option<(B, usize)> {
        let max_size = ((size as f64) * (1.0 + SIZE_TOLERANCE)).ceil() as usize;

        let mut found_key = None;
        for (&key, buffers) in self.large_exact.range(size..=max_size) {
            if !buffers.is_empty() {
                found_key = Some(key);
                break;
            }
        }

        let key = found_key?;
        let buffers = self.large_exact.get_mut(&key)?;
        let cached = buffers.pop()?;

        if buffers.is_empty() {
            self.large_exact.remove(&key);
        }

        Some((cached.buffer, key))
    }

    pub(super) fn return_small_buffer(
        &mut self,
        buffer: B,
        bucket_size: usize,
    ) {
        let tick = self.current_tick;
        self.small_buckets.entry(bucket_size).or_default().push(CachedBuffer {
            buffer,
            last_used_tick: tick,
        });
    }

    pub(super) fn return_large_buffer(
        &mut self,
        buffer: B,
        size: usize,
    ) {
        let tick = self.current_tick;
        self.large_exact.entry(size).or_default().push(CachedBuffer {
            buffer,
            last_used_tick: tick,
        });
    }

    pub(super) fn evict_stale(
        &mut self,
        max_age: u64,
    ) -> usize {
        let current = self.current_tick;
        let threshold = current.saturating_sub(max_age);

        let mut evicted = 0;

        let keys_to_check: Vec<_> = self.small_buckets.keys().copied().collect();
        for key in keys_to_check {
            if let Some(buffers) = self.small_buckets.get_mut(&key) {
                let before = buffers.len();
                buffers.retain(|b| b.last_used_tick >= threshold);
                evicted += before - buffers.len();
                if buffers.is_empty() {
                    self.small_buckets.remove(&key);
                }
            }
        }

        let keys_to_check: Vec<_> = self.large_exact.keys().copied().collect();
        for key in keys_to_check {
            if let Some(buffers) = self.large_exact.get_mut(&key) {
                let before = buffers.len();
                buffers.retain(|b| b.last_used_tick >= threshold);
                evicted += before - buffers.len();
                if buffers.is_empty() {
                    self.large_exact.remove(&key);
                }
            }
        }

        evicted
    }

    pub(super) fn available_size(&self) -> usize {
        let small: usize = self
            .small_buckets
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum();

        let large: usize = self
            .large_exact
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum();

        small + large
    }

    pub(super) fn clear(&mut self) -> usize {
        let small_count: usize = self.small_buckets.values().map(|v| v.len()).sum();
        let large_count: usize = self.large_exact.values().map(|v| v.len()).sum();
        let freed = self.available_size();

        self.small_buckets.clear();
        self.large_exact.clear();
        self.total_allocated -= freed;

        small_count + large_count
    }
}
