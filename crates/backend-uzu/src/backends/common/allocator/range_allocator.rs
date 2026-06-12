use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum AllocationType {
    Global,
    Pooled {
        pool: usize,
        can_alias_before: bool,
        can_alias_after: bool,
    },
}

#[derive(Default)]
struct AvailableRanges {
    by_start: BTreeMap<usize, usize>,
    by_len: BTreeSet<(usize, usize, usize)>,
    total_len: usize,
}

#[derive(Clone, Copy)]
struct AvailableRangeFit {
    available_start: usize,
    available_end: usize,
    allocated_start: usize,
    allocated_end: usize,
}

impl AvailableRangeFit {
    fn available_key(self) -> (usize, usize) {
        (self.available_end - self.available_start, self.available_start)
    }

    fn allocated_range(self) -> Range<usize> {
        self.allocated_start..self.allocated_end
    }

    fn allocated_len(self) -> usize {
        self.allocated_end - self.allocated_start
    }
}

impl AvailableRanges {
    fn new(range: Range<usize>) -> Self {
        let mut available_ranges = Self::default();
        available_ranges.insert(range);
        available_ranges
    }

    fn total_len(&self) -> usize {
        self.total_len
    }

    fn insert(
        &mut self,
        mut range: Range<usize>,
    ) {
        if range.is_empty() {
            return;
        }

        if let Some((&start, &end)) = self.by_start.range(..=range.start).next_back()
            && end >= range.start
        {
            self.remove_entry(start, end);
            range.start = usize::min(range.start, start);
            range.end = usize::max(range.end, end);
        }

        while let Some((&start, &end)) = self.by_start.range(range.start..).next() {
            if start > range.end {
                break;
            }

            self.remove_entry(start, end);
            range.end = usize::max(range.end, end);
        }

        self.insert_entry(range.start, range.end);
    }

    fn remove_fit(
        &mut self,
        fit: AvailableRangeFit,
    ) -> Range<usize> {
        self.remove_entry(fit.available_start, fit.available_end);

        if fit.available_start < fit.allocated_start {
            self.insert_entry(fit.available_start, fit.allocated_start);
        }

        if fit.allocated_end < fit.available_end {
            self.insert_entry(fit.allocated_end, fit.available_end);
        }

        fit.allocated_range()
    }

    fn best_fit(
        &self,
        size: usize,
        alignment: usize,
    ) -> Option<AvailableRangeFit> {
        for &(range_len, range_start, range_end) in self.by_len.range((size, 0, 0)..) {
            let allocated_start = range_start.next_multiple_of(alignment);

            if range_end.saturating_sub(allocated_start) >= size {
                debug_assert_eq!(range_end - range_start, range_len);

                return Some(AvailableRangeFit {
                    available_start: range_start,
                    available_end: range_end,
                    allocated_start,
                    allocated_end: allocated_start + size,
                });
            }
        }

        None
    }

    fn into_ranges(self) -> impl Iterator<Item = Range<usize>> {
        self.by_start.into_iter().map(|(start, end)| start..end)
    }

    fn insert_entry(
        &mut self,
        start: usize,
        end: usize,
    ) {
        debug_assert!(start < end);

        let len = end - start;
        let old_end = self.by_start.insert(start, end);
        debug_assert!(old_end.is_none());

        self.by_len.insert((len, start, end));
        self.total_len += len;
    }

    fn remove_entry(
        &mut self,
        start: usize,
        end: usize,
    ) {
        let len = end - start;

        assert_eq!(self.by_start.remove(&start), Some(end));
        assert!(self.by_len.remove(&(len, start, end)));
        self.total_len -= len;
    }
}

pub struct RangeAllocator {
    full_len: usize,
    free_ranges: AvailableRanges,
    aliasable_ranges_by_pool: Vec<AvailableRanges>,
    pool_ranges_by_pool: Vec<AvailableRanges>,
    pool_live_allocations: Vec<usize>,
    total_available: usize,
}

impl RangeAllocator {
    pub fn new(full_range: Range<usize>) -> Self {
        let full_len = full_range.len();

        Self {
            full_len,
            free_ranges: AvailableRanges::new(full_range),
            aliasable_ranges_by_pool: Vec::new(),
            pool_ranges_by_pool: Vec::new(),
            pool_live_allocations: Vec::new(),
            total_available: full_len,
        }
    }

    pub fn allocate_range_aligned(
        &mut self,
        size: usize,
        alignment: usize,
        allocation_type: AllocationType,
    ) -> Option<Range<usize>> {
        let aliasable_pool = match allocation_type {
            AllocationType::Pooled {
                pool,
                can_alias_before: true,
                can_alias_after: _,
            } => Some(pool),
            _ => None,
        };

        let free_candidate = self.free_ranges.best_fit(size, alignment);
        let aliasable_candidate = aliasable_pool.and_then(|pool| {
            self.aliasable_ranges_by_pool.get(pool).and_then(|ranges| ranges.best_fit(size, alignment))
        });

        let (selected_aliasable_pool, fit) = match (free_candidate, aliasable_candidate) {
            (None, None) => return None,
            (Some(fit), None) => (None, fit),
            (None, Some(fit)) => (aliasable_pool, fit),
            (Some(free_fit), Some(aliasable_fit)) if aliasable_fit.available_key() < free_fit.available_key() => {
                (aliasable_pool, aliasable_fit)
            },
            (Some(fit), Some(_)) => (None, fit),
        };

        let allocated_range = if let Some(pool) = selected_aliasable_pool {
            self.aliasable_ranges_by_pool[pool].remove_fit(fit)
        } else {
            self.free_ranges.remove_fit(fit)
        };
        self.total_available -= fit.allocated_len();

        match allocation_type {
            AllocationType::Global => {},
            AllocationType::Pooled {
                pool,
                can_alias_before: _,
                can_alias_after: _,
            } => {
                self.ensure_pool(pool);
                self.pool_live_allocations[pool] += 1;

                if selected_aliasable_pool.is_none() {
                    self.pool_ranges_by_pool[pool].insert(allocated_range.clone());
                }
            },
        };

        Some(allocated_range)
    }

    pub fn free_range(
        &mut self,
        range: Range<usize>,
        allocation_type: AllocationType,
    ) {
        match allocation_type {
            AllocationType::Global => {
                self.free_ranges.insert(range.clone());
                self.total_available += range.len();
            },
            AllocationType::Pooled {
                pool,
                can_alias_before: _,
                can_alias_after,
            } => {
                self.pool_live_allocations[pool] -= 1;

                if can_alias_after {
                    self.aliasable_ranges_by_pool[pool].insert(range.clone());
                    self.total_available += range.len();
                }
            },
        };
    }

    pub fn free_pool(
        &mut self,
        pool: usize,
    ) {
        if let Some(live_allocations) = self.pool_live_allocations.get(pool) {
            assert_eq!(*live_allocations, 0, "attempted to free a pool that has live allocations");
        }

        if let Some(aliasable_ranges) = self.aliasable_ranges_by_pool.get_mut(pool) {
            self.total_available -= aliasable_ranges.total_len();
            *aliasable_ranges = AvailableRanges::default();
        }

        let Some(pool_ranges) = self.pool_ranges_by_pool.get_mut(pool) else {
            return;
        };
        let pool_ranges = std::mem::take(pool_ranges);

        for pool_range in pool_ranges.into_ranges() {
            self.free_ranges.insert(pool_range.clone());
            self.total_available += pool_range.len();
        }
    }

    pub fn total_available(&self) -> usize {
        self.total_available
    }

    pub fn is_empty(&self) -> bool {
        self.free_ranges.total_len() == self.full_len
    }

    fn ensure_pool(
        &mut self,
        pool: usize,
    ) {
        let pool_len = pool + 1;

        self.aliasable_ranges_by_pool.resize_with(pool_len, AvailableRanges::default);
        self.pool_ranges_by_pool.resize_with(pool_len, AvailableRanges::default);
        self.pool_live_allocations.resize(pool_len, 0);
    }
}
