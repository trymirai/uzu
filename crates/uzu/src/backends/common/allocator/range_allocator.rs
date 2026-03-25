use std::ops::Range;

use rangemap::{RangeMap, RangeSet};

#[derive(PartialEq, Eq, Clone)]
pub enum AllocationType<P> {
    Global,
    Pooled {
        pool: P,
        can_alias_before: bool,
        can_alias_after: bool,
    },
}

#[derive(PartialEq, Eq, Clone)]
enum PooledAllocationRangeStatus {
    Allocated {
        can_alias_after: bool,
    },
    Aliasable,
    Frozen,
}

#[derive(PartialEq, Eq, Clone)]
enum AllocationRangeType<P: Eq + Clone> {
    Global,
    Pooled {
        pool: P,
        status: PooledAllocationRangeStatus,
    },
}

pub struct RangeAllocator<P: Eq + Clone> {
    full_range: Range<usize>,
    ranges: RangeMap<usize, AllocationRangeType<P>>,
}

impl<P: Eq + Clone> RangeAllocator<P> {
    pub fn new(full_range: Range<usize>) -> Self {
        Self {
            full_range,
            ranges: RangeMap::new(),
        }
    }

    pub fn allocate_range_aligned(
        &mut self,
        size: usize,
        alignment: usize,
        allocation_type: AllocationType<P>,
    ) -> Option<Range<usize>> {
        let mut available_range = RangeSet::new();
        available_range.extend(self.ranges.gaps(&self.full_range));

        if let AllocationType::Pooled {
            pool,
            can_alias_before: true,
            can_alias_after: _,
        } = &allocation_type
        {
            available_range.extend(self.ranges.iter().flat_map(|(range, range_type)| {
                if let AllocationRangeType::Pooled {
                    pool: range_pool,
                    status: PooledAllocationRangeStatus::Aliasable,
                } = range_type
                    && range_pool == pool
                {
                    Some(range.clone())
                } else {
                    None
                }
            }));
        }

        let Some(smallest_fitting_free_range) = available_range
            .iter()
            .filter(|range| range.end.saturating_sub(range.start.next_multiple_of(alignment)) >= size)
            .min_by_key(|range| range.len())
        else {
            return None;
        };

        let allocated_range_start = smallest_fitting_free_range.start.next_multiple_of(alignment);
        let allocated_range = allocated_range_start..(allocated_range_start + size);
        let allocated_range_status = match allocation_type {
            AllocationType::Global => AllocationRangeType::Global,
            AllocationType::Pooled {
                pool,
                can_alias_before: _,
                can_alias_after,
            } => AllocationRangeType::Pooled {
                pool,
                status: PooledAllocationRangeStatus::Allocated {
                    can_alias_after,
                },
            },
        };

        self.ranges.insert(allocated_range.clone(), allocated_range_status);

        Some(allocated_range)
    }

    pub fn free_range(
        &mut self,
        range: Range<usize>,
    ) {
        let overlapping_ranges = self.ranges.overlapping(range.clone()).collect::<Vec<_>>();
        assert!(overlapping_ranges.len() == 1, "expected only 1 overlapping range");

        let (overlapping_range, range_type) = overlapping_ranges[0];
        assert!(
            overlapping_range.start <= range.start && overlapping_range.end >= range.end,
            "overlapping range doesn't fully cover the range"
        );

        match range_type {
            AllocationRangeType::Global => self.ranges.remove(range),
            AllocationRangeType::Pooled {
                pool,
                status,
            } => {
                let status = match status {
                    PooledAllocationRangeStatus::Allocated {
                        can_alias_after: true,
                    } => PooledAllocationRangeStatus::Aliasable,
                    PooledAllocationRangeStatus::Allocated {
                        can_alias_after: false,
                    } => PooledAllocationRangeStatus::Frozen,
                    _ => panic!("attempted to free a range that isn't allocated"),
                };

                self.ranges.insert(
                    range,
                    AllocationRangeType::Pooled {
                        pool: pool.clone(),
                        status,
                    },
                )
            },
        };
    }

    pub fn free_pool(
        &mut self,
        pool: P,
    ) {
        let pool_ranges = self
            .ranges
            .iter()
            .filter_map(|(range, range_type)| {
                if let AllocationRangeType::Pooled {
                    pool: range_pool,
                    status,
                } = range_type
                    && range_pool == &pool
                {
                    assert!(
                        !matches!(status, PooledAllocationRangeStatus::Allocated { .. }),
                        "attempted to free a pool that has live allocations"
                    );
                    Some(range.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for pool_range in pool_ranges {
            self.ranges.remove(pool_range);
        }
    }

    pub fn total_available(&self) -> usize {
        self.ranges
            .gaps(&self.full_range)
            .map(|gap| gap.len())
            .chain(self.ranges.iter().filter_map(|(range, range_type)| {
                if matches!(
                    range_type,
                    AllocationRangeType::Pooled {
                        pool: _,
                        status: PooledAllocationRangeStatus::Aliasable
                    }
                ) {
                    Some(range.len())
                } else {
                    None
                }
            }))
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}
