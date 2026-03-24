use std::ops::Range;

use rangemap::RangeSet;

use crate::backends::common::AccessFlags;

#[derive(Debug, Clone)]
pub struct Access {
    pub range: Range<usize>,
    pub flags: AccessFlags,
}

pub struct HazardTracker {
    compute_reads: RangeSet<usize>,
    compute_writes: RangeSet<usize>,
    copy_reads: RangeSet<usize>,
    copy_writes: RangeSet<usize>,
}

impl HazardTracker {
    pub fn new() -> Self {
        Self {
            compute_reads: RangeSet::new(),
            compute_writes: RangeSet::new(),
            copy_reads: RangeSet::new(),
            copy_writes: RangeSet::new(),
        }
    }

    #[must_use]
    pub fn access(
        &mut self,
        accesses: &[Access],
    ) -> Option<(AccessFlags, AccessFlags)> {
        let mut after = AccessFlags::empty();
        let mut before = AccessFlags::empty();

        // Get existing hazards (before the barrier, aka barrier after this)
        for access in accesses {
            // Any access should wait for earlier overlapping writes to complete (RAW/WAW)
            if access.flags.compute_read
                || access.flags.compute_write
                || access.flags.copy_read
                || access.flags.copy_write
            {
                if self.compute_writes.overlaps(&access.range) {
                    after.compute_write = true;
                    self.compute_writes.clear();
                }
                if self.copy_writes.overlaps(&access.range) {
                    after.copy_write = true;
                    self.copy_writes.clear();
                }
            }
            // Write accesses also must wait for previous reads to complete (WAR)
            if access.flags.compute_write || access.flags.copy_write {
                if self.compute_reads.overlaps(&access.range) {
                    after.compute_read = true;
                    self.compute_reads.clear();
                }
                if self.copy_reads.overlaps(&access.range) {
                    after.copy_read = true;
                    self.copy_reads.clear();
                }
            }
        }

        // Track new ones (after the barrier, aka barrier before this)
        for access in accesses {
            if access.flags.compute_read {
                before.compute_read = true;
                self.compute_reads.insert(access.range.clone());
            }
            if access.flags.compute_write {
                before.compute_write = true;
                self.compute_writes.insert(access.range.clone());
            }
            if access.flags.copy_read {
                before.copy_read = true;
                self.copy_reads.insert(access.range.clone());
            }
            if access.flags.copy_write {
                before.copy_write = true;
                self.copy_writes.insert(access.range.clone());
            }
        }

        if after.compute_read || after.compute_write || after.copy_read || after.copy_write {
            Some((after, before))
        } else {
            None
        }
    }
}

#[cfg(test)]
#[path = "../../../tests_unit/hazard_tracker_test.rs"]
mod hazard_tracker_test;
