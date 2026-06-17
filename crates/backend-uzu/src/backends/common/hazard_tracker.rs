use std::{ffi::c_void, ops::Range, ptr::NonNull};

use rangemap::RangeSet;

use crate::backends::common::AccessFlags;

/// Opaque, backend-specific handle to the resource an access touches.
///
/// Metal stores a pointer to the underlying `MTLResource` so a barrier can be scoped to the
/// exact buffers involved (`memoryBarrierWithResources:`) instead of all buffer memory. The CPU
/// backend (and anything that does not override `Buffer::resource_handle`) leaves this `None`.
pub type ResourceHandle = Option<NonNull<c_void>>;

#[derive(Debug, Clone)]
pub struct Access {
    pub range: Range<usize>,
    pub flags: AccessFlags,
    pub resource: ResourceHandle,
}

/// A required barrier: which prior access classes to wait on (`after`), which upcoming classes
/// it precedes (`before`), and the concrete resources that must be synchronized (`resources`,
/// possibly empty when handles are unavailable — callers should fall back to a global barrier).
#[derive(Debug, Clone)]
pub struct Barrier {
    pub after: AccessFlags,
    pub before: AccessFlags,
    pub resources: Vec<NonNull<c_void>>,
}

pub struct HazardTracker {
    compute_reads: RangeSet<usize>,
    compute_writes: RangeSet<usize>,
    copy_reads: RangeSet<usize>,
    copy_writes: RangeSet<usize>,
    // Resources tracked alongside each range set, drained in lockstep with the matching
    // `clear()` so an emitted barrier names exactly the resources it synchronizes.
    compute_read_resources: Vec<NonNull<c_void>>,
    compute_write_resources: Vec<NonNull<c_void>>,
    copy_read_resources: Vec<NonNull<c_void>>,
    copy_write_resources: Vec<NonNull<c_void>>,
}

fn drain_into(
    dst: &mut Vec<NonNull<c_void>>,
    src: &mut Vec<NonNull<c_void>>,
) {
    for resource in src.drain(..) {
        if !dst.contains(&resource) {
            dst.push(resource);
        }
    }
}

impl HazardTracker {
    pub fn new() -> Self {
        Self {
            compute_reads: RangeSet::new(),
            compute_writes: RangeSet::new(),
            copy_reads: RangeSet::new(),
            copy_writes: RangeSet::new(),
            compute_read_resources: Vec::new(),
            compute_write_resources: Vec::new(),
            copy_read_resources: Vec::new(),
            copy_write_resources: Vec::new(),
        }
    }

    #[must_use]
    pub fn access(
        &mut self,
        accesses: &[Access],
    ) -> Option<Barrier> {
        let mut after = AccessFlags::empty();
        let mut before = AccessFlags::empty();
        let mut resources: Vec<NonNull<c_void>> = Vec::new();

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
                    drain_into(&mut resources, &mut self.compute_write_resources);
                }
                if self.copy_writes.overlaps(&access.range) {
                    after.copy_write = true;
                    self.copy_writes.clear();
                    drain_into(&mut resources, &mut self.copy_write_resources);
                }
            }
            // Write accesses also must wait for previous reads to complete (WAR)
            if access.flags.compute_write || access.flags.copy_write {
                if self.compute_reads.overlaps(&access.range) {
                    after.compute_read = true;
                    self.compute_reads.clear();
                    drain_into(&mut resources, &mut self.compute_read_resources);
                }
                if self.copy_reads.overlaps(&access.range) {
                    after.copy_read = true;
                    self.copy_reads.clear();
                    drain_into(&mut resources, &mut self.copy_read_resources);
                }
            }
        }

        // Track new ones (after the barrier, aka barrier before this)
        for access in accesses {
            if access.flags.compute_read {
                before.compute_read = true;
                self.compute_reads.insert(access.range.clone());
                if let Some(resource) = access.resource {
                    self.compute_read_resources.push(resource);
                }
            }
            if access.flags.compute_write {
                before.compute_write = true;
                self.compute_writes.insert(access.range.clone());
                if let Some(resource) = access.resource {
                    self.compute_write_resources.push(resource);
                }
            }
            if access.flags.copy_read {
                before.copy_read = true;
                self.copy_reads.insert(access.range.clone());
                if let Some(resource) = access.resource {
                    self.copy_read_resources.push(resource);
                }
            }
            if access.flags.copy_write {
                before.copy_write = true;
                self.copy_writes.insert(access.range.clone());
                if let Some(resource) = access.resource {
                    self.copy_write_resources.push(resource);
                }
            }
        }

        if after.compute_read || after.compute_write || after.copy_read || after.copy_write {
            Some(Barrier {
                after,
                before,
                resources,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
#[path = "../../../unit/backends/common/hazard_tracker_test.rs"]
mod tests;
