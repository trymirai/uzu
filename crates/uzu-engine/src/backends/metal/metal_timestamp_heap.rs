use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use metal::{
    MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapExt, MTL4CounterHeapType,
    MTL4TimestampGranularity, MTLDevice, MTLDeviceExt,
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};

const CAPACITY: usize = 4096;

pub struct MetalTimestampHeap {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    frequency: u64,
    cells: Vec<Arc<OnceLock<Instant>>>,
}

impl MetalTimestampHeap {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.set_type(MTL4CounterHeapType::TIMESTAMP);
        descriptor.set_count(CAPACITY);

        Self {
            device: device.retain(),
            heap: device.new_counter_heap_with_descriptor(&descriptor).expect("Failed to create timestamp heap"),
            frequency: device.query_timestamp_frequency(),
            cells: Vec::new(),
        }
    }

    pub fn write(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        cell: Arc<OnceLock<Instant>>,
    ) {
        if self.cells.len() < CAPACITY {
            encoder.write_timestamp_with_granularity_into_heap_at_index(
                MTL4TimestampGranularity::RELAXED,
                &self.heap,
                self.cells.len(),
            );
            self.cells.push(cell);
        }
    }

    pub fn resolve(self) {
        let bytes = self.heap.resolve_counter_range(0..self.cells.len()).unwrap_or_default();
        let now = Instant::now();
        let (mut cpu_now, mut gpu_now) = (0, 0);
        self.device.sample_timestamps_gpu_timestamp(&mut cpu_now, &mut gpu_now);
        let (entries, _) = bytes.as_chunks::<{ size_of::<u64>() }>();
        for (cell, entry) in self.cells.iter().zip(entries) {
            let nanoseconds = (u64::from_ne_bytes(*entry) as u128 * 1_000_000_000 / self.frequency as u128) as u64;
            let elapsed = Duration::from_nanos(gpu_now.saturating_sub(nanoseconds));
            if let Some(instant) = now.checked_sub(elapsed) {
                let _ = cell.set(instant);
            }
        }
    }
}
