use std::time::{Duration, Instant};

use metal::{
    MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapExt, MTL4CounterHeapType,
    MTL4TimestampGranularity, MTLDevice, MTLDeviceExt,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{BlockName, CommandBufferTimestamps, TimestampSlot, TimestampSpanRecorder},
    metal::error::MetalError,
};

// "Maximum counter sample buffer length" is 32 KB, i.e. 4096 timestamp entries of 8 bytes (`MTL4TimestampHeapEntry`):
// https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
const COUNTER_HEAP_CAPACITY: usize = 4096;

pub struct MetalTimestampRecorder {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    spans: TimestampSpanRecorder,
    counter_heaps: Vec<Retained<ProtocolObject<dyn MTL4CounterHeap>>>,
}

impl MetalTimestampRecorder {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            spans: TimestampSpanRecorder::default(),
            counter_heaps: Vec::new(),
        }
    }

    pub fn start(
        &mut self,
        name: BlockName,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) {
        let slot = self.spans.start(name);
        self.write(slot, encoder);
    }

    pub fn end(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) {
        let slot = self.spans.end();
        self.write(slot, encoder);
    }

    fn write(
        &mut self,
        slot: TimestampSlot,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) {
        if slot / COUNTER_HEAP_CAPACITY == self.counter_heaps.len() {
            let descriptor = MTL4CounterHeapDescriptor::new();
            descriptor.set_type(MTL4CounterHeapType::TIMESTAMP);
            descriptor.set_count(COUNTER_HEAP_CAPACITY);
            let counter_heap =
                self.device.new_counter_heap_with_descriptor(&descriptor).expect("Failed to create a counter heap");
            counter_heap.invalidate_counter_range(0..COUNTER_HEAP_CAPACITY);
            self.counter_heaps.push(counter_heap);
        }
        encoder.write_timestamp_with_granularity_into_heap_at_index(
            MTL4TimestampGranularity::PRECISE,
            &self.counter_heaps[slot / COUNTER_HEAP_CAPACITY],
            slot % COUNTER_HEAP_CAPACITY,
        );
    }

    pub fn resolve(&self) -> Result<CommandBufferTimestamps, MetalError> {
        let count = self.spans.slot_count();
        let mut ticks = Vec::with_capacity(count);
        for (index, counter_heap) in self.counter_heaps.iter().enumerate() {
            let slots = 0..(count - index * COUNTER_HEAP_CAPACITY).min(COUNTER_HEAP_CAPACITY);
            let bytes = counter_heap.resolve_counter_range(slots).ok_or(MetalError::CannotResolveCounterHeap)?;
            let (entries, _) = bytes.as_chunks::<{ size_of::<u64>() }>();
            ticks.extend(entries.iter().map(|entry| u64::from_ne_bytes(*entry)));
        }
        let frequency = self.device.query_timestamp_frequency() as u128;
        let (mut cpu_now, mut gpu_now) = (0, 0);
        let now = Instant::now();
        self.device.sample_timestamps_gpu_timestamp(&mut cpu_now, &mut gpu_now);
        let to_instant = |ticks: u64| {
            let nanoseconds = (ticks as u128 * 1_000_000_000 / frequency) as u64;
            now - Duration::from_nanos(gpu_now - nanoseconds)
        };
        let instants = ticks
            .iter()
            .enumerate()
            .scan(None, |previous_timestamp, (slot, &slot_ticks)| {
                let timestamp = match slot_ticks {
                    0 => previous_timestamp.ok_or(MetalError::UnwrittenTimestamp(slot)),
                    slot_ticks => Ok(to_instant(slot_ticks)),
                };
                *previous_timestamp = timestamp.as_ref().ok().copied();
                Some(timestamp)
            })
            .collect::<Result<Box<[Instant]>, MetalError>>()?;
        Ok(self.spans.spans(&instants))
    }
}
