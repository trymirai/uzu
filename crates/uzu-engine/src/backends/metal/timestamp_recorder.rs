use std::{
    ops::Range,
    time::{Duration, Instant},
};

use metal::{
    MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapExt, MTL4CounterHeapType,
    MTL4TimestampGranularity, MTLDevice, MTLDeviceExt,
};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{TimestampSource, TimestampSpan},
    metal::error::MetalError,
};

// "Maximum counter sample buffer length" is 32 KB, i.e. 4096 timestamp entries of 8 bytes (`MTL4TimestampHeapEntry`):
// https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
const COUNTER_HEAP_CAPACITY: usize = 4096;

// Enough tries for one to avoid preemption between the host reads around `sampleTimestamps`, cheap since it runs once.
const CLOCK_SAMPLE_ATTEMPTS: usize = 8;

pub struct MetalGpuClock {
    instant: Instant,
    nanoseconds: u64,
    frequency: u128,
}

impl MetalGpuClock {
    pub fn sample(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let (_, instant, nanoseconds) = (0..CLOCK_SAMPLE_ATTEMPTS)
            .map(|_| {
                let (mut cpu_nanoseconds, mut gpu_nanoseconds) = (0, 0);
                let before = Instant::now();
                device.sample_timestamps_gpu_timestamp(&mut cpu_nanoseconds, &mut gpu_nanoseconds);
                let bracket = before.elapsed();
                (bracket, before + bracket / 2, gpu_nanoseconds)
            })
            .min_by_key(|&(bracket, ..)| bracket)
            .expect("at least one clock sample");

        Self {
            instant,
            nanoseconds,
            frequency: device.query_timestamp_frequency() as u128,
        }
    }

    fn to_instant(
        &self,
        ticks: u64,
    ) -> Instant {
        let nanoseconds = (ticks as u128 * 1_000_000_000 / self.frequency) as u64;
        match nanoseconds.checked_sub(self.nanoseconds) {
            Some(after) => self.instant + Duration::from_nanos(after),
            None => self.instant - Duration::from_nanos(self.nanoseconds - nanoseconds),
        }
    }
}

#[derive(Default)]
pub struct MetalTimestampRecorder {
    spans: Vec<(String, Range<usize>)>,
    open_spans: Vec<usize>,
    slot_count: usize,
    counter_heaps: Vec<Retained<ProtocolObject<dyn MTL4CounterHeap>>>,
}

impl MetalTimestampRecorder {
    pub fn start(
        &mut self,
        label: &str,
        device: &ProtocolObject<dyn MTLDevice>,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) {
        let name = match self.open_spans.last() {
            Some(&parent) => format!("{}/{label}", self.spans[parent].0),
            None => label.to_owned(),
        };
        let slot = self.write(device, encoder);
        self.open_spans.push(self.spans.len());
        self.spans.push((name, slot..slot));
    }

    pub fn end(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) {
        let span = self.open_spans.pop().expect("timestamp ended without a start");
        self.spans[span].1.end = self.write(device, encoder);
    }

    fn write(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) -> usize {
        let slot = self.slot_count;
        if slot / COUNTER_HEAP_CAPACITY == self.counter_heaps.len() {
            let descriptor = MTL4CounterHeapDescriptor::new();
            descriptor.set_type(MTL4CounterHeapType::TIMESTAMP);
            descriptor.set_count(COUNTER_HEAP_CAPACITY);
            let counter_heap =
                device.new_counter_heap_with_descriptor(&descriptor).expect("Failed to create a counter heap");
            counter_heap.invalidate_counter_range(0..COUNTER_HEAP_CAPACITY);
            self.counter_heaps.push(counter_heap);
        }
        encoder.write_timestamp_with_granularity_into_heap_at_index(
            MTL4TimestampGranularity::PRECISE,
            &self.counter_heaps[slot / COUNTER_HEAP_CAPACITY],
            slot % COUNTER_HEAP_CAPACITY,
        );
        self.slot_count += 1;
        slot
    }

    pub fn resolve(
        &self,
        clock: &MetalGpuClock,
    ) -> Result<Box<[TimestampSpan]>, MetalError> {
        let count = self.slot_count;
        let mut ticks = Vec::with_capacity(count);
        for (index, counter_heap) in self.counter_heaps.iter().enumerate() {
            let slots = 0..(count - index * COUNTER_HEAP_CAPACITY).min(COUNTER_HEAP_CAPACITY);
            let bytes = counter_heap.resolve_counter_range(slots).ok_or(MetalError::CannotResolveCounterHeap)?;
            let (entries, _) = bytes.as_chunks::<{ size_of::<u64>() }>();
            ticks.extend(entries.iter().map(|entry| u64::from_ne_bytes(*entry)));
        }
        let instants = ticks
            .iter()
            .enumerate()
            .scan(None, |previous_timestamp, (slot, &slot_ticks)| {
                let timestamp = match slot_ticks {
                    0 => previous_timestamp.ok_or(MetalError::UnwrittenTimestamp(slot)),
                    slot_ticks => Ok(clock.to_instant(slot_ticks)),
                };
                *previous_timestamp = timestamp.as_ref().ok().copied();
                Some(timestamp)
            })
            .collect::<Result<Box<[Instant]>, MetalError>>()?;
        Ok(self
            .spans
            .iter()
            .map(|(name, slots)| TimestampSpan {
                name: name.clone(),
                source: TimestampSource::Gpu,
                start: instants[slots.start],
                end: instants[slots.end],
            })
            .collect())
    }
}
