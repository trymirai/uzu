use std::{
    collections::{HashMap, hash_map::Entry},
    fs::File,
    io::BufReader,
    sync::Arc,
    time::Instant,
};

use proc_macros::uzu_test;
use serde::{Deserialize, Serialize};

use crate::backends::{
    common::{
        Allocation, AllocationPool, AllocationType, Allocator, AsBufferRangeMut, AsBufferRangeRef, Backend,
        BufferArgMut, Context,
    },
    metal::Metal,
};

#[derive(Serialize, Deserialize)]
struct TraceEvent {
    allocator_id: usize,
    event_type: TraceEventType,
}

#[derive(Serialize, Deserialize)]
pub enum TraceEventAllocationType {
    Global,
    Pooled {
        pool_id: usize,
        cpu_available: bool,
    },
}

#[derive(Serialize, Deserialize)]
enum TraceEventType {
    Create,
    CreatePool {
        reusable: bool,
        pool_id: usize,
    },
    Allocate {
        size: usize,
        allocation_type: TraceEventAllocationType,
        allocation_id: usize,
    },
    Free {
        allocation_id: usize,
    },
    FreePool {
        pool_id: usize,
    },
}

#[uzu_test]
#[ignore]
fn bench_allocator_generation_trace() {
    let file = BufReader::with_capacity(1 << 20, File::open("/tmp/allocator_trace.jsonl").unwrap());
    let events: Vec<TraceEvent> = serde_json::Deserializer::from_reader(file)
        .into_iter::<TraceEvent>()
        .map(|r| r.expect("malformed trace event"))
        .collect();

    let context = <<Metal as Backend>::Context as Context>::new().unwrap();

    let mut allocators: HashMap<usize, Arc<Allocator<Metal>>> = HashMap::new();
    let mut pools: HashMap<(usize, usize), AllocationPool<Metal>> = HashMap::new();
    let mut allocations: HashMap<(usize, usize), Allocation<Metal>> = HashMap::new();

    let start = Instant::now();

    for event in events {
        match event.event_type {
            TraceEventType::Create => {
                let Entry::Vacant(vacant) = allocators.entry(event.allocator_id) else {
                    panic!("Duplicate allocator id: {}", event.allocator_id);
                };

                vacant.insert(Allocator::new(Arc::downgrade(&context)));
            },
            TraceEventType::CreatePool {
                reusable,
                pool_id,
            } => {
                let Entry::Vacant(vacant) = pools.entry((event.allocator_id, pool_id)) else {
                    panic!("Duplicate pool id {} in allocator {}", pool_id, event.allocator_id);
                };

                vacant.insert(allocators[&event.allocator_id].create_pool(reusable));
            },
            TraceEventType::Allocate {
                size,
                allocation_type,
                allocation_id,
            } => {
                let Entry::Vacant(vacant) = allocations.entry((event.allocator_id, allocation_id)) else {
                    panic!("Duplicate allocation id {} in allocator {}", allocation_id, event.allocator_id);
                };

                let allocation_type = match allocation_type {
                    TraceEventAllocationType::Global => AllocationType::Global,
                    TraceEventAllocationType::Pooled {
                        pool_id,
                        cpu_available,
                    } => AllocationType::Pooled {
                        pool: &pools[&(event.allocator_id, pool_id)],
                        cpu_available,
                    },
                };

                vacant.insert(allocators[&event.allocator_id].allocate(size, allocation_type).unwrap());
            },
            TraceEventType::Free {
                allocation_id,
            } => {
                allocations.remove(&(event.allocator_id, allocation_id)).unwrap();
            },
            TraceEventType::FreePool {
                pool_id,
            } => {
                pools.remove(&(event.allocator_id, pool_id)).unwrap();
            },
        }
    }

    let elapsed = start.elapsed();

    eprintln!();
    eprintln!("--------------------");
    eprintln!("Cpu time: {} ms", elapsed.as_millis());
    eprintln!("--------------------");
}

#[uzu_test]
#[ignore]
fn allocation_split_at_mut_returns_disjoint_buffer_arguments() {
    let context = <<Metal as Backend>::Context as Context>::new().unwrap();
    let mut allocation = context.create_allocation(64, AllocationType::Global).unwrap();
    let allocation_range = allocation.as_buffer_range_ref().range();

    let (head, tail) = allocation.as_buffer_range_mut().split_at(24);
    let (_, head_offset, head_length) = BufferArgMut::<'_, Metal>::into_parts(head);
    let (_, tail_offset, tail_length) = BufferArgMut::<'_, Metal>::into_parts(tail);

    assert_eq!((head_offset, head_length), (allocation_range.start, 24));
    assert_eq!((tail_offset, tail_length), (allocation_range.start + 24, 40));
}
