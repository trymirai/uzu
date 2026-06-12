use std::{
    collections::{HashMap, hash_map::Entry},
    fs::File,
    io::BufReader,
    rc::Rc,
    time::Instant,
};

use serde::{Deserialize, Serialize};

use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context},
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

#[test]
#[ignore]
fn bench_allocator_generation_trace() {
    let file = BufReader::with_capacity(1 << 20, File::open("/tmp/allocator_trace.jsonl").unwrap());
    let events: Vec<TraceEvent> = serde_json::Deserializer::from_reader(file)
        .into_iter::<TraceEvent>()
        .map(|r| r.expect("malformed trace event"))
        .collect();

    let context = <<Metal as Backend>::Context as Context>::new().unwrap();

    let mut allocators: HashMap<usize, Rc<Allocator<Metal>>> = HashMap::new();
    let mut pools: HashMap<(usize, usize), AllocationPool<Metal>> = HashMap::new();
    let mut allocations: HashMap<(usize, usize), Allocation<Metal>> = HashMap::new();

    let start = Instant::now();

    for event in events {
        match event.event_type {
            TraceEventType::Create => {
                let Entry::Vacant(vacant) = allocators.entry(event.allocator_id) else {
                    panic!("Duplicate allocator id: {}", event.allocator_id);
                };

                vacant.insert(Allocator::new(Rc::downgrade(&context)));
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
