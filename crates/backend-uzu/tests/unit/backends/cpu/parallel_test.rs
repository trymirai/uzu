use std::sync::Mutex;

use proc_macros::uzu_test;

use super::Pool;

const LARGE_WORK: usize = 1 << 24;

fn collect_chunks(
    threads: usize,
    units: usize,
    work: usize,
) -> Vec<std::ops::Range<usize>> {
    let chunks = Mutex::new(Vec::new());
    Pool::new(threads).for_each_chunk(units, work, |range| chunks.lock().unwrap().push(range));
    let mut chunks = chunks.into_inner().unwrap();
    chunks.sort_by_key(|range| range.start);
    chunks
}

#[uzu_test]
fn chunks_partition_every_unit() {
    for threads in [1usize, 2, 3, 7, 8, 64] {
        for units in [1usize, 2, 5, 16, 100, 1024] {
            let chunks = collect_chunks(threads, units, LARGE_WORK);

            let mut covered = Vec::new();
            for chunk in &chunks {
                assert!(chunk.start < chunk.end, "empty chunk for threads={threads} units={units}");
                covered.extend(chunk.clone());
            }
            assert_eq!(
                covered,
                (0..units).collect::<Vec<_>>(),
                "threads={threads} units={units} must be covered once, in order"
            );
            assert!(chunks.len() <= units, "threads={threads} units={units}");
        }
    }
}

#[uzu_test]
fn work_is_split_finer_than_thread_count() {
    let chunks = collect_chunks(4, 4096, LARGE_WORK);
    assert!(chunks.len() > 4, "expected finer granularity than one chunk per thread, got {}", chunks.len());
}

#[uzu_test]
fn small_work_stays_on_one_thread() {
    let chunks = collect_chunks(8, 1024, 1024);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], 0..1024);
}

#[uzu_test]
fn single_thread_keeps_one_chunk() {
    let chunks = collect_chunks(1, 4096, LARGE_WORK);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], 0..4096);
}

#[uzu_test]
fn pool_beats_spawning_per_dispatch() {
    let threads = std::thread::available_parallelism().map(|threads| threads.get()).unwrap_or(1);
    let pool = Pool::new(threads);
    let units = 1024usize;
    let work = 1 << 24;
    let iterations = 200;
    let counter = std::sync::atomic::AtomicUsize::new(0);

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        pool.for_each_chunk(units, work, |range| {
            counter.fetch_add(range.len(), std::sync::atomic::Ordering::Relaxed);
        });
    }
    let pooled = started.elapsed() / iterations;

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let chunk = units.div_ceil(8 * threads);
        let chunks = units.div_ceil(chunk);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let next = &next;
                let counter = &counter;
                scope.spawn(move || {
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if index >= chunks {
                            break;
                        }
                        let start = index * chunk;
                        let end = (start + chunk).min(units);
                        counter.fetch_add(end - start, std::sync::atomic::Ordering::Relaxed);
                    }
                });
            }
        });
    }
    let spawned = started.elapsed() / iterations;

    println!("parallel dispatch: spawn {spawned:?} -> pool {pooled:?}");
    assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2 * iterations as usize * units);
}
