use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::sync_channel,
    },
    thread,
    time::Duration,
};

use proc_macros::uzu_test;

use super::{StageRequest, StageState, run_worker};
use crate::backends::common::SharedEvent;

#[derive(Clone, Default)]
struct TestEvent(Arc<AtomicU64>);

impl SharedEvent for TestEvent {
    fn signaled_value(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    fn signal(
        &self,
        value: u64,
    ) {
        self.0.fetch_max(value, Ordering::Release);
    }
}

fn start_worker(
    slot_count: usize,
    slot_bytes: usize,
) -> (
    [Vec<u8>; 2],
    std::sync::mpsc::SyncSender<StageRequest<u64>>,
    TestEvent,
    TestEvent,
    Arc<StageState>,
    thread::JoinHandle<()>,
) {
    assert_eq!(slot_count, 2);
    let mut buffers = [vec![0_u8; slot_bytes], vec![0_u8; slot_bytes]];
    let pointers = buffers.each_mut().map(|buffer| buffer.as_mut_ptr() as usize).to_vec();
    let state = Arc::new(StageState::new(slot_count));
    let ready = TestEvent::default();
    let consumed = TestEvent::default();
    let (tx, rx) = sync_channel(slot_count);
    let worker_state = Arc::clone(&state);
    let worker_ready = ready.clone();
    let worker_consumed = consumed.clone();
    let worker = thread::spawn(move || {
        run_worker(
            rx,
            pointers,
            slot_bytes,
            worker_ready,
            worker_consumed,
            worker_state,
            |request: StageRequest<u64>, _cancelled, destination| {
                destination.copy_from_slice(&request.request.to_le_bytes());
                Ok(())
            },
        );
    });
    (buffers, tx, ready, consumed, state, worker)
}

#[uzu_test]
fn worker_loads_requests_in_order_and_signals_generations() {
    let (buffers, tx, ready, _consumed, _state, worker) = start_worker(2, 8);
    tx.send(StageRequest {
        generation: 1,
        slot: 0,
        bytes: 8,
        wait_for_consumed: None,
        request: 11,
    })
    .unwrap();
    tx.send(StageRequest {
        generation: 2,
        slot: 1,
        bytes: 8,
        wait_for_consumed: None,
        request: 22,
    })
    .unwrap();
    drop(tx);
    worker.join().unwrap();

    assert_eq!(ready.signaled_value(), 2);
    assert_eq!(buffers[0], 11_u64.to_le_bytes());
    assert_eq!(buffers[1], 22_u64.to_le_bytes());
}

#[uzu_test]
fn worker_waits_for_consumed_generation_before_reusing_slot() {
    let (buffers, tx, ready, consumed, _state, worker) = start_worker(2, 8);
    tx.send(StageRequest {
        generation: 3,
        slot: 0,
        bytes: 8,
        wait_for_consumed: Some(1),
        request: 33,
    })
    .unwrap();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(ready.signaled_value(), 0);

    consumed.signal(1);
    assert!(ready.wait_until_signaled_value_timeout_ms(3, 1_000));
    drop(tx);
    worker.join().unwrap();
    assert_eq!(buffers[0], 33_u64.to_le_bytes());
}

#[uzu_test]
fn worker_zero_fills_and_records_io_errors() {
    let mut buffers = [vec![0_u8; 8], vec![0_u8; 8]];
    buffers[0].fill(0xA5);
    let pointers = buffers.each_mut().map(|buffer| buffer.as_mut_ptr() as usize).to_vec();
    let state = Arc::new(StageState::new(2));
    let ready = TestEvent::default();
    let consumed = TestEvent::default();
    let (tx, rx) = sync_channel(2);
    let worker_state = Arc::clone(&state);
    let worker_ready = ready.clone();
    let worker = thread::spawn(move || {
        run_worker(rx, pointers, 8, worker_ready, consumed, worker_state, |_request, _cancelled, _destination| {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"))
        });
    });
    tx.send(StageRequest {
        generation: 1,
        slot: 0,
        bytes: 8,
        wait_for_consumed: None,
        request: 0,
    })
    .unwrap();
    drop(tx);
    worker.join().unwrap();

    assert!(ready.signaled_value() >= 1);
    assert_eq!(state.take_error(0).unwrap().kind(), io::ErrorKind::UnexpectedEof);
    assert!(buffers[0].iter().all(|byte| *byte == 0));
}

#[uzu_test]
fn cancellation_unblocks_consumed_wait() {
    let (buffers, tx, ready, _consumed, state, worker) = start_worker(2, 8);
    tx.send(StageRequest {
        generation: 1,
        slot: 0,
        bytes: 8,
        wait_for_consumed: Some(1),
        request: 1,
    })
    .unwrap();
    thread::sleep(Duration::from_millis(20));
    state.cancelled.store(true, Ordering::Release);
    drop(tx);
    worker.join().unwrap();
    assert!(ready.signaled_value() >= 1);
    assert!(buffers[0].iter().all(|byte| *byte == 0));
}
