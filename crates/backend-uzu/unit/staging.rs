use std::{
    io,
    sync::{Arc, mpsc::sync_channel},
    thread,
};

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, DenseBuffer, SharedEvent},
        cpu::{Cpu, CpuSharedEvent},
    },
    staging::{AsyncStager, StageLease, StageRequest, StageState, run_worker},
};

#[uzu_test]
fn requests_must_be_enqueued_in_generation_order() {
    let context = <Cpu as Backend>::Context::new().unwrap();
    let mut stager: AsyncStager<Cpu, u64> =
        AsyncStager::new(&*context, 2, 8, |request: StageRequest<u64>, _, destination| {
            destination.copy_from_slice(&request.request.to_le_bytes());
            Ok(())
        })
        .unwrap();
    let first = stager.reserve(false).unwrap();
    let mut second = stager.reserve(false).unwrap();

    assert_eq!(stager.enqueue(&mut second, 2, 8).unwrap_err().kind(), io::ErrorKind::InvalidInput);
    drop(first);
    stager.enqueue(&mut second, 2, 8).unwrap();
    second.ready_event.wait_until_signaled_value_timeout_ms(2, 1_000);
    let staged = stager.view(&second).allocation;
    let bytes = unsafe { std::slice::from_raw_parts(staged.cpu_ptr().as_ptr().cast::<u8>(), 8) };
    assert_eq!(bytes, 2_u64.to_le_bytes());
    second.complete().unwrap();
}

#[uzu_test]
fn only_unsubmitted_leases_return_their_slot() {
    let (state, free_slots) = StageState::new(1);
    let state = Arc::new(state);
    let lease = StageLease::<Cpu> {
        slot: Some(free_slots.recv().unwrap()),
        generation: 1,
        ready_event: CpuSharedEvent::default(),
        state: Arc::clone(&state),
        submitted: false,
    };
    drop(lease);
    let slot = free_slots.recv().unwrap();

    let lease = StageLease::<Cpu> {
        slot: Some(slot),
        generation: 2,
        ready_event: CpuSharedEvent::default(),
        state,
        submitted: true,
    };
    drop(lease);
    assert!(free_slots.try_recv().is_err());
}

#[uzu_test]
fn worker_zero_fills_and_records_io_errors() {
    let mut buffers = [vec![0_u8; 8], vec![0_u8; 8]];
    buffers[0].fill(0xA5);
    let pointers = buffers.each_mut().map(|buffer| buffer.as_mut_ptr() as usize).to_vec();
    let (state, _free_slots) = StageState::new(2);
    let state = Arc::new(state);
    let ready = CpuSharedEvent::default();
    let (tx, rx) = sync_channel(2);
    let worker_state = Arc::clone(&state);
    let worker_ready = ready.clone();
    let worker = thread::spawn(move || {
        run_worker(rx, pointers, 8, worker_ready, worker_state, |_request, _cancelled, _destination| {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"))
        });
    });
    tx.send(StageRequest {
        generation: 1,
        slot: 0,
        bytes: 8,
        request: 0,
    })
    .unwrap();
    drop(tx);
    worker.join().unwrap();

    assert!(ready.signaled_value() >= 1);
    assert_eq!(state.errors[0].lock().unwrap().take().unwrap().kind(), io::ErrorKind::UnexpectedEof);
    assert!(buffers[0].iter().all(|byte| *byte == 0));
}
