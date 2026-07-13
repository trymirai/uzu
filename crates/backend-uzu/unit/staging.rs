use std::{io, sync::mpsc::sync_channel};

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, DenseBuffer, SharedEvent},
        cpu::Cpu,
    },
    staging::{AsyncStager, StageRequest},
};

fn value_stager(slot_count: usize) -> AsyncStager<Cpu, u64> {
    let context = <Cpu as Backend>::Context::new().unwrap();
    AsyncStager::new(&*context, slot_count, 8, |request: StageRequest<u64>, _, destination| {
        destination.copy_from_slice(&request.request.to_le_bytes());
        Ok(())
    })
    .unwrap()
}

#[uzu_test]
fn dropped_reservation_returns_its_slot() {
    let stager = value_stager(1);
    drop(stager.try_reserve().unwrap());
    assert!(stager.try_reserve().is_ok());
}

#[uzu_test]
fn reservations_can_be_submitted_out_of_order() {
    let mut stager = value_stager(2);
    let first = stager.try_reserve().unwrap();
    let second = stager.try_reserve().unwrap();

    let second = stager.submit(second, 2, 8).unwrap();
    let first = stager.submit(first, 1, 8).unwrap();
    assert_eq!(second.generation(), 1);
    assert_eq!(first.generation(), 2);

    for (stage, expected) in [(&second, 2_u64), (&first, 1_u64)] {
        let view = stager.view(stage);
        assert!(view.ready_event.wait_until_signaled_value_timeout_ms(view.value, 1_000));
        let bytes = unsafe { std::slice::from_raw_parts(view.allocation.cpu_ptr().as_ptr().cast::<u8>(), 8) };
        assert_eq!(bytes, expected.to_le_bytes());
    }
    second.complete().unwrap();
    first.complete().unwrap();
}

#[uzu_test]
fn failed_submit_returns_its_slot() {
    let mut stager = value_stager(1);
    let reservation = stager.try_reserve().unwrap();

    assert_eq!(stager.submit(reservation, 1, 9).err().unwrap().kind(), io::ErrorKind::InvalidInput);
    assert!(stager.try_reserve().is_ok());
}

#[uzu_test]
fn disconnected_worker_returns_the_reserved_slot() {
    let mut stager = value_stager(1);
    drop(stager.requests.take());
    let reservation = stager.try_reserve().unwrap();

    assert_eq!(stager.submit(reservation, 1, 8).err().unwrap().kind(), io::ErrorKind::BrokenPipe);
    assert!(stager.try_reserve().is_ok());
}

#[uzu_test]
fn early_completion_does_not_recycle_the_slot() {
    let context = <Cpu as Backend>::Context::new().unwrap();
    let (started_tx, started_rx) = sync_channel(1);
    let (release_tx, release_rx) = sync_channel(1);
    let mut stager: AsyncStager<Cpu, ()> = AsyncStager::new(&*context, 1, 8, move |_request, _, _| {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        Ok(())
    })
    .unwrap();
    let reservation = stager.try_reserve().unwrap();
    let stage = stager.submit(reservation, (), 8).unwrap();
    started_rx.recv().unwrap();

    assert_eq!(stage.complete().unwrap_err().kind(), io::ErrorKind::WouldBlock);
    assert_eq!(stager.try_reserve().err().unwrap().kind(), io::ErrorKind::WouldBlock);
    release_tx.send(()).unwrap();
}

#[uzu_test]
fn worker_zero_fills_reports_error_and_recycles_on_completion() {
    let context = <Cpu as Backend>::Context::new().unwrap();
    let mut stager: AsyncStager<Cpu, ()> = AsyncStager::new(&*context, 1, 8, |_request, _, destination| {
        destination.fill(0xA5);
        Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"))
    })
    .unwrap();
    let reservation = stager.try_reserve().unwrap();
    let stage = stager.submit(reservation, (), 8).unwrap();
    let view = stager.view(&stage);
    assert!(view.ready_event.wait_until_signaled_value_timeout_ms(view.value, 1_000));
    let bytes = unsafe { std::slice::from_raw_parts(view.allocation.cpu_ptr().as_ptr().cast::<u8>(), 8) };
    assert!(bytes.iter().all(|byte| *byte == 0));

    assert_eq!(stage.complete().unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    assert!(stager.try_reserve().is_ok());
}

#[uzu_test]
fn worker_panic_reports_error_and_recycles_on_completion() {
    let context = <Cpu as Backend>::Context::new().unwrap();
    let mut stager: AsyncStager<Cpu, ()> =
        AsyncStager::new(&*context, 1, 8, |_request, _, _| panic!("injected")).unwrap();
    let reservation = stager.try_reserve().unwrap();
    let stage = stager.submit(reservation, (), 8).unwrap();
    let view = stager.view(&stage);
    assert!(view.ready_event.wait_until_signaled_value_timeout_ms(view.value, 1_000));

    assert_eq!(stage.complete().unwrap_err().to_string(), "staging worker panicked");
    assert!(stager.try_reserve().is_ok());
}

#[uzu_test]
fn generation_overflow_returns_the_reserved_slot() {
    let mut stager = value_stager(1);
    stager.next_generation = u64::MAX;
    let reservation = stager.try_reserve().unwrap();

    assert_eq!(stager.submit(reservation, 1, 8).err().unwrap().kind(), io::ErrorKind::Other);
    assert!(stager.try_reserve().is_ok());
}
