use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use backend_uzu::{
    backends::common::{AllocationType, Backend, Context, Encoder},
    data_type::DataType,
};

#[cfg(backend = "cpu")]
type TestBackend = backend_uzu::backends::cpu::Cpu;
#[cfg(all(backend = "metal", not(backend = "cpu")))]
type TestBackend = backend_uzu::backends::metal::Metal;

struct Counting;

// `const` init: no lazy setup, no destructor, so counting cannot recurse.
thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        COUNT.set(COUNT.get() + 1);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(
        &self,
        pointer: *mut u8,
        layout: Layout,
    ) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        COUNT.set(COUNT.get() + 1);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(
        &self,
        pointer: *mut u8,
        layout: Layout,
        new_size: usize,
    ) -> *mut u8 {
        COUNT.set(COUNT.get() + 1);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn count(work: impl FnOnce()) -> usize {
    COUNT.set(0);
    work();
    COUNT.get()
}

/// The two repeat counts are far apart, so a single allocation per dispatch shows up as
/// hundreds and cannot hide in the fixed-setup noise.
fn per_dispatch(mut work: impl FnMut(usize)) -> usize {
    let few = count(|| work(64));
    let many = count(|| work(1024));
    many.saturating_sub(few) / 960
}

#[test]
fn encode_copy_budget() {
    let context = <TestBackend as Backend>::Context::new().expect("context");
    let bytes = 64 * DataType::U32.size_in_bytes();
    let source = context.create_allocation(bytes, AllocationType::Global).expect("source");
    let mut destination = context.create_allocation(bytes, AllocationType::Global).expect("destination");

    let allocations = per_dispatch(|repeats| {
        let mut encoder = Encoder::<TestBackend>::new(&context).expect("encoder");
        for _ in 0..repeats {
            encoder.encode_copy(&source, 0..bytes, &mut destination, 0..bytes);
        }
        encoder.end_encoding().submit().wait_until_completed().expect("run");
    });

    println!("encode_copy: {allocations} allocations per dispatch");
    assert!(allocations <= 4, "encode_copy allocates {allocations} per dispatch");
}

#[test]
fn debug_group_budget() {
    let context = <TestBackend as Backend>::Context::new().expect("context");

    let allocations = per_dispatch(|repeats| {
        let mut encoder = Encoder::<TestBackend>::new(&context).expect("encoder");
        for _ in 0..repeats {
            encoder.push_debug_group("budget probe");
            encoder.pop_debug_group();
        }
        encoder.end_encoding().submit().wait_until_completed().expect("run");
    });

    println!("debug group: {allocations} allocations per push/pop");
    assert_eq!(allocations, 0, "a debug group must not allocate");
}
