#![cfg(target_os = "macos")]

use std::time::Instant;

use metal::prelude::*;
use metal::MTLHeapDescriptor;
use objc2::rc::autoreleasepool;

const MB: usize = 1024 * 1024;
const GB: usize = 1024 * MB;

const BUFFER_SIZES: &[usize] = &[
    128 * MB,
    256 * MB,
    512 * MB,
    1 * GB,
    2 * GB,
    4 * GB,
    6 * GB,
    8 * GB,
    10 * GB,
];

const HEAP_SIZE: usize = 12 * GB;

fn format_bytes(bytes: usize) -> String {
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() {
    autoreleasepool(|_| {
        let device = <dyn MTLDevice>::system_default().expect("No Metal device available");
        println!("Device: {}", device.name());
        println!();

        let descriptor = MTLHeapDescriptor::new();
        descriptor.set_size(HEAP_SIZE);
        descriptor.set_storage_mode(MTLStorageMode::Shared);

        let heap = device
            .new_heap_with_descriptor(&descriptor)
            .expect("Failed to create heap");

        println!("Heap size: {}", format_bytes(heap.size()));
        println!();

        // Test: Allocate and deallocate buffers of various sizes
        println!("=== Buffer Allocation & Deallocation ===");
        println!("{:>10} | {:>12} | {:>12} | {:>12} | {:>12}", 
            "Size", "Alloc Time", "Heap Used", "Drop Time", "Reclaimed");
        println!("{}", "-".repeat(70));

        for &size in BUFFER_SIZES {
            let (buffer, alloc_time, used_after_alloc) = autoreleasepool(|_| {
                let start = Instant::now();
                let buffer = heap
                    .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
                    .expect(&format!("Failed to allocate {} buffer", format_bytes(size)));
                let alloc_time = start.elapsed().as_nanos();
                let used = heap.used_size();
                (buffer, alloc_time, used)
            });

            let (drop_time, reclaimed) = autoreleasepool(|_| {
                let before = heap.used_size();
                let start = Instant::now();
                drop(buffer);
                let drop_time = start.elapsed().as_nanos();
                let after = heap.used_size();
                let reclaimed = if before > after { before - after } else { 0 };
                (drop_time, reclaimed)
            });

            println!("{:>10} | {:>9} ns | {:>12} | {:>9} ns | {:>12}",
                format_bytes(size),
                alloc_time,
                format_bytes(used_after_alloc),
                drop_time,
                format_bytes(reclaimed)
            );
        }
        println!();

        // Final check
        println!("=== Final State ===");
        println!("Heap used: {}", format_bytes(heap.used_size()));
        if heap.used_size() == 0 {
            println!("✓ All memory reclaimed!");
        } else {
            println!("⚠ Memory leak: {}", format_bytes(heap.used_size()));
        }
    });
}
