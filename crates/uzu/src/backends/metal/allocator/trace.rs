use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationPhase {
    SessionInit,
    Prefill {
        step: usize,
        batch_size: usize,
    },
    Generate {
        step: usize,
        batch_size: usize,
    },
}

#[derive(Clone, Debug)]
pub struct AllocationEvent {
    pub label: String,
    pub size: usize,
    pub timestamp_ns: u64,
    pub phase: AllocationPhase,
}

#[derive(Debug)]
pub struct AllocationSummary {
    pub total_allocations: usize,
    pub total_bytes: usize,
    pub peak_bytes: usize,
    pub by_phase: Vec<PhaseSummary>,
}

#[derive(Debug)]
pub struct PhaseSummary {
    pub phase: AllocationPhase,
    pub allocations: usize,
    pub bytes: usize,
}

pub struct AllocationTracer {
    start_time: Instant,
    current_phase: Mutex<AllocationPhase>,
    events: Mutex<Vec<AllocationEvent>>,
    total_allocated: AtomicU64,
    peak_allocated: AtomicU64,
}

impl AllocationTracer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            current_phase: Mutex::new(AllocationPhase::SessionInit),
            events: Mutex::new(Vec::new()),
            total_allocated: AtomicU64::new(0),
            peak_allocated: AtomicU64::new(0),
        }
    }

    pub fn set_phase(
        &self,
        phase: AllocationPhase,
    ) {
        if let Ok(mut current) = self.current_phase.lock() {
            *current = phase;
        }
    }

    pub fn record_allocation(
        &self,
        label: &str,
        size: usize,
    ) {
        let timestamp_ns = self.start_time.elapsed().as_nanos() as u64;
        let phase = self
            .current_phase
            .lock()
            .map(|p| *p)
            .unwrap_or(AllocationPhase::SessionInit);

        let event = AllocationEvent {
            label: label.to_string(),
            size,
            timestamp_ns,
            phase,
        };

        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }

        let new_total =
            self.total_allocated.fetch_add(size as u64, Ordering::Relaxed)
                + size as u64;
        self.peak_allocated.fetch_max(new_total, Ordering::Relaxed);
    }

    pub fn record_deallocation(
        &self,
        size: usize,
    ) {
        self.total_allocated.fetch_sub(size as u64, Ordering::Relaxed);
    }

    pub fn events(&self) -> Vec<AllocationEvent> {
        self.events.lock().map(|e| e.clone()).unwrap_or_default()
    }

    pub fn summary(&self) -> AllocationSummary {
        let events = self.events();

        let total_allocations = events.len();
        let total_bytes: usize = events.iter().map(|e| e.size).sum();
        let peak_bytes = self.peak_allocated.load(Ordering::Relaxed) as usize;

        let mut phase_map: std::collections::HashMap<
            String,
            (AllocationPhase, usize, usize),
        > = std::collections::HashMap::new();

        for event in &events {
            let key = format!("{:?}", event.phase);
            let entry = phase_map.entry(key).or_insert((event.phase, 0, 0));
            entry.1 += 1;
            entry.2 += event.size;
        }

        let by_phase: Vec<PhaseSummary> = phase_map
            .into_values()
            .map(|(phase, allocations, bytes)| PhaseSummary {
                phase,
                allocations,
                bytes,
            })
            .collect();

        AllocationSummary {
            total_allocations,
            total_bytes,
            peak_bytes,
            by_phase,
        }
    }

    pub fn print_summary(&self) {
        let summary = self.summary();
        let events = self.events();

        println!("\n=== Allocation Trace Summary ===\n");
        println!("Total allocations: {}", summary.total_allocations);
        println!(
            "Total bytes allocated: {} MB",
            summary.total_bytes / (1024 * 1024)
        );
        println!("Peak memory: {} MB", summary.peak_bytes / (1024 * 1024));

        println!("\n--- By Phase ---\n");
        for phase in &summary.by_phase {
            println!(
                "{:?}: {} allocations, {} MB",
                phase.phase,
                phase.allocations,
                phase.bytes / (1024 * 1024)
            );
        }

        println!("\n--- All Allocations ---\n");
        println!("{:<50} {:>15} {:>20}", "Label", "Size (bytes)", "Phase");
        println!("{}", "-".repeat(90));

        for event in &events {
            println!(
                "{:<50} {:>15} {:>20?}",
                if event.label.len() > 48 {
                    format!("{}...", &event.label[..45])
                } else {
                    event.label.clone()
                },
                event.size,
                event.phase
            );
        }
    }

    pub fn print_unique_buffers(&self) {
        let events = self.events();

        let mut unique: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();

        for event in &events {
            let entry = unique.entry(event.label.clone()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.max(event.size);
        }

        let mut sorted: Vec<_> = unique.into_iter().collect();
        sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));

        println!("\n=== Unique Buffers (by max size) ===\n");
        println!("{:<50} {:>10} {:>15}", "Label", "Count", "Max Size (bytes)");
        println!("{}", "-".repeat(80));

        for (label, (count, max_size)) in sorted {
            println!(
                "{:<50} {:>10} {:>15}",
                if label.len() > 48 {
                    format!("{}...", &label[..45])
                } else {
                    label
                },
                count,
                max_size
            );
        }
    }
}

impl Default for AllocationTracer {
    fn default() -> Self {
        Self::new()
    }
}
