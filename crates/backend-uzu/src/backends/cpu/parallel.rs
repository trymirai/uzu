use std::{
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::JoinHandle,
};

const MIN_WORK_PER_THREAD: usize = 1 << 16;
// More than one chunk per thread: P and E cores differ several-fold in throughput.
const CHUNKS_PER_THREAD: usize = 8;

pub(crate) fn available_threads() -> usize {
    std::env::var("UZU_CPU_THREADS")
        .ok()
        .and_then(|threads| threads.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|threads| threads.get()).unwrap_or(1))
}

struct Job {
    run: unsafe fn(*const (), Range<usize>),
    data: *const (),
    chunk: usize,
    chunks: usize,
    units: usize,
    next: AtomicUsize,
}

unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    unsafe fn run_claimed(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            if index >= self.chunks {
                return;
            }
            let start = index * self.chunk;
            unsafe { (self.run)(self.data, start..(start + self.chunk).min(self.units)) };
        }
    }
}

#[derive(Default)]
struct State {
    job: Option<&'static Job>,
    generation: u64,
    active: usize,
    panicked: bool,
    stop: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    idle: Condvar,
}

pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// Retracts the job and drains workers even if the submitter unwinds, so the job never
/// outlives the closure it points at.
struct Retract<'a>(&'a Shared);

impl Drop for Retract<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|error| error.into_inner());
        state.job = None;
        while state.active > 0 {
            state = self.0.idle.wait(state).unwrap_or_else(|error| error.into_inner());
        }
    }
}

impl Pool {
    pub(crate) fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared::default());
        let workers = (1..threads)
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || worker(&shared))
            })
            .collect();

        Self {
            shared,
            workers,
        }
    }

    pub(crate) fn for_each_chunk<F>(
        &self,
        units: usize,
        work: usize,
        compute: F,
    ) where
        F: Fn(Range<usize>) + Sync,
    {
        let threads = (self.workers.len() + 1).min(units).min((work / MIN_WORK_PER_THREAD).max(1));
        if threads <= 1 {
            compute(0..units);
            return;
        }

        unsafe fn call<F: Fn(Range<usize>)>(
            data: *const (),
            range: Range<usize>,
        ) {
            unsafe { (*(data as *const F))(range) }
        }

        let chunk = units.div_ceil(threads * CHUNKS_PER_THREAD).max(1);
        let job = Job {
            run: call::<F>,
            data: (&raw const compute) as *const (),
            chunk,
            chunks: units.div_ceil(chunk),
            units,
            next: AtomicUsize::new(0),
        };

        // SAFETY: `Retract` below clears the job and waits for every worker that took it,
        // on the normal path and while unwinding, so workers never see it after this call.
        let published: &'static Job = unsafe { std::mem::transmute::<&Job, &'static Job>(&job) };

        {
            let mut state = self.shared.state.lock().unwrap();
            state.job = Some(published);
            state.generation = state.generation.wrapping_add(1);
        }
        for _ in 1..threads {
            self.shared.wake.notify_one();
        }

        {
            let _retract = Retract(&self.shared);
            unsafe { job.run_claimed() };
        }

        let mut state = self.shared.state.lock().unwrap();
        if std::mem::take(&mut state.panicked) {
            drop(state);
            panic!("cpu kernel panicked on a pool worker");
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|error| error.into_inner());
            state.stop = true;
        }
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut state = shared.state.lock().unwrap_or_else(|error| error.into_inner());
            loop {
                if state.stop {
                    return;
                }
                match state.job {
                    Some(job) if state.generation != seen => {
                        seen = state.generation;
                        state.active += 1;
                        break job;
                    },
                    _ => {
                        state = shared.wake.wait(state).unwrap_or_else(|error| error.into_inner());
                    },
                }
            }
        };

        let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { job.run_claimed() }));

        let mut state = shared.state.lock().unwrap_or_else(|error| error.into_inner());
        state.active -= 1;
        state.panicked |= outcome.is_err();
        if state.active == 0 {
            shared.idle.notify_all();
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/cpu/parallel_test.rs"]
mod tests;
