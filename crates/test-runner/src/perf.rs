use std::time::Instant;

pub struct PerfResult {
    pub name: String,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
}

impl PerfResult {
    fn new(
        name: String,
        times_ms: &[f64],
    ) -> Self {
        let mut sorted = times_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        let variance = sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
        let std_dev = variance.sqrt();

        Self {
            name,
            mean_ms: mean,
            median_ms: median,
            min_ms: min,
            max_ms: max,
            std_dev_ms: std_dev,
        }
    }

    pub fn print(&self) {
        eprintln!(
            "  {:<30} mean={:8.3}ms  median={:8.3}ms  min={:8.3}ms  max={:8.3}ms  std={:6.3}ms",
            self.name, self.mean_ms, self.median_ms, self.min_ms, self.max_ms, self.std_dev_ms
        );
    }
}

pub fn run_perf<F: FnMut()>(
    name: &str,
    iterations: usize,
    f: F,
) -> PerfResult {
    run_perf_with_warmup(name, 0, iterations, f)
}

pub fn run_perf_with_warmup<F: FnMut()>(
    name: &str,
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> PerfResult {
    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Measure
    let mut times_ms = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times_ms.push(elapsed.as_secs_f64() * 1000.0);
    }

    PerfResult::new(name.to_string(), &times_ms)
}
