use keisoku::Session;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSummary {
    pub gpu_watts_avg: f64,
    pub gpu_watts_peak: f64,
    pub total_watts_avg: f64,
    pub ane_watts_avg: f64,
    pub energy_joules: f64,
    pub samples: usize,
}

pub fn summarize(session: &Session) -> Option<PowerSummary> {
    let power: Vec<_> = session.snapshots.iter().filter_map(|snapshot| snapshot.power.as_ref()).collect();
    let samples = power.len();
    if samples == 0 {
        return None;
    }

    let count = samples as f64;
    let gpu_watts_avg = power.iter().map(|metrics| metrics.gpu.value() as f64).sum::<f64>() / count;
    let gpu_watts_peak = power.iter().map(|metrics| metrics.gpu.value() as f64).fold(0.0, f64::max);
    let total_watts_avg = power.iter().map(|metrics| metrics.total.value() as f64).sum::<f64>() / count;
    let ane_watts_avg = power.iter().map(|metrics| metrics.ane.value() as f64).sum::<f64>() / count;
    let duration_seconds =
        session.snapshots.last().map(|snapshot| snapshot.elapsed.value() as f64 / 1000.0).unwrap_or_default();

    Some(PowerSummary {
        gpu_watts_avg,
        gpu_watts_peak,
        total_watts_avg,
        ane_watts_avg,
        energy_joules: total_watts_avg * duration_seconds,
        samples,
    })
}
