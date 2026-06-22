use std::collections::BTreeMap;

use keisoku::{Device, Session};
use serde::{Deserialize, Serialize};

pub type Data = BTreeMap<String, BTreeMap<String, f64>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub device: Device,
    pub interval_ms: u64,
    pub windows: Vec<Window>,
    pub session: Session,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Window {
    pub label: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub gpu_watts_avg: f64,
    pub gpu_watts_peak: f64,
    pub total_watts_avg: f64,
    pub energy_joules: f64,
    pub samples: usize,
    pub data: BTreeMap<String, f64>,
}

pub fn build(
    session: Session,
    data: Data,
) -> Report {
    let windows = windows(&session, &data);
    Report {
        device: session.device.clone(),
        interval_ms: session.interval.value(),
        windows,
        session,
    }
}

fn windows(
    session: &Session,
    data: &Data,
) -> Vec<Window> {
    let end_of_run = session.snapshots.last().map(|snapshot| snapshot.elapsed.value()).unwrap_or_default();
    session
        .markers
        .iter()
        .enumerate()
        .map(|(index, marker)| {
            let start_ms = marker.elapsed.value();
            let end_ms = session.markers.get(index + 1).map(|next| next.elapsed.value()).unwrap_or(end_of_run);
            window(session, data, marker.label.clone(), start_ms, end_ms)
        })
        .collect()
}

fn window(
    session: &Session,
    data: &Data,
    label: String,
    start_ms: u64,
    end_ms: u64,
) -> Window {
    let mut gpu_sum = 0.0_f64;
    let mut gpu_peak = 0.0_f64;
    let mut total_sum = 0.0_f64;
    let mut samples = 0usize;
    let mut points: Vec<(f64, f64)> = Vec::new();
    for snapshot in &session.snapshots {
        let elapsed = snapshot.elapsed.value();
        if elapsed < start_ms || elapsed > end_ms {
            continue;
        }
        let Some(power) = &snapshot.power else {
            continue;
        };
        let gpu = power.gpu.value() as f64;
        gpu_sum += gpu;
        gpu_peak = gpu_peak.max(gpu);
        total_sum += power.total.value() as f64;
        samples += 1;
        points.push((elapsed as f64 / 1000.0, power.total.value() as f64));
    }
    let count = samples.max(1) as f64;
    Window {
        label: label.clone(),
        start_ms,
        end_ms,
        gpu_watts_avg: gpu_sum / count,
        gpu_watts_peak: gpu_peak,
        total_watts_avg: total_sum / count,
        energy_joules: trapezoid(&points),
        samples,
        data: data.get(&label).cloned().unwrap_or_default(),
    }
}

fn trapezoid(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|pair| {
            let (start_seconds, start_watts) = pair[0];
            let (end_seconds, end_watts) = pair[1];
            (end_seconds - start_seconds) * (start_watts + end_watts) / 2.0
        })
        .sum()
}
