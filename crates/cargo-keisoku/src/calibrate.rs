use crate::{
    model::{self, Row},
    report::{Report, Window},
};

pub fn run(args: &[String]) {
    let mut report_path: Option<String> = None;
    let mut out_path = String::from("keisoku-coeffs.json");
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--report" => report_path = iter.next().cloned(),
            "--out" => out_path = iter.next().cloned().unwrap_or(out_path),
            other => {
                eprintln!("calibrate: unknown argument {other}");
                std::process::exit(2);
            },
        }
    }

    let report_path = report_path.unwrap_or_else(|| {
        eprintln!("calibrate requires --report <report.json>");
        std::process::exit(2);
    });
    let json = std::fs::read_to_string(&report_path).expect("failed to read report");
    let report: Report = serde_json::from_str(&json).expect("failed to parse report");

    let rows: Vec<Row> = report.windows.iter().filter_map(row).collect();
    let coefficients = model::fit(report.device.chip.clone(), &rows).unwrap_or_else(|| {
        eprintln!("calibrate: need at least 3 GEMM windows carrying iters data (found {})", rows.len());
        std::process::exit(1);
    });

    let json = serde_json::to_string_pretty(&coefficients).expect("failed to serialize coefficients");
    std::fs::write(&out_path, json).expect("failed to write coefficients");
    eprintln!(
        "keisoku: a={:.3e} J/byte  b={:.3e} J/FLOP  P_idle={:.2} W  bw={:.1} GB/s  (n={}, rms={:.3} J) -> {out_path}",
        coefficients.a_j_per_byte,
        coefficients.b_j_per_flop,
        coefficients.p_idle_w,
        coefficients.peak_bandwidth_bytes_per_s / 1e9,
        coefficients.samples,
        coefficients.rms_error_j,
    );
}

fn row(window: &Window) -> Option<Row> {
    let (m, k, n) = model::parse_mkn(&window.label)?;
    let iters = *window.data.get("iters")?;
    let gpu_seconds = window.data.get("gpu_ns").copied().unwrap_or_default() / 1e9;
    if iters <= 0.0 {
        return None;
    }
    let work = model::gemm_work(m, k, n);
    Some(Row {
        bytes: work.bytes * iters,
        flops: work.flops * iters,
        gpu_seconds,
        duration_seconds: window.end_ms.saturating_sub(window.start_ms) as f64 / 1000.0,
        energy_joules: window.energy_joules,
    })
}
