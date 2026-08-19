use std::time::{Duration, Instant};

use proc_macros::uzu_test;
use test_runner::perf::run_perf_with_warmup;

use super::{
    Cell, Matmul,
    matrix::{QUANTIZATIONS, shapes},
    mlx::{MlxMatmul, assert_available},
    summary::{self, Sample},
    table::{Block, Slot},
    uzu::UzuMatmul,
};
use crate::tests::util::shared_metal_context;

const WARMUP: usize = 5;
const SAMPLES: usize = 15;
const TARGET_SAMPLE: Duration = Duration::from_millis(10);
const MAX_DISPATCHES: u64 = 64;
const RAMP: Duration = Duration::from_millis(500);

fn ramp_gpu_clocks(engine: &mut dyn Matmul) {
    let cell = Cell {
        layer: "ramp",
        m: 512,
        k: 4096,
        n: 4096,
        bits: 4,
        group_size: 64,
    };
    if engine.prepare(cell).is_err() {
        return;
    }
    let start = Instant::now();
    while start.elapsed() < RAMP {
        if engine.dispatch(8).is_err() {
            return;
        }
    }
}

fn dispatches_per_sample(engine: &mut dyn Matmul) -> Result<u64, String> {
    let start = Instant::now();
    engine.dispatch(1)?;
    let single = start.elapsed().as_secs_f64();
    if single <= 0.0 {
        return Ok(MAX_DISPATCHES);
    }
    Ok(((TARGET_SAMPLE.as_secs_f64() / single) as u64).clamp(1, MAX_DISPATCHES))
}

#[uzu_test]
#[ignore]
fn quantized_matmul_benchmark() {
    assert_available();

    let context = shared_metal_context();
    let mut engines: Vec<Box<dyn Matmul>> = UzuMatmul::all(&context).into_iter().chain(MlxMatmul::all()).collect();
    let columns: Vec<&'static str> = engines.iter().map(|engine| engine.name()).collect();

    ramp_gpu_clocks(&mut *engines[0]);

    let mut samples: Vec<Sample> = Vec::new();

    for &(bits, group_size) in QUANTIZATIONS {
        let cells: Vec<Cell> = shapes(bits, group_size).collect();
        let mut block = Block::new(format!("{bits}-bit, group size {group_size}"), columns.clone(), &cells);

        for (row, cell) in cells.iter().enumerate() {
            for offset in 0..engines.len() {
                let column = (offset + row) % engines.len();
                let engine = &mut engines[column];
                let name = engine.name();

                let slot = match engine.prepare(*cell).and_then(|()| dispatches_per_sample(&mut **engine)) {
                    Err(_) => Slot::Unsupported,
                    Ok(dispatches) => {
                        let measured = run_perf_with_warmup(name, WARMUP, SAMPLES, || {
                            engine.dispatch(dispatches).expect("dispatch after successful prepare");
                        });
                        Slot::Micros(measured.min_ms * 1000.0 / dispatches as f64)
                    },
                };

                if let Slot::Micros(micros) = slot {
                    samples.push(Sample {
                        cell: *cell,
                        engine: engine.engine(),
                        micros,
                    });
                }
                block.set(row, column, slot);
            }
        }

        block.finish();
    }

    let ratio = summary::mlx_over_uzu(&samples).expect("no shape was measured by both uzu and mlx");
    println!("mlx / uzu: {ratio:.2}x (geometric mean over shapes measured by both)");
}
