use std::time::{Duration, Instant};

use proc_macros::uzu_test;
use test_runner::perf::run_perf_with_warmup;

use super::{
    Cell, Matmul,
    matrix::{cases, cells},
    mlx::{MlxMatmul, assert_available},
    summary::{self, Sample},
    table::{Block, Slot},
    uzu::UzuMatmul,
};
use crate::tests::util::shared_metal_context;

const WARMUP: usize = 5;
const SAMPLES: usize = 15;
const DISPATCHES: u64 = 100;
const RAMP: Duration = Duration::from_millis(500);

fn ramp_gpu_clocks(engine: &mut dyn Matmul) {
    let cell = Cell {
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

#[uzu_test]
#[ignore]
fn quantized_matmul_benchmark() {
    assert_available();

    let context = shared_metal_context();
    let mut engines: Vec<Box<dyn Matmul>> = UzuMatmul::all(&context).into_iter().chain(MlxMatmul::all()).collect();
    let columns: Vec<&'static str> = engines.iter().map(|engine| engine.name()).collect();

    ramp_gpu_clocks(&mut *engines[0]);

    let mut samples: Vec<Sample> = Vec::new();

    for case in cases() {
        let ladder: Vec<Cell> = cells(case).collect();
        let title = format!("{}-bit, group size {}, K {}, N {}", case.bits, case.group_size, case.k, case.n);
        let mut block = Block::new(title, columns.clone(), &ladder);

        for (row, cell) in ladder.iter().enumerate() {
            for offset in 0..engines.len() {
                let column = (offset + row) % engines.len();
                let engine = &mut engines[column];
                let name = engine.name();

                let slot = match engine.prepare(*cell) {
                    Err(_) => Slot::Unsupported,
                    Ok(()) => {
                        let measured = run_perf_with_warmup(name, WARMUP, SAMPLES, || {
                            engine.dispatch(DISPATCHES).expect("dispatch after successful prepare");
                        });
                        Slot::Micros(measured.min_ms * 1000.0 / DISPATCHES as f64)
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
