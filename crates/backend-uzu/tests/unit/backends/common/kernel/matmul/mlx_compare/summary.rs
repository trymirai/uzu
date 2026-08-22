use super::{Cell, Engine};

type Key = (u32, u32, u32, u32, u32);

pub struct Sample {
    pub cell: Cell,
    pub engine: Engine,
    pub micros: f64,
}

fn key(cell: &Cell) -> Key {
    (cell.k, cell.n, cell.m, cell.bits, cell.group_size)
}

fn fastest(
    samples: &[Sample],
    cell: Key,
    engine: Engine,
) -> Option<f64> {
    samples
        .iter()
        .filter(|sample| sample.engine == engine && key(&sample.cell) == cell && sample.micros > 0.0)
        .map(|sample| sample.micros)
        .reduce(f64::min)
}

fn geomean(ratios: &[f64]) -> Option<f64> {
    let logs: Vec<f64> = ratios.iter().filter(|ratio| **ratio > 0.0).map(|ratio| ratio.ln()).collect();
    (!logs.is_empty()).then(|| (logs.iter().sum::<f64>() / logs.len() as f64).exp())
}

pub fn mlx_over_uzu(samples: &[Sample]) -> Option<f64> {
    let mut seen: Vec<Key> = Vec::new();
    let mut ratios = Vec::new();

    for sample in samples {
        let cell = key(&sample.cell);
        if seen.contains(&cell) {
            continue;
        }
        seen.push(cell);

        if let (Some(uzu), Some(mlx)) = (fastest(samples, cell, Engine::Uzu), fastest(samples, cell, Engine::Mlx)) {
            ratios.push(mlx / uzu);
        }
    }

    geomean(&ratios)
}
