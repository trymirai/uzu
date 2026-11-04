use anyhow::ensure;

use crate::system::get_gpu_energy;

pub fn gpu_energy() -> anyhow::Result<f64> {
    let mut joules = 0.0;
    let result = unsafe { get_gpu_energy(&mut joules) };
    ensure!(result == 0, "get_gpu_energy failed ({result})");
    Ok(joules)
}
