#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_family = "wasm")]
use std::os::wasi::fs::FileExt;
use std::{fs::File, io};

use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::data_type::DataType;

pub enum ParameterBytes<'a> {
    File(&'a File),
    Random(u64),
}

impl ParameterBytes<'_> {
    pub fn read_into(
        &self,
        destination: &mut [u8],
        offset: u64,
        data_type: DataType,
    ) -> io::Result<()> {
        match self {
            Self::File(file) => file.read_exact_at(destination, offset),
            Self::Random(seed) => {
                fill_random(destination, data_type, *seed);
                Ok(())
            },
        }
    }
}

fn fill_random(
    destination: &mut [u8],
    data_type: DataType,
    seed: u64,
) {
    if destination.is_empty() {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let head_len = destination.len().min(65536);
    let (head, tail) = destination.split_at_mut(head_len);
    match data_type {
        DataType::BF16 => {
            for chunk in head.as_chunks_mut::<2>().0 {
                *chunk = bf16::from_f32(rng.random_range(-0.1f32..0.1f32)).to_le_bytes();
            }
        },
        DataType::F16 => {
            for chunk in head.as_chunks_mut::<2>().0 {
                *chunk = f16::from_f32(rng.random_range(-0.1f32..0.1f32)).to_le_bytes();
            }
        },
        DataType::F32 => {
            for chunk in head.as_chunks_mut::<4>().0 {
                *chunk = rng.random_range(-0.1f32..0.1f32).to_le_bytes();
            }
        },
        DataType::F64 => {
            for chunk in head.as_chunks_mut::<8>().0 {
                *chunk = rng.random_range(-0.1f64..0.1f64).to_le_bytes();
            }
        },
        _ => {
            for chunk in head.chunks_mut(8) {
                let bytes = rng.random::<u64>().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        },
    }
    for target in tail.chunks_mut(head_len) {
        target.copy_from_slice(&head[..target.len()]);
    }
}
