use std::ffi::CStr;

use half::bf16;
use mlx_rs::{Array, Stream, random::normal, transforms::eval};

use super::{Cell, Engine, Matmul};

struct Prepared {
    x: Array,
    weights: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
}

pub struct MlxMatmul {
    mode: &'static CStr,
    name: &'static str,
    prepared: Option<Prepared>,
}

impl MlxMatmul {
    pub fn all() -> Vec<Box<dyn Matmul>> {
        [(c"affine", "mlx affine"), (c"mxfp4", "mlx mxfp4"), (c"mxfp8", "mlx mxfp8")]
            .into_iter()
            .map(|(mode, name)| {
                Box::new(Self {
                    mode,
                    name,
                    prepared: None,
                }) as Box<dyn Matmul>
            })
            .collect()
    }
}

pub fn assert_available() {
    let probe = Array::from_slice(&[0.0f32], &[1]);
    eval([&probe]).expect("mlx is unusable");
}

fn optional(value: i32) -> mlx_sys::mlx_optional_int {
    mlx_sys::mlx_optional_int {
        value,
        has_value: true,
    }
}

fn quantize(
    source: &Array,
    group_size: i32,
    bits: i32,
    mode: &CStr,
) -> Result<(Array, Array, Option<Array>), String> {
    unsafe {
        let mut vector = mlx_sys::mlx_vector_array_new();
        let status = mlx_sys::mlx_quantize(
            &mut vector,
            source.as_ptr(),
            optional(group_size),
            optional(bits),
            mode.as_ptr(),
            Stream::gpu().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_vector_array_free(vector);
            return Err(format!("mlx_quantize rejected bits {bits} group size {group_size}"));
        }

        let count = mlx_sys::mlx_vector_array_size(vector);
        let mut parts = Vec::with_capacity(count);
        for index in 0..count {
            let mut array = mlx_sys::mlx_array_new();
            mlx_sys::mlx_vector_array_get(&mut array, vector, index);
            parts.push(Array::from_ptr(array));
        }
        mlx_sys::mlx_vector_array_free(vector);

        let mut parts = parts.into_iter();
        let weights = parts.next().ok_or_else(|| "mlx_quantize returned nothing".to_owned())?;
        let scales = parts.next().ok_or_else(|| "mlx_quantize returned no scales".to_owned())?;
        Ok((weights, scales, parts.next()))
    }
}

fn matmul(
    prepared: &Prepared,
    mode: &CStr,
) -> Result<Array, String> {
    let absent = mlx_sys::mlx_array {
        ctx: std::ptr::null_mut(),
    };
    let biases = prepared.biases.as_ref().map_or(absent, Array::as_ptr);
    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_quantized_matmul(
            &mut result,
            prepared.x.as_ptr(),
            prepared.weights.as_ptr(),
            prepared.scales.as_ptr(),
            biases,
            true,
            optional(prepared.group_size),
            optional(prepared.bits),
            mode.as_ptr(),
            Stream::gpu().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(format!("mlx_quantized_matmul rejected {}", mode.to_string_lossy()));
        }
        Ok(Array::from_ptr(result))
    }
}

impl Matmul for MlxMatmul {
    fn engine(&self) -> Engine {
        Engine::Mlx
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn prepare(
        &mut self,
        cell: Cell,
    ) -> Result<(), String> {
        self.prepared = None;
        unsafe { mlx_sys::mlx_clear_cache() };

        let (m, k, n) = (cell.m as i32, cell.k as i32, cell.n as i32);
        let group_size = cell.group_size as i32;
        let bits = cell.bits as i32;

        let x = normal::<bf16>(&[m, k][..], None, None, None).map_err(|error| format!("{error:?}"))?;
        let source = normal::<bf16>(&[n, k][..], None, None, None).map_err(|error| format!("{error:?}"))?;
        eval([&x, &source]).map_err(|error| format!("{error:?}"))?;

        let (weights, scales, biases) = quantize(&source, group_size, bits, self.mode)?;
        eval([&weights, &scales]).map_err(|error| format!("{error:?}"))?;
        if let Some(biases) = &biases {
            eval([biases]).map_err(|error| format!("{error:?}"))?;
        }

        self.prepared = Some(Prepared {
            x,
            weights,
            scales,
            biases,
            group_size,
            bits,
        });
        self.dispatch(1)
    }

    fn dispatch(
        &mut self,
        count: u64,
    ) -> Result<(), String> {
        let mode = self.mode;
        let prepared = self.prepared.as_ref().ok_or_else(|| "dispatch before prepare".to_owned())?;

        let outputs = (0..count).map(|_| matmul(prepared, mode)).collect::<Result<Vec<_>, _>>()?;
        eval(outputs.iter()).map_err(|error| format!("{error:?}"))
    }
}
