use std::sync::Arc;

use half::bf16;

use super::{Cell, Engine, Matmul};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{Kernels, matmul::MatmulKernel},
        },
        metal::{Metal, MetalContext},
    },
    tests::matmul::{QuantBuffers, QuantInput, quant_arguments},
};

const SEED: u64 = 0x5EED;

type Kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

struct Prepared {
    input: QuantInput<bf16>,
    buffers: QuantBuffers<Metal, bf16>,
    kernel: Kernel,
}

pub struct UzuMatmul {
    context: Arc<MetalContext>,
    method: QuantizationMethod,
    name: String,
    prepared: Option<Prepared>,
}

impl UzuMatmul {
    pub fn all(context: &Arc<MetalContext>) -> Vec<Box<dyn Matmul>> {
        [QuantizationMethod::ScaleBias, QuantizationMethod::ScaleZeroPoint, QuantizationMethod::ScaleSymmetric]
            .into_iter()
            .map(|method| {
                Box::new(Self {
                    context: Arc::clone(context),
                    method,
                    name: format!("uzu {method}"),
                    prepared: None,
                }) as Box<dyn Matmul>
            })
            .collect()
    }
}

impl Matmul for UzuMatmul {
    fn engine(&self) -> Engine {
        Engine::Uzu
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(
        &mut self,
        cell: Cell,
    ) -> Result<(), String> {
        self.prepared = None;

        let input = QuantInput::<bf16>::new(cell.m, cell.k, cell.n, cell.group_size, cell.bits, self.method, SEED);
        let buffers = QuantBuffers::<Metal, bf16>::allocate(&self.context, &input);
        let kernel = Kernel::new(&self.context, bf16::data_type(), bf16::data_type(), bf16::data_type())
            .map_err(|error| format!("{error:?}"))?;

        self.prepared = Some(Prepared {
            input,
            buffers,
            kernel,
        });
        self.dispatch(1)
    }

    fn dispatch(
        &mut self,
        count: u64,
    ) -> Result<(), String> {
        let context = Arc::clone(&self.context);
        let prepared = self.prepared.as_mut().ok_or_else(|| "dispatch before prepare".to_owned())?;

        let mut encoder = Encoder::<Metal>::new(&context).map_err(|error| format!("{error:?}"))?;
        for _ in 0..count {
            let arguments = quant_arguments(&mut prepared.buffers, &prepared.input);
            prepared.kernel.encode(arguments, &mut encoder).map_err(|error| format!("{error:?}"))?;
        }
        encoder.end_encoding().submit().wait_until_completed().map_err(|error| format!("{error:?}"))?;
        Ok(())
    }
}
