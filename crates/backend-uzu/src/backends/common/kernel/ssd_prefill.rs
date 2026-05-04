use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        kernel::{SSDPrefill64Kernel, SSDPrefillKernel, SSDPrefillSequentialKernel},
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
}

pub struct SSDPrefillArguments<'a, B: Backend> {
    pub x: &'a B::DenseBuffer,
    pub dt: &'a B::DenseBuffer, // raw dt values
    pub b: &'a B::DenseBuffer,
    pub c: &'a B::DenseBuffer,
    pub d: &'a B::DenseBuffer,
    pub z: &'a B::DenseBuffer,
    pub state: &'a mut B::DenseBuffer,
    pub y: &'a mut B::DenseBuffer,
    pub suffix_len: usize,
    pub group_size: u32,
    pub state_size: u32,
    pub x_strides: [usize; 3],
    pub dt_strides: [usize; 2],
    pub cb_strides: [usize; 3],
    pub state_strides: [usize; 3],
    pub channels: usize,
    pub head_dim: usize,
}

pub struct SSDPrefillKernels<B: Backend> {
    single: <B::Kernels as Kernels>::SSDPrefillKernel,
    single_64: <B::Kernels as Kernels>::SSDPrefill64Kernel,
    sequential: <B::Kernels as Kernels>::SSDPrefillSequentialKernel,
}

impl<B: Backend> SSDPrefillKernels<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let single = <B::Kernels as Kernels>::SSDPrefillKernel::new(context, data_type)?;
        let single_64 = <B::Kernels as Kernels>::SSDPrefill64Kernel::new(context, data_type)?;
        let sequential = <B::Kernels as Kernels>::SSDPrefillSequentialKernel::new(context, data_type)?;
        Ok(Self {
            single,
            single_64,
            sequential,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        args: SSDPrefillArguments<B>,
        mode: SSDPrefillMode,
    ) {
        let x_strides: [u32; 3] = args.x_strides.map(|x| x as u32);
        let dt_strides: [u32; 2] = args.dt_strides.map(|x| x as u32);
        let cb_strides: [u32; 3] = args.cb_strides.map(|x| x as u32);
        let state_strides: [u32; 3] = args.state_strides.map(|x| x as u32);

        if mode == SSDPrefillMode::SinglePass {
            if args.state_size == 64 {
                self.single_64.encode(
                    args.x,
                    args.dt,
                    args.b,
                    args.c,
                    args.d,
                    args.z,
                    args.state,
                    args.y,
                    args.suffix_len as u32,
                    args.group_size,
                    args.state_size,
                    &x_strides,
                    &dt_strides,
                    &cb_strides,
                    &state_strides,
                    args.channels as u32,
                    args.head_dim as u32,
                    encoder,
                )
            } else {
                self.single.encode(
                    args.x,
                    args.dt,
                    args.b,
                    args.c,
                    args.d,
                    args.z,
                    args.state,
                    args.y,
                    args.suffix_len as u32,
                    args.group_size,
                    args.state_size,
                    &x_strides,
                    &dt_strides,
                    &cb_strides,
                    &state_strides,
                    args.channels as u32,
                    args.head_dim as u32,
                    encoder,
                )
            }
        } else if mode == SSDPrefillMode::Sequential {
            self.sequential.encode(
                args.x,
                args.dt,
                args.b,
                args.c,
                args.d,
                args.z,
                args.state,
                args.y,
                args.suffix_len as u32,
                args.group_size,
                args.state_size,
                &x_strides,
                &dt_strides,
                &cb_strides,
                &state_strides,
                args.channels as u32,
                args.head_dim as u32,
                encoder,
            )
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/ssd_prefill_test.rs"]
mod tests;
