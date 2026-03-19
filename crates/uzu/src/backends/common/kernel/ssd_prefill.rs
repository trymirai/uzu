use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{SSDPrefill64Kernel, SSDPrefillKernel, SSDPrefillSequentialKernel},
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
}

pub struct SSDPrefillArguments<'a, B: Backend> {
    pub x: &'a B::Buffer,
    pub dt: &'a B::Buffer, // raw dt values
    pub b: &'a B::Buffer,
    pub c: &'a B::Buffer,
    pub d: &'a B::Buffer,
    pub z: &'a B::Buffer,
    pub state: &'a mut B::Buffer,
    pub y: &'a mut B::Buffer,
    pub suffix_len: usize,
    pub group_size: i32,
    pub state_size: i32,
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
        compute_encoder: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        args: SSDPrefillArguments<B>,
        mode: SSDPrefillMode,
    ) {
        let x_strides: Vec<u32> = args.x_strides.iter().map(|x| *x as u32).collect();
        let dt_strides: Vec<u32> = args.dt_strides.iter().map(|x| *x as u32).collect();
        let cb_strides: Vec<u32> = args.cb_strides.iter().map(|x| *x as u32).collect();
        let state_strides: Vec<u32> = args.state_strides.iter().map(|x| *x as u32).collect();

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
                    x_strides.as_slice(),
                    dt_strides.as_slice(),
                    cb_strides.as_slice(),
                    state_strides.as_slice(),
                    args.channels as u32,
                    args.head_dim as u32,
                    compute_encoder,
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
                    x_strides.as_slice(),
                    dt_strides.as_slice(),
                    cb_strides.as_slice(),
                    state_strides.as_slice(),
                    args.channels as u32,
                    args.head_dim as u32,
                    compute_encoder,
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
                x_strides.as_slice(),
                dt_strides.as_slice(),
                cb_strides.as_slice(),
                state_strides.as_slice(),
                args.channels as u32,
                args.head_dim as u32,
                compute_encoder,
            )
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/ssd_prefill_test.rs"]
mod tests;
