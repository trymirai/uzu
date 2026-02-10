use objc2::Message;

use crate::backends::{
    common::kernel::{SSDPrefill64Kernel as _, SSDPrefillKernel as _, SSDPrefillSequentialKernel as _},
    metal::{
        KernelDataType, MTLBuffer, MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
        kernel::dsl::{SSDPrefill64MetalKernel, SSDPrefillMetalKernel, SSDPrefillSequentialMetalKernel},
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SSDPrefillMode {
    Sequential,
    SinglePass,
}

pub struct SSDPrefillArguments<'a> {
    pub x: &'a ProtocolObject<dyn MTLBuffer>,
    pub dt: &'a ProtocolObject<dyn MTLBuffer>, // raw dt values
    pub b: &'a ProtocolObject<dyn MTLBuffer>,
    pub c: &'a ProtocolObject<dyn MTLBuffer>,
    pub d: &'a ProtocolObject<dyn MTLBuffer>,
    pub z: &'a ProtocolObject<dyn MTLBuffer>,
    pub state: &'a ProtocolObject<dyn MTLBuffer>,
    pub y: &'a ProtocolObject<dyn MTLBuffer>,
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

pub struct SSDPrefillKernels {
    single: SSDPrefillMetalKernel,
    single_64: SSDPrefill64MetalKernel,
    sequential: SSDPrefillSequentialMetalKernel,
}

impl SSDPrefillKernels {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MTLError> {
        let single = SSDPrefillMetalKernel::new(context, data_type.into())?;
        let single_64 = SSDPrefill64MetalKernel::new(context, data_type.into())?;
        let sequential = SSDPrefillSequentialMetalKernel::new(context, data_type.into())?;
        Ok(Self {
            single,
            single_64,
            sequential,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: SSDPrefillArguments,
        mode: SSDPrefillMode,
    ) {
        if mode == SSDPrefillMode::SinglePass {
            if args.state_size == 64 {
                self.single_64.encode(
                    &args.x.retain(),
                    &args.dt.retain(),
                    &args.b.retain(),
                    &args.c.retain(),
                    &args.d.retain(),
                    &args.z.retain(),
                    &args.state.retain(),
                    &args.y.retain(),
                    args.suffix_len as u32,
                    args.group_size,
                    args.state_size,
                    args.x_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.dt_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.cb_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.state_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.channels as u32,
                    args.head_dim as u32,
                    compute_encoder,
                )
            } else {
                self.single.encode(
                    &args.x.retain(),
                    &args.dt.retain(),
                    &args.b.retain(),
                    &args.c.retain(),
                    &args.d.retain(),
                    &args.z.retain(),
                    &args.state.retain(),
                    &args.y.retain(),
                    args.suffix_len as u32,
                    args.group_size,
                    args.state_size,
                    args.x_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.dt_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.cb_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.state_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                    args.channels as u32,
                    args.head_dim as u32,
                    compute_encoder,
                )
            }
        } else if mode == SSDPrefillMode::Sequential {
            self.sequential.encode(
                &args.x.retain(),
                &args.dt.retain(),
                &args.b.retain(),
                &args.c.retain(),
                &args.d.retain(),
                &args.z.retain(),
                &args.state.retain(),
                &args.y.retain(),
                args.suffix_len as u32,
                args.group_size,
                args.state_size,
                args.x_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                args.dt_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                args.cb_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                args.state_strides.iter().map(|x| *x as u32).collect::<Vec<_>>().as_slice(),
                args.channels as u32,
                args.head_dim as u32,
                compute_encoder,
            )
        }
    }
}
