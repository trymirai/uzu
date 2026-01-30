use metal::{MTLBuffer, MTLComputeCommandEncoder};
use objc2::__framework_prelude::ProtocolObject;
use crate::backends::metal::kernel::dsl::{SSDPrefillSequentialKernel, SSDPrefillSingle64Kernel, SSDPrefillSingleKernel};
use crate::backends::metal::{KernelDataType, MTLContext};

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
    pub suffix_len: u32,
    pub group_size: u32,
    pub state_size: u32,
    pub x_strides: [u32; 3],
    pub dt_strides: [u32; 2],
    pub cb_strides: [u32; 3],
    pub state_strides: [u32; 3],
    pub channels: u32,
    pub head_dim: u32,
}

pub struct SSDPrefillKernels {
    sequential: SSDPrefillSequentialKernel,
    single: SSDPrefillSingleKernel,
    single_64: SSDPrefillSingle64Kernel,
}

impl SSDPrefillKernels {
    pub fn new(
        mtl_context: &MTLContext,
        data_type: KernelDataType
    ) -> Self {
        let sequential = SSDPrefillSequentialKernel::new(mtl_context, data_type)
            .expect("Failed to create SSD prefill sequential kernel");
        let single = SSDPrefillSingleKernel::new(mtl_context, data_type)
            .expect("Failed to create SSD prefill single kernel");
        let single_64 = SSDPrefillSingle64Kernel::new(mtl_context, data_type)
            .expect("Failed to create SSD prefill single 64 kernel");
        Self { sequential, single, single_64 }
    }

    pub fn encode(
        &self,
        mode: SSDPrefillMode,
        args: &SSDPrefillArguments,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>
    ) {
        if mode == SSDPrefillMode::SinglePass {
            if args.state_size == 64 {
                self.single_64.encode(
                    &args.x,
                    &args.dt,
                    &args.b,
                    &args.c,
                    &args.d,
                    &args.z,
                    &args.state,
                    &args.y,
                    args.suffix_len,
                    args.group_size,
                    args.state_size,
                    args.x_strides.as_slice(),
                    args.dt_strides.as_slice(),
                    args.cb_strides.as_slice(),
                    args.state_strides.as_slice(),
                    args.channels,
                    args.head_dim,
                    encoder
                )
            } else {
                self.single.encode(
                    &args.x,
                    &args.dt,
                    &args.b,
                    &args.c,
                    &args.d,
                    &args.z,
                    &args.state,
                    &args.y,
                    args.suffix_len,
                    args.group_size,
                    args.state_size,
                    args.x_strides.as_slice(),
                    args.dt_strides.as_slice(),
                    args.cb_strides.as_slice(),
                    args.state_strides.as_slice(),
                    args.channels,
                    args.head_dim,
                    encoder
                )
            }
        } else if mode == SSDPrefillMode::Sequential {
            self.sequential.encode(
                &args.x,
                &args.dt,
                &args.b,
                &args.c,
                &args.d,
                &args.z,
                &args.state,
                &args.y,
                args.suffix_len,
                args.group_size,
                args.state_size,
                args.x_strides.as_slice(),
                args.dt_strides.as_slice(),
                args.cb_strides.as_slice(),
                args.state_strides.as_slice(),
                args.channels,
                args.head_dim,
                encoder
            )
        }
    }
}