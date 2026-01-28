use super::{SSMKernelError, fn_suffix};
use crate::backends::metal::{
    KernelDataType, MTLBuffer, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLContext, MTLSize, ProtocolObject, Retained,
    metal_extensions::ComputeEncoderSetValue,
};

pub struct SSDUpdateKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct SSDUpdateArguments<'a> {
    pub x: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0)  (b, h, dh)
    pub dt: &'a ProtocolObject<dyn MTLBuffer>, // buffer(1)  (b, h) - raw dt values
    pub b: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(2)  (b, g, n)
    pub c: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(3)  (b, g, n)
    pub d: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(4)  (h)
    pub z: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(5)  (b, d)
    pub state: &'a ProtocolObject<dyn MTLBuffer>, // buffer(6)  (b, h, dh, n)
    pub y: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(7)
    pub next_state: &'a ProtocolObject<dyn MTLBuffer>, // buffer(8)
    pub group_size: i32,                       // buffer(9)
    pub state_size: i32,                       // buffer(10)
    pub x_strides: [usize; 3],                 // buffer(11)
    pub dt_strides: [usize; 2],                // buffer(12)
    pub cb_strides: [usize; 3],                // buffer(13)
    pub state_strides: [usize; 4],             // buffer(14)
    pub b_size: usize,
    pub h_size: usize,
    pub dh_size: usize,
}

impl SSDUpdateKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, SSMKernelError> {
        let fn_name = format!("ssd_update_kernel_{}", fn_suffix(data_type));
        let pipeline = context
            .compute_pipeline_state(&fn_name, None)
            .map_err(SSMKernelError::MetalError)?;
        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: SSDUpdateArguments,
    ) -> Result<(), SSMKernelError> {
        compute_encoder.set_compute_pipeline_state(&self.pipeline);

        compute_encoder.set_buffer(Some(args.x), 0, 0);
        compute_encoder.set_buffer(Some(args.dt), 0, 1);
        compute_encoder.set_buffer(Some(args.b), 0, 2);
        compute_encoder.set_buffer(Some(args.c), 0, 3);
        compute_encoder.set_buffer(Some(args.d), 0, 4);
        compute_encoder.set_buffer(Some(args.z), 0, 5);
        compute_encoder.set_buffer(Some(args.state), 0, 6);
        compute_encoder.set_buffer(Some(args.y), 0, 7);
        compute_encoder.set_buffer(Some(args.next_state), 0, 8);

        compute_encoder.set_value(&args.group_size, 9);
        compute_encoder.set_value(&args.state_size, 10);
        compute_encoder.set_value(&args.x_strides, 11);
        compute_encoder.set_value(&args.dt_strides, 12);
        compute_encoder.set_value(&args.cb_strides, 13);
        compute_encoder.set_value(&args.state_strides, 14);

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 32,
            depth: 1,
        };
        let total_threads = MTLSize {
            width: args.b_size,
            height: args.h_size,
            depth: args.dh_size,
        };
        compute_encoder
            .dispatch_threads(total_threads, threads_per_threadgroup);
        Ok(())
    }
}
