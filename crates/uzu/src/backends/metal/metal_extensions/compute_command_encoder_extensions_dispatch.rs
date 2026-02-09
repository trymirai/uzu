use super::{
    compute_pipeline_state_extensions_threads::ComputePipelineStateThreads,
    device_extensions_features::{DeviceFeatures, Feature},
};
use crate::backends::metal::{
    MTLCommandEncoderExt, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLSize, ProtocolObject,
};

/// Extensions for MTLComputeCommandEncoder to simplify dispatch operations
pub trait ComputeEncoderDispatch {
    /// Dispatches a 1D compute grid, covering at least the specified size.
    fn dispatch_1d_covering(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: usize,
        threadgroup_width: Option<usize>,
    );

    /// Dispatches a 1D compute grid with exactly the specified size.
    fn dispatch_1d_exactly(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: usize,
        threadgroup_width: Option<usize>,
    );

    /// Dispatches a 1D compute grid, using the most efficient method available on the device.
    fn dispatch_1d(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: usize,
        threadgroup_width: Option<usize>,
    );

    /// Dispatches a 2D compute grid, covering at least the specified size.
    fn dispatch_2d_covering(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    );

    /// Dispatches a 2D compute grid with exactly the specified size.
    fn dispatch_2d_exactly(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    );

    /// Dispatches a 2D compute grid, using the most efficient method available on the device.
    fn dispatch_2d(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    );

    /// Dispatches a 3D compute grid, covering at least the specified size.
    fn dispatch_3d_covering(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    );

    /// Dispatches a 3D compute grid with exactly the specified size.
    fn dispatch_3d_exactly(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    );

    /// Dispatches a 3D compute grid, using the most efficient method available on the device.
    fn dispatch_3d(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    );
}

impl ComputeEncoderDispatch for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn dispatch_1d_covering(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: usize,
        threadgroup_width: Option<usize>,
    ) {
        let tg_width = threadgroup_width
            .unwrap_or(state.thread_execution_width() as usize);
        let tg_size = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };

        let count = MTLSize {
            width: (size + tg_width - 1) / tg_width,
            height: 1,
            depth: 1,
        };

        self.set_compute_pipeline_state(state);
        self.dispatch_threadgroups(count, tg_size);
    }

    fn dispatch_1d_exactly(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: usize,
        threadgroup_width: Option<usize>,
    ) {
        let tg_size = MTLSize {
            width: threadgroup_width
                .unwrap_or(state.thread_execution_width() as usize),
            height: 1,
            depth: 1,
        };

        self.set_compute_pipeline_state(state);
        self.dispatch_threads(
            MTLSize {
                width: size,
                height: 1,
                depth: 1,
            },
            tg_size,
        );
    }

    fn dispatch_1d(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: usize,
        threadgroup_width: Option<usize>,
    ) {
        let device = self.device();

        if device.supports_feature(Feature::NonUniformThreadgroups) {
            self.dispatch_1d_exactly(state, size, threadgroup_width);
        } else {
            self.dispatch_1d_covering(state, size, threadgroup_width);
        }
    }

    fn dispatch_2d_covering(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    ) {
        let tg_size =
            threadgroup_size.unwrap_or_else(|| state.max_2d_threadgroup_size());

        let count = MTLSize {
            width: (size.width + tg_size.width - 1) / tg_size.width,
            height: (size.height + tg_size.height - 1) / tg_size.height,
            depth: 1,
        };

        self.set_compute_pipeline_state(state);
        self.dispatch_threadgroups(count, tg_size);
    }

    fn dispatch_2d_exactly(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    ) {
        let tg_size =
            threadgroup_size.unwrap_or_else(|| state.max_2d_threadgroup_size());

        self.set_compute_pipeline_state(state);
        self.dispatch_threads(size, tg_size);
    }

    fn dispatch_2d(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    ) {
        let device = self.device();

        if device.supports_feature(Feature::NonUniformThreadgroups) {
            self.dispatch_2d_exactly(state, size, threadgroup_size);
        } else {
            self.dispatch_2d_covering(state, size, threadgroup_size);
        }
    }

    fn dispatch_3d_covering(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    ) {
        let tg_size =
            threadgroup_size.unwrap_or_else(|| state.max_2d_threadgroup_size());

        let count = MTLSize {
            width: (size.width + tg_size.width - 1) / tg_size.width,
            height: (size.height + tg_size.height - 1) / tg_size.height,
            depth: (size.depth + tg_size.depth - 1) / tg_size.depth,
        };

        self.set_compute_pipeline_state(state);
        self.dispatch_threadgroups(count, tg_size);
    }

    fn dispatch_3d_exactly(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    ) {
        let tg_size =
            threadgroup_size.unwrap_or_else(|| state.max_2d_threadgroup_size());

        self.set_compute_pipeline_state(state);
        self.dispatch_threads(size, tg_size);
    }

    fn dispatch_3d(
        &self,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        size: MTLSize,
        threadgroup_size: Option<MTLSize>,
    ) {
        let device = self.device();

        if device.supports_feature(Feature::NonUniformThreadgroups) {
            self.dispatch_3d_exactly(state, size, threadgroup_size);
        } else {
            self.dispatch_3d_covering(state, size, threadgroup_size);
        }
    }
}
