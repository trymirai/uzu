use metal::{MTLComputePipelineState, MTLSize};
use objc2::runtime::ProtocolObject;

/// Extensions for ComputePipelineState to provide optimized threadgroup sizes
pub trait ComputePipelineStateThreads {
    /// Returns a threadgroup size based on the pipeline's thread execution width.
    ///
    /// This size is optimized for 1D compute kernels.
    fn execution_width_threadgroup_size(&self) -> MTLSize;

    /// Returns the maximum threadgroup size for a 1D compute kernel.
    ///
    /// This size uses the maximum total threads per threadgroup.
    fn max_1d_threadgroup_size(&self) -> MTLSize;

    /// Returns the maximum threadgroup size for a 2D compute kernel.
    ///
    /// This size balances the thread execution width and the maximum total threads per threadgroup.
    fn max_2d_threadgroup_size(&self) -> MTLSize;

    /// Calculates the maximum threadgroup size for a 3D compute kernel with a specified depth.
    ///
    /// - Parameter depth: The desired depth of the threadgroup.
    /// - Returns: The maximum threadgroup size that can be used for a 3D compute kernel.
    fn max_3d_threadgroup_size(
        &self,
        depth: usize,
    ) -> MTLSize;
}

impl ComputePipelineStateThreads for ProtocolObject<dyn MTLComputePipelineState> {
    fn execution_width_threadgroup_size(&self) -> MTLSize {
        let w = self.thread_execution_width();

        MTLSize {
            width: w,
            height: 1,
            depth: 1,
        }
    }

    fn max_1d_threadgroup_size(&self) -> MTLSize {
        let w = self.max_total_threads_per_threadgroup();

        MTLSize {
            width: w,
            height: 1,
            depth: 1,
        }
    }

    fn max_2d_threadgroup_size(&self) -> MTLSize {
        let w = self.thread_execution_width();
        let h = self.max_total_threads_per_threadgroup() / w;

        MTLSize {
            width: w,
            height: h,
            depth: 1,
        }
    }

    fn max_3d_threadgroup_size(
        &self,
        depth: usize,
    ) -> MTLSize {
        let w = self.thread_execution_width() / depth;
        let h = self.max_total_threads_per_threadgroup() / w;

        MTLSize {
            width: w,
            height: h,
            depth,
        }
    }
}
