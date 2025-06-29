use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputeCommandEncoderRef as MTLComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLResourceOptions,
    MTLSize,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;
use thiserror::Error;

use super::{
    super::MTLError, KernelDataType, MTLContext,
    metal_extensions::ComputeEncoderDispatch,
};
use crate::{Array, backends::metal::forward_pass::ForwardPassState};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Swap {
    pub source: i32,
    pub destination: i32,
}

pub struct KVLayerData {
    pub key_buffer: MTLBuffer,
    pub key_shape: [usize; 3],
    pub value_buffer: MTLBuffer,
    pub value_shape: [usize; 3],
}

pub struct KVCacheUpdate {
    pipeline_state: MTLComputePipelineState,
    indices_buffer: MTLBuffer,
    max_sequence_length: usize,
}

#[derive(Debug, Error)]
pub enum KVCacheUpdateError {
    #[error("Source and destination indices length mismatch")]
    IndicesCountMismatch,
    #[error("Shape mismatch between key and value tensors")]
    ShapeMismatch,
    #[error("Unsupported data type")]
    UnsupportedDataType,
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Swap count exceeds maximum sequence length")]
    MaxSequenceLengthExceeded,
    #[error("Buffer creation failed")]
    BufferCreationFailed,
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

impl KVCacheUpdate {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        max_sequence_length: usize,
    ) -> Result<Self, KVCacheUpdateError> {
        // Format function name with the appropriate suffix
        let function_name =
            format!("updateKVCache_{}", data_type.function_name_suffix());

        // Use MetalContext's compute_pipeline_state_with_reflection method
        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)
            .map_err(|e| KVCacheUpdateError::MetalError(e))?;

        let indices_buffer = context.device.new_buffer(
            (max_sequence_length * size_of::<Swap>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            pipeline_state,
            indices_buffer,
            max_sequence_length,
        })
    }

    pub fn encode(
        &self,
        in_place_data: &[KVLayerData],
        source_indices: &[usize],
        destination_indices: &[usize],
        command_buffer: &MTLCommandBuffer,
    ) -> Result<(), KVCacheUpdateError> {
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_label("KV Cache Update");

        self.encode_with_encoder(
            in_place_data,
            source_indices,
            destination_indices,
            &compute_encoder,
        )?;

        compute_encoder.end_encoding();
        Ok(())
    }

    /// Encode the KV cache update operation using a provided compute encoder
    pub fn encode_with_encoder(
        &self,
        in_place_data: &[KVLayerData],
        source_indices: &[usize],
        destination_indices: &[usize],
        compute_encoder: &MTLComputeCommandEncoderRef,
    ) -> Result<(), KVCacheUpdateError> {
        if source_indices.len() != destination_indices.len() {
            return Err(KVCacheUpdateError::IndicesCountMismatch);
        }

        let swaps = create_swaps_direct(source_indices, destination_indices);
        if swaps.len() > self.max_sequence_length {
            return Err(KVCacheUpdateError::MaxSequenceLengthExceeded);
        }

        let indices_ptr = self.indices_buffer.contents() as *mut Swap;
        unsafe {
            for (i, swap) in swaps.iter().enumerate() {
                *indices_ptr.add(i) = *swap;
            }
        }

        for layer_data in in_place_data {
            if layer_data.key_shape != layer_data.value_shape {
                return Err(KVCacheUpdateError::ShapeMismatch);
            }

            let [num_heads, max_sequence_length, head_dim] =
                layer_data.key_shape;

            // Set buffers and parameters
            compute_encoder.set_buffer(0, Some(&layer_data.key_buffer), 0);
            compute_encoder.set_buffer(1, Some(&layer_data.value_buffer), 0);
            compute_encoder.set_buffer(2, Some(&self.indices_buffer), 0);
            compute_encoder.set_bytes(
                3,
                size_of::<i32>() as u64,
                &(swaps.len() as i32) as *const _ as *const std::ffi::c_void,
            );
            compute_encoder.set_bytes(
                4,
                size_of::<i32>() as u64,
                &(num_heads as i32) as *const _ as *const std::ffi::c_void,
            );
            compute_encoder.set_bytes(
                5,
                size_of::<i32>() as u64,
                &(max_sequence_length as i32) as *const _
                    as *const std::ffi::c_void,
            );
            compute_encoder.set_bytes(
                6,
                size_of::<i32>() as u64,
                &(head_dim as i32) as *const _ as *const std::ffi::c_void,
            );

            compute_encoder.dispatch_2d_exactly(
                &self.pipeline_state,
                MTLSize {
                    width: num_heads as u64,
                    height: head_dim as u64,
                    depth: 1,
                },
                None,
            );
        }

        Ok(())
    }
}

pub fn create_swaps_direct(
    source_indices: &[usize],
    destination_indices: &[usize],
) -> Vec<Swap> {
    if source_indices.len() != destination_indices.len() {
        return Vec::new();
    }

    let mut swaps = Vec::with_capacity(source_indices.len());

    for (&src, &dst) in source_indices.iter().zip(destination_indices.iter()) {
        if src != dst {
            swaps.push(Swap {
                source: src as i32,
                destination: dst as i32,
            });
        }
    }

    swaps
}

/// Original implementation using cycles - kept for reference
/// This function generates the minimum set of swaps needed to move elements
/// from source positions to destination positions.
pub fn create_swaps(
    source_indices: &[usize],
    destination_indices: &[usize],
) -> Vec<Swap> {
    // The two arrays must be of the same length.
    if source_indices.len() != destination_indices.len() {
        return Vec::new();
    }

    // Build mappings between source and destination indices
    let mut forward_mapping = std::collections::HashMap::new();
    let mut reverse_mapping = std::collections::HashMap::new();

    for (&s, &d) in source_indices.iter().zip(destination_indices.iter()) {
        if s != d {
            forward_mapping.insert(s, d);
            reverse_mapping.insert(d, s);
        }
    }

    let mut swaps = Vec::new();
    let mut visited = std::collections::HashSet::new();

    // Process keys in sorted order
    let all_keys: std::collections::HashSet<_> =
        forward_mapping.keys().chain(reverse_mapping.keys()).copied().collect();

    let all_keys_vec: Vec<usize> = all_keys.into_iter().collect();
    for &s in &all_keys_vec {
        if visited.contains(&s) {
            continue;
        }

        visited.insert(s);
        let mut cycle = vec![s];
        let mut iterator = s;

        while let Some(&next) = forward_mapping.get(&iterator) {
            iterator = next;
            if visited.contains(&iterator) {
                break;
            }
            visited.insert(iterator);
            cycle.push(iterator);
        }

        // Reverse direction
        iterator = s;
        while let Some(&next) = reverse_mapping.get(&iterator) {
            iterator = next;
            if visited.contains(&iterator) {
                break;
            }
            visited.insert(iterator);
            cycle.push(iterator);
        }

        let first = cycle[0];
        for &next in cycle.iter().skip(1) {
            swaps.push(Swap {
                source: first as i32,
                destination: next as i32,
            });
        }
    }

    swaps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_swaps() {
        let sources = [0, 2, 4];
        let destinations = [1, 3, 5];
        let swaps = create_swaps_direct(&sources, &destinations);
        assert_eq!(swaps.len(), 3);
        assert_eq!(swaps[0].source, 0);
        assert_eq!(swaps[0].destination, 1);
        assert_eq!(swaps[1].source, 2);
        assert_eq!(swaps[1].destination, 3);
        assert_eq!(swaps[2].source, 4);
        assert_eq!(swaps[2].destination, 5);
    }
}
