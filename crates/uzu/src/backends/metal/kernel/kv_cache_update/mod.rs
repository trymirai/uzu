use std::{mem::size_of, ptr::NonNull};

use thiserror::Error;

use super::{
    super::MTLError, KernelDataType, MTLContext,
    metal_extensions::ComputeEncoderDispatch,
};
use crate::backends::metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandEncoderExt,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDeviceExt,
    MTLResourceOptions, MTLSize, ProtocolObject, Retained,
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Swap {
    pub source: i32,
    pub destination: i32,
}

pub struct KVLayerData {
    pub key_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub key_shape: [usize; 3],
    pub value_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub value_shape: [usize; 3],
}

pub struct KVCacheUpdate {
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    indices_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
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

        let pipeline_state = context
            .compute_pipeline_state(&function_name, None)
            .map_err(|e| KVCacheUpdateError::MetalError(e))?;

        let indices_buffer = context
            .device
            .new_buffer(
                max_sequence_length * size_of::<Swap>(),
                MTLResourceOptions::STORAGE_MODE_SHARED,
            )
            .ok_or(KVCacheUpdateError::BufferCreationFailed)?;

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
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<(), KVCacheUpdateError> {
        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        compute_encoder.set_label(Some("KV Cache Update"));

        self.encode_with_encoder(
            in_place_data,
            source_indices,
            destination_indices,
            &*compute_encoder,
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
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) -> Result<(), KVCacheUpdateError> {
        if source_indices.len() != destination_indices.len() {
            return Err(KVCacheUpdateError::IndicesCountMismatch);
        }

        let swaps = create_swaps_direct(source_indices, destination_indices);
        if swaps.len() > self.max_sequence_length {
            return Err(KVCacheUpdateError::MaxSequenceLengthExceeded);
        }

        // For small swap counts, use set_bytes to avoid shared buffer race in async mode.
        // Metal's set_bytes limit is 4KB; 32 swaps = 256 bytes, well under limit.
        let use_inline_bytes = swaps.len() <= 32;

        if !use_inline_bytes {
            let indices_ptr =
                self.indices_buffer.contents().as_ptr() as *mut Swap;
            unsafe {
                for (i, swap) in swaps.iter().enumerate() {
                    *indices_ptr.add(i) = *swap;
                }
            }
        }

        for layer_data in in_place_data {
            if layer_data.key_shape != layer_data.value_shape {
                return Err(KVCacheUpdateError::ShapeMismatch);
            }

            let [num_heads, max_sequence_length, head_dim] =
                layer_data.key_shape;

            // Set buffers and parameters
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(&layer_data.key_buffer),
                0,
                0,
            );
            MTLComputeCommandEncoder::set_buffer(
                compute_encoder,
                Some(&layer_data.value_buffer),
                0,
                1,
            );
            if use_inline_bytes {
                unsafe {
                    MTLComputeCommandEncoder::set_bytes(
                        compute_encoder,
                        NonNull::new_unchecked(swaps.as_ptr() as *mut _),
                        swaps.len() * size_of::<Swap>(),
                        2,
                    );
                }
            } else {
                MTLComputeCommandEncoder::set_buffer(
                    compute_encoder,
                    Some(&self.indices_buffer),
                    0,
                    2,
                );
            }
            unsafe {
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &(swaps.len() as i32) as *const _ as *mut _,
                    ),
                    size_of::<i32>(),
                    3,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &(num_heads as i32) as *const _ as *mut _,
                    ),
                    size_of::<i32>(),
                    4,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &(max_sequence_length as i32) as *const _ as *mut _,
                    ),
                    size_of::<i32>(),
                    5,
                );
                MTLComputeCommandEncoder::set_bytes(
                    compute_encoder,
                    NonNull::new_unchecked(
                        &(head_dim as i32) as *const _ as *mut _,
                    ),
                    size_of::<i32>(),
                    6,
                );
            }

            compute_encoder.dispatch_2d_exactly(
                &self.pipeline_state,
                MTLSize {
                    width: num_heads as usize,
                    height: head_dim as usize,
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
