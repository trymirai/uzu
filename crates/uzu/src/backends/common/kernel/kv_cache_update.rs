use std::mem::size_of;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer, Kernels, gpu_types::Swap, kernel::KVCacheUpdateKernel},
};

pub struct KVLayerData<B: Backend> {
    pub key_buffer: B::NativeBuffer,
    pub key_shape: [usize; 3],
    pub value_buffer: B::NativeBuffer,
    pub value_shape: [usize; 3],
}

pub struct KVCacheUpdate<B: Backend> {
    kernel: <B::Kernels as Kernels>::KVCacheUpdateKernel,
    max_sequence_length: usize,
}

#[derive(Debug, Error)]
pub enum KVCacheUpdateError<B: Backend> {
    #[error("Source and destination indices length mismatch")]
    IndicesCountMismatch,
    #[error("Shape mismatch between key and value tensors")]
    ShapeMismatch,
    #[error("Unsupported data type")]
    UnsupportedDataType,
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Swap count exceeds maximum sequence length")]
    MaxSequenceLengthExceeded,
    #[error("Buffer creation failed")]
    BufferCreationFailed,
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

impl<B: Backend> KVCacheUpdate<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        max_sequence_length: usize,
    ) -> Result<Self, KVCacheUpdateError<B>> {
        let kernel = <B::Kernels as Kernels>::KVCacheUpdateKernel::new(context, data_type)
            .map_err(KVCacheUpdateError::BackendError)?;

        Ok(Self {
            kernel,
            max_sequence_length,
        })
    }

    pub fn encode(
        &self,
        in_place_data: &[KVLayerData<B>],
        source_indices: &[usize],
        destination_indices: &[usize],
        command_buffer: &B::CommandBuffer,
    ) -> Result<(), KVCacheUpdateError<B>> {
        command_buffer.with_compute_encoder(|encoder| {
            self.encode_with_encoder(in_place_data, source_indices, destination_indices, encoder)
        })
    }

    /// Encode the KV cache update operation using a provided compute encoder
    pub fn encode_with_encoder(
        &self,
        in_place_data: &[KVLayerData<B>],
        source_indices: &[usize],
        destination_indices: &[usize],
        encoder: &B::ComputeEncoder,
    ) -> Result<(), KVCacheUpdateError<B>> {
        if source_indices.len() != destination_indices.len() {
            return Err(KVCacheUpdateError::IndicesCountMismatch);
        }

        let swaps = create_swaps_direct(source_indices, destination_indices);
        if swaps.len() > self.max_sequence_length {
            return Err(KVCacheUpdateError::MaxSequenceLengthExceeded);
        }
        let max_inline_swaps = (B::MAX_INLINE_BYTES / size_of::<Swap>()).max(1);

        for layer_data in in_place_data {
            if layer_data.key_shape != layer_data.value_shape {
                return Err(KVCacheUpdateError::ShapeMismatch);
            }

            let [num_heads, max_sequence_length, head_dim] = layer_data.key_shape;

            // non-inline is not supported yet (and is broken anyways due to a data race)
            for swaps_chunk in swaps.chunks(max_inline_swaps) {
                self.kernel.encode(
                    &layer_data.key_buffer,
                    &layer_data.value_buffer,
                    swaps_chunk,
                    swaps_chunk.len() as u32,
                    num_heads as u32,
                    max_sequence_length as u32,
                    head_dim as u32,
                    encoder,
                );
            }
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
                source: src as u32,
                destination: dst as u32,
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
