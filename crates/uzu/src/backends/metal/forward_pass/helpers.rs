use std::cell::RefCell;

use metal::CommandBufferRef;

use super::{
    ArrayId, CacheLayers, ForwardPassState, KVCacheLayer, KVCacheLayerState,
};
use crate::{
    Array,
    backends::metal::{
        MetalArray,
        kernel::{KVCacheUpdate, kv_cache_update::KVLayerData},
    },
};

pub fn encode_copy_array(
    state: &ForwardPassState,
    command_buffer: &CommandBufferRef,
    source_array_id: ArrayId,
    destination_array: RefCell<MetalArray>,
) {
    let arrays = state.arrays(&[source_array_id]);
    let source_cell = &arrays[0];
    let mut src_borrow = source_cell.borrow_mut();
    let mut dst_borrow = destination_array.borrow_mut();

    let src_buf = unsafe { src_borrow.mtl_buffer().clone() };
    let dst_buf = unsafe { dst_borrow.mtl_buffer().clone() };

    let copy_size_bytes = dst_borrow.size_in_bytes() as u64;
    debug_assert_eq!(dst_borrow.size_in_bytes(), src_borrow.size_in_bytes());

    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.copy_from_buffer(&src_buf, 0, &dst_buf, 0, copy_size_bytes);
    blit_encoder.end_encoding();
}

pub fn update_cache_layers_after_acceptance(
    cache_layers: &mut CacheLayers,
    accepted_suffix_indices: &[usize],
    suffix_start: Option<usize>,
    command_buffer: &CommandBufferRef,
    kv_cache_update: &KVCacheUpdate,
) {
    for layer in cache_layers.data.iter_mut() {
        if let Some(layer) = layer.as_transformer_mut() {
            update_kv_cache_layer_after_acceptance(
                layer,
                accepted_suffix_indices,
                suffix_start,
                command_buffer,
                kv_cache_update,
            );
        }
    }
}

pub fn update_kv_cache_layer_after_acceptance(
    layer: &mut KVCacheLayer,
    accepted_suffix_indices: &[usize],
    suffix_start: Option<usize>,
    command_buffer: &CommandBufferRef,
    kv_cache_update: &KVCacheUpdate,
) {
    match &layer.state {
        KVCacheLayerState::Full {
            prefix_len,
        } => {
            if accepted_suffix_indices.is_empty() {
                return;
            }

            // Absolute positions of the *source* rows.
            let source_indices: Vec<usize> = accepted_suffix_indices
                .iter()
                .map(|i| i + suffix_start.unwrap_or(*prefix_len))
                .collect();

            // Absolute positions of the *destination* rows.
            let destination_indices: Vec<usize> = (*prefix_len
                ..*prefix_len + accepted_suffix_indices.len())
                .collect();

            scatter_if_required(
                layer,
                &source_indices,
                &destination_indices,
                command_buffer,
                kv_cache_update,
            );
        },

        KVCacheLayerState::Windowed {
            ring_offset,
            ring_length,
            window_length,
        } => {
            let suffix_indices: Vec<usize> =
                if accepted_suffix_indices.is_empty() {
                    vec![0]
                } else {
                    accepted_suffix_indices.to_vec()
                };

            let source_indices: Vec<usize> =
                suffix_indices.iter().map(|i| i + *window_length).collect();

            let mut destination_indices =
                Vec::with_capacity(suffix_indices.len());

            for i in 0..suffix_indices.len() {
                destination_indices
                    .push((*ring_length + *ring_offset + i) % *window_length);
            }

            scatter_if_required(
                layer,
                &source_indices,
                &destination_indices,
                command_buffer,
                kv_cache_update,
            );
        },
    }
}

fn scatter_if_required(
    layer: &KVCacheLayer,
    source_indices: &[usize],
    destination_indices: &[usize],
    command_buffer: &CommandBufferRef,
    kv_cache_update: &KVCacheUpdate,
) {
    if source_indices == destination_indices {
        return;
    }

    let key_buffer = {
        let mut k = layer.keys.borrow_mut();
        unsafe { k.mtl_buffer() }.clone()
    };
    let value_buffer = {
        let mut v = layer.values.borrow_mut();
        unsafe { v.mtl_buffer() }.clone()
    };

    let k_shape = layer.keys.borrow().shape().to_vec();
    let v_shape = layer.values.borrow().shape().to_vec();

    let layer_data = KVLayerData {
        key_buffer,
        key_shape: [k_shape[0], k_shape[1], k_shape[2]],
        value_buffer,
        value_shape: [v_shape[0], v_shape[1], v_shape[2]],
    };

    let command_buffer = command_buffer.to_owned();
    let _ = kv_cache_update.encode(
        &[layer_data],
        source_indices,
        destination_indices,
        &command_buffer,
    );
}
