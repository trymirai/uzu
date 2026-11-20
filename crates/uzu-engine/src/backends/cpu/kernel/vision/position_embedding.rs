use half::bf16;
use uzu_engine_macros::kernel;

use crate::{data_type::DataTypeRepr, engine::language_model::vision::patch_grid::vision_patch_coordinates};

/// `index`-th of `count` points spread evenly over `[0, side - 1]`, computed as `torch.linspace` does,
/// see `vision_linspace` in `metal/kernel/vision/position_embedding.metal`.
fn vision_linspace(
    index: u32,
    count: u32,
    side: u32,
) -> f32 {
    if count == 1 {
        return 0.0;
    }

    let end = (side - 1) as f32;
    let step = end / (count - 1) as f32;
    if index < count / 2 {
        index as f32 * step
    } else {
        end - (count - index - 1) as f32 * step
    }
}

/// Adds the learned position embedding, bilinearly resized to the patch grid, to the patch embeddings,
/// see `metal/kernel/vision/position_embedding.metal`.
#[kernel(VisionPositionEmbedding)]
#[variants(T, bf16)]
pub fn vision_position_embedding<T: DataTypeRepr>(
    table: *const T,
    stream: *mut T,
    side: u32,
    grid_width: u32,
    grid_height: u32,
    merge_size: u32,
    model_dim: u32,
    length: u32,
) {
    // Every weight, product and partial sum is rounded to T, as HF does, to match it bit for bit
    let round = |value: f32| T::from(value).unwrap().to_f32().unwrap();

    for position in 0..length {
        let row = position / model_dim;
        let channel = position % model_dim;

        let (patch_x, patch_y) = vision_patch_coordinates(row, grid_width, merge_size);
        let y = vision_linspace(patch_y, grid_height, side);
        let x = vision_linspace(patch_x, grid_width, side);
        let y_floor = y as u32;
        let x_floor = x as u32;
        let y_ceil = (y_floor + 1).min(side - 1);
        let x_ceil = (x_floor + 1).min(side - 1);

        let corners =
            [y_floor * side + x_floor, y_floor * side + x_ceil, y_ceil * side + x_floor, y_ceil * side + x_ceil];

        let dy = y - y_floor as f32;
        let dx = x - x_floor as f32;
        let weights = [(1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx, dy * (1.0 - dx), dy * dx];

        let mut embedding = 0.0;
        for (corner, weight) in corners.into_iter().zip(weights) {
            let table_idx = (corner * model_dim + channel) as usize;
            let table_value = unsafe { (*table.add(table_idx)).to_f32().unwrap() };
            let value = round(table_value * round(weight));
            embedding = round(embedding + value);
        }

        unsafe {
            let stream_value = stream.add(position as usize);
            *stream_value = T::from((*stream_value).to_f32().unwrap() + embedding).unwrap();
        }
    }
}
