use image::{DynamicImage, imageops::FilterType};

use super::config::{CanvasConfig, VisionConfig};

#[derive(Debug, Clone)]
pub struct PreprocessedVision {
    /// Flattened patch embeddings: [total_patches * patch_size * patch_size * 3]
    pub patches: Vec<f32>,
    /// Patch grids for thumbnail and each slice: [(1, grid_h, grid_w)]
    pub grids: Vec<(usize, usize, usize)>,
    /// Slice grid rows and columns: Some((rows, cols)) or None
    pub slice_grid: Option<(usize, usize)>,
}

pub fn ensure_divide(
    length: f64,
    divisor: usize,
) -> usize {
    let div = divisor as f64;
    let rounded = (length / div).round() as usize * divisor;
    rounded.max(divisor)
}

pub fn find_best_resize(
    height: f64,
    width: f64,
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut h, mut w) = (height, width);
    let scale_area = (scale_resolution * scale_resolution) as f64;
    if h * w > scale_area || allow_upscale {
        let aspect = w / h;
        h = (scale_resolution as f64 / aspect.sqrt()).floor();
        w = (h * aspect).floor();
    }
    let best_w = ensure_divide(w, patch_size * 4);
    let best_h = ensure_divide(h, patch_size * 4);
    (best_h, best_w)
}

pub fn sliced_grid(
    height: usize,
    width: usize,
    max_slice_nums: usize,
    scale_resolution: usize,
) -> Option<(usize, usize)> {
    let log_ratio = (width as f64 / height as f64).ln();
    let ratio = (width * height) as f64 / (scale_resolution * scale_resolution) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1, 1);
    let mut min_error = f64::INFINITY;
    let candidates = [multiple.saturating_sub(1), multiple, multiple + 1];
    for num_slices in candidates {
        if num_slices <= 1 || num_slices > max_slice_nums {
            continue;
        }
        for num_rows in 1..=num_slices {
            if num_slices % num_rows == 0 {
                let num_cols = num_slices / num_rows;
                let error = (log_ratio - (num_cols as f64 / num_rows as f64).ln()).abs();
                if error < min_error {
                    best = (num_rows, num_cols);
                    min_error = error;
                } else if (error - min_error).abs() < 1e-9 && num_rows > best.0 {
                    best = (num_rows, num_cols);
                }
            }
        }
    }
    Some(best)
}

pub fn refine_size(
    height: usize,
    width: usize,
    grid: (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (rows, cols) = grid;
    let refine_w = ensure_divide(width as f64, cols);
    let refine_h = ensure_divide(height as f64, rows);
    let (best_h, best_w) = find_best_resize(
        refine_h as f64 / rows as f64,
        refine_w as f64 / cols as f64,
        scale_resolution,
        patch_size,
        allow_upscale,
    );
    (best_h * rows, best_w * cols)
}

#[inline]
pub fn extract_patches_from_region(
    raw_pixels: &[u8],
    img_width: usize,
    x_offset: usize,
    y_offset: usize,
    region_width: usize,
    region_height: usize,
    patch_size: usize,
    out: &mut Vec<f32>,
) -> (usize, usize, usize) {
    assert!(region_height.is_multiple_of(patch_size) && region_width.is_multiple_of(patch_size));
    let grid_h = region_height / patch_size;
    let grid_w = region_width / patch_size;
    let scale = 1.0f32 / 127.5f32; // (val / 255.0 - 0.5) / 0.5 == val / 127.5 - 1.0

    for gh in 0..grid_h {
        let base_y = y_offset + gh * patch_size;
        for gw in 0..grid_w {
            let base_x = x_offset + gw * patch_size;
            for py in 0..patch_size {
                let y = base_y + py;
                let row_start = (y * img_width + base_x) * 3;
                let row = &raw_pixels[row_start..row_start + patch_size * 3];
                for px in 0..patch_size {
                    let p = px * 3;
                    out.push(row[p] as f32 * scale - 1.0);
                    out.push(row[p + 1] as f32 * scale - 1.0);
                    out.push(row[p + 2] as f32 * scale - 1.0);
                }
            }
        }
    }

    (1, grid_h, grid_w)
}

pub fn preprocess_image(
    image: &DynamicImage,
    vision_config: &VisionConfig,
    _canvas_config: &CanvasConfig,
    max_slice_nums: usize,
    scale_resolution: usize,
) -> PreprocessedVision {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);
    let patch_size = vision_config.patch_size;
    let grid = sliced_grid(h, w, max_slice_nums, scale_resolution);

    let (thumb_h, thumb_w) = find_best_resize(h as f64, w as f64, scale_resolution, patch_size, grid.is_none());

    let (ref_h, ref_w, total_patches) = if let Some((rows, cols)) = grid {
        let (rh, rw) = refine_size(h, w, (rows, cols), scale_resolution, patch_size, true);
        let n_thumb = (thumb_h / patch_size) * (thumb_w / patch_size);
        let n_slices = (rh / patch_size) * (rw / patch_size);
        (rh, rw, n_thumb + n_slices)
    } else {
        (0, 0, (thumb_h / patch_size) * (thumb_w / patch_size))
    };

    // Resize thumbnail and full reference slices concurrently across threads
    let (thumb_img, ref_img) = std::thread::scope(|s| {
        let h_thumb = s.spawn(|| {
            image::imageops::resize(
                &rgb,
                thumb_w as u32,
                thumb_h as u32,
                FilterType::CatmullRom, // Bicubic
            )
        });
        let h_ref = s.spawn(|| {
            if grid.is_some() {
                Some(image::imageops::resize(&rgb, ref_w as u32, ref_h as u32, FilterType::CatmullRom))
            } else {
                None
            }
        });
        (h_thumb.join().unwrap(), h_ref.join().unwrap())
    });

    let patch_dim = patch_size * patch_size * 3;
    let mut all_patches = Vec::with_capacity(total_patches * patch_dim);

    let thumb_grid =
        extract_patches_from_region(thumb_img.as_raw(), thumb_w, 0, 0, thumb_w, thumb_h, patch_size, &mut all_patches);
    let mut all_grids = vec![thumb_grid];

    // 2. Slices if applicable (Zero-copy region extraction directly from ref_img)
    if let (Some((rows, cols)), Some(ref_img)) = (grid, ref_img) {
        let tile_h = ref_h / rows;
        let tile_w = ref_w / cols;
        let raw = ref_img.as_raw();
        for r in 0..rows {
            for c in 0..cols {
                let tile_grid = extract_patches_from_region(
                    raw,
                    ref_w,
                    c * tile_w,
                    r * tile_h,
                    tile_w,
                    tile_h,
                    patch_size,
                    &mut all_patches,
                );
                all_grids.push(tile_grid);
            }
        }
    }

    PreprocessedVision {
        patches: all_patches,
        grids: all_grids,
        slice_grid: grid,
    }
}

pub fn nearest_position_ids(
    height: usize,
    width: usize,
    num_patches_per_side: usize,
) -> Vec<usize> {
    let step = 1.0 / num_patches_per_side as f64;
    let count = ((1.0 - step) / step).ceil() as usize;
    let mut boundaries = Vec::with_capacity(count);
    for i in 0..count {
        boundaries.push(step + i as f64 * step);
    }

    let bucket = |i: usize, n: usize| -> usize {
        let coord = i as f64 / n as f64;
        let mut b = 0;
        for &boundary in &boundaries {
            if boundary <= coord {
                b += 1;
            }
        }
        b
    };

    let rows: Vec<usize> = (0..height).map(|r| bucket(r, height)).collect();
    let cols: Vec<usize> = (0..width).map(|c| bucket(c, width)).collect();
    let mut out = Vec::with_capacity(height * width);
    for &r in &rows {
        for &c in &cols {
            out.push(r * num_patches_per_side + c);
        }
    }
    out
}

pub fn window_index(
    height: usize,
    width: usize,
    window: usize,
    offset: usize,
) -> Vec<usize> {
    let mut index = Vec::with_capacity(height * width);
    let h_windows = height / window;
    let w_windows = width / window;
    for wi in 0..h_windows {
        for wj in 0..w_windows {
            for r in 0..window {
                for c in 0..window {
                    index.push(offset + (wi * window + r) * width + (wj * window + c));
                }
            }
        }
    }
    index
}

pub const IMAGE_PAD: u32 = 248056;
pub const IMAGE_START: u32 = 248078;
pub const IMAGE_END: u32 = 248079;
pub const SLICE_START: u32 = 248088;
pub const SLICE_END: u32 = 248089;
pub const IMAGE_ID_START: u32 = 248090;
pub const IMAGE_ID_END: u32 = 248091;
pub const NEWLINE: u32 = 198;

pub fn build_placeholder_tokens_with_config(
    grids: &[(usize, usize, usize)],
    slice_grid: Option<(usize, usize)>,
    local_index: usize,
    use_image_id: bool,
    tokens_per_patch_divisor: usize,
    canvas_config: &CanvasConfig,
) -> Vec<u32> {
    let mut out = Vec::new();
    if use_image_id {
        out.push(canvas_config.image_id_start);
        out.push((15 + local_index) as u32);
        out.push(canvas_config.image_id_end);
    }

    let thumb_count = (grids[0].1 * grids[0].2) / tokens_per_patch_divisor;
    out.push(canvas_config.image_start_id);
    out.extend(std::iter::repeat_n(canvas_config.image_token_id, thumb_count));
    out.push(canvas_config.image_end_id);

    if let Some((rows, cols)) = slice_grid {
        let per_slice = if grids.len() > 1 {
            (grids[1].1 * grids[1].2) / tokens_per_patch_divisor
        } else {
            0
        };
        for r in 0..rows {
            for _c in 0..cols {
                out.push(canvas_config.slice_start_id);
                out.extend(std::iter::repeat_n(canvas_config.image_token_id, per_slice));
                out.push(canvas_config.slice_end_id);
            }
            if r + 1 < rows {
                out.push(canvas_config.newline_id);
            }
        }
    }

    out
}

fn coords(
    canvas: usize,
    n: usize,
    offset: usize,
) -> Vec<usize> {
    if n == 1 {
        return vec![offset];
    }
    let step = (canvas - 1) as f64 / (n - 1) as f64;
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        let v = if i < n / 2 {
            i as f64 * step
        } else {
            (canvas - 1) as f64 - (n - 1 - i) as f64 * step
        };
        res.push(v.round() as usize + offset);
    }
    res
}

fn vision_position_ids(
    start: usize,
    grid_h: usize,
    grid_w: usize,
    canvas_h: usize,
    canvas_w: usize,
    h_off: usize,
    w_off: usize,
    merge_factor: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let llm_h = grid_h / merge_factor;
    let llm_w = grid_w / merge_factor;
    let ch = if canvas_h == 0 {
        llm_h
    } else {
        canvas_h
    };
    let cw = if canvas_w == 0 {
        llm_w
    } else {
        canvas_w
    };

    let hs = coords(ch, llm_h, h_off);
    let ws = coords(cw, llm_w, w_off);

    let mut t = Vec::with_capacity(llm_h * llm_w);
    let mut h = Vec::with_capacity(llm_h * llm_w);
    let mut w = Vec::with_capacity(llm_h * llm_w);

    for &hv in &hs {
        for &wv in &ws {
            t.push(start as u32);
            h.push((hv + start) as u32);
            w.push((wv + start) as u32);
        }
    }
    (t, h, w)
}

struct VisualFrame {
    thumb_start: usize,
    thumb_end: usize,
    slices: Vec<(usize, usize)>,
    frame_end: usize,
}

struct VisualGroup {
    start: usize,
    end: usize,
    frames: Vec<VisualFrame>,
}

pub fn compute_canvas_mrope_with_config(
    prompt_tokens: &[u64],
    image_grids: &[(usize, usize, usize)],
    merge_factor: usize,
    canvas_config: &CanvasConfig,
) -> (Vec<[u32; 3]>, i64) {
    let seq_len = prompt_tokens.len();
    let pad_id = canvas_config.image_token_id as u64;
    let is_image_pad: Vec<bool> = prompt_tokens.iter().map(|&t| t == pad_id).collect();
    let is_marker = |t: u64| {
        t == canvas_config.slice_start_id as u64
            || t == canvas_config.slice_end_id as u64
            || t == canvas_config.image_start_id as u64
            || t == canvas_config.image_end_id as u64
            || t == canvas_config.newline_id as u64
            || t == 271 // double newline
    };

    // 1. Group visual frames
    let mut groups: Vec<VisualGroup> = Vec::new();
    let mut open_frame_owner: Option<(usize, usize)> = None;
    let mut previous_end = 0;
    let mut i = 0;

    while i < seq_len {
        let is_img = is_image_pad[i];
        let mut j = i;
        while j < seq_len && is_image_pad[j] == is_img {
            j += 1;
        }
        if !is_img {
            i = j;
            continue;
        }
        let crop = (i, j);
        if crop.0 > 0
            && prompt_tokens[crop.0 - 1] == SLICE_START as u64
            && let Some((g_idx, f_idx)) = open_frame_owner
        {
            groups[g_idx].frames[f_idx].slices.push(crop);
        } else {
            let gap_start = previous_end;
            let gap_end = previous_end.max(crop.0.saturating_sub(1));
            let adjacent = !groups.is_empty() && prompt_tokens[gap_start..gap_end].iter().all(|&t| is_marker(t));
            let frame = VisualFrame {
                thumb_start: crop.0,
                thumb_end: crop.1,
                slices: Vec::new(),
                frame_end: 0,
            };
            if adjacent {
                let last_g = groups.len() - 1;
                groups[last_g].frames.push(frame);
                open_frame_owner = Some((last_g, groups[last_g].frames.len() - 1));
            } else {
                let start = crop.0.saturating_sub(1);
                groups.push(VisualGroup {
                    start,
                    end: 0,
                    frames: vec![frame],
                });
                let last_g = groups.len() - 1;
                open_frame_owner = Some((last_g, 0));
            }
        }
        previous_end = crop.1;
        i = j;
    }

    let get_frame_end = |last_crop_end: usize| -> usize {
        let mut end = last_crop_end;
        while end < seq_len && is_marker(prompt_tokens[end]) {
            end += 1;
        }
        end
    };

    for g in &mut groups {
        for f in &mut g.frames {
            let last_crop_end = f.slices.last().map(|s| s.1).unwrap_or(f.thumb_end);
            f.frame_end = get_frame_end(last_crop_end);
        }
        if let Some(last_f) = g.frames.last() {
            g.end = last_f.frame_end;
        }
    }

    // 2. Assign 3D coordinates (T, H, W)
    let mut pos_t: Vec<u32> = (0..seq_len as u32).collect();
    let mut pos_h: Vec<u32> = (0..seq_len as u32).collect();
    let mut pos_w: Vec<u32> = (0..seq_len as u32).collect();

    let mut grid_cursor = 0;
    let mut next_grid = || -> (usize, usize) {
        if grid_cursor < image_grids.len() {
            let g = image_grids[grid_cursor];
            grid_cursor += 1;
            (g.1, g.2)
        } else {
            (28, 36)
        }
    };

    let fill_range =
        |start_idx: usize, stop_idx: usize, base: usize, pt: &mut [u32], ph: &mut [u32], pw: &mut [u32]| {
            for (offset, idx) in (start_idx..stop_idx).enumerate() {
                let v = (base + offset) as u32;
                pt[idx] = v;
                ph[idx] = v;
                pw[idx] = v;
            }
        };

    let mut current_pos = 0;
    let mut current_idx = 0;

    for group in &groups {
        if group.start > current_idx {
            fill_range(current_idx, group.start, current_pos, &mut pos_t, &mut pos_h, &mut pos_w);
            current_pos += group.start - current_idx;
        }

        let mut frame_cursor = group.start;
        for frame in &group.frames {
            let thumb = next_grid();
            let frame_start = frame.thumb_start.saturating_sub(1);
            if frame_start > frame_cursor {
                fill_range(frame_cursor, frame_start, current_pos, &mut pos_t, &mut pos_h, &mut pos_w);
                current_pos += (frame_start - frame_cursor) + 1;
            }

            let frame_end = frame.frame_end.min(group.end);
            let canvas_origin = current_pos;
            let halo_before_canvas = canvas_origin.saturating_sub(1);

            let mut llm_slice_h = 0;
            let mut llm_slice_w = 0;
            let mut num_cols = 0;
            let mut canvas_h = thumb.0 / merge_factor;
            let mut canvas_w = thumb.1 / merge_factor;
            let mut first_slice = None;

            if !frame.slices.is_empty() {
                num_cols = frame.slices.len();
                for k in 0..frame.slices.len() - 1 {
                    if frame.slices[k + 1].0 - frame.slices[k].1 > 2 {
                        num_cols = k + 1;
                        break;
                    }
                }
                let mut num_rows = frame.slices.len().checked_div(num_cols).unwrap_or(1);
                if num_rows * num_cols != frame.slices.len() {
                    num_rows = 1;
                    num_cols = frame.slices.len();
                }
                let fs = next_grid();
                llm_slice_h = fs.0 / merge_factor;
                llm_slice_w = fs.1 / merge_factor;
                first_slice = Some(fs);
                canvas_h = num_rows * llm_slice_h;
                canvas_w = num_cols * llm_slice_w;
            }

            for idx in frame_start..frame_end {
                pos_t[idx] = canvas_origin as u32;
                pos_h[idx] = canvas_origin as u32;
                pos_w[idx] = canvas_origin as u32;
            }
            pos_h[frame_start] = halo_before_canvas as u32;
            pos_w[frame_start] = halo_before_canvas as u32;

            if frame.thumb_end < frame_end {
                pos_h[frame.thumb_end] = (canvas_origin + canvas_h) as u32;
                pos_w[frame.thumb_end] = (canvas_origin + canvas_w) as u32;
            }

            let (tt, th, tw) =
                vision_position_ids(canvas_origin, thumb.0, thumb.1, canvas_h, canvas_w, 0, 0, merge_factor);
            pos_t[frame.thumb_start..frame.thumb_end].copy_from_slice(&tt);
            pos_h[frame.thumb_start..frame.thumb_end].copy_from_slice(&th);
            pos_w[frame.thumb_start..frame.thumb_end].copy_from_slice(&tw);

            for (k, slice_crop) in frame.slices.iter().enumerate() {
                let h_off = (k / num_cols) * llm_slice_h;
                let w_off = (k % num_cols) * llm_slice_w;
                let grid = if k == 0 {
                    first_slice.unwrap()
                } else {
                    next_grid()
                };
                let slice_h = grid.0 / merge_factor;
                let slice_w = grid.1 / merge_factor;
                let start_marker = slice_crop.0.saturating_sub(1);
                if start_marker >= frame_start {
                    pos_h[start_marker] = (canvas_origin + h_off) as u32;
                    pos_w[start_marker] = (canvas_origin + w_off) as u32;
                }
                if slice_crop.1 < frame_end {
                    pos_h[slice_crop.1] = (canvas_origin + h_off + slice_h - 1) as u32;
                    pos_w[slice_crop.1] = (canvas_origin + w_off + slice_w - 1) as u32;
                }
                let (st, sh, sw) = vision_position_ids(canvas_origin, grid.0, grid.1, 0, 0, h_off, w_off, merge_factor);
                pos_t[slice_crop.0..slice_crop.1].copy_from_slice(&st);
                pos_h[slice_crop.0..slice_crop.1].copy_from_slice(&sh);
                pos_w[slice_crop.0..slice_crop.1].copy_from_slice(&sw);
            }

            if frame.slices.len() > 1 {
                for k in 0..frame.slices.len() - 1 {
                    let gap_start = frame.slices[k].1;
                    let gap_end = frame.slices[k + 1].0;
                    if gap_end - gap_start <= 2 {
                        continue;
                    }
                    let boundary_h = (k / num_cols + 1) * llm_slice_h - 1;
                    let right_edge_w = num_cols * llm_slice_w;
                    for newline_idx in (gap_start + 1)..gap_end.saturating_sub(1) {
                        pos_h[newline_idx] = (canvas_origin + boundary_h) as u32;
                        pos_w[newline_idx] = (canvas_origin + right_edge_w) as u32;
                    }
                }
            }

            current_pos = canvas_origin + canvas_h.max(canvas_w) + 1;
            frame_cursor = frame_end;
        }

        if frame_cursor < group.end {
            fill_range(frame_cursor, group.end, current_pos, &mut pos_t, &mut pos_h, &mut pos_w);
            current_pos += group.end - frame_cursor;
        }
        current_idx = group.end;
    }

    if current_idx < seq_len {
        fill_range(current_idx, seq_len, current_pos, &mut pos_t, &mut pos_h, &mut pos_w);
    }

    let mut max_pos = 0;
    for i in 0..seq_len {
        max_pos = max_pos.max(pos_t[i]).max(pos_h[i]).max(pos_w[i]);
    }
    let delta = (max_pos as i64 + 1) - (seq_len as i64);

    let mrope_3d: Vec<[u32; 3]> = (0..seq_len).map(|i| [pos_t[i], pos_h[i], pos_w[i]]).collect();

    (mrope_3d, delta)
}
