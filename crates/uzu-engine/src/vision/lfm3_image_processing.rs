//! LFM3-VL image preprocessing: Qwen3VL-style smart resize + patchify.
//!
//! Ports the MLX-side processor (`lfm3_vl/processing_lfm3_vl.py`, whose image
//! path delegates to `qwen3_vl/processing_qwen3_vl.py`) so pixel values match
//! the reference at quantized-input precision: resize runs on the u8 image
//! with Pillow's bicubic convolution exactly as in `Resample.c` (a = -0.5,
//! support 2, float coefficients sum-normalized, quantized to `2^22` INT32
//! fixed point round-half-away-from-zero, accumulated with a `2^21` bias and
//! an arithmetic `>> 22`, clamped; identity-size passes copy unchanged), and
//! rescale/normalize happen after resize in f32, exactly like the numpy
//! pipeline (`pixels *= rescale_factor`, `(pixels - mean) / std`).
//!
//! Token layout matches the reference processor: patch-level tokens in
//! 2x2-merge-block order, each token holding C * temporal * patch^2 values
//! (channels, temporal, patch_y, patch_x).


/// Runs `f(index, chunk)` over disjoint chunks of `slice` on worker threads.
/// Only used for element-wise work whose per-element arithmetic is unchanged,
/// so results stay bit-identical to the serial implementation.
fn parallel_chunks_mut<T: Send>(
    slice: &mut [T],
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let len = slice.len();
    if len == 0 {
        return;
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);
    let chunk = len.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (index, part) in slice.chunks_mut(chunk).enumerate() {
            let f = &f;
            let start = index * chunk;
            scope.spawn(move || f(start, part));
        }
    });
}

/// Same as [`parallel_chunks_mut`] but splits at `row_len` boundaries so each
/// worker receives whole rows and can derive absolute row indices.
fn parallel_row_chunks_mut<T: Send>(
    slice: &mut [T],
    row_len: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    if row_len == 0 || slice.is_empty() {
        return;
    }
    let rows = slice.len() / row_len;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);
    let rows_per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (index, part) in slice.chunks_mut(row_len * rows_per).enumerate() {
            let f = &f;
            let first = index * rows_per;
            scope.spawn(move || f(first, part));
        }
    });
}

pub struct Lfm3ImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f32,
}

impl Lfm3ImageProcessor {
    pub fn new(patch_size: usize, temporal_patch_size: usize, merge_size: usize) -> Self {
        Self {
            patch_size,
            temporal_patch_size,
            merge_size,
            min_pixels: 65536,
            max_pixels: 16777216,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// RGB u8 image in, patch tokens + `[grid_t, grid_h, grid_w]` out.
    /// Returns `pixel_values` shaped `[N, C * tps * ps * ps]` (f32) where
    /// `N = grid_h * grid_w` (patch-level rows, 2x2-merge-block order).
    pub fn process(&self, rgb: &[u8], width: usize, height: usize) -> (Vec<f32>, [usize; 3]) {
        let factor = self.patch_size * self.merge_size;
        let (resized_h, resized_w) =
            smart_resize(height, width, factor, self.min_pixels, self.max_pixels);
        // Pillow skips passes whose axis size is unchanged (identity copies).
        let resized_buf = (resized_w != width || resized_h != height)
            .then(|| resample_bicubic(rgb, width, height, resized_w, resized_h));
        let resized = resized_buf.as_deref().unwrap_or(rgb);

        // Normalize in f32, same op order as the numpy pipeline.
        let mut img = vec![0f32; resized.len()];
        parallel_chunks_mut(&mut img, |start, chunk| {
            for (local, out) in chunk.iter_mut().enumerate() {
                let i = start + local;
                let c = i % 3;
                let v = resized[i] as f32 * self.rescale_factor;
                *out = (v - self.image_mean[c]) / self.image_std[c];
            }
        });

        let (pixels, grid) = self.patchify(&img, resized_w, resized_h);
        (pixels, grid)
    }

    /// Block-major patch tokens: for each 2x2 merge block (row-block, then
    /// col-block), the m*m patches inside; each token flattens
    /// [channels, temporal, patch_y, patch_x]. Single images duplicate the
    /// frame `temporal_patch_size` times.
    fn patchify(&self, img: &[f32], width: usize, height: usize) -> (Vec<f32>, [usize; 3]) {
        let ps = self.patch_size;
        let tps = self.temporal_patch_size;
        let m = self.merge_size;
        let c_n = 3;
        let gh = height / ps;
        let gw = width / ps;
        let gw_m = gw / m;
        let dims = c_n * tps * ps * ps;
        let mut tokens = vec![0f32; gh * gw * dims];
        parallel_row_chunks_mut(&mut tokens, dims, |first, chunk| {
            for (local, token) in chunk.chunks_mut(dims).enumerate() {
                let token_idx = first + local;
                let block = token_idx / (m * m);
                let within = token_idx % (m * m);
                let hb = block / gw_m;
                let wb = block % gw_m;
                let ir = within / m;
                let ic = within % m;
                let row = hb * m + ir;
                let col = wb * m + ic;
                let mut k = 0;
                for c in 0..c_n {
                    for _t in 0..tps {
                        for py in 0..ps {
                            for px in 0..ps {
                                let y = row * ps + py;
                                let x = col * ps + px;
                                token[k] = img[(y * width + x) * c_n + c];
                                k += 1;
                            }
                        }
                    }
                }
            }
        });
        (tokens, [1, gh, gw])
    }
}


/// One RGB u8 video frame supplied by the caller (already sampled to the
/// model's video fps).
#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Reference video processor limits (`processor_config.json` video_processor).
const VIDEO_MIN_PIXELS: u64 = 4096;
const VIDEO_MAX_PIXELS: u64 = 25_165_824;
/// Runtime default: the reference allows 24M total pixels, but our processor
/// materializes all patch tokens at once. Keep the default at 8M (env override
/// `LFM3_VL_VIDEO_MAX_PIXELS`) to bound peak memory; per-frame attention keeps
/// the tower cost linear in the number of frames.
const VIDEO_DEFAULT_MAX_PIXELS: u64 = 8_388_608;

impl Lfm3ImageProcessor {
    /// Qwen3VL/LFM3 video preprocessing: resize every frame with a total-pixel
    /// budget, pad the frame count to `temporal_patch_size`, normalize, then
    /// patchify with the temporal dimension grouped (token value order is
    /// C, temporal, patch_y, patch_x).
    pub fn process_video(
        &self,
        frames: &[VideoFrame],
    ) -> Result<(Vec<f32>, [usize; 3]), String> {
        let tps = self.temporal_patch_size;
        let ps = self.patch_size;
        let m = self.merge_size;
        let c_n = 3;
        if frames.len() < tps {
            return Err(format!("LFM3-VL video requires at least {tps} frames"));
        }
        let (width, height) = (frames[0].width, frames[0].height);
        if frames.iter().any(|frame| frame.width != width || frame.height != height) {
            return Err("video frames must all have the same dimensions".to_string());
        }
        let t = frames.len();
        let factor = ps * m;
        let round_ties_even = |value: f64| value.round_ties_even() as usize;
        let mut target_h = round_ties_even(height as f64 / factor as f64) * factor;
        let mut target_w = round_ties_even(width as f64 / factor as f64) * factor;
        let rounded_t = round_ties_even(t as f64 / tps as f64) * tps;
        let max_pixels = std::env::var("LFM3_VL_VIDEO_MAX_PIXELS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(VIDEO_DEFAULT_MAX_PIXELS)
            .clamp(VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS);
        let total = rounded_t as u64 * target_h as u64 * target_w as u64;
        if total > max_pixels {
            let beta = ((t as f64 * height as f64 * width as f64) / max_pixels as f64).sqrt();
            target_h = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
            target_w = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
        } else if total < VIDEO_MIN_PIXELS {
            let beta = (VIDEO_MIN_PIXELS as f64 / (t as f64 * height as f64 * width as f64)).sqrt();
            target_h = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            target_w = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        // Resize + normalize every frame (Pillow-compatible bicubic).
        let mut normalized: Vec<Vec<f32>> = Vec::with_capacity(t);
        for frame in frames {
            let resized = resample_bicubic(&frame.rgb, width, height, target_w, target_h);
            let mut img = vec![0f32; resized.len()];
            parallel_chunks_mut(&mut img, |start, chunk| {
                for (local, out) in chunk.iter_mut().enumerate() {
                    let i = start + local;
                    let c = i % 3;
                    let v = resized[i] as f32 * self.rescale_factor;
                    *out = (v - self.image_mean[c]) / self.image_std[c];
                }
            });
            normalized.push(img);
        }
        // The reference pads an odd frame count by repeating the last frame.
        while !normalized.len().is_multiple_of(tps) {
            normalized.push(normalized.last().expect("non-empty video").clone());
        }

        let grid_t = normalized.len() / tps;
        let gh = target_h / ps;
        let gw = target_w / ps;
        let gh_m = gh / m;
        let gw_m = gw / m;
        let dims = c_n * tps * ps * ps;
        let rows = grid_t * gh * gw;
        let mut tokens = vec![0f32; rows * dims];
        let per_frame = gh_m * gw_m * m * m;
        parallel_row_chunks_mut(&mut tokens, dims, |first, chunk| {
            for (local, token) in chunk.chunks_mut(dims).enumerate() {
                let token_idx = first + local;
                let tf = token_idx / per_frame;
                let rem = token_idx % per_frame;
                let block = rem / (m * m);
                let within = rem % (m * m);
                let hb = block / gw_m;
                let wb = block % gw_m;
                let ir = within / m;
                let ic = within % m;
                let row = hb * m + ir;
                let col = wb * m + ic;
                let mut k = 0;
                for c in 0..c_n {
                    for tau in 0..tps {
                        let img = &normalized[tf * tps + tau];
                        for py in 0..ps {
                            for px in 0..ps {
                                let y = row * ps + py;
                                let x = col * ps + px;
                                token[k] = img[(y * target_w + x) * c_n + c];
                                k += 1;
                            }
                        }
                    }
                }
            }
        });
        Ok((tokens, [grid_t, gh, gw]))
    }
}

/// Pillow 12.1 `Image.BICUBIC` resize (`Resample.c`) over interleaved RGB u8:
/// horizontal pass only when width changes, vertical only when height changes
/// (identity sizes copy unchanged), horizontal first. u8 output.
fn resample_bicubic(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    match (dst_w != src_w, dst_h != src_h) {
        (true, true) => {
            let horizontal = resample_axis(src, src_w, src_h, dst_w, true);
            resample_axis(&horizontal, dst_w, src_h, dst_h, false)
        }
        (true, false) => resample_axis(src, src_w, src_h, dst_w, true),
        (false, true) => resample_axis(src, src_w, src_h, dst_h, false),
        (false, false) => src.to_vec(),
    }
}

/// One `ResampleHorizontal`/`ResampleVertical` 8bpc pass: float bicubic
/// window (support 2, sum-normalized) quantized to INT32 fixed point
/// (`PRECISION_BITS = 32 - 8 - 2 = 22`, round half away from zero), then a
/// `2^21`-biased INT32 accumulation with arithmetic `>> 22` (round half up),
/// clamped to u8.
///
/// `transpose` = resize along the width axis; otherwise along height.
fn resample_axis(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_len: usize,
    transpose: bool,
) -> Vec<u8> {
    const PRECISION_BITS: u32 = 32 - 8 - 2;
    let in_len = if transpose { src_w } else { src_h };
    let other = if transpose { src_h } else { src_w };

    // precompute_coeffs
    let scale = in_len as f64 / dst_len as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale; // BICUBIC filter support
    let ss = 1.0 / filterscale; // Pillow multiplies by the reciprocal

    struct Coefs {
        xmin: usize,
        coefs: Vec<i32>,
    }
    let mut rows = Vec::with_capacity(dst_len);
    for xx in 0..dst_len {
        let center = (xx as f64 + 0.5) * scale;
        // `(int)` casts truncate toward zero; the clamps below make xmin's
        // trunc-vs-floor difference unobservable.
        let xmin = ((center - support + 0.5) as isize).clamp(0, in_len as isize);
        let xmax = ((center + support + 0.5) as isize).clamp(xmin, in_len as isize);
        let window = (xmax - xmin) as usize;
        let mut w = Vec::with_capacity(window);
        for x in 0..window {
            w.push(cubic((x as f64 + xmin as f64 - center + 0.5) * ss));
        }
        let ww: f64 = w.iter().sum();
        if ww != 0.0 {
            for v in &mut w {
                *v /= ww;
            }
        }
        // normalize_coeffs_8bpc: round half away from zero into 2^22 fixed point.
        let coefs = w
            .iter()
            .map(|v| {
                let s = v * ((1i64 << PRECISION_BITS) as f64);
                if s < 0.0 {
                    (-0.5 + s) as i32
                } else {
                    (0.5 + s) as i32
                }
            })
            .collect();
        rows.push(Coefs { xmin: xmin as usize, coefs });
    }

    // ImagingResample{Horizontal,Vertical}_8bpc accumulation. Output rows are
    // independent; the transposed (horizontal) pass is split over the source
    // line (d), the vertical pass over the output line (xx).
    let mut dst = vec![0u8; dst_len * other * 3];
    if transpose {
        parallel_row_chunks_mut(&mut dst, dst_len * 3, |d0, chunk| {
            for (local, out) in chunk.chunks_mut(dst_len * 3).enumerate() {
                let d = d0 + local;
                for (xx, row) in rows.iter().enumerate() {
                    for c in 0..3usize {
                        let mut acc = 1i32 << (PRECISION_BITS - 1);
                        for (j, k) in row.coefs.iter().enumerate() {
                            let sample = src[(d * src_w + row.xmin + j) * 3 + c] as i32;
                            acc = acc.wrapping_add(sample.wrapping_mul(*k));
                        }
                        out[xx * 3 + c] = (acc >> PRECISION_BITS).clamp(0, 255) as u8;
                    }
                }
            }
        });
    } else {
        parallel_row_chunks_mut(&mut dst, other * 3, |xx0, chunk| {
            for (local, out) in chunk.chunks_mut(other * 3).enumerate() {
                let xx = xx0 + local;
                let row = &rows[xx];
                for d in 0..other {
                    for c in 0..3usize {
                        let mut acc = 1i32 << (PRECISION_BITS - 1);
                        for (j, k) in row.coefs.iter().enumerate() {
                            let sample = src[((row.xmin + j) * src_w + d) * 3 + c] as i32;
                            acc = acc.wrapping_add(sample.wrapping_mul(*k));
                        }
                        out[d * 3 + c] = (acc >> PRECISION_BITS).clamp(0, 255) as u8;
                    }
                }
            }
        });
    }
    dst
}

/// Smart resize: round both sides to `factor` multiples, bounding the pixel
/// count between `min_pixels` and `max_pixels` (ports `_smart_resize_image`).
fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: u64,
    max_pixels: u64,
) -> (usize, usize) {
    assert!(
        usize::max(height, width) / usize::min(height, width) <= 200,
        "absolute aspect ratio must be smaller than 200"
    );
    let mut h_bar = ((height as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut w_bar = ((width as f64 / factor as f64).round_ties_even() as usize) * factor;
    if (h_bar * w_bar) as u64 > max_pixels {
        let beta = ((height as f64 * width as f64) / max_pixels as f64).sqrt();
        h_bar = usize::max(factor, (height as f64 / beta / factor as f64).floor() as usize * factor);
        w_bar = usize::max(factor, (width as f64 / beta / factor as f64).floor() as usize * factor);
    } else if ((h_bar * w_bar) as u64) < min_pixels {
        let beta = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        h_bar = usize::max(factor, (height as f64 * beta / factor as f64).ceil() as usize * factor);
        w_bar = usize::max(factor, (width as f64 * beta / factor as f64).ceil() as usize * factor);
    }
    (h_bar, w_bar)
}

fn cubic(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 1.0 {
        1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0
    } else if ax < 2.0 {
        -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uzu_engine_macros::uzu_test;

    /// Parity check against the MLX reference processor, env-gated:
    /// LFM3_IMG (small test image), LFM3_IMG_CHECK (.f32 pixel_values dump,
    /// [N, C*tps*ps*ps] row-major), LFM3_IMG_GRID (.i64 [1, 3] grid).
    #[uzu_test]
    fn lfm3_image_processor_matches_mlx_reference() {
        let parse = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let (Some(img_path), Some(check_path), Some(grid_path)) =
            (parse("LFM3_IMG"), parse("LFM3_IMG_CHECK"), parse("LFM3_IMG_GRID"))
        else {
            println!("{{\"skipped\": \"set LFM3_IMG/LFM3_IMG_CHECK/LFM3_IMG_GRID to run\"}}");
            return;
        };

        let img = image::open(&img_path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let proc = Lfm3ImageProcessor::new(16, 2, 2);
        let (pixels, grid) = proc.process(img.as_raw(), w, h);

        let expected_grid: Vec<i64> = std::fs::read(&grid_path)
            .unwrap()
            .as_chunks::<8>().0.iter()
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        assert_eq!(grid, [expected_grid[0] as usize, expected_grid[1] as usize, expected_grid[2] as usize]);

        let expected: Vec<f32> = std::fs::read(&check_path)
            .unwrap()
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(pixels.len(), expected.len(), "token count mismatch");
        let max_diff = pixels
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{{\"max_abs_diff\": {max_diff}, \"tokens\": {}}}", pixels.len() / 1536);
        assert_eq!(pixels, expected, "pixel values differ from MLX reference");
    }

    #[uzu_test]
    fn caps_max_pixels_before_patchify() {
        let mut processor = Lfm3ImageProcessor::new(16, 2, 2);
        processor.max_pixels = 4_194_304;
        let (width, height) = (4032, 2688);
        let rgb = vec![128u8; width * height * 3];
        let (_pixels, grid) = processor.process(&rgb, width, height);
        let patches = grid[0] * grid[1] * grid[2];
        let pixels = patches * processor.patch_size * processor.patch_size;
        assert!(pixels <= 4_194_304, "cap exceeded: {pixels} pixels ({patches} patches)");
        // The smart resize should land near the cap, not collapse the image.
        assert!(pixels * 10 >= 4_194_304 * 9, "over-downscaled: {pixels} pixels");
    }
}
