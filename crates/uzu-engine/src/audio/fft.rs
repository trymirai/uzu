//! Small complex FFT for the Liquid-Codec STFT/ISTFT (n_fft = 1280 = 2^8 · 5).
//!
//! Radix-2/5 recursive Cooley-Tukey in f32 — exact recursion mirrors the
//! convention of `torch.fft.rfft` / `torch.fft.irfft` with `norm="backward"`:
//! the forward transform is unscaled, the inverse carries a `1/n` factor.

/// In-place complex FFT, forward (`inverse = false`) or unscaled-backward-inverse
/// sign (`inverse = true` applies `e^{+i…}` twiddles). `n` must factor into 2s and 5s.
pub fn fft_inplace(
    re: &mut [f32],
    im: &mut [f32],
    inverse: bool,
) {
    let n = re.len();
    debug_assert_eq!(re.len(), im.len());
    if n <= 1 {
        return;
    }
    let radix = if n % 2 == 0 {
        2
    } else if n % 5 == 0 {
        5
    } else {
        panic!("FFT size {n} must factor into 2s and 5s");
    };
    let m = n / radix;
    let sign = if inverse {
        1.0f32
    } else {
        -1.0f32
    };

    let mut sub_re: Vec<Vec<f32>> = Vec::with_capacity(radix);
    let mut sub_im: Vec<Vec<f32>> = Vec::with_capacity(radix);
    for s in 0..radix {
        let (mut r, mut i) = (Vec::with_capacity(m), Vec::with_capacity(m));
        for idx in (s..n).step_by(radix) {
            r.push(re[idx]);
            i.push(im[idx]);
        }
        fft_inplace(&mut r, &mut i, inverse);
        sub_re.push(r);
        sub_im.push(i);
    }

    for k in 0..n {
        let km = k % m;
        let (mut acc_re, mut acc_im) = (sub_re[0][km], sub_im[0][km]);
        for s in 1..radix {
            let ang = sign * std::f32::consts::TAU * ((s * k) as f32) / (n as f32);
            let (c, sn) = (ang.cos(), ang.sin());
            let (xr, xi) = (sub_re[s][km], sub_im[s][km]);
            acc_re += xr * c - xi * sn;
            acc_im += xr * sn + xi * c;
        }
        re[k] = acc_re;
        im[k] = acc_im;
    }
}

/// Real FFT: `n = input.len()`, bins `0..=n/2`.
pub fn rfft(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = input.len();
    let mut re = input.to_vec();
    let mut im = vec![0f32; n];
    fft_inplace(&mut re, &mut im, false);
    re.truncate(n / 2 + 1);
    im.truncate(n / 2 + 1);
    (re, im)
}

/// Inverse real FFT from `n/2 + 1` bins: rebuilds the conjugate-symmetric
/// spectrum, applies the `1/n` factor (`norm="backward"`), returns `n` real samples.
pub fn irfft(
    bins_re: &[f32],
    bins_im: &[f32],
    out: &mut [f32],
) {
    let bins = bins_re.len();
    assert_eq!(bins, out.len() / 2 + 1, "bin count mismatch");
    let n = out.len();
    let mut re = vec![0f32; n];
    let mut im = vec![0f32; n];
    re[..bins].copy_from_slice(bins_re);
    im[..bins].copy_from_slice(bins_im);
    for k in bins..n {
        let src = n - k;
        re[k] = bins_re[src];
        im[k] = -bins_im[src];
    }
    fft_inplace(&mut re, &mut im, true);
    let inv_n = 1.0f32 / n as f32;
    for (o, &r) in out.iter_mut().zip(&re) {
        *o = r * inv_n;
    }
}

#[cfg(test)]
mod tests {
    use uzu_engine_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn matches_known_spectrum() {
        // x = cos(2π n / 8): rfft bins [4, 4, 0, ...]
        let x: Vec<f32> = (0..8).map(|n| (std::f32::consts::TAU * n as f32 / 8.0).cos()).collect();
        let (re, im) = rfft(&x);
        assert!((re[0] - 0.0).abs() < 1e-4, "re0 {}", re[0]);
        assert!((re[1] - 4.0).abs() < 1e-4, "re1 {}", re[1]);
        assert!(im.iter().all(|v| v.abs() < 1e-4));
        // Round trip
        let mut out = vec![0f32; 8];
        irfft(&re, &im, &mut out);
        for (a, b) in out.iter().zip(&x) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
