//! Video decoding for the VLM CLI/server.
//!
//! The model itself only consumes pre-sampled RGB frames; container decoding is
//! an external concern. We shell out to ffmpeg/ffprobe (both available on this
//! machine) and read raw RGB24 frames from a pipe.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

/// Reference video processor defaults from `processor_config.json`.
pub const VIDEO_FPS: f64 = 2.0;
pub const VIDEO_MAX_FRAMES: usize = 768;

fn probe_dimensions(path: &Path) -> Result<(usize, usize)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .context("failed to run ffprobe (is ffmpeg installed?)")?;
    if !output.status.success() {
        anyhow::bail!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next().unwrap_or_default();
    let mut parts = first.split(',');
    let width: usize = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let height: usize = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
    anyhow::ensure!(width > 0 && height > 0, "cannot determine video dimensions");
    Ok((width, height))
}

/// Decodes `path` into RGB frames sampled at `fps` (at most `max_frames`).
pub fn decode_video_frames(
    path: impl AsRef<Path>,
    fps: f64,
    max_frames: usize,
) -> Result<Vec<image::DynamicImage>> {
    let path = path.as_ref();
    let (width, height) = probe_dimensions(path)?;
    let frame_bytes = width * height * 3;
    let filter = format!("fps={fps}");
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &filter,
            "-fps_mode",
            "passthrough",
            "-frames:v",
            &max_frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .context("failed to run ffmpeg (is it installed?)")?;
    if !output.status.success() {
        anyhow::bail!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let data = output.stdout;
    anyhow::ensure!(
        data.len() >= frame_bytes * 2,
        "video produced {} frame(s); at least 2 are required",
        data.len() / frame_bytes
    );
    let mut frames = Vec::with_capacity(data.len() / frame_bytes);
    for chunk in data.chunks_exact(frame_bytes) {
        let buffer = image::RgbImage::from_raw(width as u32, height as u32, chunk.to_vec())
            .context("failed to build RGB frame")?;
        frames.push(image::DynamicImage::ImageRgb8(buffer));
    }
    Ok(frames)
}

/// Decodes video bytes (e.g. a data URL payload) via a temporary file.
pub fn decode_video_bytes(
    bytes: &[u8],
    suffix: &str,
    fps: f64,
    max_frames: usize,
) -> Result<Vec<image::DynamicImage>> {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "uzu-video-{}-{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        suffix.trim_start_matches('.')
    ));
    std::fs::write(&path, bytes).context("failed to write temporary video")?;
    let result = decode_video_frames(&path, fps, max_frames);
    let _ = std::fs::remove_file(&path);
    result
}

/// Decodes a local path, `file://` URL or `data:` URL.
pub fn decode_video_source(url: &str) -> Result<Vec<image::DynamicImage>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_header, payload) = rest.split_once(',').context("invalid data URL")?;
        use base64::Engine;
        let bytes = base64::prelude::BASE64_STANDARD
            .decode(payload.trim())
            .or_else(|_| base64::prelude::BASE64_URL_SAFE.decode(payload.trim()))
            .context("failed to decode video base64 payload")?;
        let suffix = if rest.starts_with("video/mp4") { "mp4" } else { "bin" };
        return decode_video_bytes(&bytes, suffix, VIDEO_FPS, VIDEO_MAX_FRAMES);
    }
    let path = url.strip_prefix("file://").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(url));
    anyhow::ensure!(path.exists(), "video file not found: {}", path.display());
    decode_video_frames(&path, VIDEO_FPS, VIDEO_MAX_FRAMES)
}
