import definitions;
import vision.patch_grid;

[[Variants("T", "bf16")]]
[[Kernel]] [[Public]] func VisionPatchify<T: IStorageFloat>(
  Ptr<uint8_t, Access.Read> pixels,
  Ptr<T, Access.ReadWrite> patches,
  Ptr<float, Access.Read> image_mean,
  Ptr<float, Access.Read> image_std,
  uint width,
  uint patch_size,
  uint merge_size,
  uint temporal_patch_size,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  let patch_area = patch_size * patch_size;
  let channel_stride = temporal_patch_size * patch_area;
  let patch_dim = 3 * channel_stride;
  let patch = vision_patch_coordinates(position / patch_dim, width / patch_size, merge_size);
  let col = position % patch_dim;
  let channel = col / channel_stride;
  // Temporal frames of a still image all read the same spatial patch.
  let y = patch.y * patch_size + (col % patch_area) / patch_size;
  let x = patch.x * patch_size + col % patch_size;
  let pixel = float(pixels[(y * width + x) * 3 + channel]) / 255.0f;
  patches[position] = T.fromFloat((pixel - image_mean[channel]) / image_std[channel]);
}
