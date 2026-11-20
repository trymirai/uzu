import definitions;

// TODO: implement, see metal/kernel/vision/patchify.metal
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
}
