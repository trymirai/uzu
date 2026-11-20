import definitions;

// TODO: implement, see metal/kernel/vision/position_embedding.metal
[[Variants("T", "bf16")]]
[[Kernel]] [[Public]] func VisionPositionEmbedding<T: IStorageFloat>(
  Ptr<T, Access.Read> table,
  Ptr<T, Access.ReadWrite> stream,
  uint side,
  uint grid_width,
  uint grid_height,
  uint merge_size,
  uint model_dim,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
}
