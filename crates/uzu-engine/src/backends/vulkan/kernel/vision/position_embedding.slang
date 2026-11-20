import definitions;
import vision.patch_grid;

float vision_linspace(uint index, uint count, uint side) {
  if (count <= 1) return 0.0f;
  let end = float(side - 1);
  let step = end / float(count - 1);
  return index < count / 2 ? float(index) * step : end - float(count - index - 1) * step;
}

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
  let patch = vision_patch_coordinates(position / model_dim, grid_width, merge_size);
  let x = vision_linspace(patch.x, grid_width, side);
  let y = vision_linspace(patch.y, grid_height, side);
  let x0 = uint(x);
  let y0 = uint(y);
  let x1 = min(x0 + 1, side - 1);
  let y1 = min(y0 + 1, side - 1);
  let dx = x - float(x0);
  let dy = y - float(y0);

  uint4 corners = uint4(y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1);
  float4 weights = float4((1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx);
  let channel = position % model_dim;
  float embedding = 0.0f;
  // Match checkpoint precision for each weight, product, and partial sum.
  [unroll]
  for (uint i = 0; i < 4; ++i) {
    let sample = table[corners[i] * model_dim + channel].toFloat();
    let value = round_to<T>(sample * round_to<T>(weights[i]));
    embedding = round_to<T>(embedding + value);
  }
  stream[position] = T.fromFloat(stream[position].toFloat() + embedding);
}
