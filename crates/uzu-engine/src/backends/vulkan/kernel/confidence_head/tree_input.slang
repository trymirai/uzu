import definitions;
import confidence_head.head;
import generated.weaver;

[[Kernel]] [[Public]]
func ConfidenceHeadTreeInput(
  Ptr<bf16, Access.Read> depth_embedding,
  Ptr<bf16, Access.Read> child_in,
  Ptr<bf16, Access.Read> node_in,
  Ptr<uint, Access.Read> frontier,
  Ptr<uint, Access.Read> row_slots,
  Ptr<bf16, Access.ReadWrite> output,
  uint rows,
  uint width,
  [[Axis("rows * width", "256")]] uint index,
) {
  let row = index / width;
  let unit = index % width;
  let depth = row == 0 ? 0u : frontier[uint(FrontierIdx.Depth) * (rows - 1) + row - 1];
  var value = 0.0;
  if (row == 0 || depth != 0) {
    value = depth_embedding[depth * width + unit].toFloat();
    if (row != 0) {
      value += child_in[(row - 1) * width + unit].toFloat();
    }
    let slot = row_slots[row];
    if (slot != HEAD_NONE) {
      value += node_in[slot * width + unit].toFloat();
    }
  }
  output[index] = bf16.fromFloat(value);
}
