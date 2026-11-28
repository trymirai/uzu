// Patches within a merge block occupy consecutive rows of the encoder input.
public uint2 vision_patch_coordinates(uint row, uint grid_width, uint merge_size) {
  let block_size = merge_size * merge_size;
  let blocks_per_row = grid_width / merge_size;
  let block = row / block_size;
  let in_block = row % block_size;
  return uint2(block % blocks_per_row, block / blocks_per_row) * merge_size
       + uint2(in_block % merge_size, in_block / merge_size);
}
