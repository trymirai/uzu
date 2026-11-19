#pragma once

// Grid coordinates (x, y) of the `row`-th patch of the vision encoder input.
//
// Patches don't go in raster order: each merge_size * merge_size block of patches is contiguous, and blocks go row by
// row. The merger then turns every merge_size * merge_size consecutive rows into one LLM token without copying. Example
// for a 4x4 grid with merge_size = 2 (cell = row index):
//
//   0  1  2  3       0  1  4  5
//   4  5  6  7  =>   2  3  6  7
//   8  9 10 11       8  9 12 13
//  12 13 14 15      10 11 14 15
//
// Example:
//  Input: row = 6, grid_width = 4, merge_size = 2
//  Output: y = 1, x = 2
inline uint2 vision_patch_coordinates(uint row, uint grid_width, uint merge_size) {
  const uint block_size = merge_size * merge_size;
  const uint blocks_per_row = grid_width / merge_size;
  const uint block = row / block_size;
  const uint in_block = row % block_size;
  const uint y = (block / blocks_per_row) * merge_size + in_block / merge_size;
  const uint x = (block % blocks_per_row) * merge_size + in_block % merge_size;
  return uint2(x, y);
}