enum RoPEKind : uint32_t {
  None = 0,
  Sequential = 1,
  ContiguousGrid = 2,
  InterleavedGrid = 3,
};

struct RoPEGridParams {
  uint first_cell;
  uint grid_width;
  uint grid_height;
  uint height_pairs;
  uint width_pairs;
  uint merge_size;
};