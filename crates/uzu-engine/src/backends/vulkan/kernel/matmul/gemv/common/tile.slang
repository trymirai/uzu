import definitions;
import generated.gemm;

// Pipeline specialization supplies the tile extents.

public static const uint GEMV_MAX_INPUT_ROWS = GEMV_INPUT_ROW_TILE;
public static const uint GEMV_MAX_ROWS_PER_LANE = 8;
public static const uint GEMV_MAX_SIMDGROUPS = 8;

public struct QuantPosition {
  public uint group;
  public uint slice;

  public bool valid(uint groups) {
    return group < groups;
  }
}

public struct GemvTile {
  public uint INPUT_ROWS;
  public uint OUTPUT_ROWS;
  public uint REDUCTION_LANES;
  public uint GROUP_LANES;
  public uint NUM_SIMDGROUPS;
  public uint K_SPLIT;
  public uint ROW_BLOCKS;
  public uint SIMDGROUP_OUTPUT_ROWS;
  public uint ROWS_PER_LANE;
  public uint GROUPS_PER_STEP;

  public __init(
    uint input_row_tile,
    uint output_row_tile,
    uint reduction_lanes,
    uint group_lanes,
    uint num_simdgroups,
    uint k_split
  ) {
    INPUT_ROWS = input_row_tile;
    OUTPUT_ROWS = output_row_tile;
    REDUCTION_LANES = reduction_lanes;
    GROUP_LANES = group_lanes;
    NUM_SIMDGROUPS = num_simdgroups;
    K_SPLIT = k_split;
    ROW_BLOCKS = SIMD_SIZE / REDUCTION_LANES;
    SIMDGROUP_OUTPUT_ROWS = OUTPUT_ROWS / (NUM_SIMDGROUPS / K_SPLIT);
    ROWS_PER_LANE = SIMDGROUP_OUTPUT_ROWS / ROW_BLOCKS;
    GROUPS_PER_STEP = REDUCTION_LANES / GROUP_LANES;
  }
}

public struct OutputTile {
  public uint simd_group;
  public uint simd_lane;
  public uint reduction_lane;
  public uint row_group;
  public uint k_slice;
  public uint input_row;
  public uint tile_row;
  public uint local_row;
  public uint row0;
  public bool writer;

  // Simdgroups are runs of 32 invocations in subgroup_invocation order, whose lanes Reduce combines.
  public static OutputTile make(
    GemvTile Tile,
    bool FULL_TILE,
    uint output_tile_idx,
    uint input_tile_idx,
    uint out_vec_size
  ) {
    let invocation = subgroup_invocation();
    let simd_group = invocation / SIMD_SIZE;
    let simd_lane = invocation % SIMD_SIZE;
    let row_group = simd_group / Tile.K_SPLIT;
    let k_slice = simd_group % Tile.K_SPLIT;
    let row_block = simd_lane / Tile.REDUCTION_LANES;
    let local_row = row_group * Tile.SIMDGROUP_OUTPUT_ROWS + row_block * Tile.ROWS_PER_LANE;
    let unclamped = output_tile_idx * Tile.OUTPUT_ROWS + local_row;
    let last = out_vec_size > Tile.ROWS_PER_LANE ? out_vec_size - Tile.ROWS_PER_LANE : 0;
    OutputTile tile;
    tile.simd_group = simd_group;
    tile.simd_lane = simd_lane;
    tile.reduction_lane = simd_lane % Tile.REDUCTION_LANES;
    tile.row_group = row_group;
    tile.k_slice = k_slice;
    tile.input_row = input_tile_idx * Tile.INPUT_ROWS;
    tile.tile_row = output_tile_idx * Tile.OUTPUT_ROWS;
    tile.local_row = local_row;
    tile.row0 = FULL_TILE ? unclamped : min(unclamped, last);
    tile.writer = Tile.K_SPLIT == 1 || k_slice == 0;
    return tile;
  }

  public static bool row_in_range(bool FULL_TILE, uint row, uint out_vec_size) {
    return FULL_TILE || row < out_vec_size;
  }
}
