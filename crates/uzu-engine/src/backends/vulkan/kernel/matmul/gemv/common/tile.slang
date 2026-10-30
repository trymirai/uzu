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
  public uint GROUP_LANES;
  public uint ROWS_PER_LANE;
  public uint GROUPS_PER_STEP;

  public __init(uint input_row_tile, uint output_row_tile, uint group_lanes, uint num_simdgroups) {
    INPUT_ROWS = input_row_tile;
    OUTPUT_ROWS = output_row_tile;
    GROUP_LANES = group_lanes;
    ROWS_PER_LANE = OUTPUT_ROWS / num_simdgroups;
    GROUPS_PER_STEP = SIMD_SIZE / GROUP_LANES;
  }
}

public struct OutputTile {
  public uint reduction_lane;
  public uint input_row;
  public uint tile_row;
  public uint local_row;
  public uint row0;

  // Simdgroups are runs of 32 invocations in subgroup_invocation order, whose lanes Reduce combines.
  public static OutputTile make(
    GemvTile Tile,
    bool FULL_TILE,
    uint output_tile_idx,
    uint input_tile_idx,
    uint out_vec_size
  ) {
    let invocation = subgroup_invocation();
    let local_row = invocation / SIMD_SIZE * Tile.ROWS_PER_LANE;
    let unclamped = output_tile_idx * Tile.OUTPUT_ROWS + local_row;
    let last = out_vec_size > Tile.ROWS_PER_LANE ? out_vec_size - Tile.ROWS_PER_LANE : 0;
    OutputTile tile;
    tile.reduction_lane = invocation % SIMD_SIZE;
    tile.input_row = input_tile_idx * Tile.INPUT_ROWS;
    tile.tile_row = output_tile_idx * Tile.OUTPUT_ROWS;
    tile.local_row = local_row;
    tile.row0 = FULL_TILE ? unclamped : min(unclamped, last);
    return tile;
  }

  public static bool row_in_range(bool FULL_TILE, uint row, uint out_vec_size) {
    return FULL_TILE || row < out_vec_size;
  }
}
