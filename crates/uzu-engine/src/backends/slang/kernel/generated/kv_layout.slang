static const uint TILE_KEYS = 32;

static const uint TILE_QUERIES = 16;

static const uint TILE_DIMS = 16;

static const uint FRAGMENT_BYTES = TILE_KEYS * TILE_DIMS;

static const uint SLICE_DIMS = 64;

static const int16_t V_CODE_BIAS = 128;

static const uint GROUP_ROWS = 8;

static const uint LANE_COLS = 4;

static const uint LANE_BYTES = TILE_KEYS / GROUP_ROWS * LANE_COLS;

static const uint WORD_BITS = 32;

static const uint ROTATION_SIGN_BITS[8] = {2654435769, 2246822507, 3266489909, 668265263, 374761393, 3210233709, 2496678331, 3605593784};