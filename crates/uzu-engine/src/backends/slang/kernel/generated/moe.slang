enum MoeExpertFormat : uint32_t {
  Nvfp4 = 0,
  AffineU4 = 1,
};

static const uint MOE_MAX_TOP_K = 8;

static const uint MOE_MAX_EXPERTS = 256;

static const uint MOE_GEMV_MAX_TOKENS = 32;

static const uint MOE_GEMV_SIMDGROUPS = 8;

static const uint MOE_UP_TILE_HIDDEN = 4;

static const uint MOE_DOWN_TILE_ROWS = 16;

static const uint MOE_ROUTE_EXPERTS_PER_GROUP = 4;

static const uint MOE_ROUTE_FINALIZE_THREADS = 1024;

static const uint MOE_ACTIVE_EXPERT_BITS = 8;

static const uint MOE_ACTIVE_COUNT_BITS = 11;