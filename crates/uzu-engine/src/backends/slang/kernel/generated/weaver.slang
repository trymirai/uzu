static const uint CANDIDATES_MAX = 512;

enum WeaverSelectionMode : uint32_t {
  Children = 0,
  Frontier = 1,
  Fused = 2,
};

enum FrontierIdx : uint32_t {
  TokenId = 0,
  ParentSlot = 1,
  Depth = 2,
  PathLogprobBits = 3,
  EdgeLogprobBits = 4,
  PathScoreKey = 5,
  Active = 6,
};

enum TreeIdx : uint32_t {
  TokenId = 0,
  ParentSlot = 1,
  Depth = 2,
  PathLogprobBits = 3,
  EdgeLogprobBits = 4,
  Valid = 5,
};

enum MetadataIdx : uint32_t {
  Depth = 0,
  AncestorCount = 1,
  TreeSlot = 2,
};

static const uint FRONTIER_NO_WINNER = ~0;

static const uint FRONTIER_MAX_SLOTS = 2048;

static const uint FRONTIER_MAX_WIDTH = 32;

static const uint PADDING_DEPTH = 0;