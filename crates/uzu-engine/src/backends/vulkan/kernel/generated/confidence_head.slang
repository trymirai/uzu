static const uint MODEL_MAX = 64;

static const uint POOL_HIDDEN_MAX = 16;

static const uint EDGE_HIDDEN_MAX = 128;

static const uint CHILDREN_MAX = 4;

static const uint SLOT_NUMBERS = 4;

static const uint POOL_FEATURES = 3;

static const uint POOL_INPUTS = 3;

static const uint RELATIONS = 6;

static const uint TREE_ROWS_MAX = 1024;

struct ConfidenceHeadParams {
  uint candidates;
  uint children;
  uint tree_slot_count;
  uint vocab_size;
  float logit_cap;
  float epsilon;
};