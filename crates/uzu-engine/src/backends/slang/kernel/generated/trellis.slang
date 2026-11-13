static const uint COLUMN_GROUP_COUNT = 4;

static const uint MAX_DIMENSION = 20480;

static const uint MIN_HADAMARD_SIZE = 512;

static const uint MAX_HADAMARD_SIZE = 4096;

struct TrellisCodebook {
  float scale;
  float offsets[4];
};