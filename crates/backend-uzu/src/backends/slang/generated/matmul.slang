struct GemmParams {
  uint M;
  uint N;
  uint K;
  uint leading_dimension_a;
  uint leading_dimension_b;
  uint leading_dimension_d;
  uint threadgroups_per_column;
  uint threadgroups_per_row;
  uint swizzle_log;
  uint aligned_inner_iterations;
  bool use_morton;
};