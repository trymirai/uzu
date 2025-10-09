// Shared test utilities
#[path = "moe_test_utils.rs"]
mod test_utils;

// Unit tests for individual kernels
#[path = "moe_block_e2e_test.rs"]
mod block_e2e_test;
#[path = "moe_bucket_counts_test.rs"]
mod bucket_counts_test;
#[path = "moe_experts_perf_test.rs"]
mod experts_perf_test;
#[path = "moe_experts_test.rs"]
mod experts_test;
#[path = "moe_finalize_test.rs"]
mod finalize_test;
#[path = "moe_gather_test.rs"]
mod gather_test;
#[path = "moe_offsets_scan_test.rs"]
mod offsets_scan_test;
#[path = "moe_perf_test.rs"]
mod perf_test;
#[path = "moe_router_test.rs"]
mod router_test;
#[path = "moe_scatter_test.rs"]
mod scatter_test;
#[path = "moe_tiles_test.rs"]
mod tiles_test;
#[path = "moe_topk_select_test.rs"]
mod topk_select_test;
