use half::bf16;
use uzu_engine_macros::kernel;

#[kernel(KVCacheUpdate)]
pub fn kv_cache_update(
    in_place_k_codes: *mut i8,
    #[optional(!keys_only)] in_place_v_codes: Option<*mut u8>,
    in_place_k_scales: *mut bf16,
    #[optional(!keys_only)] in_place_v_scales: Option<*mut bf16>,
    copies: &[crate::backends::common::gpu_types::kv_cache_update::Copy],
    copy_count: u32,
    #[specialize] head_dim: u32,
    #[specialize] num_kv_heads: u32,
    #[specialize] backend_opaque_layout: u32,
    #[specialize] keys_only: bool,
) {
    assert_eq!(backend_opaque_layout, 0);
    assert_eq!(in_place_v_codes.is_some(), !keys_only);
    let row_width = (num_kv_heads * head_dim) as usize;
    for copy in copies.iter().take(copy_count as usize) {
        let source = copy.source as usize * row_width;
        let destination = copy.destination as usize * row_width;
        unsafe {
            std::ptr::copy(in_place_k_codes.add(source), in_place_k_codes.add(destination), row_width);
            if let Some(v_codes) = in_place_v_codes {
                std::ptr::copy(v_codes.add(source), v_codes.add(destination), row_width);
            }
            let source = (copy.source * num_kv_heads) as usize;
            let destination = (copy.destination * num_kv_heads) as usize;
            std::ptr::copy(in_place_k_scales.add(source), in_place_k_scales.add(destination), num_kv_heads as usize);
            if let Some(v_scales) = in_place_v_scales {
                std::ptr::copy(v_scales.add(source), v_scales.add(destination), num_kv_heads as usize);
            }
        }
    }
}
