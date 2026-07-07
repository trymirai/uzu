mod arguments;
mod kernel;

pub use arguments::AttentionGemmArgs;
pub use kernel::AttentionGemmDispatch;

use crate::backends::common::gpu_types::attention::AttnParams;

pub(crate) fn retile_params(
    mut params: AttnParams,
    bq: u32,
    bk: u32,
) -> AttnParams {
    params.nq_aligned = params.q_len / bq;
    params.q_rem = params.q_len % bq;
    params.nk = params.k_len.div_ceil(bk);
    params.nk_aligned = params.k_len / bk;
    params.k_rem = params.k_len % bk;
    params
}
