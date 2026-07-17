use proc_macros::kernel;

#[kernel(DFlashTopK)]
pub fn dflash_top_k(
    logits: *const f32,
    output_ids: *mut u32,
    output_scores: *mut f32,
    rows: u32,
    vocab_size: u32,
    k: u32,
) {
    let rows = rows as usize;
    let vocab_size = vocab_size as usize;
    let k = (k as usize).min(vocab_size);
    for row in 0..rows {
        let values = unsafe { std::slice::from_raw_parts(logits.add(row * vocab_size), vocab_size) };
        let mut indices = (0..vocab_size).collect::<Vec<_>>();
        let compare =
            |&left: &usize, &right: &usize| values[right].total_cmp(&values[left]).then_with(|| left.cmp(&right));
        if k < vocab_size {
            indices.select_nth_unstable_by(k, compare);
            indices.truncate(k);
        }
        indices.sort_by(compare);
        for (rank, index) in indices.into_iter().enumerate() {
            unsafe {
                *output_ids.add(row * k + rank) = index as u32;
                *output_scores.add(row * k + rank) = values[index];
            }
        }
    }
}
