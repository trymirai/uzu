use proc_macros::kernel;

#[kernel(RadixTopK)]
pub fn radix_top_k(
    input: *const f32,
    output_ids: *mut u32,
    output_scores: *mut f32,
    rows: u32,
    columns: u32,
    k: u32,
) {
    let rows = rows as usize;
    let columns = columns as usize;
    let k = (k as usize).min(columns);
    for row in 0..rows {
        let values = unsafe { std::slice::from_raw_parts(input.add(row * columns), columns) };
        let mut indices = (0..columns).collect::<Vec<_>>();
        let compare =
            |&left: &usize, &right: &usize| values[right].total_cmp(&values[left]).then_with(|| left.cmp(&right));
        if k < columns {
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
