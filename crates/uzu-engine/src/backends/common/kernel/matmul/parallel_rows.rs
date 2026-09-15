pub(super) fn for_each_block<F>(
    rows: &mut [u8],
    row_bytes: usize,
    entries: usize,
    task: F,
) where
    F: Fn(usize, &mut [u8]) + Sync,
{
    const ENTRIES_PER_WORKER: usize = 1 << 17;

    let row_count = rows.len() / row_bytes;
    let num_workers = std::thread::available_parallelism()
        .map_or(1, |parallelism| parallelism.get().saturating_sub(1).max(1))
        .min(row_count)
        .min((entries / ENTRIES_PER_WORKER).max(1));
    let rows_per_worker = row_count.div_ceil(num_workers);
    if num_workers == 1 {
        return task(0, rows);
    }
    std::thread::scope(|scope| {
        for (index, block) in rows.chunks_mut(rows_per_worker * row_bytes).enumerate() {
            let task = &task;
            scope.spawn(move || task(index * rows_per_worker, block));
        }
    });
}
