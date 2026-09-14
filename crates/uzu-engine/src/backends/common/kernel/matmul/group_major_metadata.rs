const GROUP_STRIDE_ALIGNMENT: u32 = 4;
const COLUMN_TILE: usize = 512;

pub const fn row_stride(columns: u32) -> u32 {
    columns.next_multiple_of(GROUP_STRIDE_ALIGNMENT)
}

pub fn plane_bytes(
    columns: u32,
    groups: u32,
    bits: u32,
) -> usize {
    groups as usize * row_stride(columns) as usize * bits as usize / u8::BITS as usize
}

/// Rewrites a checkpoint-order plane into group-major order inside its own allocation.
pub fn transpose(
    plane: &mut [u8],
    columns: u32,
    groups: u32,
    bits: u32,
) {
    let source_bytes = columns as usize * (groups as usize * bits as usize).div_ceil(u8::BITS as usize);
    assert!(plane.len() >= source_bytes, "checkpoint layout plane does not fit its allocation");
    assert!(plane.len() >= plane_bytes(columns, groups, bits), "[G, N] layout plane does not fit its allocation");
    let source = plane[..source_bytes].to_vec();
    transpose_from(&source, plane, columns, groups, bits);
}

fn transpose_from(
    source: &[u8],
    output: &mut [u8],
    columns: u32,
    groups: u32,
    bits: u32,
) {
    let row_bytes = row_stride(columns) as usize * bits as usize / u8::BITS as usize;
    let (groups, columns) = (groups as usize, columns as usize);
    let jobs = columns * groups;
    for_each_row_block(output, row_bytes, groups, jobs, |first_row, block| {
        block.fill(0);
        match bits {
            4 => transpose_u4(source, columns, groups, row_bytes, first_row, block),
            8 => transpose_values::<1>(source, columns, groups, row_bytes, first_row, block),
            16 => transpose_values::<2>(source, columns, groups, row_bytes, first_row, block),
            32 => transpose_values::<4>(source, columns, groups, row_bytes, first_row, block),
            _ => panic!("no [G, N] transpose for {bits}-bit entries"),
        }
    });
}

fn for_each_row_block<F>(
    destination: &mut [u8],
    row_bytes: usize,
    rows: usize,
    jobs: usize,
    task: F,
) where
    F: Fn(usize, &mut [u8]) + Sync,
{
    const JOBS_PER_WORKER: usize = 1 << 16;
    let num_workers = std::thread::available_parallelism()
        .map_or(1, |parallelism| parallelism.get().saturating_sub(1).max(1))
        .min(rows)
        .min((jobs / JOBS_PER_WORKER).max(1));
    let rows_per_worker = rows.div_ceil(num_workers);
    std::thread::scope(|scope| {
        for (index, block) in destination[..row_bytes * rows].chunks_mut(rows_per_worker * row_bytes).enumerate() {
            let task = &task;
            scope.spawn(move || task(index * rows_per_worker, block));
        }
    });
}

fn transpose_values<const WIDTH: usize>(
    source: &[u8],
    columns: usize,
    groups: usize,
    row_bytes: usize,
    first_row: usize,
    block: &mut [u8],
) {
    const GROUP_BLOCK: usize = 8;
    let (source, _) = source[..columns * groups * WIDTH].as_chunks::<WIDTH>();
    let row_entries = row_bytes / WIDTH;
    for (index, row_block) in block.chunks_mut(GROUP_BLOCK * row_bytes).enumerate() {
        let first_group = first_row + index * GROUP_BLOCK;
        let (entries, _) = row_block.as_chunks_mut::<WIDTH>();
        let block_rows = entries.len() / row_entries;
        for column_start in (0..columns).step_by(COLUMN_TILE) {
            let column_end = (column_start + COLUMN_TILE).min(columns);
            for column in column_start..column_end {
                let source_run = &source[column * groups + first_group..][..block_rows];
                for (row, value) in source_run.iter().enumerate() {
                    entries[row * row_entries + column] = *value;
                }
            }
        }
    }
}

fn transpose_u4(
    source: &[u8],
    columns: usize,
    groups: usize,
    row_bytes: usize,
    first_row: usize,
    block: &mut [u8],
) {
    const BITS_PER_NIBBLE: usize = 4;
    const NIBBLES_PER_BYTE: usize = u8::BITS as usize / BITS_PER_NIBBLE;
    const NIBBLE_MASK: u8 = (1 << BITS_PER_NIBBLE) - 1;

    let source_group_stride = groups.div_ceil(NIBBLES_PER_BYTE);
    for column_start in (0..columns).step_by(COLUMN_TILE) {
        let column_end = (column_start + COLUMN_TILE).min(columns);
        for (row, destination_row) in block.chunks_mut(row_bytes).enumerate() {
            let group = first_row + row;
            let source_byte = group / NIBBLES_PER_BYTE;
            let nibble_shift = (group % NIBBLES_PER_BYTE) * BITS_PER_NIBBLE;
            let nibble =
                |column: usize| (source[column * source_group_stride + source_byte] >> nibble_shift) & NIBBLE_MASK;
            for byte_index in column_start / NIBBLES_PER_BYTE..column_end.div_ceil(NIBBLES_PER_BYTE) {
                let low_column = byte_index * NIBBLES_PER_BYTE;
                let high_column = low_column + 1;
                let high_nibble = (high_column < columns).then(|| nibble(high_column)).unwrap_or(0);
                destination_row[byte_index] = nibble(low_column) | (high_nibble << BITS_PER_NIBBLE);
            }
        }
    }
}
