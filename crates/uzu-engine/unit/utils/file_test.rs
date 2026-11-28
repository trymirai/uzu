use std::{io, path::PathBuf};

use uzu_engine_macros::uzu_test;

use super::read_exact_at;

/// Byte `index` of the file written with `seed`. A hash of both, so a read from
/// the wrong offset or the wrong file shows up whatever the distance.
fn byte(
    seed: u8,
    index: usize,
) -> u8 {
    let hash = ((index as u64) ^ ((seed as u64) << 56)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (hash >> 56) as u8
}

fn write_file(
    name: &str,
    seed: u8,
    len: usize,
) -> PathBuf {
    let path = std::env::temp_dir().join(format!("uzu-{name}-{}.bin", std::process::id()));
    std::fs::write(&path, (0..len).map(|index| byte(seed, index)).collect::<Vec<_>>()).expect("write file");
    path
}

/// `(file index, offset, len)` spread over both files, in no particular order.
fn read_plan(
    count: usize,
    file_len: usize,
) -> Vec<(usize, u64, usize)> {
    (0..count)
        .map(|index| {
            let len = 1 + index * 7 % 300;
            let offset = (index * 7919) % (file_len - len);
            (index % 2, offset as u64, len)
        })
        .collect()
}

fn check_reads(
    files: &[std::fs::File; 2],
    seeds: [u8; 2],
    plan: &[(usize, u64, usize)],
) {
    let mut buffers = plan.iter().map(|&(_, _, len)| vec![0u8; len]).collect::<Vec<_>>();
    let reads = plan
        .iter()
        .zip(&mut buffers)
        .map(|(&(file, offset, _), buffer)| (&files[file], offset, buffer.as_mut_slice()))
        .collect();
    read_exact_at(reads).expect("read");
    for (&(file, offset, len), buffer) in plan.iter().zip(&buffers) {
        let expected =
            (offset as usize..offset as usize + len).map(|index| byte(seeds[file], index)).collect::<Vec<_>>();
        assert_eq!(buffer, &expected, "read of {len} bytes at {offset} from file {file}");
    }
}

#[uzu_test]
fn reads_every_buffer() {
    const LEN: usize = 1 << 20;
    let paths = [write_file("read-exact-at-a", 1, LEN), write_file("read-exact-at-b", 2, LEN)];
    let files = paths.each_ref().map(|path| std::fs::File::open(path).expect("open"));

    check_reads(&files, [1, 2], &read_plan(5000, LEN));
    check_reads(&files, [1, 2], &read_plan(1, LEN));

    for path in paths {
        std::fs::remove_file(path).expect("remove file");
    }
}

#[uzu_test]
fn concurrent_calls_get_their_own_rows() {
    const LEN: usize = 1 << 20;
    let paths = [write_file("read-exact-at-concurrent-a", 3, LEN), write_file("read-exact-at-concurrent-b", 4, LEN)];
    let files = paths.each_ref().map(|path| std::fs::File::open(path).expect("open"));

    std::thread::scope(|scope| {
        for caller in 0..8 {
            let files = &files;
            scope.spawn(move || {
                for round in 0..20 {
                    let plan = read_plan(1 + (caller * 37 + round * 101) % 500, LEN);
                    check_reads(files, [3, 4], &plan);
                }
            });
        }
    });

    for path in paths {
        std::fs::remove_file(path).expect("remove file");
    }
}

#[uzu_test]
fn empty_reads_succeed() {
    read_exact_at(Vec::new()).expect("no reads");
}

#[uzu_test]
fn short_file_fails_and_pool_keeps_working() {
    const LEN: usize = 1 << 20;
    let path = write_file("read-exact-at-short", 5, LEN);
    let file = std::fs::File::open(&path).expect("open");

    // The failing read goes first, so the many after it are skipped or still in
    // flight when it fails, and all buffers are dropped as soon as the call returns.
    for _ in 0..10 {
        let mut past_end = vec![0u8; 100];
        let mut buffers =
            read_plan(3000, LEN).iter().map(|&(_, offset, len)| (offset, vec![0u8; len])).collect::<Vec<_>>();
        let reads = std::iter::once((&file, LEN as u64 - 50, past_end.as_mut_slice()))
            .chain(buffers.iter_mut().map(|(offset, buffer)| (&file, *offset, buffer.as_mut_slice())))
            .collect();
        let error = read_exact_at(reads).expect_err("reading past the end must fail");
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    let mut after = vec![0u8; 10];
    read_exact_at(vec![(&file, 10, after.as_mut_slice())]).expect("read after a failed call");
    assert_eq!(after, (10..20).map(|index| byte(5, index)).collect::<Vec<_>>());

    std::fs::remove_file(path).expect("remove file");
}
