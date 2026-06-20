use proc_macros::kernel;

#[kernel(BuildPathMatrix)]
pub fn build_path_matrix(
    parent: *const i32,
    path_matrix: *mut u8,
    batch_size: u32,
    tree_size: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;

    unsafe {
        for batch in 0..batch_size {
            for row in 0..tree_size {
                let base = batch * tree_size * tree_size + row * tree_size;
                for col in 0..tree_size {
                    *path_matrix.add(base + col) = 0;
                }

                let mut cur = row as i32;
                for _ in 0..tree_size {
                    if cur < 0 {
                        break;
                    }
                    *path_matrix.add(base + cur as usize) = 1;
                    cur = *parent.add(batch * tree_size + cur as usize);
                }
            }
        }
    }
}
