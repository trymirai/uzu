use itertools::Itertools;
use xxhash_rust::xxh3::xxh3_64;

use super::wrapper::{KernelWrappers, VariantWrapper};

const MIN_VARIANTS_PER_SHARD: usize = 8;
const MAX_VARIANTS_PER_SHARD: usize = 64;

pub fn shard_footers(kernel_wrappers: &[KernelWrappers]) -> Vec<String> {
    let total_variants: usize = kernel_wrappers.iter().map(|kernel| kernel.variants.len()).sum();
    let min_shards = total_variants.div_ceil(MAX_VARIANTS_PER_SHARD);
    let ncpu = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let num_shards = total_variants.div_ceil(MIN_VARIANTS_PER_SHARD).clamp(min_shards, min_shards.max(ncpu));
    let num_shards = if num_shards >= ncpu {
        num_shards.div_ceil(ncpu) * ncpu
    } else {
        num_shards
    };

    let mut footers = vec![String::new(); num_shards];
    for kernel in kernel_wrappers {
        let mut emit = |index: usize, variants: Vec<&VariantWrapper>| {
            let footer = &mut footers[index];
            footer.push_str(kernel.header.as_deref().unwrap_or(""));
            footer.push_str(&variants.iter().map(|variant| variant.source.as_ref()).join(""));
            footer.push_str(kernel.footer.as_deref().unwrap_or(""));
        };

        if num_shards == 1 {
            emit(0, kernel.variants.iter().collect());
        } else {
            for (index, variants) in kernel
                .variants
                .iter()
                .into_group_map_by(|variant| (xxh3_64(variant.name.as_bytes()) % num_shards as u64) as usize)
                .into_iter()
                .sorted_unstable_by_key(|(index, _)| *index)
            {
                emit(index, variants);
            }
        }
    }
    footers
}
