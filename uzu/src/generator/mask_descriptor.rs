#[derive(Debug, Clone)]
pub struct GeneratorPrefixLength {
    pub real: usize,
    pub step: Option<usize>,
}

impl GeneratorPrefixLength {
    pub fn padded(&self) -> usize {
        if let Some(step) = self.step {
            let remainder = self.real % step;
            self.real
                + if remainder == 0 {
                    0
                } else {
                    step - remainder
                }
        } else {
            self.real
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorKVCacheUpdateTask {
    pub sources_indices: Vec<usize>,
    pub destination_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct GeneratorMaskDescriptor {
    pub suffix_length: usize,
    pub prefix_length: GeneratorPrefixLength,
    pub casual_mask: Option<Vec<Vec<bool>>>,
}

impl GeneratorMaskDescriptor {
    pub fn prefill_last_step_casual_mask(
        suffix_length: usize,
        speculated_tokens_start: usize,
        speculated_casual_mask: Vec<Vec<bool>>,
    ) -> Vec<Vec<bool>> {
        let mut mask: Vec<Vec<bool>> =
            vec![vec![false; suffix_length]; suffix_length];
        let mask_len = mask.len();

        for row in 0..mask_len {
            for column in 0..mask_len {
                mask[row][column] = row >= column;
            }
        }
        for (row, row_values) in speculated_casual_mask.iter().enumerate() {
            for (column, &value) in row_values.iter().enumerate() {
                mask[speculated_tokens_start + row]
                    [speculated_tokens_start + column] = value;
            }
        }
        return mask;
    }

    pub fn create_attention_bias_closure_with_prefix(
        &self,
        effective_prefix_length: usize,
    ) -> impl Fn(usize, usize) -> bool {
        // Compute padding based on the *effective* prefix length so that the attention
        // bias matrix is consistent with the real size that the GPU kernels expect.
        let step_opt = self.prefix_length.step;
        let prefix_length_real_effective = effective_prefix_length;
        let prefix_length_padded_effective = if let Some(step) = step_opt {
            let remainder = prefix_length_real_effective % step;
            prefix_length_real_effective
                + if remainder == 0 {
                    0
                } else {
                    step - remainder
                }
        } else {
            prefix_length_real_effective
        };

        let number_of_padded_tokens =
            prefix_length_padded_effective - prefix_length_real_effective;
        let casual_mask_cloned = self.casual_mask.clone();
        move |row: usize, column: usize| {
            // Columns < prefix_length_padded_effective correspond to prefix (incl. padding)
            if column < prefix_length_padded_effective {
                // Columns in the artificial padding area should be -inf
                if column < number_of_padded_tokens {
                    return true;
                } else {
                    // Real prefix token – always visible
                    return false;
                }
            }
            let suffix_column = column - prefix_length_padded_effective;
            // Now we are inside the suffix portion (0-based index)
            if let Some(suffix_mask) = &casual_mask_cloned {
                if row >= suffix_mask.len()
                    || suffix_column >= suffix_mask[0].len()
                {
                    return true; // outside provided mask => block
                }
                return !suffix_mask[row][suffix_column];
            }
            // Standard causal mask: block future tokens
            row < suffix_column
        }
    }

    pub fn kv_cache_update_task(&self) -> Option<GeneratorKVCacheUpdateTask> {
        let prefix_length_real = self.prefix_length.real;
        let prefix_length_padded = self.prefix_length.padded();
        if prefix_length_padded == prefix_length_real {
            return None;
        }

        let suffix_length = self.suffix_length;
        let sources_indices: Vec<usize> = (prefix_length_padded
            ..prefix_length_padded + suffix_length)
            .collect();
        let destination_indices: Vec<usize> =
            (prefix_length_real..prefix_length_real + suffix_length).collect();
        Some(GeneratorKVCacheUpdateTask {
            sources_indices,
            destination_indices,
        })
    }
}
