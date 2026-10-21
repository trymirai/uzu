use std::{fs::File, path::Path, sync::OnceLock};

use futures::lock::Mutex;

use crate::{
    backends::{
        Backend,
        common::{BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    },
    data_type::DataType,
    encodable_block::{
        conf_head::{
            config::ConfidenceHeadConfig,
            error::ConfidenceHeadError,
            tree::ConfidenceHeadTree,
            util::{create_normalization, create_normalization_config, map_norm_new_error},
            weights::get_local_weights_buffer,
        },
        embedding::Embedding,
        linear::Linear,
        normalization::{Normalization, PostLayerScalar, ShortcutMode},
    },
    safetensors::ParameterLoader,
    utils,
};

pub struct ConfidenceHead<B: Backend> {
    config: ConfidenceHeadConfig,
    local_weights: B::GlobalBuffer,
    tree: Option<ConfidenceHeadTree<B>>,
    token_features: OnceLock<B::GlobalBuffer>,
    token_features_lock: Mutex<()>,
    token_norm: Normalization<B>,
    token_proj: Box<dyn Linear<B>>,
    draft_norm: Normalization<B>,
    draft_proj: Box<dyn Linear<B>>,
}

impl<B: Backend> ConfidenceHead<B> {
    pub fn new(
        context: &B::Context,
        model_dir: &Path,
        conf_head_dir: &Path,
        config: ConfidenceHeadConfig,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        validate_config(&config)?;

        let model_path = model_dir.join("model.safetensors");
        let model_sha256_bytes = utils::hash::sha256_file(&model_path).map_err(ConfidenceHeadError::Io)?;
        let model_sha256_str: String = model_sha256_bytes.into_iter().map(|byte| format!("{byte:02x}")).collect();
        if model_sha256_str != config.speculator_sha256 {
            return Err(ConfidenceHeadError::ModelMismatch);
        }

        let weights_file_path = conf_head_dir.join("model.safetensors");
        let weights_file = File::open(weights_file_path)?;
        let weights_loader = ParameterLoader::new(&weights_file, context)?;
        let weights_tree = weights_loader.tree();

        let local_weights = get_local_weights_buffer(context, &config, &weights_tree)?;
        let tree: Option<ConfidenceHeadTree<B>> = if let Some(ref tree_config) = config.tree {
            Some(ConfidenceHeadTree::new(context, &weights_tree, &config, tree_config)?)
        } else {
            None
        };

        let norm_config = create_normalization_config(config.layer_norm_eps);
        let token_norm = Normalization::new_mixed_types(
            config.model_width,
            DataType::BF16,
            DataType::F32,
            DataType::F32,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &norm_config,
            &weights_tree.subtree("token_norm"),
            context,
        )
        .map_err(map_norm_new_error)?;

        let token_proj = <dyn Linear<B>>::new(
            config.model_width,
            config.hidden,
            DataType::F32,
            true,
            &weights_tree.subtree("token_projection"),
            context,
        )
        .map_err(ConfidenceHeadError::LinearBlock)?;

        let draft_norm = create_normalization(
            context,
            &weights_tree.subtree("draft_norm"),
            config.model_width,
            ShortcutMode::None,
            config.layer_norm_eps,
        )?;

        let draft_proj = <dyn Linear<B>>::new(
            config.model_width,
            config.hidden,
            DataType::F32,
            true,
            &weights_tree.subtree("draft"),
            context,
        )
        .map_err(ConfidenceHeadError::LinearBlock)?;

        Ok(Self {
            config,
            local_weights,
            tree,
            token_features: OnceLock::new(),
            token_features_lock: Mutex::new(()),
            token_norm,
            token_proj,
            draft_norm,
            draft_proj,
        })
    }

    pub fn config(&self) -> &ConfidenceHeadConfig {
        &self.config
    }

    /// Builds and caches an F32 [vocab_size, hidden] table from the target model's BF16 input embedding
    /// and the confidence head's learned projection:
    ///
    /// E_t       = target input embedding lookup(t)       // model_width values
    /// table[t]  = W_token × LayerNorm(E_t) + b_token     // hidden values
    pub async fn token_table(
        &self,
        context: &B::Context,
        embedding: &Embedding<B>,
    ) -> Result<&B::GlobalBuffer, ConfidenceHeadError<B>> {
        if embedding.model_dim() != self.config.model_width {
            return Err(ConfidenceHeadError::Unsupported(format!(
                "the target embedding is {} wide, the head reads {}",
                embedding.model_dim(),
                self.config.model_width
            )));
        }

        if let Some(buffer) = self.token_features.get() {
            return Ok(buffer);
        }
        let _guard = self.token_features_lock.lock().await;
        if let Some(buffer) = self.token_features.get() {
            return Ok(buffer);
        }

        let vocab_size = embedding.vocab_size() as usize;
        let hidden_size = self.config.hidden as usize;
        let table_chunk = 4096usize;
        let table_buffer_size = vocab_size * hidden_size * DataType::F32.size_in_bytes();
        let row_bytes = hidden_size * DataType::F32.size_in_bytes();

        let mut table_buffer = context.create_buffer(table_buffer_size).map_err(ConfidenceHeadError::Backend)?;
        let mut command_buffer = context
            .create_command_buffer(Some("confidence head token table"), None)
            .map_err(ConfidenceHeadError::Backend)?;

        let mut ids_buf =
            command_buffer.allocate_constant(vocab_size * size_of::<u32>()).map_err(ConfidenceHeadError::Backend)?;
        for (id, value) in ids_buf.as_slice_mut::<u32>().iter_mut().enumerate() {
            *value = id as u32;
        }

        for start in (0..vocab_size).step_by(table_chunk) {
            let count = table_chunk.min(vocab_size - start) as u32;
            let ids_start = start * size_of::<u32>();
            let ids_end = ids_start + count as usize * size_of::<u32>();
            let ids = (&ids_buf).subrange(ids_start..ids_end);
            let emb_rows = embedding.encode_lookup(ids, count, &mut command_buffer)?;

            let normalized = self
                .token_norm
                .encode(&emb_rows, 0, count, None::<&mut B::ScratchBuffer>, &mut command_buffer)
                .map_err(ConfidenceHeadError::Backend)?;
            let projected = self
                .token_proj
                .encode(normalized, count, &mut command_buffer)
                .map_err(ConfidenceHeadError::LinearEncode)?;

            let start_byte = start * row_bytes;
            let end_byte = start_byte + count as usize * row_bytes;
            command_buffer.encode_copy(&projected, (&mut table_buffer).subrange_mut(start_byte..end_byte));
        }
        command_buffer.end_encoding().submit().wait_until_completed().await.map_err(ConfidenceHeadError::Backend)?;

        Ok(self.token_features.get_or_init(|| table_buffer))
    }
}

fn validate_config<B: Backend>(config: &ConfidenceHeadConfig) -> Result<(), ConfidenceHeadError<B>> {
    let mut problems = Vec::new();

    let format = 2;
    if config.format != format {
        problems.push(format!("format {} (expected {format})", config.format));
    }

    let hidden_max = 64;
    if config.hidden == 0 || config.hidden > hidden_max {
        problems.push(format!("hidden {} outside 1..={hidden_max}", config.hidden));
    }

    if config.children == 0 || config.children > 4 || config.children * config.hidden > 256 {
        problems.push(format!("{} children of width {}", config.children, config.hidden));
    }

    let candidates_max = 512;
    if config.pool == 0 || config.pool > candidates_max {
        problems.push(format!("pool {} outside 1..={candidates_max}", config.pool));
    }

    let pool_hidden_max = 16;
    if config.pool_hidden == 0 || config.pool_hidden > pool_hidden_max {
        problems.push(format!("pool width {} outside 1..={pool_hidden_max}", config.pool_hidden));
    }

    let edge_hidden_max = 128;
    if config.edge_hidden == 0 || config.edge_hidden > edge_hidden_max {
        problems.push(format!("edge width {} outside 1..={edge_hidden_max}", config.edge_hidden));
    }

    // The Metal dense layers read four adjacent inputs at once:
    // every layer's input width is a multiple of four (hidden, 2 * pool width, 3 * hidden + 4, edge width).
    if !config.hidden.is_multiple_of(4)
        || !config.pool_hidden.is_multiple_of(2)
        || !config.edge_hidden.is_multiple_of(4)
    {
        problems.push(format!(
            "widths hidden {}, pool {}, edge {} not multiples of 4, 2, 4",
            config.hidden, config.pool_hidden, config.edge_hidden
        ));
    }

    if config.max_depth + 1 > 32 {
        problems.push(format!("max depth {}", config.max_depth));
    }

    if config.speculator_sha256.len() != 64
        || !config.speculator_sha256.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        problems.push(format!("speculator sha256 {:?} is not 64 lowercase hex digits", config.speculator_sha256));
    }

    if let Some(tree) = &config.tree {
        // Tree relations the attention biases: self, parent, child, ancestor two or more edges up, descendant likewise, sibling.
        let relations = 6;
        if tree.relations != relations {
            problems.push(format!("{} tree relations (expected {relations})", tree.relations));
        }

        if tree.heads == 0
            || tree.width % tree.heads != 0
            || 256 % (tree.width / tree.heads) != 0
            || tree.width / tree.heads > 64
            || (tree.width / tree.heads) % 8 != 0
        {
            problems.push(format!("{} heads over width {}", tree.heads, tree.width));
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(ConfidenceHeadError::Unsupported(problems.join("; ")))
    }
}
