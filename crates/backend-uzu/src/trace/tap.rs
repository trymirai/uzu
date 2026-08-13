//! Tap trees. Field names are the safetensors path segments, so these
//! declarations *are* the trace layout; `#[tap(rename)]` maps uzu's block names
//! onto the ones lalamo expects.

use proc_macros::taps;

taps! {
    pub DecoderTap {
        // uzu materializes the embedding lookup, lalamo does not name it.
        #[tap(skip)]
        embedded,
        #[tap(rename = "activation_trace")]
        transformer: TransformerTap,
        logits,
    }
}

taps! {
    pub TransformerTap {
        // Host-built i32 arrays with no device counterpart; filled by the model's
        // `record_trace` once encoding is done, not by `Transformer::encode`.
        token_ids,
        token_positions,
        rope_embeddings: [RopeTap],
        #[tap(rename = "layer_results")]
        layers: [TransformerLayerTap],
        output_norm,
    }
}

taps! {
    pub TransformerLayerTap {
        outputs,
        #[tap(rename = "activation_trace")]
        activations: TransformerLayerActivationsTap,
    }
}

taps! {
    pub TransformerLayerActivationsTap {
        inputs,
        pre_mixer_norm,
        mixer,
        post_mixer_norm,
        mlp_inputs,
        pre_mlp_norm,
        mlp,
        post_mlp_norm,
    }
}

taps! {
    pub RopeTap {
        cosines,
        sines,
    }
}

taps! {
    pub ClassifierTap {
        logits,
        #[tap(rename = "activation_trace")]
        activations: ClassifierActivationsTap,
    }
}

taps! {
    pub ClassifierActivationsTap {
        embedding_norm_output,
        // lalamo folds the transformer's arrays straight into the classifier's
        // activation trace rather than nesting them.
        #[tap(flatten)]
        transformer: TransformerTap,
        output_pooling,
        // Also present at the root; `Array` owns its allocation and is not `Clone`,
        // so this one is captured separately.
        logits,
    }
}
