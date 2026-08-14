use proc_macros::taps;

taps! {
    pub DecoderTap {
        #[tap(skip)]
        embedded,
        #[tap(rename = "activation_trace")]
        transformer: TransformerTap,
        logits,
    }
}

taps! {
    pub TransformerTap {
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
        #[tap(flatten)]
        transformer: TransformerTap,
        output_pooling,
        logits,
    }
}
