use super::trace_debug::TraceDebugSnapshot;

#[derive(Debug, Clone)]
pub struct TargetHiddenLayerSnapshot {
    pub layer_index: usize,
    pub active_row_count: usize,
    pub model_dim: usize,
    pub outputs: Box<[f32]>,
}

#[derive(Debug, Clone)]
pub struct TargetHiddenSnapshot {
    pub layers: Box<[TargetHiddenLayerSnapshot]>,
}

impl TargetHiddenSnapshot {
    pub fn from_trace_snapshot(
        snapshot: &TraceDebugSnapshot,
        sample_count: usize,
    ) -> Self {
        assert!(sample_count > 0, "sample_count must be positive");
        assert!(!snapshot.layers.is_empty(), "trace snapshot must contain at least one layer");

        let sampled_layer_indices = sampled_layer_indices(snapshot.layers.len(), sample_count);
        let layers = sampled_layer_indices
            .into_iter()
            .map(|layer_index| {
                let layer = &snapshot.layers[layer_index];
                TargetHiddenLayerSnapshot {
                    layer_index: layer.layer_index,
                    active_row_count: layer.active_row_count,
                    model_dim: layer.model_dim,
                    outputs: layer.outputs.clone().into_boxed_slice(),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            layers,
        }
    }
}

fn sampled_layer_indices(
    layer_count: usize,
    sample_count: usize,
) -> Box<[usize]> {
    let sampled_count = sample_count.min(layer_count);
    let last_layer = layer_count - 1;

    (0..sampled_count).map(|step| step * last_layer / (sampled_count - 1).max(1)).collect::<Vec<_>>().into_boxed_slice()
}
