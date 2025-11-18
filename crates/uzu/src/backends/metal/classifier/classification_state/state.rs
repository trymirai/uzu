use std::{any::Any, cell::RefCell, collections::HashMap, rc::Rc};

use super::{aux_buffers::AuxBuffers, types::ArrayCell};
#[cfg(feature = "tracing")]
use crate::backends::metal::classifier::ClassifierActivationTrace;
use crate::{
    DataType, DeviceContext,
    backends::metal::{
        MTLContext, MetalArray,
        forward_pass::{
            ArrayId, EmbeddingsBuffers, ForwardPassBuffers, ForwardPassState,
            HashMapId, KVCache, ModelShape, RopeType, SharedBuffers,
            traces::DecoderActivationTrace,
        },
    },
};

pub struct ClassificationForwardPassState {
    context: Rc<MTLContext>,
    token_ids: ArrayCell,
    token_positions: ArrayCell,
    attention_bias: HashMap<Option<usize>, ArrayCell>,
    pub shared_buffers: Rc<RefCell<SharedBuffers>>,
    aux_buffers: AuxBuffers,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ClassifierActivationTrace>>,
}

impl ClassificationForwardPassState {
    pub fn new(
        context: Rc<MTLContext>,
        model_shape: &ModelShape,
        scratch: &ForwardPassBuffers,
        shared_buffers: Rc<RefCell<SharedBuffers>>,
        token_ids: &[u64],
        token_positions: &[usize],
        bidirectional_attention: bool,
        num_labels: usize,
    ) -> Self {
        let suffix_length = token_ids.len();
        assert_eq!(suffix_length, token_positions.len());

        let aux_buffers = AuxBuffers::new(
            scratch,
            model_shape,
            suffix_length,
            &context,
            num_labels,
        );

        let mut token_ids_array = unsafe {
            MetalArray::new(
                scratch.token_ids.clone(),
                &[suffix_length],
                DataType::U64,
            )
        };
        context.copy_from_view(&mut token_ids_array, token_ids.into());
        let token_ids = RefCell::new(token_ids_array);

        let mut token_positions_array = unsafe {
            MetalArray::new(
                scratch.token_positions.clone(),
                &[suffix_length],
                DataType::I32,
            )
        };
        let token_positions_i32: Box<[i32]> =
            token_positions.iter().map(|p| *p as i32).collect();
        context.copy_from_view(
            &mut token_positions_array,
            token_positions_i32.as_ref().into(),
        );
        let token_positions = RefCell::new(token_positions_array);

        let act_dtype = model_shape.activation_data_type();
        let mut attention_bias_map: HashMap<Option<usize>, MetalArray> =
            scratch
                .attention_window_size_to_bias
                .iter()
                .map(|(window_size, buffer)| {
                    let attention_bias_shape = [suffix_length, suffix_length];
                    let array = unsafe {
                        MetalArray::new(
                            buffer.clone(),
                            &attention_bias_shape,
                            act_dtype,
                        )
                    };
                    (*window_size, array)
                })
                .collect();

        for (window, bias_array) in attention_bias_map.iter_mut() {
            if bidirectional_attention {
                if let Some(window_size) = window {
                    let half_window = (window_size / 2) as isize;
                    context.fill_attention_bias(
                        bias_array,
                        suffix_length,
                        0,
                        |row, col| {
                            let distance = (row as isize) - (col as isize);
                            distance.abs() > half_window
                        },
                    );
                } else {
                    context.fill_attention_bias(
                        bias_array,
                        suffix_length,
                        0,
                        |_row, _col| false,
                    );
                }
            } else {
                context.fill_attention_bias(
                    bias_array,
                    suffix_length,
                    0,
                    |row, col| row < col,
                );
            }
        }

        let attention_bias: HashMap<Option<usize>, ArrayCell> =
            attention_bias_map
                .into_iter()
                .map(|(k, v)| (k, RefCell::new(v)))
                .collect();

        #[cfg(feature = "tracing")]
        let traces = Rc::new(RefCell::new(ClassifierActivationTrace::new(
            &context,
            model_shape,
            suffix_length,
            num_labels,
        )));

        Self {
            context,
            token_ids,
            token_positions,
            attention_bias,
            shared_buffers,
            aux_buffers,
            #[cfg(feature = "tracing")]
            traces,
        }
    }

    fn array_cell(
        &self,
        id: ArrayId,
    ) -> ArrayCell {
        match id {
            ArrayId::TokenIds => self.token_ids.clone(),
            ArrayId::TokenPositions => self.token_positions.clone(),
            ArrayId::Main => self.aux_buffers.main.clone(),
            ArrayId::Shortcut => self.aux_buffers.shortcut.clone(),
            ArrayId::QKV => self.aux_buffers.qkv.clone(),
            ArrayId::AttentionOutput => {
                self.aux_buffers.attention_output.clone()
            },
            ArrayId::MlpFusedUp => self.aux_buffers.mlp_fused_up.clone(),
            ArrayId::MlpHidden => self.aux_buffers.mlp_hidden.clone(),
            ArrayId::RotatedQueries => self.aux_buffers.rotated_queries.clone(),
            ArrayId::RotatedKeys => self.aux_buffers.rotated_keys.clone(),
            ArrayId::RotatedValues => self.aux_buffers.rotated_values.clone(),
            ArrayId::AttentionPartials => {
                self.aux_buffers.attention_partials.clone()
            },
            ArrayId::AttentionSums => self.aux_buffers.attention_sums.clone(),
            ArrayId::AttentionMaxs => self.aux_buffers.attention_maxs.clone(),

            ArrayId::EmbeddingsInputWeights => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Tied {
                        weights,
                    } => weights.clone(),
                    EmbeddingsBuffers::Untied {
                        input_weights,
                        ..
                    } => input_weights.clone(),
                    EmbeddingsBuffers::QuantizedTied {
                        weights,
                        ..
                    } => weights.clone(),
                }
            },
            ArrayId::EmbeddingsOutputWeights => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::Tied {
                        weights,
                    } => weights.clone(),
                    EmbeddingsBuffers::Untied {
                        output_weights,
                        ..
                    } => output_weights.clone(),
                    EmbeddingsBuffers::QuantizedTied {
                        weights,
                        ..
                    } => weights.clone(),
                }
            },
            ArrayId::EmbeddingsScales => {
                match &self.shared_buffers.borrow().embeddings {
                    EmbeddingsBuffers::QuantizedTied {
                        scales,
                        ..
                    } => scales.clone(),
                    _ => panic!("Expected EmbeddingsBuffers::QuantizedTied"),
                }
            },
            ArrayId::RopeCosines(rope_type) => match rope_type {
                RopeType::Global => {
                    self.shared_buffers.borrow().global_rope.cosines.clone()
                },
                RopeType::Local => self
                    .shared_buffers
                    .borrow()
                    .local_rope
                    .as_ref()
                    .expect("Local rope requested but not initialized")
                    .cosines
                    .clone(),
            },
            ArrayId::RopeSines(rope_type) => match rope_type {
                RopeType::Global => {
                    self.shared_buffers.borrow().global_rope.sines.clone()
                },
                RopeType::Local => self
                    .shared_buffers
                    .borrow()
                    .local_rope
                    .as_ref()
                    .expect("Local rope requested but not initialized")
                    .sines
                    .clone(),
            },

            ArrayId::ClassifierPooling => self.aux_buffers.pooling.clone(),
            ArrayId::ClassifierPredictionHeadDense => {
                self.aux_buffers.dense.clone()
            },
            ArrayId::ClassifierPredictionHeadNorm => {
                self.aux_buffers.norm.clone()
            },
            ArrayId::ClassifierPredictionHeadLogits => {
                self.aux_buffers.logits.clone()
            },
            _ => panic!("Unsupported ArrayId for classifier: {:?}", id),
        }
    }

    pub fn aux_buffers_suffix_length(&self) -> usize {
        self.aux_buffers.suffix_length()
    }
    pub fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }
    #[cfg(feature = "tracing")]
    pub fn classifier_traces(&self) -> &Rc<RefCell<ClassifierActivationTrace>> {
        &self.traces
    }
}

impl ForwardPassState for ClassificationForwardPassState {
    fn arrays(
        &self,
        ids: &[ArrayId],
    ) -> Box<[ArrayCell]> {
        ids.iter().map(|id| self.array_cell(*id)).collect()
    }
    fn hashmaps(
        &self,
        ids: &[HashMapId],
    ) -> Box<[HashMap<Option<usize>, ArrayCell>]> {
        ids.iter().map(|id| self.hashmap_cell(id).clone()).collect()
    }
    fn aux_buffers_suffix_length(&self) -> usize {
        self.aux_buffers.suffix_length()
    }
    fn mtl_context(&self) -> &Rc<MTLContext> {
        &self.context
    }
    fn shared_buffers(&self) -> &Rc<RefCell<SharedBuffers>> {
        &self.shared_buffers
    }
    fn kv_cache(&self) -> Option<&Rc<RefCell<KVCache>>> {
        None
    }
    fn sampling_output(&self) -> Option<&ArrayCell> {
        None
    }
    fn traces(&self) -> Option<&Rc<RefCell<DecoderActivationTrace>>> {
        None
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ClassificationForwardPassState {
    fn hashmap_cell(
        &self,
        id: &HashMapId,
    ) -> &HashMap<Option<usize>, ArrayCell> {
        match id {
            HashMapId::AttentionBias => &self.attention_bias,
        }
    }
}
