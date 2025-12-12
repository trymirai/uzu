//! Prediction head encodable for classification output.

use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{EncodableBlock, EncodingParameters};
#[cfg(feature = "tracing")]
use crate::Array;
#[cfg(feature = "tracing")]
use crate::backends::metal::forward_pass::ArrayId;
use crate::backends::metal::forward_pass::ForwardPassState;

pub struct ClassifierPredictionHead {
    dense: Box<dyn EncodableBlock>,
    activation: Box<dyn EncodableBlock>,
    norm: Box<dyn EncodableBlock>,
    readout: Box<dyn EncodableBlock>,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    num_labels: usize,
}

impl ClassifierPredictionHead {
    pub fn new(
        dense: Box<dyn EncodableBlock>,
        activation: Box<dyn EncodableBlock>,
        norm: Box<dyn EncodableBlock>,
        readout: Box<dyn EncodableBlock>,
        num_labels: usize,
    ) -> Self {
        Self {
            dense,
            activation,
            norm,
            readout,
            num_labels,
        }
    }
}

impl EncodableBlock for ClassifierPredictionHead {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        self.dense.encode(state, command_buffer, parameters);
        self.activation.encode(state, command_buffer, parameters);
        self.norm.encode(state, command_buffer, parameters);
        self.readout.encode(state, command_buffer, parameters);

        #[cfg_attr(feature = "tracing", allow(unused_variables))]
        let root_after_head = command_buffer.root_command_buffer().to_owned();
        command_buffer.commit_and_continue();

        #[cfg(not(feature = "tracing"))]
        root_after_head.wait_until_completed();

        #[cfg(feature = "tracing")]
        {
            let traces_rc = state.traces().clone();
            let logits_arrays =
                state.arrays(&[ArrayId::ClassifierPredictionHeadLogits]);
            let mut logits_array_ref = logits_arrays[0].borrow_mut();
            let linear_output_buffer =
                unsafe { logits_array_ref.mtl_buffer().to_owned() };
            let data_type = Array::data_type(&*logits_array_ref);
            let batch_size = Array::shape(&*logits_array_ref)[0];
            drop(logits_array_ref);

            let traces_ref = traces_rc.borrow();
            let mut trace_logits = traces_ref.logits.borrow_mut();
            let dst_trace_buf = unsafe { trace_logits.mtl_buffer().to_owned() };
            drop(trace_logits);
            drop(traces_ref);

            let copy_size_bytes =
                (batch_size * self.num_labels * data_type.size_in_bytes())
                    as u64;

            let root = command_buffer.root_command_buffer();
            let blit = root.new_blit_command_encoder();
            blit.copy_from_buffer(
                &linear_output_buffer,
                0,
                &dst_trace_buf,
                0,
                copy_size_bytes,
            );
            blit.end_encoding();

            let root_owned = root.to_owned();
            command_buffer.commit_and_continue();
            root_owned.wait_until_completed();
        }
    }
}
