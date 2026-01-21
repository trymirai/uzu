//! MLP block encodable.

use std::{cell::RefCell, rc::Rc};

use crate::backends::metal::{
    Buffer, BufferRef, CommandBufferRef, ComputeCommandEncoderRef,
    MTLCommandBuffer, MTLCommandEncoder,
};

use super::{EncodableBlock, EncodingParameters};
use crate::{
    Array, DataType,
    backends::metal::{
        MTLContext,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::{
            mlp::{MlpActivationType, MlpGateActMulEncodable},
            mlp_fused::{MlpFusedArguments, MlpFusedKernel},
            quant_matmul::{
                MlpFusedQmmArguments, MlpFusedQmmKernel, MlpFusedQmvArguments,
                MlpFusedQmvKernel,
            },
        },
    },
    config::Activation,
};

pub struct MlpBlock {
    up: Box<dyn EncodableBlock>,
    gate: MlpGateActMulEncodable,
    down: Box<dyn EncodableBlock>,
}

impl MlpBlock {
    pub fn new(
        up: Box<dyn EncodableBlock>,
        gate: MlpGateActMulEncodable,
        down: Box<dyn EncodableBlock>,
    ) -> Self {
        Self {
            up,
            gate,
            down,
        }
    }
}

impl EncodableBlock for MlpBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: CommandBufferRef<'_>,
        params: &EncodingParameters,
    ) {
        if self.supports_shared_encoder() {
            let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
            self.encode_with_shared_encoder(state, &encoder, params);
            encoder.end_encoding();
        } else {
            // Up
            self.up.encode(state, command_buffer, params);

            // Gate act+mul (fused_up -> hidden)
            {
                let arrays =
                    state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
                let mut fused = arrays[0].borrow_mut();
                let mut hidden = arrays[1].borrow_mut();
                let m = fused.shape()[0] as i32;
                let fused_buf = unsafe { fused.mtl_buffer() };
                let hidden_buf = unsafe { hidden.mtl_buffer() };

                let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
                self.gate
                    .encode(&encoder, fused_buf, hidden_buf, m)
                    .expect("Failed to encode MLP activation/mul kernel");
                encoder.end_encoding();
            }

            // Down
            self.down.encode(state, command_buffer, params);
        }

        if params.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.up.supports_shared_encoder() && self.down.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: ComputeCommandEncoderRef<'_>,
        params: &EncodingParameters,
    ) {
        // Up
        self.up.encode_with_shared_encoder(state, encoder, params);

        // Gate act+mul (fused_up -> hidden)
        let arrays = state.arrays(&[ArrayId::MlpFusedUp, ArrayId::MlpHidden]);
        let mut fused = arrays[0].borrow_mut();
        let mut hidden = arrays[1].borrow_mut();
        let m = fused.shape()[0] as i32;
        let fused_buf = unsafe { fused.mtl_buffer() };
        let hidden_buf = unsafe { hidden.mtl_buffer() };
        self.gate
            .encode(encoder, fused_buf, hidden_buf, m)
            .expect("Failed to encode MLP activation/mul kernel");
        drop(fused);
        drop(hidden);

        // Down
        self.down.encode_with_shared_encoder(state, encoder, params);
    }
}

/// MLP Fused Up Projection Variant
/// Selects between full precision and quantized fused kernels
pub enum MlpFusedUpKernel {
    FullPrecision {
        kernel: RefCell<MlpFusedKernel>,
    },
    Quantized {
        qmv: MlpFusedQmvKernel,
        qmm: MlpFusedQmmKernel,
    },
}

/// MLP block with fused up projection and activation
/// Combines up * activation(gate) into a single kernel call
pub struct MlpFusedBlock {
    context: Rc<MTLContext>,
    fused_up: MlpFusedUpKernel,
    down: Box<dyn EncodableBlock>,
    weights_buffer: Buffer,
    scales_buffer: Option<Buffer>,
    zero_points_or_biases_buffer: Option<Buffer>,
    input_dim: usize,
    hidden_dim: usize,
    activation: MlpActivationType,
    input_array_id: ArrayId,
    hidden_array_id: ArrayId,
}

impl MlpFusedBlock {
    /// Create MLP fused block with full precision weights
    pub fn new_full_precision(
        context: Rc<MTLContext>,
        data_type: DataType,
        weights_buffer: Buffer,
        input_dim: usize,
        hidden_dim: usize,
        activation: &Activation,
        down: Box<dyn EncodableBlock>,
        input_array_id: ArrayId,
        hidden_array_id: ArrayId,
    ) -> Result<Self, crate::backends::metal::MTLError> {
        let kernel = MlpFusedKernel::new(data_type, true)?; // Weights transposed

        Ok(Self {
            context,
            fused_up: MlpFusedUpKernel::FullPrecision {
                kernel: RefCell::new(kernel),
            },
            down,
            weights_buffer,
            scales_buffer: None,
            zero_points_or_biases_buffer: None,
            input_dim,
            hidden_dim,
            activation: MlpActivationType::from(activation),
            input_array_id,
            hidden_array_id,
        })
    }

    /// Create MLP fused block with quantized weights
    pub fn new_quantized(
        context: Rc<MTLContext>,
        data_type: DataType,
        weights_buffer: Buffer,
        scales_buffer: Buffer,
        zero_points_or_biases_buffer: Buffer,
        input_dim: usize,
        hidden_dim: usize,
        group_size: usize,
        mode: crate::config::QuantizationMode,
        quantization_type: crate::backends::metal::kernel::quant_matmul::QuantizationType,
        activation: &Activation,
        down: Box<dyn EncodableBlock>,
        input_array_id: ArrayId,
        hidden_array_id: ArrayId,
    ) -> Result<Self, crate::backends::metal::MTLError> {
        let qmv = MlpFusedQmvKernel::new(
            &context,
            data_type,
            group_size,
            mode,
            quantization_type,
        )
        .map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!("{:?}", e))
        })?;

        let qmm = MlpFusedQmmKernel::new(
            &context,
            data_type,
            group_size,
            mode,
            quantization_type,
        )
        .map_err(|e| {
            crate::backends::metal::MTLError::Generic(format!("{:?}", e))
        })?;

        Ok(Self {
            context,
            fused_up: MlpFusedUpKernel::Quantized {
                qmv,
                qmm,
            },
            down,
            weights_buffer,
            scales_buffer: Some(scales_buffer),
            zero_points_or_biases_buffer: Some(zero_points_or_biases_buffer),
            input_dim,
            hidden_dim,
            activation: MlpActivationType::from(activation),
            input_array_id,
            hidden_array_id,
        })
    }

    fn encode_fused_up(
        &self,
        encoder: ComputeCommandEncoderRef<'_>,
        input: BufferRef<'_>,
        input_offset: u64,
        output: BufferRef<'_>,
        batch: i32,
    ) {
        match &self.fused_up {
            MlpFusedUpKernel::FullPrecision {
                kernel,
            } => {
                let args = MlpFusedArguments {
                    input,
                    input_offset,
                    weights: &self.weights_buffer,
                    output,
                    batch,
                    input_dim: self.input_dim as i32,
                    hidden_dim: self.hidden_dim as i32,
                    lda: self.input_dim as i32,
                    ldb: self.input_dim as i32,
                    ldd: self.hidden_dim as i32,
                    batch_count: batch,
                    activation: self.activation,
                };
                kernel
                    .borrow_mut()
                    .encode(&self.context, encoder, &args)
                    .expect("Failed to encode MLP fused kernel");
            },
            MlpFusedUpKernel::Quantized {
                qmv,
                qmm,
            } => {
                let scales = self.scales_buffer.as_ref().unwrap();
                let zp_or_biases =
                    self.zero_points_or_biases_buffer.as_ref().unwrap();

                let is_decode = batch == 1;
                if is_decode {
                    let args = MlpFusedQmvArguments {
                        weights: &self.weights_buffer,
                        scales,
                        zero_points_or_biases: zp_or_biases,
                        input,
                        input_offset,
                        output,
                        input_dim: self.input_dim as i32,
                        hidden_dim: self.hidden_dim as i32,
                        batch_count: batch,
                    };
                    qmv.encode(encoder, &args)
                        .expect("Failed to encode MLP fused QMV");
                } else {
                    let args = MlpFusedQmmArguments {
                        weights: &self.weights_buffer,
                        scales,
                        zero_points_or_biases: zp_or_biases,
                        input,
                        input_offset,
                        output,
                        batch,
                        input_dim: self.input_dim as i32,
                        hidden_dim: self.hidden_dim as i32,
                    };
                    qmm.encode(encoder, &args)
                        .expect("Failed to encode MLP fused QMM");
                }
            },
        }
    }
}

impl EncodableBlock for MlpFusedBlock {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: CommandBufferRef<'_>,
        params: &EncodingParameters,
    ) {
        if self.supports_shared_encoder() {
            let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
            self.encode_with_shared_encoder(state, &encoder, params);
            encoder.end_encoding();
        } else {
            // Fused up + activation
            {
                let arrays =
                    state.arrays(&[self.input_array_id, self.hidden_array_id]);
                let mut input = arrays[0].borrow_mut();
                let mut hidden = arrays[1].borrow_mut();
                let batch = input.shape()[0] as i32;
                let input_buf = unsafe { input.mtl_buffer() };
                let hidden_buf = unsafe { hidden.mtl_buffer() };

                let encoder = command_buffer.new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
                self.encode_fused_up(&encoder, input_buf, 0, hidden_buf, batch);
                encoder.end_encoding();
            }

            // Down
            self.down.encode(state, command_buffer, params);
        }

        if params.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        self.down.supports_shared_encoder()
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: ComputeCommandEncoderRef<'_>,
        params: &EncodingParameters,
    ) {
        // Fused up + activation
        let arrays = state.arrays(&[self.input_array_id, self.hidden_array_id]);
        let mut input = arrays[0].borrow_mut();
        let mut hidden = arrays[1].borrow_mut();
        let batch = input.shape()[0] as i32;
        let input_buf = unsafe { input.mtl_buffer() };
        let hidden_buf = unsafe { hidden.mtl_buffer() };

        self.encode_fused_up(encoder, input_buf, 0, hidden_buf, batch);

        drop(input);
        drop(hidden);

        // Down
        self.down.encode_with_shared_encoder(state, encoder, params);
    }
}
