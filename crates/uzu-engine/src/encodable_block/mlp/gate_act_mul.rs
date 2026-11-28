use std::sync::Arc;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::activation_type::ActivationType,
        kernel::{
            GatedActMul, GatedActMulSettings,
            matmul::{ActivationFormat, MatmulA, packed_half_bytes},
        },
    },
    config::{activation::AnyActivation, clipping::ClippingBounds},
    data_type::DataType,
    encodable_block::linear::{LinearInput, LinearInputPreparation},
};

pub struct MlpGateActMulEncodable<B: Backend> {
    fp_kernel: GatedActMul<B>,
    activation: AnyActivation,
    hidden_dim: u32,
    data_type: DataType,
    hadamard_factors: Option<Arc<B::GlobalGpuBuffer>>,
    hadamard_size: Option<u32>,
    quantized_kernel: Option<GatedActMul<B>>,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        hidden_dim: u32,
        data_type: DataType,
        activation: AnyActivation,
        gate_clipping: ClippingBounds,
        value_clipping: ClippingBounds,
        input_preparation: Option<LinearInputPreparation<B>>,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let (rht, format) = input_preparation.map_or((None, ActivationFormat::Bf16), |preparation| {
            (Some(preparation.rht), preparation.activation_format)
        });
        let settings = GatedActMulSettings {
            activation_alpha: activation.custom_alpha(),
            gate_clipping,
            value_clipping,
        };
        let hadamard_size = rht.as_ref().map(|rht| rht.hadamard_size);
        let fp_kernel = GatedActMul::full_precision(context, data_type, true, hadamard_size, settings)?;
        let quantized_kernel = hadamard_size
            .filter(|_| format != ActivationFormat::Bf16)
            .map(|hadamard_size| GatedActMul::quantized(context, data_type, hadamard_size, format, settings))
            .transpose()?;
        Ok(Self {
            fp_kernel,
            activation,
            hidden_dim,
            data_type,
            hadamard_factors: rht.map(|rht| rht.signs),
            hadamard_size,
            quantized_kernel,
        })
    }

    /// The down projection's input RHT signs this block applies, if any.
    pub fn input_rht(&self) -> Option<&B::GlobalGpuBuffer> {
        self.hadamard_factors.as_deref()
    }

    pub fn input_hadamard_size(&self) -> Option<u32> {
        self.hadamard_size
    }

    pub fn encode_for_linear(
        &self,
        fused_up: impl BufferRef<Backend = B>,
        batch_dim: u32,
        act_format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }

        if let Some(kernel) = self.quantized_kernel.as_ref().filter(|kernel| kernel.format() == act_format) {
            let hadamard_factors = self.hadamard_factors.as_deref().expect("quantized input requires RHT factors");
            if let ActivationFormat::PackedHalf {
                block_k,
                bits,
            } = act_format
            {
                let mut values = command_buffer.allocate_scratch(packed_half_bytes(self.hidden_dim, block_k, bits))?;
                kernel.encode_pack_half(
                    fused_up,
                    &mut values,
                    hadamard_factors,
                    self.hidden_dim,
                    batch_dim,
                    self.activation.act_type(),
                    command_buffer,
                );
                return Ok(MatmulA::PackedHalf {
                    values,
                    block_k,
                    bits,
                });
            }
            let ActivationFormat::Int8(quantization) = act_format else {
                unreachable!()
            };
            let mut values = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.hidden_dim], DataType::I8)?;
            let mut scales = command_buffer.allocate_scratch_for_shape(
                &[batch_dim, self.hidden_dim.div_ceil(quantization.scale_group_size())],
                DataType::F32,
            )?;
            let mut group_sums = quantization
                .sum_group_size()
                .map(|group_size| {
                    command_buffer
                        .allocate_scratch_for_shape(&[batch_dim, self.hidden_dim.div_ceil(group_size)], DataType::I32)
                })
                .transpose()?;
            kernel.encode_quantized(
                fused_up,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                hadamard_factors,
                self.hidden_dim,
                batch_dim,
                self.activation.act_type(),
                command_buffer,
            );
            return Ok(MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                quantization,
            });
        }

        let mut hidden = command_buffer.allocate_scratch_for_shape(&[batch_dim, self.hidden_dim], self.data_type)?;
        self.fp_kernel.encode_fp(
            fused_up,
            None::<&B::ScratchBuffer>,
            &mut hidden,
            self.hadamard_factors.as_deref(),
            self.hidden_dim,
            batch_dim,
            0,
            0,
            self.activation.act_type(),
            command_buffer,
        );
        Ok(MatmulA::full_precision(hidden))
    }
}
