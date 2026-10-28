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
    hadamard_factors: Option<B::GlobalGpuBuffer>,
    encoder: Option<GatedActMul<B>>,
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
        let (hadamard_factors, format) = input_preparation.map_or((None, ActivationFormat::Bf16), |preparation| {
            (Some(preparation.rht_signs), preparation.activation_format)
        });
        let settings = GatedActMulSettings {
            activation_alpha: activation.custom_alpha(),
            gate_clipping,
            value_clipping,
        };
        let fp_kernel = GatedActMul::full_precision(context, data_type, true, hadamard_factors.is_some(), settings)?;
        let encoder = (format != ActivationFormat::Bf16)
            .then(|| GatedActMul::quantize(context, data_type, format, settings))
            .transpose()?;
        Ok(Self {
            fp_kernel,
            activation,
            hidden_dim,
            data_type,
            hadamard_factors,
            encoder,
        })
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

        let encoder = self.encoder.as_ref().filter(|kernel| kernel.format() == act_format);
        assert!(
            act_format == ActivationFormat::Bf16 || encoder.is_some(),
            "requested activation format was not prepared"
        );
        if let (
            ActivationFormat::PackedHalf {
                group_size,
            },
            Some(kernel),
        ) = (act_format, encoder)
        {
            let mut values = command_buffer.allocate_scratch(packed_half_bytes(self.hidden_dim, group_size))?;
            kernel.encode_packed_half(
                fused_up,
                &mut values,
                self.hadamard_factors.as_ref().expect("packed half requires RHT factors"),
                self.hidden_dim,
                batch_dim,
                self.activation.act_type(),
                command_buffer,
            );
            return Ok(MatmulA::PackedHalf {
                values,
                group_size,
            });
        }

        if let (ActivationFormat::Int8(quantization), Some(kernel)) = (act_format, encoder) {
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
            kernel.encode_quantize(
                fused_up,
                &mut values,
                &mut scales,
                group_sums.as_mut(),
                self.hadamard_factors.as_ref().expect("INT8 requires RHT factors"),
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
            self.hadamard_factors.as_ref(),
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
