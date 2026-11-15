use std::f32;

use anyhow::{Result, anyhow};
use half::bf16;

use super::{
    config::HaruHeadConfig,
    linear::{BiasKind, HaruLinear, MetalGpuBuffer, upload_f32},
};
use crate::{
    backends::{
        common::{
            BufferMut, BufferRef, CommandBufferEncoding,
            kernel::matmul::MatmulKernel,
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            kernel::{
                HaruAdaLnModulateMetalKernel, HaruAddSiluMetalKernel, HaruGatedResidualMetalKernel,
                HaruLatentAffineMetalKernel, HaruVelocityReadoutMetalKernel, matmul::MatmulMetalKernel,
            },
        },
    },
    config::{
        activation::AnyActivation,
        activation::silu::SiLU,
        linear::LinearConfig,
        mlp::{AnyMLPConfig, dense_mlp::DenseMLPConfig},
    },
    data_type::DataType,
    encodable_block::mlp::Mlp,
    safetensors::ParameterTree,
};

struct ScalarEmbedWeights {
    fc1: Box<[bf16]>,
    fc2: Box<[bf16]>,
    hidden_size: usize,
    half_dim: usize,
}

impl ScalarEmbedWeights {
    async fn load(
        tree: &ParameterTree<'_, Metal>,
        hidden_size: u32,
        emb_dim: u32,
    ) -> Result<Self> {
        let fc1 = tree
            .subtree("fc1")
            .read_slice::<bf16>("weight", &[hidden_size, emb_dim])
            .await
            .map_err(|e| anyhow!("{e:?}"))?;
        let fc2 = tree
            .subtree("fc2")
            .read_slice::<bf16>("weight", &[hidden_size, hidden_size])
            .await
            .map_err(|e| anyhow!("{e:?}"))?;
        Ok(Self {
            fc1,
            fc2,
            hidden_size: hidden_size as usize,
            half_dim: (emb_dim / 2) as usize,
        })
    }

    fn eval(&self, val: f32) -> Vec<f32> {
        let in_dim = self.half_dim * 2;
        let max_period = 10000.0f32;
        let mut feat = vec![0.0f32; in_dim];
        for i in 0..self.half_dim {
            let freq = (-(max_period.ln()) * (i as f32) / (self.half_dim as f32)).exp();
            let arg = val * freq;
            feat[i] = bf16::from_f32(arg.cos()).to_f32();
            feat[self.half_dim + i] = bf16::from_f32(arg.sin()).to_f32();
        }
        let mut h = vec![0.0f32; self.hidden_size];
        for (o, h_out) in h.iter_mut().enumerate() {
            let row = &self.fc1[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for (&x, &w) in feat.iter().zip(row.iter()) {
                acc = x.mul_add(w.to_f32(), acc);
            }
            let x_b = bf16::from_f32(acc).to_f32();
            *h_out = bf16::from_f32(x_b / (1.0f32 + (-x_b).exp())).to_f32();
        }
        let mut out = vec![0.0f32; self.hidden_size];
        for (o, out_val) in out.iter_mut().enumerate() {
            let row = &self.fc2[o * self.hidden_size..(o + 1) * self.hidden_size];
            let mut acc = 0.0f32;
            for (&x, &w) in h.iter().zip(row.iter()) {
                acc = x.mul_add(w.to_f32(), acc);
            }
            *out_val = bf16::from_f32(acc).to_f32();
        }
        out
    }
}

struct MeanFlowLayer {
    norm_w: MetalGpuBuffer,
    ffn: Box<dyn Mlp<Metal>>,
}

pub struct MeanFlowHead {
    config: HaruHeadConfig,
    guidance_scale: f32,
    t_dt_h_sum: Vec<f32>,
    omega_embed: ScalarEmbedWeights,
    static_time_emb: MetalGpuBuffer,
    latent_proj: HaruLinear,
    cond_proj: HaruLinear,
    ada: HaruLinear,
    final_proj: HaruLinear,
    layers: Vec<MeanFlowLayer>,
    pub scale_one: MetalGpuBuffer,
    pub bias_zero: MetalGpuBuffer,
    dummy_norm_w: MetalGpuBuffer,
    matmul: MatmulMetalKernel,
    add_silu: HaruAddSiluMetalKernel,
    latent_affine: HaruLatentAffineMetalKernel,
    adaln_modulate: HaruAdaLnModulateMetalKernel,
    gated_residual: HaruGatedResidualMetalKernel,
    velocity_readout: HaruVelocityReadoutMetalKernel,
}

impl MeanFlowHead {
    pub async fn load(
        tree: &ParameterTree<'_, Metal>,
        config: &HaruHeadConfig,
        context: &MetalContext,
    ) -> Result<Self> {
        let dim = config.hidden_size;
        let lat_dim = config.latent_dim;
        let ffn_dim = ((dim as f32) * config.ffn_ratio).round() as u32;
        let emb_dim = 256u32;

        let matmul = MatmulMetalKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16)
            .map_err(|e| anyhow!("{e:?}"))?;

        let t_embed = ScalarEmbedWeights::load(&tree.subtree("t_embed"), dim, emb_dim).await?;
        let dt_embed = ScalarEmbedWeights::load(&tree.subtree("dt_embed"), dim, emb_dim).await?;
        let h_embed = ScalarEmbedWeights::load(&tree.subtree("h_embed"), dim, emb_dim).await?;
        let omega_embed = ScalarEmbedWeights::load(&tree.subtree("omega_embed"), dim, emb_dim).await?;

        let t_v = t_embed.eval(0.0);
        let dt_v = dt_embed.eval(config.interval_embed);
        let h_v = h_embed.eval(1.0 / (config.num_steps as f32));
        let mut t_dt_h_sum = vec![0.0f32; dim as usize];
        for i in 0..dim as usize {
            t_dt_h_sum[i] = t_v[i] + dt_v[i] + h_v[i];
        }

        let guidance_scale = config.guidance_scale;
        let omega_v = omega_embed.eval(1.0 - 1.0 / guidance_scale);
        let static_f32: Vec<f32> = t_dt_h_sum.iter().zip(omega_v.iter()).map(|(&a, &b)| a + b).collect();
        let static_time_emb = upload_f32(context, &static_f32)?;

        let latent_proj =
            HaruLinear::load_from_tree(&tree.subtree("latent_proj"), &matmul, context, dim, lat_dim, BiasKind::None)
                .await?;
        let cond_proj =
            HaruLinear::load_from_tree(&tree.subtree("cond_proj"), &matmul, context, dim, dim, BiasKind::None).await?;
        let ada_out_dim = (config.num_layers * 3 + 2) * dim;
        let ada = HaruLinear::load_from_tree(&tree.subtree("ada"), &matmul, context, ada_out_dim, dim, BiasKind::None)
            .await?;
        let final_proj = HaruLinear::load_from_tree(
            &tree.subtree("final").subtree("proj"),
            &matmul,
            context,
            lat_dim,
            dim,
            BiasKind::None,
        )
        .await?;

        let mlp_cfg = AnyMLPConfig::DenseMLPConfig(DenseMLPConfig::unclipped(
            LinearConfig {},
            AnyActivation::SiLU(SiLU::default()),
            false,
            false,
        ));
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let lt = tree.subtree("layers").subtree(&i.to_string());
            let norm_w = lt.subtree("norm").read_buffer("weight", &[dim], DataType::F32).await.map_err(|e| anyhow!("{e:?}"))?;
            let (ffn, _) = <dyn Mlp<Metal>>::new(dim, ffn_dim, DataType::BF16, &mlp_cfg, &lt.subtree("ffn"), context)
                .await
                .map_err(|e| anyhow!("{e:?}"))?;
            layers.push(MeanFlowLayer {
                norm_w,
                ffn,
            });
        }

        Ok(Self {
            config: config.clone(),
            guidance_scale,
            t_dt_h_sum,
            omega_embed,
            static_time_emb,
            latent_proj,
            cond_proj,
            ada,
            final_proj,
            layers,
            scale_one: upload_f32(context, &[1.0f32])?,
            bias_zero: upload_f32(context, &[0.0f32])?,
            dummy_norm_w: upload_f32(context, &[0.0f32])?,
            matmul,
            add_silu: HaruAddSiluMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            latent_affine: HaruLatentAffineMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            adaln_modulate: HaruAdaLnModulateMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            gated_residual: HaruGatedResidualMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
            velocity_readout: HaruVelocityReadoutMetalKernel::new(context).map_err(|e| anyhow!("{e:?}"))?,
        })
    }

    pub fn set_guidance_scale(&mut self, guidance_scale: f32, context: &MetalContext) -> Result<()> {
        if (self.guidance_scale - guidance_scale).abs() < 1e-6 {
            return Ok(());
        }
        let omega_v = self.omega_embed.eval(1.0 - 1.0 / guidance_scale);
        let static_f32: Vec<f32> = self.t_dt_h_sum.iter().zip(omega_v.iter()).map(|(&a, &b)| a + b).collect();
        self.static_time_emb = upload_f32(context, &static_f32)?;
        self.guidance_scale = guidance_scale;
        Ok(())
    }

    pub fn encode(
        &mut self,
        cond_bf16: impl BufferRef<Backend = Metal>,
        noise_f32: impl BufferRef<Backend = Metal>,
        out_latent_f32: impl BufferMut<Backend = Metal>,
        cb: &mut MetalCommandBufferEncoding,
    ) -> Result<()> {
        let dim = self.config.hidden_size;
        let lat_dim = self.config.latent_dim;
        let ada_out_dim = (self.config.num_layers * 3 + 2) * dim;

        let mut c_proj = cb.allocate_scratch_for_shape(&[1, dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.cond_proj.encode_into(&mut self.matmul, None, cond_bf16, 1, &mut c_proj, cb)?;

        let mut c_cond = cb.allocate_scratch_for_shape(&[1, dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.add_silu.encode(&c_proj, &self.static_time_emb, &mut c_cond, dim, cb);

        let mut modulation =
            cb.allocate_scratch_for_shape(&[1, ada_out_dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.ada.encode_into(&mut self.matmul, None, &c_cond, 1, &mut modulation, cb)?;

        let mut z_bf16 = cb.allocate_scratch_for_shape(&[1, lat_dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.latent_affine.encode(noise_f32, &self.scale_one, &self.bias_zero, &mut z_bf16, lat_dim, cb);

        let mut x = cb.allocate_scratch_for_shape(&[1, dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.latent_proj.encode_into(&mut self.matmul, None, &z_bf16, 1, &mut x, cb)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let mut x_mod = cb.allocate_scratch_for_shape(&[1, dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
            let mod_offset = (i as u32) * 3 * dim;
            self.adaln_modulate.encode(&x, &layer.norm_w, &modulation, &mut x_mod, dim, mod_offset, 1, self.config.norm_eps, cb);
            let ffn_out = layer.ffn.encode(x_mod, 1, cb).map_err(|e| anyhow!("{e:?}"))?;
            self.gated_residual.encode(&mut x, &ffn_out, &modulation, dim, mod_offset + 2 * dim, cb);
        }

        let mut final_mod = cb.allocate_scratch_for_shape(&[1, dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.adaln_modulate.encode(
            &x,
            &self.dummy_norm_w,
            &modulation,
            &mut final_mod,
            dim,
            self.config.num_layers * 3 * dim,
            0,
            self.config.norm_eps,
            cb,
        );

        let mut v_bf16 = cb.allocate_scratch_for_shape(&[1, lat_dim], DataType::BF16).map_err(|e| anyhow!("{e:?}"))?;
        self.final_proj.encode_into(&mut self.matmul, None, &final_mod, 1, &mut v_bf16, cb)?;

        let step_h = 1.0f32 / (self.config.num_steps as f32);
        self.velocity_readout.encode(noise_f32, &v_bf16, out_latent_f32, step_h, lat_dim, cb);
        Ok(())
    }
}
