mod activation;
mod attention;
mod audio;
mod conv;
mod embedding;
mod matmul;
mod moe;
mod norm;
mod pooling;
mod rope;
mod sampling;
mod ssd;
mod tensor;

use crate::backends::{
    common::Kernels,
    cpu::{
        backend::Cpu,
        kernel::{
            activation::ActivationCpuKernel,
            attention::{
                AttentionGemmCpuKernel, AttentionSinglePassCpuKernel, AttentionTwoPass1CpuKernel,
                AttentionTwoPass2CpuKernel, AttentionUpdateKVCacheCpuKernel,
            },
            audio::{
                AudioAddCpuKernel, AudioCausalConv1dCpuKernel,
                AudioCausalConvTranspose1dCpuKernel, AudioClampCpuKernel, AudioConv1dCpuKernel,
                AudioFsqDecodeCpuKernel, AudioFsqEncodeCpuKernel, AudioHalfSnakeCpuKernel,
                AudioLeakyReluCpuKernel, AudioScaleCpuKernel, AudioTanhCpuKernel,
            },
            conv::{
                Conv1dDecodeCpuKernel, Conv1dPackCpuKernel, Conv1dScanCpuKernel,
                ShortConvDecodeCpuKernel, ShortConvPackCpuKernel, ShortConvPrefillCpuKernel,
                ShortConvTrieCpuKernel, SigmoidCpuKernel, SplitInProjCpuKernel,
            },
            embedding::{FullPrecisionEmbeddingLookupCpuKernel, QuantizedEmbeddingLookupCpuKernel},
            matmul::{
                QuantizedMatmulQmmCpuKernel, QuantizedMatmulQmmTransposed64x64CpuKernel,
                QuantizedMatmulQmmTransposedCpuKernel, QuantizedMatmulQmvCpuKernel,
                QuantizedMatmulQmvFastCpuKernel, QuantizedMatmulQvmCpuKernel,
            },
            moe::{
                MoeBuildTileMapCpuKernel, MoeBlockBasesFromPartialsCpuKernel,
                MoeCountsOffsetsFusedCpuKernel, MoeExpertsDecodeDownFused2DCpuKernel,
                MoeExpertsDecodePassACpuKernel, MoeExpertsDecodeSinglePassACpuKernel,
                MoeExpertsDecodeSinglePassBCpuKernel, MoeExpertsPrefillPassACpuKernel,
                MoeExpertsPrefillPassBCpuKernel, MoeFinalizeCpuKernel, MoeGatherXPerm1DCpuKernel,
                MoeGatherXPerm2DCpuKernel, MoePassABuildRowMapCpuKernel,
                MoePassABuildTileMapCpuKernel, MoePassATileCountsCpuKernel,
                MoePassATileScanCpuKernel, MoePassAWriteDispatchArgsCpuKernel,
                MoeRouterTopKCpuKernel, MoeScatterBucketsCpuKernel, MoeScatterBucketsMapCpuKernel,
                MoeTileCountsCpuKernel, MoeTileScanCpuKernel, MoeWriteDispatchArgsCpuKernel,
            },
            norm::{
                KVCacheUpdateCpuKernel, LayerNormCpuKernel, MaskUpdateCpuKernel,
                MlpGateActMulCpuKernel, QKNormCpuKernel, RMSNormCpuKernel,
            },
            pooling::{PoolingClsCpuKernel, PoolingMeanCpuKernel},
            rope::RopeCpuKernel,
            sampling::{
                ArgmaxFinalCpuKernel, ArgmaxMainCpuKernel, ArgmaxSingleCpuKernel, BitmaskCpuKernel,
                GumbelCpuKernel, MinPCpuKernel, TemperatureCpuKernel, TopKCpuKernel,
                TopPCpuKernel,
            },
            ssd::{
                SSDPrefill64CpuKernel, SSDPrefillCpuKernel, SSDPrefillSequentialCpuKernel,
                SSDUpdateCpuKernel,
            },
            tensor::{
                TensorAddBiasCpuKernel, TensorAddSwapCpuKernel, TensorCopyCpuKernel,
                TokenCopySampledCpuKernel, TokenCopyToResultsCpuKernel,
            },
        },
    },
};

pub struct CpuKernels;

impl Kernels for CpuKernels {
    type Backend = Cpu;
    type ActivationKernel = ActivationCpuKernel;
    type AttentionGemmKernel = AttentionGemmCpuKernel;
    type AttentionSinglePassKernel = AttentionSinglePassCpuKernel;
    type AttentionTwoPass1Kernel = AttentionTwoPass1CpuKernel;
    type AttentionTwoPass2Kernel = AttentionTwoPass2CpuKernel;
    type AttentionUpdateKVCacheKernel = AttentionUpdateKVCacheCpuKernel;
    type AudioFsqDecodeKernel = AudioFsqDecodeCpuKernel;
    type AudioLeakyReluKernel = AudioLeakyReluCpuKernel;
    type AudioTanhKernel = AudioTanhCpuKernel;
    type AudioAddKernel = AudioAddCpuKernel;
    type AudioScaleKernel = AudioScaleCpuKernel;
    type AudioCausalConv1dKernel = AudioCausalConv1dCpuKernel;
    type AudioCausalConvTranspose1dKernel = AudioCausalConvTranspose1dCpuKernel;
    type AudioHalfSnakeKernel = AudioHalfSnakeCpuKernel;
    type AudioClampKernel = AudioClampCpuKernel;
    type AudioConv1dKernel = AudioConv1dCpuKernel;
    type AudioFsqEncodeKernel = AudioFsqEncodeCpuKernel;
    type FullPrecisionEmbeddingLookupKernel = FullPrecisionEmbeddingLookupCpuKernel;
    type QuantizedEmbeddingLookupKernel = QuantizedEmbeddingLookupCpuKernel;
    type KVCacheUpdateKernel = KVCacheUpdateCpuKernel;
    type LayerNormKernel = LayerNormCpuKernel;
    type MaskUpdateKernel = MaskUpdateCpuKernel;
    type MlpGateActMulKernel = MlpGateActMulCpuKernel;
    type MoeCountsOffsetsFusedKernel = MoeCountsOffsetsFusedCpuKernel;
    type MoeExpertsDecodeSinglePassAKernel = MoeExpertsDecodeSinglePassACpuKernel;
    type MoeExpertsDecodeSinglePassBKernel = MoeExpertsDecodeSinglePassBCpuKernel;
    type MoeExpertsDecodePassAKernel = MoeExpertsDecodePassACpuKernel;
    type MoeExpertsDecodeDownFused2DKernel = MoeExpertsDecodeDownFused2DCpuKernel;
    type MoeExpertsPrefillPassAKernel = MoeExpertsPrefillPassACpuKernel;
    type MoeExpertsPrefillPassBKernel = MoeExpertsPrefillPassBCpuKernel;
    type MoeFinalizeKernel = MoeFinalizeCpuKernel;
    type MoeGatherXPerm2DKernel = MoeGatherXPerm2DCpuKernel;
    type MoeGatherXPerm1DKernel = MoeGatherXPerm1DCpuKernel;
    type MoeRouterTopKKernel = MoeRouterTopKCpuKernel;
    type MoeBlockBasesFromPartialsKernel = MoeBlockBasesFromPartialsCpuKernel;
    type MoeScatterBucketsKernel = MoeScatterBucketsCpuKernel;
    type MoeScatterBucketsMapKernel = MoeScatterBucketsMapCpuKernel;
    type MoeTileCountsKernel = MoeTileCountsCpuKernel;
    type MoeTileScanKernel = MoeTileScanCpuKernel;
    type MoeBuildTileMapKernel = MoeBuildTileMapCpuKernel;
    type MoeWriteDispatchArgsKernel = MoeWriteDispatchArgsCpuKernel;
    type MoePassATileCountsKernel = MoePassATileCountsCpuKernel;
    type MoePassATileScanKernel = MoePassATileScanCpuKernel;
    type MoePassABuildRowMapKernel = MoePassABuildRowMapCpuKernel;
    type MoePassABuildTileMapKernel = MoePassABuildTileMapCpuKernel;
    type MoePassAWriteDispatchArgsKernel = MoePassAWriteDispatchArgsCpuKernel;
    type PoolingClsKernel = PoolingClsCpuKernel;
    type PoolingMeanKernel = PoolingMeanCpuKernel;
    type QuantizedMatmulQmmKernel = QuantizedMatmulQmmCpuKernel;
    type QuantizedMatmulQmmTransposedKernel = QuantizedMatmulQmmTransposedCpuKernel;
    type QuantizedMatmulQmmTransposed64x64Kernel = QuantizedMatmulQmmTransposed64x64CpuKernel;
    type QuantizedMatmulQmvKernel = QuantizedMatmulQmvCpuKernel;
    type QuantizedMatmulQmvFastKernel = QuantizedMatmulQmvFastCpuKernel;
    type QuantizedMatmulQvmKernel = QuantizedMatmulQvmCpuKernel;
    type QKNormKernel = QKNormCpuKernel;
    type RMSNormKernel = RMSNormCpuKernel;
    type RopeKernel = RopeCpuKernel;
    type ArgmaxSingleKernel = ArgmaxSingleCpuKernel;
    type ArgmaxMainKernel = ArgmaxMainCpuKernel;
    type ArgmaxFinalKernel = ArgmaxFinalCpuKernel;
    type BitmaskKernel = BitmaskCpuKernel;
    type GumbelKernel = GumbelCpuKernel;
    type MinPKernel = MinPCpuKernel;
    type TemperatureKernel = TemperatureCpuKernel;
    type TopKKernel = TopKCpuKernel;
    type TopPKernel = TopPCpuKernel;
    type ShortConvPackKernel = ShortConvPackCpuKernel;
    type ShortConvPrefillKernel = ShortConvPrefillCpuKernel;
    type ShortConvDecodeKernel = ShortConvDecodeCpuKernel;
    type ShortConvTrieKernel = ShortConvTrieCpuKernel;
    type SigmoidKernel = SigmoidCpuKernel;
    type Conv1dPackKernel = Conv1dPackCpuKernel;
    type Conv1dDecodeKernel = Conv1dDecodeCpuKernel;
    type Conv1dScanKernel = Conv1dScanCpuKernel;
    type SplitInProjKernel = SplitInProjCpuKernel;
    type SSDPrefill64Kernel = SSDPrefill64CpuKernel;
    type SSDPrefillKernel = SSDPrefillCpuKernel;
    type SSDPrefillSequentialKernel = SSDPrefillSequentialCpuKernel;
    type SSDUpdateKernel = SSDUpdateCpuKernel;
    type TensorAddBiasKernel = TensorAddBiasCpuKernel;
    type TensorAddSwapKernel = TensorAddSwapCpuKernel;
    type TensorCopyKernel = TensorCopyCpuKernel;
    type TokenCopySampledKernel = TokenCopySampledCpuKernel;
    type TokenCopyToResultsKernel = TokenCopyToResultsCpuKernel;
}
