use serde::{Deserialize, Serialize};

use super::RopeType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ArrayId {
    TokenIds,
    TokenPositions,
    TokenParents,
    TokenBitmask,
    Logits,
    TokenSeeds,

    Main,
    Shortcut,
    QKV,
    Gate,
    AttentionOutput,
    MlpFusedUp,
    MlpHidden,
    SsmInProj,

    Keys(usize),
    Values(usize),
    SsmConvState(usize),
    SsmState(usize),
    SsmPacked(usize),
    SsmX(usize),
    SsmB(usize),
    SsmC(usize),
    SsmDt(usize),
    SsmZ(usize),
    ShortConvState(usize),
    ShortConvSuffixState(usize),
    DeltaNetConvState(usize),
    DeltaNetSsmState(usize),

    RotatedQueries,
    RotatedKeys,
    ExtractedValues,

    AttentionPartials,
    AttentionSums,
    AttentionMaxs,

    RopeCosines(RopeType),
    RopeSines(RopeType),

    AttentionSinks(usize),

    MoeTopkIds,
    MoeTopkProbs,
    MoeOffsets,
    MoeSumK,
    MoeBucketedTokenIds,
    MoeBucketedProbs,
    MoeXPerm,
    MoeTok2Row,
    MoeYPartial,
    MoeHidden,
    MoeTwoPassRowExpertMap,
    MoeTileCounts,
    MoeTileOffsets,
    MoeTileMap,
    MoeTotalTiles,
    MoeDispatchArgs,
    MoeScatterPartials,
    MoeScatterBlockBases,
    MoeBlockAlloc,

    // Per-Layer Embedding (PLE) buffers
    /// PLE per-layer embedding lookup output [batch, seq, num_layers * ple_dim]
    PleEmbeddings,
    /// PLE projection output (main embeddings projected into PLE space) [batch, seq, num_layers * ple_dim]
    PleProjection,
    /// PLE combined per-layer inputs [batch, seq, num_layers, ple_dim]
    PlePerLayerInputs,
    /// PLE gate output within a layer [batch, seq, ple_dim]
    PleGate,

    // Classifier prediction head buffers
    ClassifierPooling,
    ClassifierPredictionHeadDense,
    ClassifierPredictionHeadNorm,
    ClassifierPredictionHeadLogits,
}
