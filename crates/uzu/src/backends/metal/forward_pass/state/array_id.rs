use serde::{Deserialize, Serialize};

use super::RopeType;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
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

    // Classifier prediction head buffers
    ClassifierPooling,
    ClassifierPredictionHeadDense,
    ClassifierPredictionHeadNorm,
    ClassifierPredictionHeadLogits,
}
