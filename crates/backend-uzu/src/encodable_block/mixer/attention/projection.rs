use crate::{
    backends::common::{Allocation, Backend, Encoder},
    encodable_block::{linear::Linear, mixer::attention::qkv_norm::QKVNorm},
};

pub(super) enum QkvProjection<B: Backend> {
    Fused {
        qkv: Box<dyn Linear<B>>,
        norm: Option<QKVNorm<B>>,
    },
    QueryOnly {
        q: Box<dyn Linear<B>>,
        norm: Option<QKVNorm<B>>,
    },
    #[allow(dead_code)]
    Split {
        q: Box<dyn Linear<B>>,
        kv: Box<dyn Linear<B>>,
        q_norm: Option<QKVNorm<B>>,
        kv_norm: Option<QKVNorm<B>>,
    },
}

pub(super) enum ProjectedQkv<B: Backend> {
    Fused(Allocation<B>),
    Query(Allocation<B>),
    Split {
        query: Allocation<B>,
        key_value: Allocation<B>,
    },
}

impl<B: Backend> QkvProjection<B> {
    pub(super) fn encode_attend(
        &self,
        hidden: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<ProjectedQkv<B>, B::Error> {
        match self {
            Self::Fused {
                qkv,
                norm,
            } => {
                let mut qkv = qkv.encode(hidden, batch_dim, encoder)?;
                if let Some(norm) = norm {
                    norm.encode(&mut qkv, batch_dim, encoder)?;
                }
                Ok(ProjectedQkv::Fused(qkv))
            },
            Self::QueryOnly {
                q,
                norm,
            } => {
                let mut query = q.encode(hidden, batch_dim, encoder)?;
                if let Some(norm) = norm {
                    norm.encode(&mut query, batch_dim, encoder)?;
                }
                Ok(ProjectedQkv::Query(query))
            },
            Self::Split {
                q,
                kv,
                q_norm,
                kv_norm,
            } => {
                // Linear::encode may consume/mutate its input; split Q/KV attention needs the same hidden for both projections.
                let mut hidden_for_key_value = encoder.allocate_scratch(hidden.size())?;
                encoder.encode_copy(&hidden, .., &mut hidden_for_key_value, ..);

                let mut query = q.encode(hidden, batch_dim, encoder)?;
                let mut key_value = kv.encode(hidden_for_key_value, batch_dim, encoder)?;
                if let Some(q_norm) = q_norm {
                    q_norm.encode(&mut query, batch_dim, encoder)?;
                }
                if let Some(kv_norm) = kv_norm {
                    kv_norm.encode(&mut key_value, batch_dim, encoder)?;
                }
                Ok(ProjectedQkv::Split {
                    query,
                    key_value,
                })
            },
        }
    }

    #[allow(dead_code)]
    pub(super) fn encode_key_value(
        &self,
        hidden: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match self {
            Self::Split {
                kv,
                kv_norm,
                ..
            } => {
                let mut key_value = kv.encode(hidden, batch_dim, encoder)?;
                if let Some(kv_norm) = kv_norm {
                    kv_norm.encode(&mut key_value, batch_dim, encoder)?;
                }
                Ok(key_value)
            },
            _ => panic!("append-only attention requires split Q/KV projection"),
        }
    }
}
