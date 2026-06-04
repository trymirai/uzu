use derive_more::Display;

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
#[display("M[{m}]K[{k}]N[{n}]")]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl Shape {
    pub const fn new(
        m: usize,
        k: usize,
        n: usize,
    ) -> Self {
        Self {
            m,
            k,
            n,
        }
    }
}

const SHAPES_TINY: &[Shape] = &[Shape::new(64, 64, 64), Shape::new(16, 128, 256), Shape::new(128, 256, 128)];

const SHAPES_UNALIGNED: &[Shape] = &[
    Shape::new(7, 33, 11),
    Shape::new(33, 2048, 2048),
    Shape::new(64, 2048, 33),
    Shape::new(200, 2048, 2048),
    Shape::new(128, 2048, 200),
];

const SHAPES_MEDIUM: &[Shape] = &[Shape::new(128, 2048, 2048), Shape::new(256, 4096, 4096)];

pub fn all_correctness_shapes() -> impl Iterator<Item = Shape> {
    SHAPES_TINY.iter().chain(SHAPES_UNALIGNED.iter()).chain(SHAPES_MEDIUM.iter()).copied()
}

const BENCH_FP_GEMM: &[Shape] = &[
    Shape::new(128, 2048, 8192),
    Shape::new(128, 4096, 14336),
    Shape::new(256, 4096, 4096),
    Shape::new(512, 8192, 2048),
];

pub fn bench_fp_gemm_shapes() -> impl Iterator<Item = Shape> {
    BENCH_FP_GEMM.iter().copied()
}

const BENCH_NK: &[(usize, usize)] =
    &[(2048, 2048), (2048, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];

pub fn bench_quant_gemm_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };
    let ms = &[4usize, 5, 6, 7, 8, 16, 32, 48, 64];
    BENCH_NK
        .iter()
        .filter(move |&&(n, k)| n % 32 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

pub fn bench_quant_gemv_shapes(bits: u32) -> impl Iterator<Item = Shape> {
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };
    let nk: &[(usize, usize)] = &[(4096, 4096), (4096, 14336), (14336, 4096), (14336, 14336)];
    let ms = &[1usize, 2, 4];
    nk.iter()
        .filter(move |&&(n, k)| n % 8 == 0 && k % block_size == 0)
        .flat_map(move |&(n, k)| ms.iter().map(move |&m| Shape::new(m, k, n)))
}

const QWEN35_DECODE_AND_CUTOFF_MS: &[usize] = &[1, 2, 4, 5, 8];
const QWEN35_READOUT_MS: &[usize] = &[1];

struct Qwen35LayerShape {
    label: &'static str,
    k: usize,
    n: usize,
    ms: &'static [usize],
}

const QWEN35_LAYERS: &[Qwen35LayerShape] = &[
    // Qwen3.5-0.8B, workspace/models/0.5.3/Qwen3.5-0.8B-gs128emb/config.json:
    // model_dim=1024, hidden_dim=3584, 18 DeltaNet layers, 6 attention layers.
    Qwen35LayerShape {
        label: "qwen35_0.8b_delta_in_proj",
        k: 1024,
        n: 8224,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_delta_out_proj",
        k: 2048,
        n: 1024,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_attention_qkv",
        k: 1024,
        n: 3072,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_attention_out",
        k: 2048,
        n: 1024,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_attention_gate",
        k: 1024,
        n: 2048,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_mlp_gateup",
        k: 1024,
        n: 7168,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_mlp_down",
        k: 3584,
        n: 1024,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_0.8b_readout",
        k: 1024,
        n: 248320,
        ms: QWEN35_READOUT_MS,
    },
    // Qwen3.5-2B public config values used by the layout benchmark:
    // model_dim=2048, hidden_dim=6144.
    Qwen35LayerShape {
        label: "qwen35_2b_delta_in_proj",
        k: 2048,
        n: 8224,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_delta_out_proj",
        k: 2048,
        n: 2048,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_attention_qkv",
        k: 2048,
        n: 6144,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_attention_out",
        k: 2048,
        n: 2048,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_attention_gate",
        k: 2048,
        n: 2048,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_mlp_gateup",
        k: 2048,
        n: 12288,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_mlp_down",
        k: 6144,
        n: 2048,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_2b_readout",
        k: 2048,
        n: 248320,
        ms: QWEN35_READOUT_MS,
    },
    // Qwen3.5-4B, workspace/models/0.5.3/Qwen3.5-4B/config.json:
    // model_dim=2560, hidden_dim=9216, 24 DeltaNet layers, 8 attention layers.
    Qwen35LayerShape {
        label: "qwen35_4b_delta_in_proj",
        k: 2560,
        n: 12352,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_delta_out_proj",
        k: 4096,
        n: 2560,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_attention_qkv",
        k: 2560,
        n: 6144,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_attention_out",
        k: 4096,
        n: 2560,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_attention_gate",
        k: 2560,
        n: 4096,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_mlp_gateup",
        k: 2560,
        n: 18432,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_mlp_down",
        k: 9216,
        n: 2560,
        ms: QWEN35_DECODE_AND_CUTOFF_MS,
    },
    Qwen35LayerShape {
        label: "qwen35_4b_readout",
        k: 2560,
        n: 248320,
        ms: QWEN35_READOUT_MS,
    },
];

pub fn qwen3_layer_shapes(bits: u32) -> impl Iterator<Item = (&'static str, Shape)> {
    let block_size: usize = if bits == 4 {
        512
    } else {
        256
    };
    QWEN35_LAYERS.iter().flat_map(move |layer| {
        layer
            .ms
            .iter()
            .filter(move |_| layer.k % block_size == 0 && layer.n % 8 == 0)
            .map(move |&m| (layer.label, Shape::new(m, layer.k, layer.n)))
    })
}
