use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulDtypeCombo {
    pub a_dtype: String,
    pub b_dtype: String,
    pub output_dtype: String,
}

impl std::fmt::Display for MatmulDtypeCombo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}*{}->{}",  self.a_dtype, self.b_dtype, self.output_dtype)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl std::fmt::Display for MatmulShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.m, self.n, self.k)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulBenchmarkTask {
    pub combos: Vec<MatmulDtypeCombo>,
    pub shapes: Vec<MatmulShape>,
    pub warmup_iterations: u64,
    pub benchmark_iterations: u64,
}

impl MatmulBenchmarkTask {
    pub fn default_mpp() -> Self {
        let combos = vec![
            MatmulDtypeCombo { a_dtype: "i8".into(), b_dtype: "i8".into(), output_dtype: "i32".into() },
            MatmulDtypeCombo { a_dtype: "i8".into(), b_dtype: "bf16".into(), output_dtype: "bf16".into() },
            MatmulDtypeCombo { a_dtype: "bf16".into(), b_dtype: "bf16".into(), output_dtype: "bf16".into() },
        ];

        let mut shapes = Vec::new();
        for &m in &[512, 1024, 2048] {
            for &n in &[512, 1024, 2048] {
                for &k in &[1, 2, 4, 8, 16, 32, 64] {
                    shapes.push(MatmulShape { m, n, k });
                }
            }
        }

        Self {
            combos,
            shapes,
            warmup_iterations: 3,
            benchmark_iterations: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulBenchmarkResult {
    pub combo: MatmulDtypeCombo,
    pub shape: MatmulShape,
    pub duration_ms: f64,
    pub gflops: f64,
    pub status: String,
    pub error_message: Option<String>,
}
