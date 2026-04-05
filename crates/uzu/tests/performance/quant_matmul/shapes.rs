pub struct TestShape {
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl std::fmt::Display for TestShape {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.batch, self.input_dim, self.output_dim)
    }
}
