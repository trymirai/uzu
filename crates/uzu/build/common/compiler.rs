use async_trait::async_trait;

#[async_trait]
pub trait Compiler {
    async fn build(&self) -> anyhow::Result<()>;
}
