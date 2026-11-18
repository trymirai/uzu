use clap::Parser;

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(short, long)]
    pub model: String,
    /// Context length in tokens; the model's default when omitted.
    #[arg(long)]
    pub context_length: Option<i64>,
}
