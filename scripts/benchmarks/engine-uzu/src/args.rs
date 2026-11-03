use clap::Parser;

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(short, long)]
    pub model: String,
}
