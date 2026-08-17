use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
};

use crate::trie::TrieNode;

pub const RECORD_GENERATION_ENV_VAR: &str = "UZU_RECORD_GENERATION";

const TOKEN_RECORD: u8 = 0;
const STEP_RECORD: u8 = 1;

pub struct GenerationRecorder {
    writer: BufWriter<File>,
}

impl GenerationRecorder {
    pub fn from_env() -> Option<Self> {
        let path = std::env::var(RECORD_GENERATION_ENV_VAR).ok()?;
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(Self {
                writer: BufWriter::new(file),
            }),
            Err(error) => {
                eprintln!("{RECORD_GENERATION_ENV_VAR}: failed to create {path}: {error}");
                None
            },
        }
    }

    fn record(
        &mut self,
        write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
    ) {
        if let Err(error) = write(&mut self.writer) {
            eprintln!("{RECORD_GENERATION_ENV_VAR}: failed to write record: {error}");
        }
    }

    pub fn record_token(
        &mut self,
        token: u64,
    ) {
        self.record(|writer| {
            writer.write_all(&[TOKEN_RECORD])?;
            writer.write_all(&(token as u32).to_le_bytes())
        });
    }

    // context_length, budget, vocab_size, the accepted path as (index, input token, output
    // token) triples, then the trie; all integers little-endian u32.
    pub fn record_step(
        &mut self,
        context_length: usize,
        budget: usize,
        trie: &TrieNode,
        accepted: &[(usize, u64, u64)],
        vocab_size: u32,
    ) {
        self.record(|writer| {
            writer.write_all(&[STEP_RECORD])?;
            writer.write_all(&(context_length as u32).to_le_bytes())?;
            writer.write_all(&(budget as u32).to_le_bytes())?;
            writer.write_all(&vocab_size.to_le_bytes())?;
            writer.write_all(&(accepted.len() as u32).to_le_bytes())?;
            for &(index, input_token, output_token) in accepted {
                writer.write_all(&(index as u32).to_le_bytes())?;
                writer.write_all(&(input_token as u32).to_le_bytes())?;
                writer.write_all(&(output_token as u32).to_le_bytes())?;
            }
            trie.write_trace(writer)
        });
    }
}
