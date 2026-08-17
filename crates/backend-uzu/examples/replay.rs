use std::io::{BufReader, Read};

use backend_uzu::trie::{MainStats, TrieNode};
use tokenizers::Tokenizer;

fn read_u32(
    reader: &mut impl Read,
) -> std::io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or("usage: replay <model-path> <trace> [budget] [alpha]")?;
    let trace_path = args.next().ok_or("usage: replay <model-path> <trace> [budget] [alpha]")?;
    let budget_override = args
        .next()
        .map(|arg| arg.parse::<usize>().map_err(|error| format!("invalid budget {arg:?}: {error}")))
        .transpose()?;
    let alpha = args
        .next()
        .map(|arg| arg.parse::<f32>().map_err(|error| format!("invalid alpha {arg:?}: {error}")))
        .transpose()?
        .unwrap_or(0.0);

    let tokenizer = Tokenizer::from_file(std::path::Path::new(&model_path).join("tokenizer.json"))
        .map_err(|error| error.to_string())?;

    // The generated sequence is determined by the target model and the per-position seeds
    // alone, so an accept walk for any re-pruning of a recorded tree follows the one true
    // continuation. Rebuild it from the token events; a step's continuation starts right
    // after the tokens that were emitted before the step was recorded.
    let mut reader = BufReader::new(std::fs::File::open(&trace_path)?);
    let mut tokens = Vec::new();
    let mut steps = Vec::new();
    loop {
        let mut tag = [0];
        match reader.read_exact(&mut tag) {
            Ok(()) => {},
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        match tag[0] {
            0 => tokens.push(u64::from(read_u32(&mut reader)?)),
            1 => {
                let _context_length = read_u32(&mut reader)?;
                let budget = read_u32(&mut reader)? as usize;
                let vocab_size = read_u32(&mut reader)?;
                let num_accepted = read_u32(&mut reader)?;
                let mut accepted = Vec::with_capacity(num_accepted as usize);
                for _ in 0..num_accepted {
                    accepted.push((
                        read_u32(&mut reader)? as usize,
                        u64::from(read_u32(&mut reader)?),
                        u64::from(read_u32(&mut reader)?),
                    ));
                }
                let trie = TrieNode::read_trace(&mut reader)?;
                steps.push((tokens.len(), budget, vocab_size, trie, accepted));
            },
            tag => return Err(format!("invalid record tag {tag}").into()),
        }
    }

    let mut num_forward_passes = 0usize;
    let mut total_accepted = 0usize;
    let mut total_accepted_recorded = 0usize;
    for (start, recorded_budget, vocab_size, mut trie, recorded_accepted) in steps {        // Ignore the recorded pruning and re-prune the full proposed tree from scratch.
        let budget = budget_override.unwrap_or(recorded_budget);
        trie.prune_to_budget_with_gumbel(budget, alpha);
        let (accepted, tip, has_bonus) = trie.linearize().accept_continuation(&tokens[start..]);
        let highlight = accepted
            .iter()
            .map(|(index, ..)| *index)
            .chain(tip)
            .collect::<Box<[usize]>>();

        trie.pretty_print(
            &tokenizer,
            &highlight,
            if has_bonus { accepted.last().map(|&(.., output_token)| output_token) } else { None },
            std::env::var_os("UZU_PRINT_PRUNED").is_some(),
            Some(MainStats {
                vocab_size,
            }),
        );
        println!();

        num_forward_passes += 1;
        total_accepted += accepted.len();
        total_accepted_recorded += recorded_accepted.len();
    }

    println!("forward passes: {num_forward_passes}");
    println!("accepted tokens: {total_accepted} (recorded: {total_accepted_recorded})");
    println!(
        "tokens per forward pass: {:.2} (recorded: {:.2})",
        total_accepted as f64 / num_forward_passes.max(1) as f64,
        total_accepted_recorded as f64 / num_forward_passes.max(1) as f64,
    );

    Ok(())
}
