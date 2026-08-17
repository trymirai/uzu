use std::path::Path;

use backend_uzu::{
    backends::{common::gpu_types::trie::TrieNode as GpuTrieNode, metal::Metal},
    engine::{
        Engine,
        language_model::forward::{MockTree, chain_nodes},
    },
};

const CONFIGS: &[(&str, Option<MockTree>)] = &[
    // ("non-spec", None),
    (
        "chain-w1",
        Some(MockTree {
            max_width: 1,
            balanced: true,
        }),
    ),
    // (
    //     "bal-w2",
    //     Some(MockTree {
    //         max_width: 2,
    //         balanced: true,
    //     }),
    // ),
    // (
    //     "bal-w4",
    //     Some(MockTree {
    //         max_width: 4,
    //         balanced: true,
    //     }),
    // ),
    // (
    //     "unbal-w2",
    //     Some(MockTree {
    //         max_width: 2,
    //         balanced: false,
    //     }),
    // ),
    // (
    //     "unbal-w4",
    //     Some(MockTree {
    //         max_width: 4,
    //         balanced: false,
    //     }),
    // ),
    // (
    //     "unbal-w6",
    //     Some(MockTree {
    //         max_width: 6,
    //         balanced: false,
    //     }),
    // ),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: sweep <model path> [prefix length]");
    let prefix_length: u32 = args.next().map(|arg| arg.parse().expect("prefix length")).unwrap_or(0);

    let engine = Engine::<Metal>::new()?;
    let model = engine.load_language_model(Path::new(&model_path))?;

    const WARMUP: usize = 3;
    const ITERATIONS: usize = 10;
    const COLUMN_WIDTH: usize = 20;

    // let sizes = (1..=16u32).chain((16..64).step_by(16).skip(1)).chain((64..=256).step_by(32));
    let sizes = 1..=64u32;

    print!("{:>COLUMN_WIDTH$}", "tokens");
    for (name, _) in CONFIGS {
        print!("{:>COLUMN_WIDTH$}", name);
    }
    println!();

    for size in sizes {
        print!("{:>COLUMN_WIDTH$}", size);
        let mut measured: Vec<(usize, (Box<[GpuTrieNode]>, bool))> = Vec::new();
        for (index, (name, spec)) in CONFIGS.iter().enumerate() {
            let key = match spec {
                Some(tree) => (tree.linearize(size).0, false),
                None => (chain_nodes(size), true),
            };
            if let Some((equivalent, _)) = measured.iter().find(|(_, other)| *other == key) {
                print!("{:>COLUMN_WIDTH$}", format!("(same as {})", CONFIGS[*equivalent].0));
                continue;
            }
            measured.push((index, key));

            let result = (|| {
                for _ in 0..WARMUP {
                    model.forward(prefix_length, size, *spec)?;
                }
                let mut samples = Vec::with_capacity(ITERATIONS);
                for _ in 0..ITERATIONS {
                    samples.push(model.forward(prefix_length, size, *spec)?);
                }
                samples.sort_unstable();
                Ok::<_, Box<dyn std::error::Error>>((samples[ITERATIONS / 2 - 1] + samples[ITERATIONS / 2]) / 2)
            })();
            match result {
                Ok(median) => print!("{:>COLUMN_WIDTH$}", format!("{:.3}", median.as_secs_f64() * 1000.0)),
                Err(error) => {
                    eprintln!("tokens={size} {name}: {error}");
                    print!("{:>COLUMN_WIDTH$}", "err");
                },
            }
        }
        println!();
    }
    Ok(())
}
