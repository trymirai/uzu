use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use uzu::session::{ClassificationSession, types::Input};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = if let Some(arg) = env::args().nth(1) {
        PathBuf::from(arg)
    } else {
        PathBuf::from("models/modern_bert")
    };

    if !model_path.exists() {
        eprintln!("Error: Model not found at {:?}", model_path);
        eprintln!("Usage: bert_classifier [model_path]");
        std::process::exit(1);
    }

    println!("🤖 Classifying input with {}:", model_path.display());

    let _session_start = Instant::now();
    let mut session = match ClassificationSession::new(model_path.clone()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error creating session: {:?}", e);
            eprintln!("Model path: {:?}", model_path);
            return Err(e.into());
        },
    };

    loop {
        print!("user> ");
        io::stdout().flush()?;

        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line)?;

        let input_text = input_line.trim();
        if input_text.is_empty() || input_text == "exit" || input_text == "quit"
        {
            break;
        }

        match session.classify(Input::Text(input_text.to_string())) {
            Ok(result) => {
                for (label, prob) in &result.probabilities {
                    println!("assistant> {} : {:.6}", label, prob);
                }
            },
            Err(e) => {
                eprintln!("Error: {:?}", e);
            },
        }
        println!();
    }

    Ok(())
}

