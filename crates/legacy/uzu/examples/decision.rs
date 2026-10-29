use std::{io, io::Write};

use shoji::types::session::decision::{DecisionAnswer, DecisionInput, DecisionOption, DecisionQuestion};
use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("trymirai/d3-flash-M".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {:.2}%", update.progress() * 100.0);
        io::stdout().flush()?;
    }
    println!();

    let session = engine.decision(model).await?;
    let questions = vec![
        DecisionQuestion::Choice {
            instructions: Some("Which team should handle this request?".to_string()),
            options: vec![
                DecisionOption::from("returns", "Refunds, replacements and damaged deliveries"),
                DecisionOption::from("billing", "Payments, invoices and charges"),
                DecisionOption::from("technical", "Product setup and faults"),
            ],
        },
        DecisionQuestion::YesNo {
            instructions: Some("Does the customer have a receipt?".to_string()),
            no_description: None,
            yes_description: None,
        },
        DecisionQuestion::Score {
            instructions: Some("How urgent is this request?".to_string()),
            levels: vec!["Routine".to_string(), "Soon".to_string(), "Today".to_string()],
        },
    ];
    let input = DecisionInput {
        state: "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today."
            .to_string(),
        questions: questions.clone(),
    };
    let output = session.decide(input).await?;

    for (question, answer) in questions.iter().zip(&output.answers) {
        match (question, answer) {
            (
                DecisionQuestion::Choice {
                    options,
                    ..
                },
                DecisionAnswer::Choice {
                    key,
                    probabilities,
                    ..
                },
            ) => {
                println!("choice: {key}");
                for (option, probability) in options.iter().zip(probabilities) {
                    println!("  {:<10} {probability:.4}", option.key);
                }
            },
            (
                DecisionQuestion::Score {
                    levels,
                    ..
                },
                DecisionAnswer::Score {
                    score,
                    probabilities,
                },
            ) => {
                println!("score: {score:.4}");
                for (level, probability) in levels.iter().zip(probabilities) {
                    println!("  {level:<10} {probability:.4}");
                }
            },
            (
                _,
                DecisionAnswer::YesNo {
                    yes_probability,
                },
            ) => println!("yes/no: {yes_probability:.4}"),
            _ => unreachable!("answers follow the question types"),
        }
    }
    println!("{} tokens in {:.2} s", output.stats.tokens_count, output.stats.total_duration);

    Ok(())
}
