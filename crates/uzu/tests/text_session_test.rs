mod common;
use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    parameter::SamplingSeed,
    types::{Input, Message, Output},
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

fn build_decoding_config() -> DecodingConfig {
    DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42))
}

fn build_default_text() -> String {
    let text = String::from("
London is one of the world’s most historically rich, culturally diverse, and economically influential cities. Located along the River Thames in southeastern England, it serves as the capital of both the United Kingdom and England. The city is known for its deep historical layers, ranging from Roman ruins and medieval fortresses to cutting-edge skyscrapers and modern cultural institutions.\n\n## 🌆 What London Is\nLondon is a global metropolis composed of 32 boroughs and the City of London, the latter being its historic financial core. It is home to over 9 million residents and attracts tens of millions of visitors each year. The city's character is shaped by its extraordinary diversity—people from hundreds of nationalities and cultures live, work, and study here. More than 300 languages are spoken, making London one of the most linguistically diverse cities on Earth.\n\n## 🏛️ History in Depth\n- **Roman Foundation (1st century CE):** The Romans founded Londinium around 47–50 CE. The city grew rapidly because of its strategic riverside location and became a major commercial center.\n- **Medieval Era:** After the fall of Rome, London persisted as a trading hub. In the medieval period it grew around institutions like Westminster Abbey and the Tower of London, becoming a political and economic heartland.\n- **Tudor & Stuart London:** The 16th and 17th centuries saw major cultural and political developments such as the reign of Henry VIII, the rise of Shakespeare’s theatre, the Great Plague (1665), and the Great Fire of London (1666), which reshaped the city.\n- **18th–19th Century Expansion:** During the Industrial Revolution and the height of the British Empire, London became the world’s largest city and a global center for finance, trade, and innovation.\n- **20th–21st Century:** London endured extensive bombings during World War II, leading to major rebuilding. In recent decades it has reinvented itself as a global powerhouse for finance, technology, media, arts, and higher education.\n\n## 🗽 Famous Places and Landmarks\nLondon’s landmarks span more than 1,000 years of architecture and history:\n- **Big Ben & the Houses of Parliament:** Iconic symbols of British democracy and Gothic Revival architecture.\n- **The Tower of London:** A UNESCO World Heritage Site built in the 11th century; home to the Crown Jewels.\n- **Tower Bridge:** A Victorian engineering marvel combining bascule and suspension bridge elements.\n- **Buckingham Palace:** The monarch’s official residence, famous for the Changing of the Guard ceremony.\n- **Westminster Abbey:** A historic church where kings and queens have been crowned since 1066.\n- **The British Museum:** Houses millions of artifacts from around the world, including the Rosetta Stone.\n- **The London Eye:** A giant observation wheel offering panoramic city views.\n- **St Paul's Cathedral:** An architectural masterpiece by Sir Christopher Wren, known for its dome.\n- **Trafalgar Square:** A major public square dominated by Nelson’s Column.\n- **The Shard:** One of Europe's tallest buildings, offering a modern contrast to London’s older skyline.\n\n## 🎭 Culture, Arts & Lifestyle\nLondon is a cultural superpower:\n- **Museums & Galleries:** The National Gallery, Tate Modern, Victoria & Albert Museum, Science Museum, Natural History Museum, and many others—many free to the public.\n- **Performing Arts:** The West End theatre district is internationally renowned; London also has major opera houses, symphony orchestras, dance companies, and fringe theatres.\n- **Music:** From classical concerts at the Royal Albert Hall to modern arenas like The O2, London has shaped genres from punk to grime.\n- **Food:** London's culinary scene is global—Michelin-starred restaurants, historic pubs, international street food markets, and diverse immigrant cuisines.\n- **Education:** Home to world-leading institutions like University College London (UCL), Imperial College, King’s College London, and the London School of Economics.\n\n## 🚇 Transport & Infrastructure\nLondon’s transport network is one of the most extensive on the planet:\n- **The London Underground (The Tube):** Opened in 1863, it is the world’s oldest metro system, serving 270 stations across 11 lines.\n- **Buses:** The red double-decker bus remains one of the city’s most recognizable symbols.\n- **Trains & Overground:** Extensive rail links connect Greater London and surrounding regions.\n- **River Thames Services:** High-speed and commuter boats offer scenic travel routes.\n- **Cycling:** London has expanded cycle paths and a popular bike-sharing scheme.\n\n## 🌳 Green Spaces & Nature\nDespite its density, London is considered one of the greenest major cities:\n- **Hyde Park:** Famous for its lakes, open lawns, and Speakers’ Corner.\n- **Regent’s Park:** Contains the London Zoo and manicured gardens.\n- **Richmond Park:** A vast royal park known for free-roaming deer.\n- **Hampstead Heath:** Offers woodlands, ponds, and views of the city skyline from Parliament Hill.\n- **Kew Gardens:** A world-famous botanical garden and UNESCO World Heritage Site.\n\n## 🌦️ Climate & Weather\nLondon’s weather is mild but unpredictable:\n- Winters are cool and damp, rarely extremely cold.\n- Summers are warm, with occasional heatwaves.\n- Rainfall is spread throughout the year, often in light drizzle.\n- The city's maritime climate means conditions can change quickly.\n\n## 🧭 Modern London Today\nLondon today is a global hub for:\n- Finance and banking (the City & Canary Wharf)\n- Technology and innovation (Silicon Roundabout, King’s Cross tech district)\n- Media, film, publishing, and fashion\n- Tourism and cultural exchange\n- International diplomacy and global business\n\nIts combination of history, diversity, and constant reinvention makes it one of the most unique cities in the world.\n\nIf you want, I can make it even more detailed—focusing on history, architecture, neighborhoods, economic data, travel guidance, or anything else! Modern London continues to evolve rapidly. New architecture rises beside medieval churches, technological innovation reshapes work and communication, and social change constantly challenges established norms. Yet the city’s identity remains rooted in contrast: old and new,  local and global,  tradition and reinvention.
");
    return text;
}

#[test]
fn test_text_session_base() {
    run(build_default_text(), build_decoding_config(), 1280);
}

#[test]
fn test_text_session_scenario() {
    let system_prompt = String::from("You are a helpful assistant.");
    let user_prompts = vec![
        String::from("Tell about London"),
        String::from("Compare with New York"),
    ];
    run_scenario(Some(system_prompt), user_prompts);
}

#[test]
fn test_text_session_stability() {
    let mut session =
        Session::new(build_model_path(), build_decoding_config()).unwrap();
    println!("Index | TTFT, s | Prompt t/s | Generate t/s");
    for index in 0..10 {
        let input = Input::Text(build_default_text());
        let output = session
            .run(
                input,
                RunConfig::default().tokens_limit(128),
                Some(|_: Output| {
                    return true;
                }),
            )
            .unwrap();
        println!(
            "{:.5} | {:.5} | {:.5} | {:.5}",
            index,
            output.stats.prefill_stats.duration,
            output.stats.prefill_stats.processed_tokens_per_second,
            output.stats.generate_stats.unwrap().tokens_per_second
        );
    }
}

fn run(
    text: String,
    decoding_config: DecodingConfig,
    tokens_limit: u64,
) {
    let mut session =
        Session::new(build_model_path(), decoding_config).unwrap();
    let input = Input::Text(text);
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(tokens_limit),
            Some(|_: Output| {
                return true;
            }),
        )
        .unwrap();

    let empty_response = String::from("None");

    println!("-------------------------");
    println!(
        "{}",
        output.text.parsed.chain_of_thought.unwrap_or(empty_response.clone())
    );
    println!("-------------------------");
    println!(
        "{}",
        output.text.parsed.response.unwrap_or(empty_response.clone())
    );
    println!("-------------------------");
    println!("{:#?}", output.stats);
    println!("-------------------------");
    println!("Finish reason: {:?}", output.finish_reason);
    println!("-------------------------");
}

fn run_scenario(
    system_prompt: Option<String>,
    user_prompts: Vec<String>,
) {
    let mut session =
        Session::new(build_model_path(), build_decoding_config()).unwrap();

    let mut messages: Vec<Message> = vec![];
    if let Some(system_prompt) = system_prompt {
        messages.push(Message::system(system_prompt.clone()));
        println!("System > {}", system_prompt.clone());
    }

    for user_prompt in user_prompts {
        messages.push(Message::user(user_prompt.clone()));
        println!("User > {}", user_prompt.clone());

        let input = Input::Messages(messages.clone());
        let output = session
            .run(
                input,
                RunConfig::default(),
                Some(|_: Output| {
                    return true;
                }),
            )
            .unwrap();
        messages.push(Message::assistant(
            output.text.parsed.response.clone().unwrap_or(String::new()),
            output.text.parsed.chain_of_thought.clone(),
        ));
        println!("Assistant > {}", output.text.original.clone());
    }
}
