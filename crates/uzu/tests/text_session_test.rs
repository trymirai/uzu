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

#[test]
fn test_text_session_base() {
    let text = String::from("
    What are the four major types of biological macromolecules found in cells?
How does the structure of a phospholipid enable it to form cell membranes?
What is the difference between prokaryotic and eukaryotic cells?
What is the function of ribosomes in the cell?
How does the mitochondrion generate ATP?
What is the role of the Golgi apparatus in protein processing?
How do lysosomes contribute to cellular homeostasis?
What is the fluid mosaic model of the plasma membrane?
What is osmosis, and how is it different from simple diffusion?
How do carrier proteins and channel proteins differ in membrane transport?
What happens during each phase of the cell cycle (G1, S, G2, M)?
How is mitosis different from meiosis at a high level?
Why is crossing over important in meiosis?
What is a homologous chromosome pair?
How do checkpoints regulate progression through the cell cycle?
What is apoptosis, and why is it biologically important?
How can errors in cell cycle regulation contribute to cancer?
What is the role of the spindle apparatus during cell division?
How does cytokinesis differ in plant and animal cells?
What is the difference between haploid and diploid cells?
What is a gene, and how is it related to DNA?
What is the structure of the DNA double helix?
How does DNA replication ensure high fidelity?
What is the central dogma of molecular biology?
How is RNA structurally different from DNA?
What happens during transcription?
How do introns and exons differ in eukaryotic genes?
What is the role of tRNA in translation?
How do codons determine the amino acid sequence of a protein?
What is a mutation, and what are some different types of mutations?
How do point mutations differ from frameshift mutations?
What is genetic recombination, and where does it occur?
How do dominant and recessive alleles differ?
What is a genotype vs a phenotype?
How do incomplete dominance and codominance differ from simple dominance?
What is a polygenic trait, and can you give an example?
How does epistasis affect phenotypic expression?
What is linkage, and how does it affect inheritance patterns?
How can a pedigree be used to infer inheritance of a trait?
What is a genetic linkage map, and how is it constructed?
What is natural selection, and what conditions are required for it to occur?
How did Darwin’s observations on the Galápagos Islands inform his theory?
What is the difference between microevolution and macroevolution?
How do genetic drift and gene flow differ?
What is the founder effect, and when might it occur?
What is a species, and what are some limitations of the biological species concept?
How can reproductive isolation lead to speciation?
What is adaptive radiation, and can you give an example?
How do homologous structures provide evidence for evolution?
What role do fossils play in reconstructing evolutionary history?
What are enzymes, and how do they speed up chemical reactions?
What is the difference between an enzyme’s active site and allosteric site?
How do temperature and pH affect enzyme activity?
What is competitive inhibition vs noncompetitive inhibition?
What is ATP, and why is it called the energy currency of the cell?
How does glycolysis convert glucose into usable energy?
What are the main stages of cellular respiration, and where do they occur?
How does oxidative phosphorylation produce ATP?
How is fermentation different from aerobic respiration?
What is the role of NADH and FADH₂ in metabolism?
What are the main steps of photosynthesis?
How do the light-dependent reactions differ from the Calvin cycle?
What is the role of chlorophyll in capturing light energy?
Why are chloroplasts described as “semi-autonomous” organelles?
How do C₄ and CAM plants minimize photorespiration?
What is photorespiration, and why is it considered wasteful?
How do stomata help regulate gas exchange in plants?
What is the difference between xylem and phloem?
How does transpiration help move water through a plant?
What is a plant hormone, and what is the role of auxin?
What are the major levels of biological organization in ecology (from organism to biosphere)?
What is a niche, and how is it different from a habitat?
How do biotic and abiotic factors interact in an ecosystem?
What is the difference between a food chain and a food web?
What are trophic levels, and why is energy transfer between them inefficient?
What is biological magnification, and how can it affect top predators?
How do keystone species influence community structure?
What is ecological succession, and how do primary and secondary succession differ?
What is carrying capacity in a population, and what factors determine it?
How can human activities alter biogeochemical cycles such as the carbon cycle?
What are the main components of the human immune system?
How do innate and adaptive immunity differ?
What is an antigen, and how does the body recognize it as foreign?
How do B cells and T cells differ in function?
What is the difference between antibodies and antigens?
How does vaccination lead to long-term immunity?
What is an autoimmune disease, and how does it arise?
How do antibiotics target bacteria, and why don’t they work on viruses?
What is antibiotic resistance, and how does it evolve?
How do viruses replicate inside host cells?
What are the main types of animal tissues, and how do they differ?
How does the structure of neurons relate to their function in signal transmission?
What is the role of neurotransmitters at synapses?
How do hormones differ from neurotransmitters in signaling range and speed?
What is homeostasis, and how do feedback loops maintain it?
How does the structure of the human lung support efficient gas exchange?
What is the role of the kidney in regulating body fluid composition?
How does the circulatory system transport oxygen and nutrients through the body?
What is the microbiome, and how does it influence human health?
How can modern techniques like CRISPR be used to edit genomes?
    ");
    run(text, build_decoding_config(), 512);
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
