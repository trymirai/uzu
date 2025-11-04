use xgrammar::{Grammar, GrammarCompiler, TokenizerInfo, VocabType};

fn main() {
    // Build a minimal tokenizer info with empty vocab for demo
    let vocab: Vec<&str> = vec![];
    let none_reserved: Option<Box<[i32]>> = None;
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &none_reserved, false);

    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1);

    // Builtin JSON
    let grammar = Grammar::builtin_json_grammar();
    let _compiled = compiler.compile_grammar(&grammar);
    println!("compiled builtin json");

    // Regex
    let grammar_regex = Grammar::from_regex("[a-z]+", false);
    let _regex_compiled = compiler.compile_grammar(&grammar_regex);

    println!("OK");
}
