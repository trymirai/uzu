#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn dump_energy_model_channels() {
    let collector = keisoku::Collector::new();
    let channels = collector.energy_model_channels();
    println!("KEISOKU_ENERGY_BEGIN");
    for (name, unit, value) in &channels {
        println!("RAIL\t{}\t{}\t{}", name, unit, value);
    }
    println!("STATUS\t{} energy-model rails", channels.len());
    println!("KEISOKU_ENERGY_END");
}
