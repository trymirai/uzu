#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn dump_energy_model_channels() {
    let collector = keisoku::Collector::new();
    let channels = collector.energy_model_channels();
    println!("KEISOKU_ENERGY_BEGIN");
    for channel in channels.iter() {
        println!("RAIL\t{}\t{}\t{}", channel.name, channel.unit, channel.value);
    }
    println!("STATUS\t{} energy-model rails", channels.len());
    println!("KEISOKU_ENERGY_END");
}
