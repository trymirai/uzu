#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn dump_detected_frequencies() {
    let collector = keisoku::Collector::new();
    println!("KEISOKU_SOC_BEGIN");
    match collector.soc() {
        Some(soc) => {
            println!("CHIP\t{}", soc.chip_name);
            println!("ECPU_MHZ\t{:?}", soc.ecpu_frequencies);
            println!("PCPU_MHZ\t{:?}", soc.pcpu_frequencies);
            println!("GPU_MHZ\t{:?}", soc.gpu_frequencies);
        },
        None => println!("STATUS\tSocInfo unavailable"),
    }
    println!("KEISOKU_SOC_END");
}
