use keisoku::SensorKind;

fn main() {
    for kind in [SensorKind::Temperature, SensorKind::Voltage, SensorKind::Current] {
        let sensors = keisoku::sensors(kind);
        println!("\n{:?} sensors ({}):", kind, sensors.len());
        for sensor in &sensors {
            let category = sensor.category.as_deref().unwrap_or("-");
            println!(
                "  {:<28} {:>10.4} {:<3} [{:<21}] category={}",
                sensor.name,
                sensor.value,
                kind.unit(),
                sensor.component.to_string(),
                category,
            );
        }
    }
}
