use keisoku::{Component, classify};

#[test]
fn classifies_observed_names() {
    assert_eq!(classify("gas gauge battery"), Component::Battery);
    assert_eq!(classify("NAND CH0 temp"), Component::Storage);
    assert_eq!(classify("PMU TP3g"), Component::Gpu);
    assert_eq!(classify("PMU TP1s"), Component::Soc);
    assert_eq!(classify("PMU tdie1"), Component::Soc);
    assert_eq!(classify("PMU tjunc"), Component::Soc);
    assert_eq!(classify("PMU tdev1"), Component::PowerManagementUnit);
    assert_eq!(classify("PMU tcal"), Component::PowerManagementUnit);
    assert_eq!(classify("PMU vbuck0"), Component::PowerManagementUnit);
    assert_eq!(classify("PMU ildo3"), Component::PowerManagementUnit);
    assert_eq!(classify("Charger TQ0j"), Component::Charger);
    assert_eq!(classify("ANE temp"), Component::NeuralEngine);
    assert_eq!(classify("GPU die"), Component::Gpu);
    assert_eq!(classify("something weird"), Component::Unknown);
}

#[test]
fn labels_are_human_readable() {
    assert_eq!(Component::NeuralEngine.label(), "Neural Engine");
    assert_eq!(Component::PowerManagementUnit.to_string(), "Power Management Unit");
}
