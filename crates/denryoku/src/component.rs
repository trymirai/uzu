use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Component {
    Cpu,
    Gpu,
    NeuralEngine,
    Soc,
    PowerManagementUnit,
    Battery,
    Storage,
    Display,
    Unknown,
}

impl Component {
    pub fn label(self) -> &'static str {
        match self {
            Component::Cpu => "CPU",
            Component::Gpu => "GPU",
            Component::NeuralEngine => "Neural Engine",
            Component::Soc => "SoC",
            Component::PowerManagementUnit => "Power Management Unit",
            Component::Battery => "Battery",
            Component::Storage => "Storage",
            Component::Display => "Display",
            Component::Unknown => "Unknown",
        }
    }
}

impl fmt::Display for Component {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

pub fn classify(name: &str) -> Component {
    let name = name.to_ascii_lowercase();

    if name.contains("battery") || name.contains("gas gauge") || name.contains("fuel gauge") {
        return Component::Battery;
    }
    if name.contains("nand") || name.contains("ssd") || name.contains("flash") {
        return Component::Storage;
    }
    if name.contains("display") || name.contains("lcd") || name.contains("oled") || name.contains("backlight") {
        return Component::Display;
    }
    if name.contains("ane") || name.contains("neural") {
        return Component::NeuralEngine;
    }
    if name.contains("gpu") || thermal_probe_block(&name) == Some('g') {
        return Component::Gpu;
    }
    if name.contains("cpu") || name.contains("pcore") || name.contains("ecore") {
        return Component::Cpu;
    }
    if name.contains("soc") || thermal_probe_block(&name) == Some('s') {
        return Component::Soc;
    }
    if name.contains("pmu") {
        return Component::PowerManagementUnit;
    }
    Component::Unknown
}

/// Apple thermal probes are named like `tp1s` / `tp3g`, where the trailing
/// letter hints at the monitored block (`s` ≈ SoC, `g` ≈ GPU). Returns that
/// letter for any `tp<digits><letter>` token in the (lowercased) name.
fn thermal_probe_block(name: &str) -> Option<char> {
    name.split_whitespace().find_map(|token| {
        let rest = token.strip_prefix("tp")?;
        let block = rest.chars().last()?;
        let digits = &rest[..rest.len() - block.len_utf8()];
        if block.is_ascii_alphabetic() && !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) {
            Some(block)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_observed_names() {
        assert_eq!(classify("gas gauge battery"), Component::Battery);
        assert_eq!(classify("NAND CH0 temp"), Component::Storage);
        assert_eq!(classify("PMU TP3g"), Component::Gpu);
        assert_eq!(classify("PMU TP1s"), Component::Soc);
        assert_eq!(classify("PMU tdie1"), Component::PowerManagementUnit);
        assert_eq!(classify("PMU vbuck0"), Component::PowerManagementUnit);
        assert_eq!(classify("ANE temp"), Component::NeuralEngine);
        assert_eq!(classify("GPU die"), Component::Gpu);
        assert_eq!(classify("something weird"), Component::Unknown);
    }

    #[test]
    fn labels_are_human_readable() {
        assert_eq!(Component::NeuralEngine.label(), "Neural Engine");
        assert_eq!(Component::PowerManagementUnit.to_string(), "Power Management Unit");
    }
}
