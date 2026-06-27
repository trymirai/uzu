#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSort {
    #[default]
    Size,
    Name,
    Newest,
}

impl ModelSort {
    pub fn label(self) -> &'static str {
        match self {
            Self::Size => "Size",
            Self::Name => "Name",
            Self::Newest => "Newest",
        }
    }
}

pub fn sort_by_name(a: &str, b: &str) -> std::cmp::Ordering {
    a.to_lowercase().cmp(&b.to_lowercase())
}

pub fn sort_by_newest(a: &str, b: &str) -> std::cmp::Ordering {
    fn score(name: &str) -> (i64, i64) {
        let params = parse_params(name).map(|p| (p * 1000.0) as i64).unwrap_or(0);
        let dated = name
            .split(|c: char| c == '-' || c == '_')
            .filter_map(|t| t.parse::<i64>().ok())
            .filter(|n| (10_000..1_000_000).contains(n))
            .max()
            .unwrap_or(0);
        (params, dated)
    }
    score(b).cmp(&score(a)).then_with(|| sort_by_name(a, b))
}

pub fn parse_params(name: &str) -> Option<f64> {
    for raw in name.split(|c: char| c == ' ' || c == '-') {
        let token = raw.trim();
        if let Some(num) = token
            .strip_suffix('B')
            .or_else(|| token.strip_suffix('b'))
        {
            if let Ok(v) = num.parse::<f64>() {
                return Some(v * 1000.0);
            }
        }
        if let Some(num) = token
            .strip_suffix('M')
            .or_else(|| token.strip_suffix('m'))
        {
            if let Ok(v) = num.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_prefers_larger_params() {
        assert_eq!(
            sort_by_newest("Qwen3-8B", "Qwen3-4B"),
            std::cmp::Ordering::Less
        );
    }
}
