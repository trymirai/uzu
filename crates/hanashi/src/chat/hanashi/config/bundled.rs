include!(concat!(env!("OUT_DIR"), "/bundled_configs.rs"));

pub fn get_parsing_config(name: &str) -> Option<&'static str> {
    BUNDLED_PARSING_CONFIGS.iter().find(|(key, _)| *key == name).map(|(_, json)| *json)
}

pub fn get_rendering_config(name: &str) -> Option<&'static str> {
    BUNDLED_RENDERING_CONFIGS.iter().find(|(key, _)| *key == name).map(|(_, json)| *json)
}

pub fn get_tokens_config(name: &str) -> Option<&'static str> {
    BUNDLED_TOKENS_CONFIGS.iter().find(|(key, _)| *key == name).map(|(_, json)| *json)
}

pub fn get_ordering_config(name: &str) -> Option<&'static str> {
    BUNDLED_ORDERING_CONFIGS.iter().find(|(key, _)| *key == name).map(|(_, json)| *json)
}

pub fn get_config_mapping(name: &str) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
    BUNDLED_CONFIG_MAPPINGS
        .iter()
        .find(|(key, _, _, _, _)| *key == name)
        .map(|(_, parsing, rendering, tokens, ordering)| (*parsing, *rendering, *tokens, *ordering))
}
