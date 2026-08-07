use comfy_table::{
    ContentArrangement, Table,
    modifiers::{UTF8_ROUND_CORNERS, UTF8_SOLID_INNER_BORDERS},
    presets::UTF8_FULL,
};
use uzu::engine::{Engine, EngineConfig};

use crate::interactive::{app::CliApplication, list::get_families};

mod app;
mod components;
mod flows;
mod helpers;
mod list;
mod sessions;

pub async fn run_interactive(model: Option<String>) -> anyhow::Result<()> {
    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let application = CliApplication::create(engine_config).await?;
    application.run_with_model(model).await?;
    Ok(())
}

pub async fn run_list_models() -> anyhow::Result<()> {
    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let engine = Engine::new(engine_config).await?;
    let models = engine.models().await?;
    if models.is_empty() {
        return Err(anyhow::anyhow!("No models to run"));
    }

    let families = get_families(&models);
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .apply_modifier(UTF8_SOLID_INNER_BORDERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Name", "ID"]);

    for family in &families {
        table.add_row(vec![&family.name, &family.id]);
    }
    println!("{table}");

    Ok(())
}
