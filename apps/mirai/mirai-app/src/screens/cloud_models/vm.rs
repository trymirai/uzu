use crate::models_store::ModelRow;

/// Display data for one cloud-model row, projected from a [`ModelRow`].
pub(super) struct CloudVm {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub icon_url: Option<String>,
}

impl CloudVm {
    pub(super) fn from_row(
        row: &ModelRow,
        dark: bool,
    ) -> Self {
        Self {
            id: row.id().to_string(),
            name: row.name(),
            vendor: row.vendor().unwrap_or_else(|| "Other".to_string()),
            icon_url: row.icon_url(dark),
        }
    }
}
