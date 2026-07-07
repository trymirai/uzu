use super::model_view_model::ModelViewModel;

pub(super) struct FamilyViewModel {
    pub key: String,
    pub name: String,
    pub vendor: String,
    pub icon_url: Option<String>,
    pub range: Option<String>,
    pub has_mirai: bool,
    pub last_installed_at: u64,
    pub models: Vec<ModelViewModel>,
}

impl FamilyViewModel {
    pub(super) fn installed_count(&self) -> usize {
        self.models.iter().filter(|model| model.installed()).count()
    }
}
