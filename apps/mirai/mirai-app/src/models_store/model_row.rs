use uzu::{
    storage::types::{DownloadPhase, DownloadState},
    types::{
        basic::{Image, ImageTheme},
        model::Model,
    },
};

pub struct ModelRow {
    pub model: Model,
    pub state: Option<DownloadState>,
}

impl ModelRow {
    pub fn id(&self) -> &str {
        &self.model.identifier
    }

    pub fn name(&self) -> String {
        self.model.name()
    }

    pub fn vendor(&self) -> Option<String> {
        self.model.family.as_ref().map(|f| f.vendor.name())
    }

    pub fn icon_url(
        &self,
        prefer_dark: bool,
    ) -> Option<String> {
        let icons = &self.model.family.as_ref()?.vendor.metadata.icons;
        let want = if prefer_dark {
            ImageTheme::Dark
        } else {
            ImageTheme::Light
        };
        let is_svg = |i: &&Image| i.url.ends_with(".svg");
        icons
            .iter()
            .find(|i| i.theme == want && is_svg(i))
            .or_else(|| icons.iter().find(is_svg))
            .or_else(|| icons.iter().find(|i| i.theme == want))
            .or_else(|| icons.first())
            .map(|i| i.url.clone())
    }

    pub fn size_bytes(&self) -> i64 {
        self.model.properties.as_ref().map(|p| p.size).unwrap_or(0)
    }

    pub fn display_size_bytes(&self) -> i64 {
        self.state.as_ref().map(|s| s.total_bytes).filter(|b| *b > 0).unwrap_or_else(|| self.size_bytes())
    }

    pub fn phase(&self) -> DownloadPhase {
        self.state.as_ref().map(|s| s.phase.clone()).unwrap_or(DownloadPhase::NotDownloaded {})
    }

    pub fn progress(&self) -> f32 {
        self.state.as_ref().map(|s| s.progress()).unwrap_or(0.0)
    }

    pub fn is_installed(&self) -> bool {
        matches!(self.phase(), DownloadPhase::Downloaded {}) || (self.model.is_local() && !self.model.is_downloadable())
    }
}
