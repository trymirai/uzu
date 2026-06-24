//! Mirai design system as a standalone crate, mirroring the structure of
//! `@trymirai-schemas/ui-kit`: theme tokens (palette, typography, layout) plus
//! the reusable GPUI components built on them. `mirai-app` builds its entire UI
//! from this crate, so components stay free of any app/engine dependencies.

pub mod components;
pub mod theme;
