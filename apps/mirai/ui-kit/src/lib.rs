//! Mirai design system (theme + GPUI components), mirroring
//! `@trymirai-schemas/ui-kit`. The app builds its UI from this crate.

pub mod components;
pub mod theme;
pub mod tokens;

#[cfg(test)]
mod interaction_tests {
    //! Proof that GPUI's headless test harness (`#[gpui::test]` +
    //! `TestAppContext`) drives real input in this workspace — Layer 2 of
    //! `TESTING.md`. Clicks a rendered element and asserts its handler ran, with
    //! no display, window server, or engine.

    use std::{cell::Cell, rc::Rc};

    use gpui::{
        Context, IntoElement, Modifiers, Render, TestAppContext, Window, div, point, prelude::*,
        px,
    };

    struct ClickProbe {
        clicks: Rc<Cell<usize>>,
    }

    impl Render for ClickProbe {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let clicks = self.clicks.clone();
            div()
                .id("probe")
                .size_full()
                .on_click(move |_, _, _| clicks.set(clicks.get() + 1))
        }
    }

    #[gpui::test]
    fn click_fires_handler(cx: &mut TestAppContext) {
        let clicks = Rc::new(Cell::new(0));
        let (_view, cx) = cx.add_window_view({
            let clicks = clicks.clone();
            move |_, _| ClickProbe { clicks }
        });
        cx.simulate_click(point(px(25.), px(25.)), Modifiers::default());
        cx.run_until_parked();
        assert_eq!(clicks.get(), 1);
    }
}
