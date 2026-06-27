//! A horizontal slider: a track with a filled portion and a draggable thumb.
//! `value` is a normalized fraction in `0.0..=1.0`; the caller maps it to/from
//! the real parameter range. Supports click-to-set and drag.

use std::{cell::Cell, rc::Rc};

use gpui::{
    App, Bounds, DispatchPhase, Element, ElementId, GlobalElementId, Hitbox, HitboxBehavior,
    InspectorElementId, IntoElement, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, Pixels, Style, Window, fill, point, px, relative, size,
};

use crate::theme::ActiveTheme;

type ChangeHandler = Box<dyn Fn(f32, &mut Window, &mut App)>;

pub struct Slider {
    id: ElementId,
    value: f32,
    on_change: Option<ChangeHandler>,
}

impl Slider {
    /// `value` is the normalized fraction (0.0–1.0).
    pub fn new(id: impl Into<ElementId>, value: f32) -> Self {
        Self { id: id.into(), value: value.clamp(0., 1.), on_change: None }
    }

    /// Called with the new fraction (0.0–1.0) on click or drag.
    pub fn on_change(mut self, handler: impl Fn(f32, &mut Window, &mut App) + 'static) -> Self {
        self.on_change = Some(Box::new(handler));
        self
    }
}

/// Drag flag, kept in element state so it survives between frames.
#[derive(Default, Clone)]
struct SliderState {
    dragging: Rc<Cell<bool>>,
}

impl IntoElement for Slider {
    type Element = Self;
    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for Slider {
    type RequestLayoutState = ();
    type PrepaintState = Hitbox;

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = px(16.).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        _: &mut App,
    ) -> Hitbox {
        window.insert_hitbox(bounds, HitboxBehavior::Normal)
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        hitbox: &mut Hitbox,
        window: &mut Window,
        cx: &mut App,
    ) {
        let theme = cx.theme().clone();
        let track_h = px(4.);
        let thumb_r = px(7.);
        let cy = bounds.center().y;
        let left = bounds.left();
        let width = bounds.size.width;
        let frac = self.value.clamp(0., 1.);
        let thumb_x = left + width * frac;

        // Track.
        window.paint_quad(
            fill(
                Bounds::new(point(left, cy - track_h / 2.), size(width, track_h)),
                theme.border,
            )
            .corner_radii(track_h / 2.),
        );
        // Filled portion.
        window.paint_quad(
            fill(
                Bounds::new(point(left, cy - track_h / 2.), size(width * frac, track_h)),
                theme.text,
            )
            .corner_radii(track_h / 2.),
        );
        // Thumb.
        window.paint_quad(
            fill(
                Bounds::new(
                    point(thumb_x - thumb_r, cy - thumb_r),
                    size(thumb_r * 2., thumb_r * 2.),
                ),
                theme.text,
            )
            .corner_radii(thumb_r),
        );

        let Some(global_id) = global_id else { return };
        let dragging = window.with_element_state::<SliderState, _>(global_id, |state, _| {
            let state = state.unwrap_or_default();
            (state.dragging.clone(), state)
        });

        let Some(on_change) = self.on_change.take() else { return };
        let on_change = Rc::new(on_change);

        // Press: jump to the clicked position and start dragging.
        {
            let hitbox = hitbox.clone();
            let dragging = dragging.clone();
            let on_change = on_change.clone();
            window.on_mouse_event(move |e: &MouseDownEvent, phase, window, cx| {
                if phase == DispatchPhase::Bubble
                    && e.button == MouseButton::Left
                    && hitbox.is_hovered(window)
                {
                    dragging.set(true);
                    on_change(((e.position.x - left) / width).clamp(0., 1.), window, cx);
                }
            });
        }
        // Drag.
        {
            let dragging = dragging.clone();
            let on_change = on_change.clone();
            window.on_mouse_event(move |e: &MouseMoveEvent, phase, window, cx| {
                if phase == DispatchPhase::Bubble && dragging.get() {
                    on_change(((e.position.x - left) / width).clamp(0., 1.), window, cx);
                }
            });
        }
        // Release.
        {
            window.on_mouse_event(move |_: &MouseUpEvent, phase, _, _| {
                if phase == DispatchPhase::Bubble {
                    dragging.set(false);
                }
            });
        }
    }
}
