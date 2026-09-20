//! App-owned copy of the dissolved cce-ui `JsonLayoutWidget` (Phase 6ay): cloud is the
//! only consumer — the KDL/JSON-driven launcher layout host (`LauncherMode::Json`),
//! on the narrow traits wrapped in `Adapted<JsonLayoutWidget>` (Phase 6az): the paint
//! walk reaches it as the adapter, whose subtree text pass-through forwards `paint`'s
//! prims verbatim. `Justification` stayed in cce-ui (Button, cce-files, settings).

use cce_ui::widget::scroll_motion::{scroll_settings, Bounds, ScrollMotion, LINE_PX};
use cce_ui::widget::{
    WidgetHost, Widget, Checkbox, Button, Label, Spinbox, ColorSelector, TextLabel, MouseButton, ElementState, Slider, Event, UiContext,
    Key, NamedKey, Justification,
};
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct JsonWidgetConfig {
    #[serde(rename = "type")]
    pub widget_type: String,
    pub text: String,
    pub id: Option<String>,
    pub checked: Option<bool>,
    pub value: Option<i32>,
    pub min: Option<i32>,
    pub max: Option<i32>,
    pub step: Option<i32>,
    pub decimals: Option<u32>,
    pub color: Option<[u8; 3]>,
    pub value_f32: Option<f32>,
    pub min_f32: Option<f32>,
    pub max_f32: Option<f32>,
    pub target_page: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct JsonPageConfig {
    pub title: String,
    pub widgets: Vec<JsonWidgetConfig>,
    pub justify: Option<Justification>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct JsonLayoutConfig {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub widgets: Option<Vec<JsonWidgetConfig>>,
    pub pages: Option<Vec<JsonPageConfig>>,
    pub justify: Option<Justification>,
}

/// The config-constructible controls, concretely typed (Phase 6bb part 3): this was the
/// last owned type-erased widget storage in the workspace. `as_dyn`/`as_dyn_mut` serve the
/// aggregation/routing paths that dispatch heterogeneously.
pub enum JsonControl {
    Label(cce_ui::widget::Adapted<Label>),
    Checkbox(cce_ui::widget::Adapted<Checkbox>),
    Button(cce_ui::widget::Adapted<Button>),
    Spinbox(cce_ui::widget::Adapted<Spinbox>),
    Color(cce_ui::widget::Adapted<ColorSelector>),
    Slider(cce_ui::widget::Adapted<Slider>),
}

impl JsonControl {
    pub fn as_dyn(&self) -> &(dyn WidgetHost + 'static) {
        match self {
            JsonControl::Label(w) => w,
            JsonControl::Checkbox(w) => w,
            JsonControl::Button(w) => w,
            JsonControl::Spinbox(w) => w,
            JsonControl::Color(w) => w,
            JsonControl::Slider(w) => w,
        }
    }

    pub fn as_dyn_mut(&mut self) -> &mut (dyn WidgetHost + 'static) {
        match self {
            JsonControl::Label(w) => w,
            JsonControl::Checkbox(w) => w,
            JsonControl::Button(w) => w,
            JsonControl::Spinbox(w) => w,
            JsonControl::Color(w) => w,
            JsonControl::Slider(w) => w,
        }
    }

    /// Drain the one-shot click flag (6bd value shrink — `take_click` left `WidgetHost`,
    /// the drains are concrete `Adapted` methods). Only buttons carry one, and both call
    /// sites already gate on the button widget type.
    pub fn take_click(&mut self) -> bool {
        match self {
            JsonControl::Button(w) => w.take_click(),
            _ => false,
        }
    }
}

pub struct JsonWidget {
    pub id: String,
    pub widget_type: String,
    pub text: String,
    pub widget: JsonControl,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub label_text: Option<TextLabel>,
    pub page_idx: usize,
    pub target_page: Option<usize>,
}

pub struct JsonLayoutWidget {
    base: Widget,
    pub widgets: Vec<JsonWidget>,
    pub dragging_slider_idx: Option<usize>,
    /// Per-page DRAWN scroll offset; `page_scroll` drives it.
    pub page_scroll_y: Vec<f32>,
    /// Per-page scroll motion: wheel notches glide, fingers track 1:1 and
    /// fling on the lift, keyboard pages glide. `tick_scroll` carries the
    /// drawn offset after it each frame.
    pub page_scroll: Vec<ScrollMotion>,
    pub page_total_heights: Vec<f32>,
    pub active_page: usize,
}

impl JsonLayoutWidget {
    pub fn new(config: &JsonLayoutConfig) -> cce_ui::widget::Adapted<JsonLayoutWidget> {
        let mut widgets = Vec::new();
        let mut page_titles = Vec::new();

        if let Some(ref pages_conf) = config.pages {
            for (page_idx, page) in pages_conf.iter().enumerate() {
                page_titles.push(page.title.clone());
                let page_justify = page.justify.unwrap_or(Justification::Center);
                for (idx, w_conf) in page.widgets.iter().enumerate() {
                    let id = w_conf.id.clone().unwrap_or_else(|| format!("widget_{}_{}", page_idx, idx));
                    let widget_type = w_conf.widget_type.clone();
                    let text = w_conf.text.clone();

                    let widget: JsonControl = match widget_type.as_str() {
                        "checkbox" => {
                            let mut cb = Checkbox::new();
                            if let Some(ch) = w_conf.checked {
                                cb.set_checked(ch);
                            }
                            JsonControl::Checkbox(cb)
                        }
                        "button" => {
                            JsonControl::Button(Button::new_menu_item(0.0, 0.0, 0.0, 0.0)
                                .with_label(&text)
                                .with_justify(page_justify)
                                .with_bg([0.0, 0.0, 0.0, 0.0])
                                .with_hover_bg([0.20, 0.35, 0.65, 0.9]))
                        }
                        "label" => {
                            JsonControl::Label(Label::new(&text).with_font_size(13.0).with_color([0xcc, 0xcc, 0xd4]))
                        }
                        "spinbox" => {
                            let min_val = w_conf.min.unwrap_or(0);
                            let max_val = w_conf.max.unwrap_or(100);
                            let step_val = w_conf.step.unwrap_or(1);
                            let mut sb = Spinbox::new(w_conf.value.unwrap_or(0), min_val, max_val, step_val)
                                .with_label(&text);
                            if let Some(dec) = w_conf.decimals {
                                sb = sb.with_decimals(dec);
                            }
                            JsonControl::Spinbox(sb)
                        }
                        "color" | "rgb" | "rgba" => {
                            let col = w_conf.color.unwrap_or([255, 255, 255]);
                            let cs = ColorSelector::new(col).with_label(&text);
                            JsonControl::Color(cs)
                        }
                        "slider" => {
                            let min_val = w_conf.min_f32.unwrap_or(0.0);
                            let max_val = w_conf.max_f32.unwrap_or(1.0);
                            let mut sl = Slider::new()
                                .with_range(min_val, max_val)
                                .with_label(&text)
                                .with_readout(true);
                            if let Some(val) = w_conf.value_f32 {
                                let pct = if max_val > min_val { (val - min_val) / (max_val - min_val) } else { 0.0 };
                                sl = sl.with_value(pct);
                            }
                            JsonControl::Slider(sl)
                        }
                        _ => JsonControl::Button(Button::new(0.0, 0.0, 0.0, 0.0)),
                    };

                    widgets.push(JsonWidget {
                        id,
                        widget_type,
                        text,
                        widget,
                        x: 0.0,
                        y: 0.0,
                        w: 0.0,
                        h: 0.0,
                        label_text: None,
                        page_idx,
                        target_page: w_conf.target_page,
                    });
                }
            }
        } else if let Some(ref widgets_conf) = config.widgets {
            let global_justify = config.justify.unwrap_or(Justification::Center);
            for (idx, w_conf) in widgets_conf.iter().enumerate() {
                let id = w_conf.id.clone().unwrap_or_else(|| format!("widget_{}", idx));
                let widget_type = w_conf.widget_type.clone();
                let text = w_conf.text.clone();

                let widget: JsonControl = match widget_type.as_str() {
                    "checkbox" => {
                        let mut cb = Checkbox::new();
                        if let Some(ch) = w_conf.checked {
                            cb.set_checked(ch);
                        }
                        JsonControl::Checkbox(cb)
                    }
                    "button" => {
                        JsonControl::Button(Button::new_menu_item(0.0, 0.0, 0.0, 0.0)
                            .with_label(&text)
                            .with_justify(global_justify)
                            .with_bg([0.0, 0.0, 0.0, 0.0])
                            .with_hover_bg([0.20, 0.35, 0.65, 0.9]))
                    }
                    "label" => {
                        JsonControl::Label(Label::new(&text).with_font_size(13.0).with_color([0xcc, 0xcc, 0xd4]))
                    }
                    "spinbox" => {
                        let min_val = w_conf.min.unwrap_or(0);
                        let max_val = w_conf.max.unwrap_or(100);
                        let step_val = w_conf.step.unwrap_or(1);
                        let mut sb = Spinbox::new(w_conf.value.unwrap_or(0), min_val, max_val, step_val)
                            .with_label(&text);
                        if let Some(dec) = w_conf.decimals {
                            sb = sb.with_decimals(dec);
                        }
                        JsonControl::Spinbox(sb)
                    }
                    "color" | "rgb" | "rgba" => {
                        let col = w_conf.color.unwrap_or([255, 255, 255]);
                        let cs = ColorSelector::new(col).with_label(&text);
                        JsonControl::Color(cs)
                    }
                    "slider" => {
                        let min_val = w_conf.min_f32.unwrap_or(0.0);
                        let max_val = w_conf.max_f32.unwrap_or(1.0);
                        let mut sl = Slider::new()
                            .with_range(min_val, max_val)
                            .with_label(&text)
                            .with_readout(true);
                        if let Some(val) = w_conf.value_f32 {
                            let pct = if max_val > min_val { (val - min_val) / (max_val - min_val) } else { 0.0 };
                            sl = sl.with_value(pct);
                        }
                        JsonControl::Slider(sl)
                    }
                    _ => JsonControl::Button(Button::new(0.0, 0.0, 0.0, 0.0)),
                };

                widgets.push(JsonWidget {
                    id,
                    widget_type,
                    text,
                    widget,
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 0.0,
                    label_text: None,
                    page_idx: 0,
                    target_page: w_conf.target_page,
                });
            }
        }

        cce_ui::widget::Adapted::new(Self {
            base: Widget::new(),
            widgets,
            dragging_slider_idx: None,
            page_scroll_y: vec![0.0; 16],
            page_scroll: vec![ScrollMotion::new(); 16],
            page_total_heights: vec![0.0; 16],
            active_page: 0,
        })
    }

    pub fn layout_children(&mut self) {
        let (bx, by, bw, _) = self.rect();

        let pad_x = 16.0;
        let usable_w = bw - 2.0 * 16.0;
        
        let mut page_current_y = vec![16.0; 16]; // support up to 16 pages
        let spacing = 12.0;

        for i in 0..self.widgets.len() {
            let p_idx = self.widgets[i].page_idx;
            if p_idx >= page_current_y.len() {
                continue;
            }
            // Menu rows in one run share a single recess (see `paint`), so
            // they butt together inside it; the spacing returns at the run's
            // end, where the well ends too.
            let run_continues = self.widgets[i].widget_type == "button"
                && self
                    .widgets
                    .get(i + 1)
                    .map(|n| n.page_idx == p_idx && n.widget_type == "button")
                    .unwrap_or(false);
            let w_state = &mut self.widgets[i];
            let current_y = &mut page_current_y[p_idx];
            w_state.x = bx + pad_x;

            let top_room = w_state.widget.as_dyn().label_strip();

            let scroll_offset = self.page_scroll_y.get(p_idx).cloned().unwrap_or(0.0);
            w_state.y = by + *current_y - scroll_offset;
            w_state.w = usable_w;

            if w_state.widget_type == "checkbox" {
                // set_rect is an WidgetHost method; call it on the box directly (the Phase 5
                // Checkbox is an Adapted widget — as_any downcasts reach the model, not WidgetHost).
                w_state.widget.as_dyn_mut().set_rect(w_state.x, w_state.y + 2.0, 18.0, 18.0);
                w_state.h = 22.0;
                w_state.label_text = Some(TextLabel {
                    text: w_state.text.clone(),
                    x: w_state.x + 28.0,
                    y: w_state.y + 2.0,
                    font_size: 13.0,
                    color: [0xcc, 0xcc, 0xd4],
                });
            } else {
                let h = match w_state.widget_type.as_str() {
                    "button" => 24.0 + top_room,
                    "label" => 18.0 + top_room,
                    "spinbox" => cce_ui::layout::spinbox_height() + top_room,
                    "color" | "rgb" | "rgba" => cce_ui::layout::color_selector_height() + top_room,
                    "slider" => 22.0 + top_room,
                    _ => 24.0,
                };
                w_state.widget.as_dyn_mut().set_rect(w_state.x, w_state.y, usable_w, h);
                w_state.h = h;
            }

            *current_y += w_state.h + if run_continues { 0.0 } else { spacing };
        }

        // Store total height of each page (adding a little padding at the end)
        for (i, &height) in page_current_y.iter().enumerate() {
            if i < self.page_total_heights.len() {
                self.page_total_heights[i] = height + 4.0;
            }
        }
    }
}

impl JsonLayoutWidget {
    /// The laid-out rect, mirrored from the adapter by `Layout::rect_assigned`.
    fn rect(&self) -> (f32, f32, f32, f32) {
        (self.base.x, self.base.y, self.base.w, self.base.h)
    }

    /// The active page's wheel range: `0..=overflow`.
    fn page_scroll_bounds(&self, page: usize) -> Bounds {
        let (_, _, _, bh) = self.rect();
        let total = self.page_total_heights.get(page).copied().unwrap_or(0.0);
        Bounds::max(total - bh)
    }

    /// Copy a page's motion position into its drawn offset; true if it moved
    /// (the caller re-lays the children out).
    fn sync_page_scroll(&mut self, page: usize) -> bool {
        let pos = self.page_scroll[page].y.pos();
        let moved = (pos - self.page_scroll_y[page]).abs() > 1e-4;
        self.page_scroll_y[page] = pos;
        moved
    }

    /// Per-frame glide/coast of the active page's scroll. Reached from
    /// `tick_children`, which the main loop calls (through `Adapted::tick`)
    /// beside the launcher list's own `ScrollRegion::tick`. True while the
    /// offset is still moving, so the demand-driven frame loop keeps drawing.
    fn tick_scroll(&mut self, dt: f32) -> bool {
        let page = self.active_page;
        if page >= self.page_scroll.len() || page >= self.page_scroll_y.len() {
            return false;
        }
        let host = self.page_scroll_y[page];
        self.page_scroll[page].reconcile(0.0, host);
        if !self.page_scroll[page].is_animating() {
            return false;
        }
        let by = self.page_scroll_bounds(page);
        let moved = self.page_scroll[page].tick(dt, Bounds::max(0.0), by);
        if self.sync_page_scroll(page) {
            self.layout_children();
        }
        moved || self.page_scroll[page].is_animating()
    }

    /// Per-frame child state (the old `WidgetHost::tick` override): active-page widgets only.
    fn tick_children(&mut self, dt: f32, ctx: &mut UiContext) -> bool {
        let mut changed = self.tick_scroll(dt);
        let active_page = self.active_page;
        for w in &mut self.widgets {
            if w.page_idx != active_page {
                continue;
            }
            if w.widget.as_dyn_mut().tick(dt, ctx) {
                changed = true;
            }
        }
        changed
    }

    /// The whole-subtree event routing (the old `WidgetHost::handle_event` override,
    /// verbatim). Every press reaches it (`Input::gates_presses` is off), matching the
    /// ungated legacy direct-dispatch path — the trailing focus-clear on a missed press
    /// depends on that.
    fn route_event(&mut self, event: &Event, ctx: &mut UiContext) -> bool {
        // MouseEnter targets THIS container (the adapter's base-hover bookkeeping
        // synthesizes it when a move goes unconsumed). Broadcasting it to every
        // child marks them all hovered — Button's on_event trusts the router
        // contract that Enter only reaches the widget under the cursor (the
        // desktop-menu every-button-lit bug). The next PointerMove re-derives
        // child hover, so dropping it loses nothing. MouseLeave still broadcasts
        // below: clearing every child's hover is exactly what leaving the panel
        // means.
        if matches!(event, Event::MouseEnter) {
            return false;
        }
        let mut changed = false;

        match event {
            Event::PointerMove { x, y, .. } => {
                if let Some(idx) = self.dragging_slider_idx {
                    if let Some(w) = self.widgets.get_mut(idx) {
                        // Drag* events map onto the Input drag hooks in handle_event (6bd
                        // collapse); the ctx is unused on that path.
                        let mut dummy = cce_ui::context::UiContext::new();
                        let ev = Event::DragUpdate { dx: 0.0, dy: 0.0, x: *x, y: *y, local_x: *x, local_y: *y };
                        if w.widget.as_dyn_mut().handle_event(&ev, &mut dummy) {
                            changed = true;
                        }
                    }
                }
            }
            Event::MouseButton { button, state, x: _, y: _, .. } => {
                if *button == MouseButton::Left && *state == ElementState::Released {
                    if let Some(idx) = self.dragging_slider_idx {
                        if let Some(w) = self.widgets.get_mut(idx) {
                            w.widget.as_dyn_mut().handle_event(event, ctx);
                            changed = true;
                        }
                        self.dragging_slider_idx = None;
                    }
                }
            }
            _ => {}
        }

        let active_page = self.active_page;
        let mut page_switch = None;
        for (idx, w) in self.widgets.iter_mut().enumerate() {
            if w.page_idx != active_page {
                continue;
            }

            if let Event::MouseButton { button, state, x, y, .. } = event {
                if *button == MouseButton::Left && *state == ElementState::Pressed {
                    let hit = *x >= w.x && *x <= w.x + w.w && *y >= w.y && *y <= w.y + w.h;
                    if hit && w.widget_type == "slider" {
                        self.dragging_slider_idx = Some(idx);
                    }
                }
            }

            if w.widget_type == "checkbox" {
                match event {
                    Event::PointerMove { x, y, .. } => {
                        if let Some(cb) = w.widget.as_dyn_mut().as_any_mut().downcast_mut::<Checkbox>() {
                            let was = cb.hovered();
                            let hit = *x >= w.x && *x <= w.x + w.w && *y >= w.y && *y <= w.y + w.h;
                            cb.set_hovered(hit);
                            if was != hit {
                                changed = true;
                            }
                        }
                    }
                    Event::MouseButton { button, state, x, y, .. } => {
                        if *button == MouseButton::Left {
                            let hit = *x >= w.x && *x <= w.x + w.w && *y >= w.y && *y <= w.y + w.h;
                            if hit {
                                if *state == ElementState::Pressed {
                                    changed = true;
                                } else if *state == ElementState::Released {
                                    if let Some(cb) = w.widget.as_dyn_mut().as_any_mut().downcast_mut::<Checkbox>() {
                                        let new_checked = !cb.checked();
                                        cb.set_checked(new_checked);
                                    }
                                    changed = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                if w.widget.as_dyn_mut().handle_event(event, ctx) {
                    changed = true;
                }
                if w.widget_type == "button" {
                    // take_click is an WidgetHost method; the Phase 5 Button is Adapted, so call it
                    // on the box directly rather than through a concrete downcast.
                    if w.target_page.is_some() && w.widget.take_click() {
                        page_switch = Some(w.target_page.unwrap());
                    }
                }
            }
        }

        if let Some(target) = page_switch {
            self.active_page = target;
            self.layout_children();
            changed = true;
        }

        if let Event::MouseWheel { delta, x, y, .. } = event {
            let (bx, by, bw, bh) = self.rect();
            if *x >= bx && *x <= bx + bw && *y >= by && *y <= by + bh {
                if active_page < self.page_total_heights.len() {
                    let total_height = self.page_total_heights[active_page];
                    let visible_h = bh;
                    let max_scroll_y = (total_height - visible_h).max(0.0);
                    if max_scroll_y > 0.0 {
                        // A notch is LINE_PX, pixels are 1:1; the motion
                        // glides or tracks and `tick_scroll` carries the drawn
                        // offset after it. A true return is the repaint signal
                        // (the target moved even if the offset has not yet).
                        let motion = &mut self.page_scroll[active_page];
                        motion.reconcile(0.0, self.page_scroll_y[active_page]);
                        if motion.apply(delta, (LINE_PX, LINE_PX), Bounds::max(0.0), Bounds::max(max_scroll_y)) {
                            changed = true;
                        }
                        if self.sync_page_scroll(active_page) {
                            self.layout_children();
                        }
                    }
                }
            }
        }

        if let Event::KeyInput(key_event) = event {
            if key_event.state == ElementState::Pressed {
                if active_page < self.page_total_heights.len() {
                    let (_, _, _, bh) = self.rect();
                    let by = self.page_scroll_bounds(active_page);
                    if by.hi > 0.0 {
                        // Pages and Home/End glide to their target; the arrows
                        // step a line and accumulate like wheel notches (the
                        // toolkit ScrollRegion's keyboard contract).
                        let s = scroll_settings();
                        let motion = &mut self.page_scroll[active_page];
                        motion.reconcile(0.0, self.page_scroll_y[active_page]);
                        let target = motion.y.target();
                        let moved = match &key_event.logical_key {
                            Key::Named(NamedKey::PageDown) => motion.y.scroll_to(target + bh, by, &s),
                            Key::Named(NamedKey::PageUp) => motion.y.scroll_to(target - bh, by, &s),
                            Key::Named(NamedKey::Home) => motion.y.scroll_to(0.0, by, &s),
                            Key::Named(NamedKey::End) => motion.y.scroll_to(by.hi, by, &s),
                            Key::Named(NamedKey::ArrowDown) => motion.y.wheel(LINE_PX, by, &s),
                            Key::Named(NamedKey::ArrowUp) => motion.y.wheel(-LINE_PX, by, &s),
                            _ => false,
                        };
                        if moved {
                            changed = true;
                        }
                        if self.sync_page_scroll(active_page) {
                            self.layout_children();
                        }
                    }
                }
            }
        }

        if let Event::Tick(dt) = event {
            for w in &mut self.widgets {
                if w.page_idx != active_page {
                    continue;
                }
                if w.widget.as_dyn_mut().tick(*dt, ctx) {
                    changed = true;
                }
            }
        }

        if let Event::MouseButton { state, .. } = event {
            if *state == ElementState::Pressed && !changed {
                ctx.clear_focus();
            }
        }

        changed
    }
}

impl cce_ui::widget::Layout for JsonLayoutWidget {
    // The old `set_rect` override: land the rect in the model, then place the children.
    fn rect_assigned(&mut self, rect: cce_ui::scene::layout::Rect) {
        self.base.x = rect.x;
        self.base.y = rect.y;
        self.base.w = rect.width;
        self.base.h = rect.height;
        self.layout_children();
    }
}

impl cce_ui::widget::Paint for JsonLayoutWidget {
    fn color(&self) -> [f32; 4] { [0.0, 0.0, 0.0, 0.0] }

    // JsonLayout paints its whole subtree: page-filtered aggregates plus the checkbox
    // side-labels that belong to the container, not to any child widget. The paint walk
    // emits these once and does not descend (descending would draw inactive pages and
    // miss the side-labels); the adapter forwards the Text prims verbatim, bounds included.
    fn paints_own_subtree(&self) -> bool { true }

    fn paint(&self, _rect: cce_ui::scene::layout::Rect, pc: &mut cce_ui::scene::paint::PaintCtx) {
        use cce_ui::scene::layout::Rect;
        use cce_ui::scene::paint::{PaintCtx, Prim};
        // The children are Phase 5 Adapted leaves: nothing their all_* getters or the
        // label walk reads comes from the routing context, so a fresh one stands in for
        // the ctx `Paint::paint` does not carry.
        let dummy = UiContext::new();
        // Children through the real paint walk, geometry only: bevel/recess/plate prims
        // survive to the tessellator where the old quad bridges flattened them to fills.
        // Text prims are skipped — every label, container-owned and child-owned alike,
        // is served by `own_labels_with_bounds` below, and emitting the children's own
        // labels here as well would double them. Page filtering and the panel clip
        // mirror the dissolved `aggregate_quads` bounds.
        let (bx, by, bw, bh) = self.rect();
        let pad_x = 16.0;
        let clip = Rect { x: bx + pad_x - 4.0, y: by, width: bw - (pad_x - 4.0), height: bh };

        // One recess per run of adjacent menu rows, drawn BEFORE the rows so
        // they sit inside it. A menu row draws no plate and no border of its
        // own (ButtonKind::MenuItem) — the group is the carved thing, and the
        // rows butt against each other within it, which is why the run's
        // bounding box is a single continuous well rather than one per item.
        let radius = cce_ui::layout::button_corner_radius();
        let face = cce_ui::colors::button_background_color();
        let mut seams: Vec<(Rect, f32)> = Vec::new();
        let mut i = 0;
        while i < self.widgets.len() {
            let w = &self.widgets[i];
            if w.page_idx != self.active_page || w.widget_type != "button" {
                i += 1;
                continue;
            }
            let start = i;
            let mut end = i;
            while let Some(n) = self.widgets.get(end + 1) {
                if n.page_idx == self.active_page && n.widget_type == "button" {
                    end += 1;
                } else {
                    break;
                }
            }
            let (first, last) = (&self.widgets[start], &self.widgets[end]);
            let run = Rect {
                x: first.x,
                y: first.y,
                width: first.w,
                height: last.y + last.h - first.y,
            };
            // Depth from ONE row's height, not the run's: the groove has to
            // read the same as every other carved control in the DE, and a
            // tall run would otherwise cut a far deeper channel.
            let depth = cce_ui::layout::bevel_width().min(first.h * 0.2);
            pc.clip(clip, |pc| {
                pc.inset_plate(run, (radius, radius, radius, radius), cce_ui::scene::Material::face(face).as_ref(), depth);
            });
            // Seams are collected, not drawn yet: they are pure shading and
            // must land ON TOP of the rows. A hovered row fills its whole
            // rect, and the groove straddles the boundary between two rows —
            // drawn underneath, the hover would erase half of each groove it
            // touches.
            //
            // Half depth so the two walls MEET at the boundary rather than
            // leaving flat floor between them: at full depth the seam reads as
            // two separate hairlines ~9px apart instead of one groove, against
            // the well's own ring which measures a 4px dark-to-light V.
            let seam_d = depth * 0.5;
            for k in start..end {
                let seam = self.widgets[k].y + self.widgets[k].h;
                seams.push((
                    Rect { x: run.x, y: seam - seam_d, width: run.width, height: 2.0 * seam_d },
                    seam_d,
                ));
            }
            i = end + 1;
        }

        for w in &self.widgets {
            if w.page_idx != self.active_page {
                continue;
            }
            let mut tmp = PaintCtx::new();
            w.widget.as_dyn().paint_self(&dummy, &mut tmp);
            pc.clip(clip, |pc| {
                for item in tmp.finish().items {
                    let clip_circle = item.clip_circle;
                    if let Some(c) = clip_circle {
                        pc.push_clip_circle(c);
                    }
                    // One forwarding match, in cce-ui: `PaintCtx::replay` emits
                    // every prim but Text and returns Text for the caller to
                    // decide. Here the widget's own label bridge supplies the
                    // text, so the returned prim is dropped.
                    let _ = pc.replay(item.prim);
                    if clip_circle.is_some() {
                        pc.pop_clip_circle();
                    }
                }
            });
        }
        // Seam grooves last: pure shading, composed over the rows so a hovered
        // row's fill cannot erase the grooves it straddles.
        for (rect, seam_d) in seams {
            pc.clip(clip, |pc| {
                pc.recess_edges(rect, (0.0, 0.0, 0.0, 0.0), seam_d, (true, false, true, false));
            });
        }
        for (tl, bounds) in self.own_labels_with_bounds(&dummy) {
            pc.text_with(tl.text, tl.x, tl.y, tl.font_size, tl.color, None, bounds);
        }
    }
}

impl cce_ui::widget::Input for JsonLayoutWidget {
    fn scrollable(&self) -> bool { true }
    fn wants_tick(&self) -> bool { true }
    fn gates_presses(&self) -> bool { false }

    fn on_event(&mut self, event: &Event, ectx: &mut cce_ui::widget::EventCtx) -> bool {
        let Some(ctx) = ectx.ui.as_deref_mut() else { return false };
        self.route_event(event, ctx)
    }

    fn tick_ctx(&mut self, dt: f32, ectx: &mut cce_ui::widget::EventCtx) -> bool {
        let Some(ctx) = ectx.ui.as_deref_mut() else { return false };
        self.tick_children(dt, ctx)
    }
}

impl JsonLayoutWidget {
    pub(crate) fn own_labels_with_bounds(&self, ctx: &UiContext) -> Vec<(TextLabel, Option<[f32; 4]>)> {
        let mut labels = Vec::new();
        let (bx, by, bw, bh) = self.rect();

        let active_page = self.active_page;
        let pad_x = 16.0;
        let content_bounds = Some([bx + pad_x - 4.0, by, bx + bw, by + bh]);

        for w in &self.widgets {
            if w.page_idx != active_page {
                continue;
            }
            let w_labels = if w.widget_type == "checkbox" {
                if let Some(tl) = &w.label_text {
                    vec![tl.clone()]
                } else {
                    Vec::new()
                }
            } else {
                // The trait text getters are gone: read the child's text off the paint
                // walk (same prims, fonts dropped — this consumer shapes with its own
                // control font, as the legacy getter path did).
                let mut scratch = cce_ui::scene::paint::PaintCtx::new();
                cce_ui::scene::painter::append_widget_text(ctx, w.widget.as_dyn(), &mut scratch);
                scratch
                    .finish()
                    .items
                    .into_iter()
                    .filter_map(|item| match item.prim {
                        cce_ui::scene::paint::Prim::Text { text, x, y, font_size, color, .. } => {
                            Some(TextLabel { text, x, y, font_size, color })
                        }
                        _ => None,
                    })
                    .collect()
            };
            for l in w_labels {
                labels.push((l, content_bounds));
            }
        }
        labels
    }
}
