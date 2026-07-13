//! App-owned copy of the dissolved cce-ui `JsonLayoutWidget` (Phase 6ay): cloud is the
//! only consumer — the KDL/JSON-driven launcher layout host (`LauncherMode::Json`),
//! on the narrow traits wrapped in `Adapted<JsonLayoutWidget>` (Phase 6az): the paint
//! walk reaches it as the adapter, whose subtree text pass-through forwards `paint`'s
//! prims verbatim. `Justification` stayed in cce-ui (Button, cce-files, settings).

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
    pub page_scroll_y: Vec<f32>,
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
                            JsonControl::Button(Button::new(0.0, 0.0, 0.0, 0.0)
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
                        JsonControl::Button(Button::new(0.0, 0.0, 0.0, 0.0)
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

        for w_state in &mut self.widgets {
            let p_idx = w_state.page_idx;
            if p_idx >= page_current_y.len() {
                continue;
            }
            let current_y = &mut page_current_y[p_idx];
            w_state.x = bx + pad_x;

            let top_room = cce_ui::widget::label_offset(w_state.widget.as_dyn());

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

            *current_y += w_state.h + spacing;
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

    /// The page-filtered plain-quad aggregate (the old `WidgetHost::all_quads` override):
    /// active-page children's backgrounds, decoration quads, and highlight, clipped to
    /// the content area. External readers reach it through the adapter's reverse bridge
    /// (`jl.all_quads(ctx)` serves `paint`'s plain prims, which come from here).
    fn aggregate_quads(&self, ctx: &UiContext) -> Vec<(f32, f32, f32, f32, [f32; 4])> {
        let mut quads = Vec::new();

        let active_page = self.active_page;
        let (bx, by, bw, bh) = self.rect();
        let pad_x = 16.0;
        let min_x = bx + pad_x - 4.0;
        let max_x = bx + bw;
        let min_y = by;
        let max_y = by + bh;

        let push_clipped = |qx: f32, qy: f32, qw: f32, qh: f32, qc: [f32; 4], q: &mut Vec<(f32, f32, f32, f32, [f32; 4])>| {
            let rx1 = qx.max(min_x);
            let ry1 = qy.max(min_y);
            let rx2 = (qx + qw).min(max_x);
            let ry2 = (qy + qh).min(max_y);
            let rw = rx2 - rx1;
            let rh = ry2 - ry1;
            if rw > 0.0 && rh > 0.0 {
                q.push((rx1, ry1, rw, rh, qc));
            }
        };

        for w in &self.widgets {
            if w.page_idx != active_page {
                continue;
            }
            let (wx, wy, ww, wh) = w.widget.as_dyn().rect();
            let has_rounded = w.widget.as_dyn().corner_style().1 != (false, false, false, false);
            if !has_rounded {
                push_clipped(wx, wy, ww, wh, w.widget.as_dyn().color(), &mut quads);
            }
            for q in w.widget.as_dyn().all_quads(ctx) {
                if has_rounded && (q.0 - wx).abs() < 0.1 && (q.1 - wy).abs() < 0.1 && (q.2 - ww).abs() < 0.1 && (q.3 - wh).abs() < 0.1 {
                    continue;
                }
                push_clipped(q.0, q.1, q.2, q.3, q.4, &mut quads);
            }
            if let Some(hq) = w.widget.as_dyn().highlight_quad(ctx) {
                push_clipped(hq.0, hq.1, hq.2, hq.3, hq.4, &mut quads);
            }
        }
        quads
    }

    /// Per-frame child state (the old `WidgetHost::tick` override): active-page widgets only.
    fn tick_children(&mut self, dt: f32, ctx: &mut UiContext) -> bool {
        let mut changed = false;
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
                        let scroll_amount = match delta {
                            cce_ui::widget::MouseScrollDelta::LineDelta(_x, y) => -*y * 24.0,
                            cce_ui::widget::MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
                        };
                        let old_scroll = self.page_scroll_y[active_page];
                        self.page_scroll_y[active_page] = (old_scroll + scroll_amount).clamp(0.0, max_scroll_y);
                        if (self.page_scroll_y[active_page] - old_scroll).abs() > 0.01 {
                            self.layout_children();
                            changed = true;
                        }
                    }
                }
            }
        }

        if let Event::KeyInput(key_event) = event {
            if key_event.state == ElementState::Pressed {
                if active_page < self.page_total_heights.len() {
                    let total_height = self.page_total_heights[active_page];
                    let (_, _, _, bh) = self.rect();
                    let max_scroll_y = (total_height - bh).max(0.0);
                    if max_scroll_y > 0.0 {
                        let old_scroll = self.page_scroll_y[active_page];
                        match &key_event.logical_key {
                            Key::Named(NamedKey::PageDown) => {
                                self.page_scroll_y[active_page] = (old_scroll + bh).clamp(0.0, max_scroll_y);
                            }
                            Key::Named(NamedKey::PageUp) => {
                                self.page_scroll_y[active_page] = (old_scroll - bh).clamp(0.0, max_scroll_y);
                            }
                            Key::Named(NamedKey::Home) => {
                                self.page_scroll_y[active_page] = 0.0;
                            }
                            Key::Named(NamedKey::End) => {
                                self.page_scroll_y[active_page] = max_scroll_y;
                            }
                            Key::Named(NamedKey::ArrowDown) => {
                                self.page_scroll_y[active_page] = (old_scroll + 24.0).clamp(0.0, max_scroll_y);
                            }
                            Key::Named(NamedKey::ArrowUp) => {
                                self.page_scroll_y[active_page] = (old_scroll - 24.0).clamp(0.0, max_scroll_y);
                            }
                            _ => {}
                        }
                        if (self.page_scroll_y[active_page] - old_scroll).abs() > 0.01 {
                            self.layout_children();
                            changed = true;
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
        // The children are Phase 5 Adapted leaves: nothing their all_* getters or the
        // label walk reads comes from the routing context, so a fresh one stands in for
        // the ctx `Paint::paint` does not carry.
        let dummy = UiContext::new();
        // Rounded: the deleted WidgetHost default's shape — no own background (transparent,
        // sharp corners), every child unfiltered, in `widgets` order.
        for w in &self.widgets {
            for (x, y, qw, qh, r, c, corners) in w.widget.as_dyn().all_rounded_quads(&dummy) {
                pc.rounded_rect(Rect { x, y, width: qw, height: qh }, r, corners, c);
            }
        }
        for (x, y, w, h, c) in self.aggregate_quads(&dummy) {
            pc.quad(Rect { x, y, width: w, height: h }, c);
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
