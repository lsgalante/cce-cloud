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

/// style: deliberate — how far west of the content inset the panel's paint
/// clip and text bounds begin, so a child drawn out to its own rect edge (a
/// label's side bearing, a well's rim) is not cut at the inset. Slack, not a
/// rung of the spacing ladder.
const CLIP_SLACK: f32 = 4.0;

/// A button's glyphs, by cce-ui's context-menu conventions, so a JSON menu
/// shows its marks and page turns the way every other menu in the DE does:
///
/// - a label that BEGINS with `MARK_CHECK` ("✓ "), `MARK_ON` ("● ") or
///   `MARK_OFF` ("○ ") wears the check, circle or circle-outline glyph at
///   its left, and the text is drawn without the mark;
/// - a button with a `target_page` LOWER than its own page turns back, and
///   wears the chevron-left glyph at its left; one with any other target
///   leads to a page, and wears the chevron-right glyph at its right end.
///
/// Returns `(left glyph, label without its mark, right glyph)`. Every
/// argument is in the protocol already, so a layout written before the
/// glyphs needs nothing new — though one that spelled its own marks ("< ",
/// " >", "[x] ") would now show them twice, which is why the DE's scripts
/// were rewritten with this.
pub fn button_glyphs(
    text: &str,
    page_idx: usize,
    target_page: Option<usize>,
) -> (Option<&'static str>, &str, Option<&'static str>) {
    let (mark, text) = cce_ui::widget::context_menu::split_mark(text);
    match target_page {
        Some(t) if t < page_idx => (Some("chevron-left"), text, None),
        Some(_) => (mark, text, Some("chevron-right")),
        None => (mark, text, None),
    }
}

/// The side of a mark glyph, and the gap after it, at button font size
/// `size` — cce-ui's context-menu proportions: a mark about as tall as a
/// capital, a chevron smaller, since it points rather than labels.
fn mark_side(size: f32) -> f32 {
    (size * 0.95).round()
}
fn chevron_side(size: f32) -> f32 {
    (size * 0.8).round()
}
const GLYPH_GAP: f32 = 6.0;
/// Where a capital's middle stands in a line box, as a share of the font
/// size from its top: what a glyph beside the text is centred on.
const CAP_MIDDLE: f32 = 0.66;

/// The room a button's glyphs take beside its text: the left column (on
/// every button of a page where any button has a left glyph, so the labels
/// share one edge) and the right chevron.
pub fn glyph_room(lead_column: bool, trail: bool, size: f32) -> f32 {
    let lead = if lead_column { mark_side(size) + GLYPH_GAP } else { 0.0 };
    let trail = if trail { GLYPH_GAP + chevron_side(size) } else { 0.0 };
    lead + trail
}

/// The button font's size, which a glyph is sized from.
pub fn button_font_size() -> f32 {
    cce_ui::layout::parse_font_string(&cce_ui::layout::button_font()).1.unwrap_or(12.0)
}

/// Whether a page's buttons reserve the left glyph column: any one of them
/// has a left glyph.
pub fn page_has_lead(widgets: &[JsonWidgetConfig], page_idx: usize) -> bool {
    widgets
        .iter()
        .any(|w| w.widget_type == "button" && button_glyphs(&w.text, page_idx, w.target_page).0.is_some())
}

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
    /// A button's glyphs (see [`button_glyphs`]); `text` is the label
    /// without its mark.
    pub lead: Option<&'static str>,
    pub trail: Option<&'static str>,
    /// Whether this button reserves the left glyph column ([`page_has_lead`]).
    pub lead_column: bool,
    pub justify: Justification,
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
                let lead_column = page_has_lead(&page.widgets, page_idx);
                for (idx, w_conf) in page.widgets.iter().enumerate() {
                    let id = w_conf.id.clone().unwrap_or_else(|| format!("widget_{}_{}", page_idx, idx));
                    let widget_type = w_conf.widget_type.clone();
                    let (lead, text, trail) = if widget_type == "button" {
                        button_glyphs(&w_conf.text, page_idx, w_conf.target_page)
                    } else {
                        (None, w_conf.text.as_str(), None)
                    };
                    let text = text.to_string();

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
                        lead,
                        trail,
                        lead_column: lead_column && w_conf.widget_type == "button",
                        justify: page_justify,
                    });
                }
            }
        } else if let Some(ref widgets_conf) = config.widgets {
            let global_justify = config.justify.unwrap_or(Justification::Center);
            let lead_column = page_has_lead(widgets_conf, 0);
            for (idx, w_conf) in widgets_conf.iter().enumerate() {
                let id = w_conf.id.clone().unwrap_or_else(|| format!("widget_{}", idx));
                let widget_type = w_conf.widget_type.clone();
                let (lead, text, trail) = if widget_type == "button" {
                    button_glyphs(&w_conf.text, 0, w_conf.target_page)
                } else {
                    (None, w_conf.text.as_str(), None)
                };
                let text = text.to_string();

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
                    lead,
                    trail,
                    lead_column: lead_column && w_conf.widget_type == "button",
                    justify: global_justify,
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

        // The widgets stand straight on the popup's root plate: the root-plate
        // inset from its edge, the root-plate gap between them.
        let pad_x = cce_ui::layout::root_plate_inset();
        let usable_w = bw - 2.0 * pad_x;

        let mut page_current_y = vec![pad_x; 16]; // support up to 16 pages
        let spacing = cce_ui::layout::root_plate_gap();

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
                // The toolkit's checkbox row: toggle height, the box square
                // at it (as a labelled `Checkbox::field` draws it).
                let h = cce_ui::layout::toggle_height();
                w_state.widget.as_dyn_mut().set_rect(w_state.x, w_state.y, h, h);
                w_state.h = h;
                w_state.label_text = Some(TextLabel {
                    text: w_state.text.clone(),
                    x: w_state.x + h + 10.0,
                    y: w_state.y + (h - 18.0) / 2.0,
                    font_size: 13.0,
                    color: [0xcc, 0xcc, 0xd4],
                });
            } else {
                let h = match w_state.widget_type.as_str() {
                    // Menu-styled rows (`new_menu_item`): the context menu's pitch.
                    "button" => cce_ui::widget::context_menu::ROW_H + top_room,
                    "label" => 18.0 + top_room,
                    "spinbox" => cce_ui::layout::spinbox_height() + top_room,
                    "color" | "rgb" | "rgba" => cce_ui::layout::color_selector_height() + top_room,
                    "slider" => 22.0 + top_room,
                    // An unknown type is built as a bare Button.
                    _ => cce_ui::layout::button_height(),
                };
                w_state.widget.as_dyn_mut().set_rect(w_state.x, w_state.y, usable_w, h);
                w_state.h = h;
            }

            *current_y += w_state.h + if run_continues { 0.0 } else { spacing };
        }

        // Store total height of each page: the walk left a gap after the last
        // widget, and what stands below it is the inset, not a gap.
        for (i, &height) in page_current_y.iter().enumerate() {
            if i < self.page_total_heights.len() {
                self.page_total_heights[i] = (height - spacing).max(pad_x) + pad_x;
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
        // is served by `labels_and_glyphs` below, and emitting the children's own
        // labels here as well would double them. Page filtering and the panel clip
        // mirror the dissolved `aggregate_quads` bounds.
        let (bx, by, bw, bh) = self.rect();
        let pad_x = cce_ui::layout::root_plate_inset();
        // style: deliberate — the clip starts CLIP_SLACK west of the content
        // edge so a child painting out to its rect edge is not cut there.
        let clip = Rect { x: bx + pad_x - CLIP_SLACK, y: by, width: bw - (pad_x - CLIP_SLACK), height: bh };

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
        let (labels, glyphs) = self.labels_and_glyphs(&dummy);
        // The buttons' glyphs (see `button_glyphs`), tinted the label's own
        // colour. A missing icon set draws nothing in their place: the label
        // beside each is the word it always was.
        pc.clip(clip, |pc| {
            for (name, rect, color) in glyphs {
                pc.icon(name, rect, color);
            }
        });
        for (tl, bounds) in labels {
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

/// A glyph to draw: its cce-icons name, rect and tint.
type GlyphPrim = (&'static str, cce_ui::scene::layout::Rect, [f32; 4]);

impl JsonLayoutWidget {
    /// Every label on the active page, moved aside for its button's glyphs,
    /// and the glyphs themselves. A glyph is placed off its label — it is
    /// sized from the label's font and tinted its colour, and the left one
    /// stands where the label would have started (left-justified; centred,
    /// label and glyph are centred together).
    pub(crate) fn labels_and_glyphs(&self, ctx: &UiContext) -> (Vec<(TextLabel, Option<[f32; 4]>)>, Vec<GlyphPrim>) {
        let mut labels = Vec::new();
        let mut glyphs = Vec::new();
        let (bx, by, bw, bh) = self.rect();
        let pad_x = cce_ui::layout::root_plate_inset();
        let content_bounds = Some([bx + pad_x - CLIP_SLACK, by, bx + bw, by + bh]);
        for w in &self.widgets {
            if w.page_idx != self.active_page {
                continue;
            }
            let mut wl = self.widget_labels(ctx, w);
            self.place_glyphs(w, wl.first_mut(), &mut glyphs);
            labels.extend(wl.into_iter().map(|l| (l, content_bounds)));
        }
        (labels, glyphs)
    }

    /// A button's glyphs, placed off its label `tl` (which moves aside for
    /// the left glyph column).
    fn place_glyphs(&self, w: &JsonWidget, tl: Option<&mut TextLabel>, glyphs: &mut Vec<GlyphPrim>) {
        use cce_ui::scene::layout::Rect;
        if w.widget_type != "button" {
            return;
        }
        let Some(tl) = tl else { return };
        let size = tl.font_size;
        let color = [tl.color[0] as f32 / 255.0, tl.color[1] as f32 / 255.0, tl.color[2] as f32 / 255.0, 1.0];
        let column = if w.lead_column { mark_side(size) + GLYPH_GAP } else { 0.0 };
        // Level with the text's capitals, not with the row: the button sets
        // its label a little below the row's middle, and a capital's middle
        // stands below its line box's (measured in a shadow: the line box's
        // middle put the glyphs 2 px high at the 12 px button font).
        let mid = tl.y + size * CAP_MIDDLE;
        // Where the left column starts, and the label moved aside for it.
        let col_x = match w.justify {
            Justification::Left => {
                let x = tl.x;
                tl.x += column;
                x
            }
            Justification::Center => {
                let x = tl.x - column / 2.0;
                tl.x += column / 2.0;
                x
            }
            Justification::Right => tl.x - column,
        };
        if let Some(name) = w.lead {
            let side = if name == "chevron-left" { chevron_side(size) } else { mark_side(size) };
            let rect = Rect {
                x: col_x + (mark_side(size) - side) / 2.0,
                y: mid - side / 2.0,
                width: side,
                height: side,
            };
            glyphs.push((name, rect, color));
        }
        if let Some(name) = w.trail {
            let side = chevron_side(size);
            // The button's own text inset from its right edge.
            let rect = Rect { x: w.x + w.w - 8.0 - side, y: mid - side / 2.0, width: side, height: side };
            glyphs.push((name, rect, color));
        }
    }

    /// One widget's labels, read off the paint walk.
    fn widget_labels(&self, ctx: &UiContext, w: &JsonWidget) -> Vec<TextLabel> {
        if w.widget_type == "checkbox" {
            return w.label_text.iter().cloned().collect();
        }
        // The trait text getters are gone: read the child's text off the paint
        // walk (same prims, fonts dropped — this consumer shapes with its own
        // control font, as the legacy getter path did).
        let mut scratch = cce_ui::scene::paint::PaintCtx::new();
        cce_ui::widget::painter::append_widget_text(ctx, w.widget.as_dyn(), &mut scratch);
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
    }
}
