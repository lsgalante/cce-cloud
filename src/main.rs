mod scroll_region;
use scroll_region::ScrollRegion;

use std::sync::{Arc, Mutex};
use std::io::{self, BufRead, IsTerminal};

use cce_ui::widget::{Element, TextLabel};
use crate::json_layout::{JsonLayoutWidget, JsonLayoutConfig};
mod json_layout;
#[cfg(test)]
use cce_ui::widget::JsonWidgetConfig;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm, delegate_layer, delegate_output,
    delegate_xdg_shell, delegate_xdg_window,
    registry::{ProvidesRegistryState, RegistryState},
    output::{OutputHandler, OutputState},
    seat::{
        keyboard::KeyboardHandler,
        pointer::PointerHandler,
        Capability, SeatHandler, SeatState,
    },
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler,
            LayerSurface, LayerSurfaceConfigure,
        },
        xdg::{
            window::{Window as XdgWindow, WindowConfigure, WindowHandler, WindowDecorations},
            XdgShell,
        },
        WaylandSurface,
    },
    shm::{Shm, ShmHandler},
};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_surface},
    Connection, QueueHandle, Proxy,
};
use calloop_wayland_source::WaylandSource;

use glyphon::{
    Attrs, Buffer, Cache, FontSystem, Metrics, Resolution, SwashCache, TextArea, TextAtlas,
    TextBounds, TextRenderer, Viewport,
};

// Vertex and quad_vertices are shared from the cce-ui engine.
pub(crate) use cce_ui::engine::{quad_vertices, Vertex};

fn rounded_rect_vertices_corners(
    x: f32, y: f32, ww: f32, h: f32,
    r: f32,
    sw: f32, sh: f32,
    color: [f32; 4],
    corners: (bool, bool, bool, bool),
) -> Vec<Vertex> {
    let mut verts = Vec::new();
    let r = r.min(ww * 0.5).min(h * 0.5);

    let push_quad = |verts: &mut Vec<Vertex>, qx: f32, qy: f32, qw: f32, qh: f32| {
        let x0 = qx;
        let y0 = qy;
        let x1 = qx + qw;
        let y1 = qy + qh;
        
        let ndc_x0 = (x0 / sw) * 2.0 - 1.0;
        let ndc_y0 = 1.0 - (y0 / sh) * 2.0;
        let ndc_x1 = (x1 / sw) * 2.0 - 1.0;
        let ndc_y1 = 1.0 - (y1 / sh) * 2.0;
        
        let clip_circle = [0.0, 0.0, 0.0];
        verts.push(Vertex { position: [ndc_x0, ndc_y0], color, clip_circle });
        verts.push(Vertex { position: [ndc_x1, ndc_y0], color, clip_circle });
        verts.push(Vertex { position: [ndc_x0, ndc_y1], color, clip_circle });
        verts.push(Vertex { position: [ndc_x1, ndc_y0], color, clip_circle });
        verts.push(Vertex { position: [ndc_x1, ndc_y1], color, clip_circle });
        verts.push(Vertex { position: [ndc_x0, ndc_y1], color, clip_circle });
    };

    if r <= 0.1 || (!corners.0 && !corners.1 && !corners.2 && !corners.3) {
        push_quad(&mut verts, x, y, ww, h);
        return verts;
    }

    push_quad(&mut verts, x + r, y, ww - 2.0 * r, h);
    push_quad(&mut verts, x, y + r, r, h - 2.0 * r);
    push_quad(&mut verts, x + ww - r, y + r, r, h - 2.0 * r);

    let corner_configs = [
        (corners.0, x, y, x + r, y + r, std::f32::consts::PI, 1.5 * std::f32::consts::PI),
        (corners.1, x + ww - r, y, x + ww - r, y + r, 1.5 * std::f32::consts::PI, 2.0 * std::f32::consts::PI),
        (corners.2, x + ww - r, y + h - r, x + ww - r, y + h - r, 0.0, 0.5 * std::f32::consts::PI),
        (corners.3, x, y + h - r, x + r, y + h - r, 0.5 * std::f32::consts::PI, std::f32::consts::PI),
    ];

    let segments = 16;
    for &(is_rounded, sqx, sqy, cx, cy, start, end) in &corner_configs {
        if is_rounded {
            for i in 0..segments {
                let theta1 = start + (i as f32) * (end - start) / (segments as f32);
                let theta2 = start + ((i + 1) as f32) * (end - start) / (segments as f32);
                
                let x0 = cx;
                let y0 = cy;
                let x1 = cx + r * theta1.cos();
                let y1 = cy + r * theta1.sin();
                let x2 = cx + r * theta2.cos();
                let y2 = cy + r * theta2.sin();
                
                let ndc_x0 = (x0 / sw) * 2.0 - 1.0;
                let ndc_y0 = 1.0 - (y0 / sh) * 2.0;
                let ndc_x1 = (x1 / sw) * 2.0 - 1.0;
                let ndc_y1 = 1.0 - (y1 / sh) * 2.0;
                let ndc_x2 = (x2 / sw) * 2.0 - 1.0;
                let ndc_y2 = 1.0 - (y2 / sh) * 2.0;
                
                let clip_circle = [0.0, 0.0, 0.0];
                verts.push(Vertex { position: [ndc_x0, ndc_y0], color, clip_circle });
                verts.push(Vertex { position: [ndc_x1, ndc_y1], color, clip_circle });
                verts.push(Vertex { position: [ndc_x2, ndc_y2], color, clip_circle });
            }
        } else {
            push_quad(&mut verts, sqx, sqy, r, r);
        }
    }

    verts
}

fn widget_vertices(w: &dyn Element, sw: f32, sh: f32) -> Vec<Vertex> {
    let (x, y, ww, h) = w.rect();
    let mut verts = quad_vertices(x, y, ww, h, sw, sh, w.color()).to_vec();
    for (qx, qy, qw, qh, qc) in w.extra_quads() {
        verts.extend(quad_vertices(qx, qy, qw, qh, sw, sh, qc));
    }
    verts
}

fn make_text_buffer(font_system: &mut FontSystem, text: &str, size: f32) -> Buffer {
    let metrics = Metrics::new(size, size * 1.4);
    let mut buffer = Buffer::new(font_system, metrics);
    let font_family = cce_ui::layout::control_label_font_parsed().0;
    let attrs = Attrs::new().family(glyphon::Family::Name(&font_family));
    buffer.set_text(font_system, text, attrs, glyphon::Shaping::Advanced);
    buffer.shape_until_scroll(font_system, true);
    buffer
}

/// A widget subtree's text via the paint walk (not the legacy text_labels getter),
/// reduced to the plain labels this renderer shapes: the buffer font and window bounds
/// stay exactly as before (make_text_buffer applies the control font to every label).
fn walk_text_labels(ui: &cce_ui::context::UiContext, w: &dyn Element) -> Vec<TextLabel> {
    let mut pc = cce_ui::scene::paint::PaintCtx::new();
    cce_ui::scene::painter::append_widget_text(ui, w, &mut pc);
    pc.finish()
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

fn filter_and_sort_items(items: &[String], query: &str) -> Vec<String> {
    if query.is_empty() {
        return items.to_vec();
    }
    let query_lower = query.to_lowercase();
    
    let mut scored: Vec<(i32, usize, &String)> = items
        .iter()
        .enumerate()
        .filter_map(|(idx, item)| {
            let item_lower = item.to_lowercase();
            if item_lower == query_lower {
                Some((100, idx, item))
            } else if item_lower.starts_with(&query_lower) {
                Some((80, idx, item))
            } else if item_lower.contains(&query_lower) {
                Some((50, idx, item))
            } else {
                // Character sequence match
                let mut query_chars = query_lower.chars().peekable();
                for c in item_lower.chars() {
                    if let Some(&qc) = query_chars.peek() {
                        if c == qc {
                            query_chars.next();
                        }
                    }
                }
                if query_chars.peek().is_none() {
                    Some((10, idx, item))
                } else {
                    None
                }
            }
        })
        .collect();
        
    scored.sort_by(|a, b| {
        let score_cmp = b.0.cmp(&a.0);
        if score_cmp != std::cmp::Ordering::Equal {
            score_cmp
        } else {
            a.1.cmp(&b.1)
        }
    });
    scored.into_iter().map(|(_, _, item)| item.clone()).collect()
}

fn scan_path() -> Vec<String> {
    let mut executables = std::collections::BTreeSet::new();
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if path.is_file() {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::PermissionsExt;
                                if let Ok(metadata) = entry.metadata() {
                                    if metadata.permissions().mode() & 0o111 != 0 {
                                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                            executables.insert(name.to_string());
                                        }
                                    }
                                }
                            }
                            #[cfg(not(unix))]
                            {
                                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                    executables.insert(name.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    executables.into_iter().collect()
}

fn clean_exec_command(exec: &str) -> String {
    let mut words = Vec::new();
    for word in exec.split_whitespace() {
        match word {
            "%f" | "%F" | "%u" | "%U" | "%d" | "%D" | "%n" | "%N" | "%i" | "%c" | "%k" | "%v" => {
                // Skip these field codes
            }
            _ => {
                let cleaned = word
                    .replace("%f", "")
                    .replace("%F", "")
                    .replace("%u", "")
                    .replace("%U", "")
                    .replace("%d", "")
                    .replace("%D", "")
                    .replace("%n", "")
                    .replace("%N", "")
                    .replace("%i", "")
                    .replace("%c", "")
                    .replace("%k", "")
                    .replace("%v", "")
                    .replace("%%", "%");
                if !cleaned.is_empty() {
                    words.push(cleaned);
                }
            }
        }
    }
    words.join(" ")
}

fn parse_desktop_file(path: &std::path::Path) -> Option<AppInfo> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    
    let mut in_desktop_entry = false;
    let mut name = None;
    let mut exec = None;
    let mut is_application = true;
    let mut no_display = false;

    for line in reader.lines() {
        let line = line.ok()?;
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if trimmed == "[Desktop Entry]" {
                in_desktop_entry = true;
            } else {
                in_desktop_entry = false;
            }
            continue;
        }
        if in_desktop_entry {
            if let Some(pos) = trimmed.find('=') {
                let key = trimmed[..pos].trim();
                let value = trimmed[pos + 1..].trim();
                match key {
                    "Name" => {
                        if name.is_none() {
                            name = Some(value.to_string());
                        }
                    }
                    "Exec" => {
                        if exec.is_none() {
                            exec = Some(clean_exec_command(value));
                        }
                    }
                    "Type" => {
                        if value != "Application" {
                            is_application = false;
                        }
                    }
                    "NoDisplay" => {
                        if value == "true" {
                            no_display = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if is_application && !no_display {
        if let (Some(n), Some(e)) = (name, exec) {
            return Some(AppInfo { name: n, exec: e });
        }
    }
    None
}

fn scan_apps() -> Vec<AppInfo> {
    let mut apps = Vec::new();
    let mut dirs = vec![
        std::path::PathBuf::from("/usr/share/applications"),
        std::path::PathBuf::from("/usr/local/share/applications"),
    ];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(std::path::PathBuf::from(home).join(".local/share/applications"));
    }

    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() && path.extension().map_or(false, |ext| ext == "desktop") {
                        if let Some(app) = parse_desktop_file(&path) {
                            apps.push(app);
                        }
                    }
                }
            }
        }
    }
    
    apps.sort_by(|a, b| a.name.cmp(&b.name));
    apps.dedup_by(|a, b| a.name == b.name);
    apps
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
struct AppLaunchHistory {
    count: u32,
    last_launch: u64,
}

fn get_cache_path() -> Option<std::path::PathBuf> {
    let cache_dir = if let Ok(cache_home) = std::env::var("XDG_CACHE_HOME") {
        std::path::PathBuf::from(cache_home)
    } else if let Ok(home) = std::env::var("HOME") {
        std::path::PathBuf::from(home).join(".cache")
    } else {
        return None;
    };
    Some(cache_dir.join("cce-cloud-apps.json"))
}

fn load_history() -> std::collections::HashMap<String, AppLaunchHistory> {
    if let Some(path) = get_cache_path() {
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(history) = serde_json::from_str(&content) {
                return history;
            }
        }
    }
    std::collections::HashMap::new()
}

fn save_history(history: &std::collections::HashMap<String, AppLaunchHistory>) {
    if let Some(path) = get_cache_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(serialized) = serde_json::to_string_pretty(history) {
            let _ = std::fs::write(path, serialized);
        }
    }
}

fn record_app_launch(app_name: &str) {
    let mut history = load_history();
    let entry = history.entry(app_name.to_string()).or_default();
    entry.count += 1;
    entry.last_launch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    save_history(&history);
}

fn sort_apps_by_history(apps: &mut Vec<AppInfo>) {
    let history = load_history();
    apps.sort_by(|a, b| {
        let hist_a = history.get(&a.name);
        let hist_b = history.get(&b.name);
        match (hist_a, hist_b) {
            (Some(a_val), Some(b_val)) => {
                let count_cmp = b_val.count.cmp(&a_val.count);
                if count_cmp != std::cmp::Ordering::Equal {
                    count_cmp
                } else {
                    let time_cmp = b_val.last_launch.cmp(&a_val.last_launch);
                    if time_cmp != std::cmp::Ordering::Equal {
                        time_cmp
                    } else {
                        a.name.cmp(&b.name)
                    }
                }
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.name.cmp(&b.name),
        }
    });
}

fn spawn_command(cmd: &str) {
    if let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/cce-spawn.log")
    {
        let mut f = file;
        use std::io::Write;
        let _ = writeln!(f, "[spawn] executing: {}", cmd);
        std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .stdout(f.try_clone().unwrap())
            .stderr(f)
            .spawn()
            .ok();
    } else {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .spawn()
            .ok();
    }
}




pub struct FuzzelWidget {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    prompt: String,
    query: String,
    all_items: Vec<String>,
    filtered_items: Vec<String>,
    selected: usize,
    pub scroll_box: ScrollRegion,
}

impl FuzzelWidget {
    pub fn new(prompt: String) -> cce_ui::widget::Adapted<FuzzelWidget> {
        cce_ui::widget::Adapted::new(Self {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
            prompt,
            query: String::new(),
            all_items: Vec::new(),
            filtered_items: Vec::new(),
            selected: 0,
            scroll_box: ScrollRegion::new(22.0, 0.0),
        })
    }

    pub fn set_items(&mut self, items: Vec<String>) {
        self.all_items = items;
        self.filter();
    }

    pub fn filter(&mut self) {
        self.filtered_items = filter_and_sort_items(&self.all_items, &self.query);
        if self.selected >= self.filtered_items.len() {
            self.selected = self.filtered_items.len().saturating_sub(1);
        }
        self.update_scroll();
        self.snap_to_selected();
    }

    pub fn update_scroll(&mut self) {
        let item_h = 25.0;
        let pad = 15.0;
        let search_h = 35.0;
        let viewport_y = self.y + pad + search_h + 10.0;
        let viewport_h = self.h - (pad + search_h + 10.0) - pad;
        let content_h = self.filtered_items.len() as f32 * item_h;
        self.scroll_box.update_bounds_raw(content_h, viewport_y, viewport_h);
    }

    pub fn snap_to_selected(&mut self) {
        let item_h = 25.0;
        let pad = 15.0;
        let search_h = 35.0;
        let viewport_h = self.h - (pad + search_h + 10.0) - pad;
        let content_h = self.filtered_items.len() as f32 * item_h;

        if self.filtered_items.is_empty() {
            return;
        }

        let virtual_selected_y = self.selected as f32 * item_h;
        if virtual_selected_y + item_h > self.scroll_box.scroll_y + viewport_h {
            self.scroll_box.scroll_y = virtual_selected_y + item_h - viewport_h;
        } else if virtual_selected_y < self.scroll_box.scroll_y {
            self.scroll_box.scroll_y = virtual_selected_y;
        }

        let max_scroll = (content_h - viewport_h).max(0.0);
        self.scroll_box.scroll_y = self.scroll_box.scroll_y.clamp(0.0, max_scroll);
    }
}

impl cce_ui::widget::Layout for FuzzelWidget {
    // The legacy `set_rect` override's body: mirror the landed rect into the model (the
    // scroll/label math reads it between events) and place the scroll region.
    fn rect_assigned(&mut self, rect: cce_ui::scene::layout::Rect) {
        self.x = rect.x;
        self.y = rect.y;
        self.w = rect.width;
        self.h = rect.height;

        let pad = 15.0;
        let search_h = 35.0;
        let viewport_y = rect.y + pad + search_h + 10.0;
        let viewport_h = rect.height - (pad + search_h + 10.0) - pad;
        self.scroll_box.set_rect(rect.x + pad, viewport_y, rect.width - pad * 2.0, viewport_h);
        self.update_scroll();
    }
}

impl cce_ui::widget::Paint for FuzzelWidget {
    fn color(&self) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn paint(&self, _rect: cce_ui::scene::layout::Rect, ctx: &mut cce_ui::scene::paint::PaintCtx) {
        use cce_ui::scene::layout::Rect;
        let pad = 15.0;
        let search_h = 35.0;

        // Search Bar Background
        ctx.quad(
            Rect { x: self.x + pad, y: self.y + pad, width: self.w - pad * 2.0, height: search_h },
            [0.10, 0.10, 0.14, 1.0],
        );

        // Search Bar Border
        let border_color = [0.25, 0.45, 0.85, 1.0];
        let bx = self.x + pad;
        let by = self.y + pad;
        let bw = self.w - pad * 2.0;
        let bh = search_h;
        ctx.quad(Rect { x: bx, y: by, width: bw, height: 1.0 }, border_color);
        ctx.quad(Rect { x: bx, y: by + bh - 1.0, width: bw, height: 1.0 }, border_color);
        ctx.quad(Rect { x: bx, y: by, width: 1.0, height: bh }, border_color);
        ctx.quad(Rect { x: bx + bw - 1.0, y: by, width: 1.0, height: bh }, border_color);

        // ScrollBox quads
        let mut quads = Vec::new();
        self.scroll_box.push_quads(&mut quads);
        for (qx, qy, qw, qh, qc) in quads {
            ctx.quad(Rect { x: qx, y: qy, width: qw, height: qh }, qc);
        }

        // Selected Item Highlight
        let item_h = 25.0;
        if !self.filtered_items.is_empty() {
            let virtual_selected_y = self.selected as f32 * item_h;
            if let Some(draw_y) = self.scroll_box.get_draw_y(virtual_selected_y, item_h) {
                let scrollbar_w = if self.scroll_box.content_h > self.scroll_box.viewport_h { 10.0 } else { 0.0 };
                ctx.quad(
                    Rect {
                        x: self.x + pad + 2.0,
                        y: draw_y,
                        width: self.w - pad * 2.0 - 4.0 - scrollbar_w,
                        height: item_h - 2.0,
                    },
                    [0.20, 0.35, 0.65, 0.9],
                );
            }
        }

        // Own labels: prompt/query line, visible items, empty-state notice.
        for l in self.own_labels() {
            ctx.text(l.text, l.x, l.y, l.font_size, l.color);
        }
    }
}

impl cce_ui::widget::Input for FuzzelWidget {
    fn on_event(&mut self, event: &cce_ui::widget::Event, _ectx: &mut cce_ui::widget::EventCtx) -> bool {
        if let cce_ui::widget::Event::MouseButton {
            button: cce_ui::widget::MouseButton::Left,
            state: cce_ui::widget::ElementState::Pressed,
            x: px,
            y: py,
            ..
        } = event
        {
            let item_h = 25.0;
            if self.scroll_box.hit(*px, *py) {
                let click_virtual_y = *py - self.scroll_box.viewport_y + self.scroll_box.scroll_y;
                let clicked_idx = (click_virtual_y / item_h).floor() as usize;
                if clicked_idx < self.filtered_items.len() {
                    self.selected = clicked_idx;
                    return true;
                }
            }
        }
        false
    }
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LauncherMode {
    Dmenu,
    Path,
    Apps,
    Json,
}

#[derive(Debug, Clone)]
struct AppInfo {
    name: String,
    exec: String,
}

struct StdinState {
    items: Vec<String>,
    new_data: bool,
    cycle_next: usize,
    select_and_close: bool,
}



#[derive(Clone)]
#[allow(dead_code)]
enum AppWindow {
    Layer(LayerSurface),
    Xdg(XdgWindow),
}

struct State {
    window: Option<AppWindow>,
    wl_surface: wl_surface::WlSurface,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,

    fuzzel: cce_ui::widget::Adapted<FuzzelWidget>,
    json_layout: Option<JsonLayoutWidget>,
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    text_viewport: Viewport,

    cursor_x: f32,
    cursor_y: f32,
    width: f32,
    height: f32,
    physical_width: u32,
    physical_height: u32,
    scale: f64,

    stdin_state: Arc<Mutex<StdinState>>,
    mode: LauncherMode,
    apps: Vec<AppInfo>,

    fade_factor: f32,
    max_width: u32,
    max_height: u32,
    select_item: Option<String>,
    switcher_mode: bool,
    last_tick: std::time::Instant,
    ui_context: cce_ui::context::UiContext,
    /// Dissolved root Backplate (Phase 6as): the plate was a pure value-holder for the
    /// window background — color (at backplate opacity), radius, rect. No border, no
    /// children, no events.
    window_rect: (f32, f32, f32, f32),
    window_bg: [f32; 4],
    window_radius: f32,
    select_and_close_requested: bool,
}

impl State {
    async fn new(
        conn: &Connection,
        qh: &QueueHandle<AppState>,
        compositor_state: &CompositorState,
        layer_shell_state: &LayerShell,
        xdg_shell_state: Option<&XdgShell>,
        cce_wm: Option<&cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1>,
        use_xdg: bool,
        prompt: String,
        stdin_sender: calloop::channel::Sender<()>,
        mode: LauncherMode,
        x_pos: Option<i32>,
        y_pos: Option<i32>,
        align_right: bool,
        scale: f64,
        select_item: Option<String>,
        switcher_mode: bool,
        json_layout_config: Option<JsonLayoutConfig>,
        parent_app_id: Option<String>,
    ) -> (Self, Option<cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1>) {
        cce_ui::scale::set_scale_factor(scale as f32);
        let (width, height) = if mode == LauncherMode::Json {
            if let Some(ref config) = json_layout_config {
                let w = config.width.unwrap_or_else(|| {
                    let mut max_widget_w = 120.0f32; // fallback minimum
                    if let Some(ref widgets) = config.widgets {
                        for w_conf in widgets {
                            let w_w = match w_conf.widget_type.as_str() {
                                "button" | "label" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 32.0,
                                "checkbox" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 44.0,
                                "spinbox" | "color" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 120.0,
                                "slider" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 160.0,
                                _ => 150.0,
                            };
                            if w_w > max_widget_w {
                                max_widget_w = w_w;
                            }
                        }
                    } else if let Some(ref pages) = config.pages {
                        for page in pages {
                            for w_conf in &page.widgets {
                                let w_w = match w_conf.widget_type.as_str() {
                                    "button" | "label" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 32.0,
                                    "checkbox" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 44.0,
                                    "spinbox" | "color" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 120.0,
                                    "slider" => cce_ui::widget::display::measure_text(&w_conf.text, 13.0) + 160.0,
                                    _ => 150.0,
                                };
                                if w_w > max_widget_w {
                                    max_widget_w = w_w;
                                }
                            }
                        }
                    }
                    max_widget_w.round() as u32
                });
                let h = config.height.unwrap_or_else(|| {
                    let mut current_y = 16.0f32;
                    if let Some(ref widgets) = config.widgets {
                        for w_conf in widgets {
                            let h = match w_conf.widget_type.as_str() {
                                "label" => 18.0,
                                "checkbox" => 22.0,
                                "button" => 24.0,
                                "spinbox" => 22.0,
                                "color" => 24.0,
                                _ => 20.0,
                            };
                            current_y += h + 12.0;
                        }
                    } else if let Some(ref pages) = config.pages {
                        let mut max_page_y = 16.0f32;
                        for page in pages {
                            let mut page_y = 16.0f32;
                            for w_conf in &page.widgets {
                                let h = match w_conf.widget_type.as_str() {
                                    "label" => 18.0,
                                    "checkbox" => 22.0,
                                    "button" => 24.0,
                                    "spinbox" => 22.0,
                                    "color" => 24.0,
                                    _ => 20.0,
                                };
                                page_y += h + 12.0;
                            }
                            if page_y > max_page_y {
                                max_page_y = page_y;
                            }
                        }
                        current_y = max_page_y;
                    }
                    current_y += 4.0;
                    std::cmp::min(current_y.round() as u32, 600)
                });
                (w, h)
            } else {
                (300, 400)
            }
        } else {
            (600, 800)
        };
        let pw = (width as f64 * scale) as u32;
        let ph = (height as f64 * scale) as u32;
        let lw = width as f32;
        let lh = height as f32;

        let wl_surface = compositor_state.create_surface(qh);
        wl_surface.set_buffer_scale(scale as i32);
        let app_id = if let Some(ref parent) = parent_app_id {
            format!("cce-cloud:{}", parent)
        } else {
            "cce-cloud".to_string()
        };

        let mut cce_toplevel = None;
        let window = if use_xdg {
            let xdg_shell = xdg_shell_state.expect("XdgShell state is required for XDG mode");
            let xdg_window = xdg_shell.create_window(wl_surface.clone(), WindowDecorations::None, qh);
            xdg_window.set_title("cce-cloud");
            xdg_window.set_app_id(app_id);
            xdg_window.set_min_size(Some((width, height)));
            if let Some(wm) = cce_wm {
                let toplevel = wm.get_cce_toplevel(&wl_surface, qh, ());
                toplevel.set_popup();
                cce_toplevel = Some(toplevel);
            }
            xdg_window.commit();
            AppWindow::Xdg(xdg_window)
        } else {
            let layer_window = layer_shell_state.create_layer_surface(
                qh,
                wl_surface.clone(),
                Layer::Overlay,
                Some(app_id),
                None,
            );
            layer_window.set_size(width, height);
            layer_window.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
            if x_pos.is_some() || y_pos.is_some() {
                let x = x_pos.unwrap_or(0);
                let y = y_pos.unwrap_or(0);
                if align_right {
                    layer_window.set_anchor(Anchor::TOP | Anchor::RIGHT);
                    layer_window.set_margin(y, x, 0, 0);
                } else {
                    layer_window.set_anchor(Anchor::TOP | Anchor::LEFT);
                    layer_window.set_margin(y, 0, 0, x);
                }
            } else {
                layer_window.set_anchor(Anchor::empty());
            }
            wl_surface.commit();
            AppWindow::Layer(layer_window)
        };

        let wayland_handle = Box::leak(Box::new(cce_ui::wayland::WaylandSurfaceHandle {
            display_ptr: conn.backend().display_id().as_ptr() as *mut std::ffi::c_void,
            surface_ptr: wl_surface.id().as_ptr() as *mut std::ffi::c_void,
        }));

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wayland_handle)
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let mut config = surface
            .get_default_config(&adapter, pw, ph)
            .expect("Failed to get surface config");

        let capabilities = surface.get_capabilities(&adapter);
        let alpha_mode = if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::PostMultiplied) {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else {
            capabilities.alpha_modes[0]
        };
        config.alpha_mode = alpha_mode;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(cce_ui::SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let font_system = cce_ui::create_font_system();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, config.format);
        let text_renderer = TextRenderer::new(&mut text_atlas, &device, wgpu::MultisampleState::default(), None);

        let mut text_viewport = Viewport::new(&device, &cache);
        text_viewport.update(&queue, Resolution { width: pw, height: ph });

        let mut fuzzel = FuzzelWidget::new(prompt);
        fuzzel.set_rect(0.0, 0.0, lw, lh);

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: 1,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stdin_state = Arc::new(Mutex::new(StdinState {
            items: Vec::new(),
            new_data: false,
            cycle_next: 0,
            select_and_close: false,
        }));

        let mut apps = Vec::new();
        if mode == LauncherMode::Dmenu {
            let stdin_state_clone = stdin_state.clone();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    if let Ok(line) = line {
                        if let Ok(mut lock_state) = stdin_state_clone.lock() {
                            if line == "__cce_switcher_next__" {
                                lock_state.cycle_next += 1;
                                lock_state.new_data = true;
                            } else if line == "__cce_switcher_select_and_close__" {
                                lock_state.select_and_close = true;
                                lock_state.new_data = true;
                            } else {
                                lock_state.items.push(line);
                                lock_state.new_data = true;
                            }
                        }
                        let _ = stdin_sender.send(());
                    }
                }
            });
        } else if mode == LauncherMode::Apps {
            apps = scan_apps();
            sort_apps_by_history(&mut apps);
            let app_names: Vec<String> = apps.iter().map(|app| app.name.clone()).collect();
            if let Ok(mut lock_state) = stdin_state.lock() {
                lock_state.items = app_names;
                lock_state.new_data = true;
            }
        } else if mode == LauncherMode::Path {
            let path_items = scan_path();
            if let Ok(mut lock_state) = stdin_state.lock() {
                lock_state.items = path_items;
                lock_state.new_data = true;
            }
        }

        let json_layout = if mode == LauncherMode::Json {
            if let Some(ref config) = json_layout_config {
                let mut jl = JsonLayoutWidget::new(config);
                jl.set_rect(0.0, 0.0, lw, lh);
                Some(jl)
            } else {
                None
            }
        } else {
            None
        };

        cce_ui::scale::set_app_id("cce-cloud".to_string());
        let bg_color = cce_ui::color::page_low_color();
        let window_rect = (0.0, 0.0, lw, lh);
        let window_radius = cce_ui::color::backplate_corner_radius();

        let mut state = Self {
            window: Some(window),
            wl_surface,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            vertex_count: 0,
            fuzzel,
            json_layout,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            text_viewport,
            cursor_x: 0.0,
            cursor_y: 0.0,
            width: lw,
            height: lh,
            physical_width: pw,
            physical_height: ph,
            scale,
            stdin_state,
            mode,
            apps,

            fade_factor: 1.0,
            max_width: width,
            max_height: height,
            select_item,
            switcher_mode,
            last_tick: std::time::Instant::now(),
            ui_context: cce_ui::context::UiContext::new(),
            window_rect,
            window_bg: bg_color,
            window_radius,
            select_and_close_requested: false,
        };

        state.check_stdin_updates();
        state.update_desired_size();
        state.apply_layout();
        state.upload_vertices();
        (state, cce_toplevel)
    }

    fn check_stdin_updates(&mut self) -> bool {
        if let Ok(mut lock) = self.stdin_state.lock() {
            if lock.new_data {
                lock.new_data = false;

                if lock.select_and_close {
                    lock.select_and_close = false;
                    self.select_and_close_requested = true;
                }

                let cycles = lock.cycle_next;
                lock.cycle_next = 0;

                let mut changed = false;
                let items = lock.items.clone();
                if self.fuzzel.all_items != items {
                    self.fuzzel.set_items(items);
                    changed = true;
                    if let Some(ref select_name) = self.select_item {
                        let select_lower = select_name.to_lowercase();
                        if let Some(idx) = self.fuzzel.filtered_items.iter().position(|item| item.to_lowercase() == select_lower) {
                            self.fuzzel.selected = idx;
                            self.fuzzel.update_scroll();
                            self.fuzzel.snap_to_selected();
                            self.select_item = None;
                        }
                    } else if self.switcher_mode && self.fuzzel.filtered_items.len() > 1 {
                        self.fuzzel.selected = 1;
                        self.fuzzel.update_scroll();
                        self.fuzzel.snap_to_selected();
                    }
                }

                if cycles > 0 && !self.fuzzel.filtered_items.is_empty() {
                    self.fuzzel.selected = (self.fuzzel.selected + cycles) % self.fuzzel.filtered_items.len();
                    self.fuzzel.update_scroll();
                    self.fuzzel.snap_to_selected();
                    changed = true;
                }

                return changed;
            }
        }
        false
    }

    fn update_desired_size(&mut self) {
        if self.mode == LauncherMode::Json {
            if let Some(ref jl) = self.json_layout {
                let mut max_widget_w = 120.0f32; // fallback minimum
                let active_page = jl.active_page;
                
                for w in &jl.widgets {
                    if w.page_idx != active_page {
                        continue;
                    }
                    let w_w = match w.widget_type.as_str() {
                        "button" | "label" => cce_ui::widget::display::measure_text(&w.text, 13.0) + 32.0,
                        "checkbox" => cce_ui::widget::display::measure_text(&w.text, 13.0) + 44.0,
                        "spinbox" | "color" => cce_ui::widget::display::measure_text(&w.text, 13.0) + 120.0,
                        "slider" => cce_ui::widget::display::measure_text(&w.text, 13.0) + 160.0,
                        _ => 150.0,
                    };
                    if w_w > max_widget_w {
                        max_widget_w = w_w;
                    }
                }
                
                let target_height = jl.page_total_heights[active_page].min(self.max_height as f32);
                let target_width = max_widget_w.clamp(120.0, self.max_width as f32);
                
                let target_height_u32 = target_height.round() as u32;
                let target_width_u32 = target_width.round() as u32;
                
                if self.width as u32 != target_width_u32 || self.height as u32 != target_height_u32 {
                    if let Some(ref window) = self.window {
                        match window {
                            AppWindow::Layer(layer) => layer.set_size(target_width_u32, target_height_u32),
                            AppWindow::Xdg(_) => {}
                        }
                    }
                    self.wl_surface.commit();
                    
                    let pw = (target_width_u32 as f64 * self.scale) as u32;
                    let ph = (target_height_u32 as f64 * self.scale) as u32;
                    self.resize(pw, ph);
                }
            }
            return;
        }
        let num_items = self.fuzzel.filtered_items.len();
        let item_count = if num_items == 0 { 1 } else { num_items };
        let needed_height = 75.0 + (item_count as f32) * 25.0;
        let target_height = needed_height.min(self.max_height as f32);

        // Calculate max text width
        let mut max_text_w: f32 = 0.0;
        
        let query_text = if self.fuzzel.query.is_empty() {
            format!("{}{}", self.fuzzel.prompt, "Type to search...")
        } else {
            format!("{}{}", self.fuzzel.prompt, self.fuzzel.query)
        };
        let buf = make_text_buffer(&mut self.font_system, &query_text, 14.0);
        let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
        if tw > max_text_w {
            max_text_w = tw;
        }

        if self.fuzzel.filtered_items.is_empty() {
            let buf = make_text_buffer(&mut self.font_system, "No matches found", 13.0);
            let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
            if tw > max_text_w {
                max_text_w = tw;
            }
        } else {
            for item in &self.fuzzel.filtered_items {
                let buf = make_text_buffer(&mut self.font_system, item, 13.0);
                let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
                if tw > max_text_w {
                    max_text_w = tw;
                }
            }
        }

        let scrollbar_w = if needed_height > self.max_height as f32 { 10.0 } else { 0.0 };
        let needed_width = max_text_w + 50.0 + scrollbar_w;
        let target_width = needed_width.clamp(300.0, self.max_width as f32);

        let target_height_u32 = target_height.round() as u32;
        let target_width_u32 = target_width.round() as u32;
        
        if self.width as u32 != target_width_u32 || self.height as u32 != target_height_u32 {
            if let Some(ref window) = self.window {
                match window {
                    AppWindow::Layer(layer) => layer.set_size(target_width_u32, target_height_u32),
                    AppWindow::Xdg(_) => {}
                }
            }
            self.wl_surface.commit();
            
            let pw = (target_width_u32 as f64 * self.scale) as u32;
            let ph = (target_height_u32 as f64 * self.scale) as u32;
            self.resize(pw, ph);
        }
    }

    fn apply_layout(&mut self) {
        let (w, h) = (self.width, self.height);
        self.window_rect = (0.0, 0.0, w, h);
        if self.mode == LauncherMode::Json {
            if let Some(jl) = &mut self.json_layout {
                cce_ui::scale::set_scale_factor(self.scale as f32);
                jl.set_rect(0.0, 0.0, w, h);
            }
        } else {
            self.fuzzel.set_rect(0.0, 0.0, w, h);
        }
    }

    fn collect_vertices(&self) -> Vec<Vertex> {
        let sw = self.width;
        let sh = self.height;
        let mut verts = Vec::new();

        // 1. Window background — the dissolved Backplate's exact emission: base color at
        // the configured backplate opacity, all corners rounded when the radius is set.
        let mut bg_color = self.window_bg;
        if bg_color[3] > 0.001 {
            bg_color[3] = cce_ui::color::active_backplate_opacity();
        }
        if bg_color[3] > 0.0 {
            let r = self.window_radius;
            let corners = if r > 0.1 { (true, true, true, true) } else { (false, false, false, false) };
            let (wx, wy, ww, wh) = self.window_rect;
            verts.extend(rounded_rect_vertices_corners(wx, wy, ww, wh, r, sw, sh, bg_color, corners));
        }

        // 3. Draw child widgets
        if self.mode == LauncherMode::Json {
            if let Some(jl) = &self.json_layout {
                for (qx, qy, qw, qh, qc) in jl.all_quads(&self.ui_context) {
                    verts.extend(quad_vertices(qx, qy, qw, qh, sw, sh, qc));
                }
                for (qx, qy, qw, qh, qr, qc, qcorners) in jl.all_rounded_quads(&self.ui_context) {
                    verts.extend(rounded_rect_vertices_corners(qx, qy, qw, qh, qr, sw, sh, qc, qcorners));
                }
            }
        } else {
            verts.extend(widget_vertices(&self.fuzzel, sw, sh));
        }

        verts
    }

    fn upload_vertices(&mut self) {
        let mut verts = self.collect_vertices();
        if self.fade_factor < 1.0 {
            for v in &mut verts {
                v.color[3] *= self.fade_factor;
            }
        }
        self.vertex_count = verts.len() as u32;
        if self.vertex_count == 0 {
            return;
        }
        let data = bytemuck::cast_slice(&verts);
        let needed = data.len() as wgpu::BufferAddress;
        if needed > self.vertex_buffer.size() {
            self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Vertex Buffer"),
                size: needed,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.queue.write_buffer(&self.vertex_buffer, 0, data);
    }

    fn prepare_text(&mut self) {
        let Self {
            ref mut text_renderer,
            ref device,
            ref queue,
            ref mut font_system,
            ref mut text_atlas,
            ref mut text_viewport,
            ref mut swash_cache,
            physical_width,
            physical_height,
            scale,
            ref fuzzel,
            ref json_layout,
            ref mode,
            ref ui_context,
            ..
        } = self;

        let viewport = Resolution { width: *physical_width, height: *physical_height };
        text_viewport.update(queue, viewport);

        let scale_f32 = *scale as f32;

        let mut areas: Vec<TextArea> = Vec::new();
        let mut widget_buffers: Vec<Buffer> = Vec::new();
        let mut widget_labels: Vec<TextLabel> = Vec::new();

        let is_json_mode = *mode == LauncherMode::Json;
        if is_json_mode {
            if let Some(jl) = json_layout {
                widget_labels.extend(walk_text_labels(ui_context, jl));
            }
        } else {
            widget_labels.extend(walk_text_labels(ui_context, fuzzel));
        }

        for label in &widget_labels {
            widget_buffers.push(make_text_buffer(font_system, &label.text, label.font_size));
        }

        for (buf, label) in widget_buffers.iter().zip(widget_labels.iter()) {
            let left = (label.x * scale_f32).round();
            let top = (label.y * scale_f32).round();
            let scale = scale_f32;
            let default_color = if self.fade_factor < 1.0 {
                let alpha = (self.fade_factor * 255.0) as u8;
                glyphon::Color::rgba(label.color[0], label.color[1], label.color[2], alpha)
            } else {
                glyphon::Color::rgb(label.color[0], label.color[1], label.color[2])
            };
            areas.push(TextArea {
                buffer: buf,
                left,
                top,
                scale,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: *physical_width as i32,
                    bottom: *physical_height as i32,
                },
                default_color,
                custom_glyphs: &[],
            });
        }

        text_renderer
            .prepare(device, queue, font_system, text_atlas, text_viewport, areas, swash_cache)
            .unwrap();
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.physical_width = width;
            self.physical_height = height;
            self.width = width as f32 / self.scale as f32;
            self.height = height as f32 / self.scale as f32;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.apply_layout();
            self.upload_vertices();
        }
    }

    fn render(&mut self, fade_factor: f32) -> bool {
        let now = std::time::Instant::now();
        self.last_tick = now;

        self.fade_factor = fade_factor;
        self.upload_vertices();
        self.prepare_text();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return false;
            }
            Err(wgpu::SurfaceError::Timeout) => return false,
            Err(e) => {
                log::error!("Surface error: {e:?}");
                return false;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if self.vertex_count > 0 {
                pass.set_pipeline(&self.render_pipeline);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..self.vertex_count, 0..1);
            }

            self.text_renderer.render(&self.text_atlas, &self.text_viewport, &mut pass).unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        false
    }
}

impl Drop for State {
    fn drop(&mut self) {
        self.window.take();
        self.wl_surface.destroy();
    }
}

#[allow(dead_code)]
struct AppState {
    registry_state: RegistryState,
    compositor_state: CompositorState,
    layer_shell_state: LayerShell,
    shm_state: Shm,
    seat_state: SeatState,
    output_state: OutputState,

    seats: Vec<wl_seat::WlSeat>,
    pointer: Option<wl_pointer::WlPointer>,
    keyboard: Option<wl_keyboard::WlKeyboard>,

    window: Option<AppWindow>,
    surface: Option<wl_surface::WlSurface>,

    state: Option<State>,
    exit: bool,
    redraw: bool,
    ctrl_pressed: bool,
    super_pressed: bool,
    switcher_mode: bool,
    fade_out: bool,
    fade_start: Option<std::time::Instant>,
    fade_factor: f32,
    cce_toplevel: Option<cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1>,
    selected_item: Option<String>,
}

impl AppState {
    fn trigger_close(&mut self) {
        if !self.fade_out {
            self.fade_out = true;
            self.fade_start = Some(std::time::Instant::now());
            self.redraw = true;
        }
    }

    fn trigger_select_and_close(&mut self) {
        let mut should_close = false;
        if let Some(st) = &mut self.state {
            if !st.fuzzel.filtered_items.is_empty() {
                if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
                    self.selected_item = Some(item.clone());
                    println!("{}", item);
                    match st.mode {
                        LauncherMode::Apps => {
                            if let Some(app) = st.apps.iter().find(|app| &app.name == item) {
                                record_app_launch(&app.name);
                                spawn_command(&app.exec);
                            }
                        }
                        LauncherMode::Path => {
                            spawn_command(item);
                        }
                        LauncherMode::Dmenu => {}
                        LauncherMode::Json => {}
                    }
                    should_close = true;
                }
            }
        }
        if should_close {
            self.trigger_close();
        }
    }
}

impl CompositorHandler for AppState {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        scale_factor: i32,
    ) {
        if let Some(state) = &mut self.state {
            state.scale = scale_factor as f64;
            state.wl_surface.set_buffer_scale(scale_factor);
            let pw = (state.width as f64 * state.scale) as u32;
            let ph = (state.height as f64 * state.scale) as u32;
            state.resize(pw, ph);
            self.redraw = true;
        }
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {}

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {}

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {}

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {}
}

impl OutputHandler for AppState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {}

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {}

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {}
}

impl SeatHandler for AppState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, seat: wl_seat::WlSeat) {
        self.seats.push(seat);
    }

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer && self.pointer.is_none() {
            let pointer = self.seat_state.get_pointer(qh, &seat).unwrap();
            self.pointer = Some(pointer);
        }
        if capability == Capability::Keyboard && self.keyboard.is_none() {
            let keyboard = self
                .seat_state
                .get_keyboard(qh, &seat, None)
                .unwrap();
            self.keyboard = Some(keyboard);
        }
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer {
            self.pointer = None;
        }
        if capability == Capability::Keyboard {
            self.keyboard = None;
        }
    }

    fn remove_seat(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, seat: wl_seat::WlSeat) {
        self.seats.retain(|s| s != &seat);
    }
}

impl ShmHandler for AppState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm_state
    }
}

impl PointerHandler for AppState {
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        events: &[smithay_client_toolkit::seat::pointer::PointerEvent],
    ) {
        use smithay_client_toolkit::seat::pointer::PointerEventKind;
        let mut should_close = false;
        for event in events {
            if let Some(ref active_surface) = self.surface {
                if active_surface != &event.surface {
                    continue;
                }
            }
            if let Some(st) = &mut self.state {
                log::debug!("Event: position={:?}, scale={}, kind={:?}", event.position, st.scale, event.kind);
                let (cx, cy) = cce_ui::wayland::scale_pointer_pos(event.position, st.scale);
                match &event.kind {
                    PointerEventKind::Motion { .. } => {
                        st.cursor_x = cx;
                        st.cursor_y = cy;
                        if st.mode == LauncherMode::Json {
                            let mut changed = false;
                            if let Some(jl) = &mut st.json_layout {
                                if jl.on_cursor_moved(event.position.0 as f32, event.position.1 as f32, &mut st.ui_context) {
                                    changed = true;
                                }
                            }
                            if changed {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        }
                    }
                    PointerEventKind::Press { button, .. } => {
                        if *button == 272 {
                            st.cursor_x = cx;
                            st.cursor_y = cy;
                            if st.mode == LauncherMode::Json {
                                let mut changed = false;
                                if let Some(jl) = &mut st.json_layout {
                                    if jl.mouse_input(
                                        cce_ui::widget::MouseButton::Left,
                                        cce_ui::widget::ElementState::Pressed,
                                        event.position.0 as f32,
                                        event.position.1 as f32,
                                        &mut st.ui_context,
                                    ) {
                                        changed = true;
                                    }
                                }
                                if changed {
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                            } else {
                                let prev_selected = st.fuzzel.selected;
                                let changed = st.fuzzel.mouse_input(
                                    cce_ui::widget::MouseButton::Left,
                                    cce_ui::widget::ElementState::Pressed,
                                    event.position.0 as f32,
                                    event.position.1 as f32,
                                    &mut st.ui_context,
                                );
                                if changed {
                                    if st.fuzzel.selected == prev_selected || st.switcher_mode || st.mode == LauncherMode::Dmenu {
                                        if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
                                            println!("{}", item);
                                            self.selected_item = Some(item.clone());
                                            match st.mode {
                                                LauncherMode::Apps => {
                                                    if let Some(app) = st.apps.iter().find(|app| &app.name == item) {
                                                        record_app_launch(&app.name);
                                                        spawn_command(&app.exec);
                                                    }
                                                }
                                                LauncherMode::Path => {
                                                    spawn_command(item);
                                                }
                                                LauncherMode::Dmenu => {}
                                                LauncherMode::Json => {}
                                            }
                                            should_close = true;
                                        }
                                    }
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                            }
                        }
                    }
                    PointerEventKind::Release { button, .. } => {
                        if *button == 272 {
                            if st.mode == LauncherMode::Json {
                                let mut changed = false;
                                let mut clicked_btn_id = None;
                                if let Some(jl) = &mut st.json_layout {
                                    if jl.mouse_input(
                                        cce_ui::widget::MouseButton::Left,
                                        cce_ui::widget::ElementState::Released,
                                        event.position.0 as f32,
                                        event.position.1 as f32,
                                        &mut st.ui_context,
                                    ) {
                                        changed = true;
                                    }
                                    for w in &mut jl.widgets {
                                        // take_click is an Element method; Phase 5 Buttons are
                                        // Adapted, so ask the box directly.
                                        if w.widget_type == "button" && w.widget.take_click() {
                                            clicked_btn_id = Some(w.id.clone());
                                            break;
                                        }
                                    }
                                }
                                if changed {
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                                if let Some(btn_id) = clicked_btn_id {
                                    let mut checkboxes = std::collections::HashMap::new();
                                    let mut spinboxes = std::collections::HashMap::new();
                                    let mut colors = std::collections::HashMap::new();
                                    let mut sliders = std::collections::HashMap::new();
                                    if let Some(jl) = &st.json_layout {
                                        for w in &jl.widgets {
                                            if let Some(cb) = w.widget.as_any().downcast_ref::<cce_ui::widget::Checkbox>() {
                                                checkboxes.insert(w.id.clone(), cb.checked());
                                            } else if let Some(sb) = w.widget.as_any().downcast_ref::<cce_ui::widget::Spinbox>() {
                                                spinboxes.insert(w.id.clone(), sb.value);
                                            } else if let Some(cs) = w.widget.as_any().downcast_ref::<cce_ui::widget::ColorSelector>() {
                                                colors.insert(w.id.clone(), cs.color);
                                            } else if let Some(sl) = w.widget.as_any().downcast_ref::<cce_ui::widget::Slider>() {
                                                sliders.insert(w.id.clone(), sl.get_scaled_value());
                                            }
                                        }
                                    }
                                    let out_val = serde_json::json!({
                                        "button": btn_id,
                                        "checkboxes": checkboxes,
                                        "spinboxes": spinboxes,
                                        "colors": colors,
                                        "sliders": sliders
                                    });
                                    let out_str = out_val.to_string();
                                    println!("{}", out_str);
                                    self.selected_item = Some(out_str);
                                    should_close = true;
                                }
                            }
                        }
                    }
                    PointerEventKind::Axis { horizontal, vertical, .. } => {
                        let h_scroll = horizontal.absolute as f32;
                        let v_scroll = vertical.absolute as f32;
                        let delta = cce_ui::widget::MouseScrollDelta::LineDelta(-h_scroll / 10.0, -v_scroll / 10.0);
                        if st.fuzzel.scroll_box.wheel(&delta, st.cursor_x, st.cursor_y) {
                            st.fuzzel.update_scroll();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        if should_close {
            self.trigger_close();
        }
    }
}

impl KeyboardHandler for AppState {
    fn enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _surface: &wl_surface::WlSurface,
        _serial: u32,
        _raw_modifiers: &[u32],
        _keysyms: &[xkeysym::Keysym],
    ) {
        log::debug!("KeyboardHandler::enter called!");
    }

    fn leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _serial: u32,
    ) {
        log::debug!("KeyboardHandler::leave called!");
        if let Some(ref active_surface) = self.surface {
            if active_surface == surface {
                self.trigger_close();
            }
        }
    }

    fn press_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        event: smithay_client_toolkit::seat::keyboard::KeyEvent,
    ) {
        log::debug!("press_key keysym={:?}, utf8={:?}", event.keysym, event.utf8);
        self.handle_key(event, cce_ui::widget::ElementState::Pressed);
    }

    fn release_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        event: smithay_client_toolkit::seat::keyboard::KeyEvent,
    ) {
        log::debug!("release_key keysym={:?}, utf8={:?}", event.keysym, event.utf8);
        self.handle_key(event, cce_ui::widget::ElementState::Released);
    }

    fn update_modifiers(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        modifiers: smithay_client_toolkit::seat::keyboard::Modifiers,
        _layout: u32,
    ) {
        let prev_super = self.super_pressed;
        self.ctrl_pressed = modifiers.ctrl;
        self.super_pressed = modifiers.logo;
        log::debug!("update_modifiers: logo={}, prev_logo={}", modifiers.logo, prev_super);

        if self.switcher_mode && prev_super && !self.super_pressed {
            log::info!("Super modifier released in switcher mode, selecting currently highlighted item");
            self.trigger_select_and_close();
        }
    }
}

impl LayerShellHandler for AppState {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (width, height) = configure.new_size;
        if let Some(state) = &mut self.state {
            let pw = (width as f64 * state.scale) as u32;
            let ph = (height as f64 * state.scale) as u32;
            state.resize(pw, ph);
        }
        self.redraw = true;
    }
}

impl WindowHandler for AppState {
    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _window: &XdgWindow,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        let (w, h) = configure.new_size;
        if let (Some(w), Some(h)) = (w, h) {
            let width = w.get();
            let height = h.get();
            if let Some(state) = &mut self.state {
                let pw = (width as f64 * state.scale) as u32;
                let ph = (height as f64 * state.scale) as u32;
                state.resize(pw, ph);
            }
        }
        self.redraw = true;
    }

    fn request_close(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _window: &XdgWindow) {
        self.exit = true;
    }
}

impl ProvidesRegistryState for AppState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    
    fn runtime_add_global(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _name: u32,
        _interface: &str,
        _version: u32,
    ) {}
    
    fn runtime_remove_global(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _name: u32,
        _interface: &str,
    ) {}
}

delegate_compositor!(AppState);
delegate_layer!(AppState);
delegate_shm!(AppState);
delegate_seat!(AppState);
delegate_pointer!(AppState);
delegate_keyboard!(AppState);
delegate_registry!(AppState);
delegate_output!(AppState);
delegate_xdg_shell!(AppState);
delegate_xdg_window!(AppState);

impl wayland_client::Dispatch<cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1, ()> for AppState {
    fn event(
        _state: &mut Self,
        _proxy: &cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1,
        _event: cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {}

    wayland_client::event_created_child!(
        AppState,
        cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1,
        [
            6 => (cce_ui::protocol::cce_window_management_v1::zcce_window_v1::ZcceWindowV1, ()),
            7 => (cce_ui::protocol::cce_window_management_v1::zcce_output_v1::ZcceOutputV1, ()),
            8 => (cce_ui::protocol::cce_window_management_v1::zcce_seat_v1::ZcceSeatV1, ()),
        ]
    );
}

impl wayland_client::Dispatch<cce_ui::protocol::cce_window_management_v1::zcce_window_v1::ZcceWindowV1, ()> for AppState {
    fn event(
        _state: &mut Self,
        _proxy: &cce_ui::protocol::cce_window_management_v1::zcce_window_v1::ZcceWindowV1,
        _event: cce_ui::protocol::cce_window_management_v1::zcce_window_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {}
}

impl wayland_client::Dispatch<cce_ui::protocol::cce_window_management_v1::zcce_output_v1::ZcceOutputV1, ()> for AppState {
    fn event(
        _state: &mut Self,
        _proxy: &cce_ui::protocol::cce_window_management_v1::zcce_output_v1::ZcceOutputV1,
        _event: cce_ui::protocol::cce_window_management_v1::zcce_output_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {}
}

impl wayland_client::Dispatch<cce_ui::protocol::cce_window_management_v1::zcce_seat_v1::ZcceSeatV1, ()> for AppState {
    fn event(
        _state: &mut Self,
        _proxy: &cce_ui::protocol::cce_window_management_v1::zcce_seat_v1::ZcceSeatV1,
        _event: cce_ui::protocol::cce_window_management_v1::zcce_seat_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {}
}

impl wayland_client::Dispatch<cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1, ()> for AppState {
    fn event(
        _state: &mut Self,
        _proxy: &cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1,
        _event: cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {}
}

impl AppState {
    fn handle_key(&mut self, event: smithay_client_toolkit::seat::keyboard::KeyEvent, state: cce_ui::widget::ElementState) {
        use cce_ui::widget::{Key, NamedKey};
        if state != cce_ui::widget::ElementState::Pressed {
            return;
        }

        let logical_key = match event.keysym {
            xkeysym::Keysym::Escape => Key::Named(NamedKey::Escape),
            xkeysym::Keysym::Return => Key::Named(NamedKey::Enter),
            xkeysym::Keysym::BackSpace => Key::Named(NamedKey::Backspace),
            xkeysym::Keysym::Down => Key::Named(NamedKey::ArrowDown),
            xkeysym::Keysym::Up => Key::Named(NamedKey::ArrowUp),
            xkeysym::Keysym::Left => Key::Named(NamedKey::ArrowLeft),
            xkeysym::Keysym::Right => Key::Named(NamedKey::ArrowRight),
            xkeysym::Keysym::Tab => Key::Named(NamedKey::Tab),
            xkeysym::Keysym::Delete => Key::Named(NamedKey::Delete),
            xkeysym::Keysym::space => Key::Named(NamedKey::Space),
            xkeysym::Keysym::n if self.ctrl_pressed => Key::Named(NamedKey::ArrowDown),
            xkeysym::Keysym::p if self.ctrl_pressed => Key::Named(NamedKey::ArrowUp),
            _ => {
                if let Some(ref text) = event.utf8 {
                    Key::Character(text.clone())
                } else {
                    return;
                }
            }
        };

        let mut should_close = false;
        if let Some(st) = &mut self.state {
            let mut handled = true;
            if st.mode == LauncherMode::Json {
                let mut widget_handled = false;
                let key_event = cce_ui::widget::KeyEvent {
                    state,
                    logical_key: logical_key.clone(),
                    text: event.utf8.clone(),
                    repeat: false,
                    ctrl: self.ctrl_pressed,
                    shift: false,
                };
                if let Some(jl) = &mut st.json_layout {
                    if jl.keyboard_input(&key_event, &mut st.ui_context) {
                        widget_handled = true;
                        st.upload_vertices();
                        self.redraw = true;
                    }
                }
                if !widget_handled {
                    match &logical_key {
                        Key::Named(NamedKey::Escape) => {
                            should_close = true;
                        }
                        _ => {
                            handled = false;
                        }
                    }
                }
            } else {
                match &logical_key {
                    Key::Named(NamedKey::Escape) => {
                        should_close = true;
                    }
                    Key::Named(NamedKey::Enter) => {
                        self.trigger_select_and_close();
                    }
                    Key::Named(NamedKey::Tab) => {
                        if !st.fuzzel.filtered_items.is_empty() {
                            st.fuzzel.selected = (st.fuzzel.selected + 1) % st.fuzzel.filtered_items.len();
                            st.fuzzel.update_scroll();
                            st.fuzzel.snap_to_selected();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if !st.fuzzel.filtered_items.is_empty() {
                            st.fuzzel.selected = (st.fuzzel.selected + 1).min(st.fuzzel.filtered_items.len() - 1);
                            st.fuzzel.update_scroll();
                            st.fuzzel.snap_to_selected();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        if st.fuzzel.selected > 0 {
                            st.fuzzel.selected -= 1;
                            st.fuzzel.update_scroll();
                            st.fuzzel.snap_to_selected();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    Key::Named(NamedKey::Backspace) => {
                        st.fuzzel.query.pop();
                        st.fuzzel.filter();
                        st.update_desired_size();
                        st.upload_vertices();
                        self.redraw = true;
                    }
                    _ => {
                        if let Some(text) = &event.utf8 {
                            for ch in text.chars().filter(|c| !c.is_control()) {
                                st.fuzzel.query.push(ch);
                            }
                            st.fuzzel.filter();
                            st.update_desired_size();
                            st.upload_vertices();
                            self.redraw = true;
                        } else {
                            handled = false;
                        }
                    }
                }
            }
            if handled {
                self.redraw = true;
            }
        }
        if should_close {
            self.trigger_close();
        }
    }
}

fn run_standalone() {
    let mut prompt = "Search: ".to_string();
    let mut mode = if !io::stdin().is_terminal() {
        LauncherMode::Dmenu
    } else {
        LauncherMode::Path
    };
    let mut x_pos: Option<i32> = None;
    let mut y_pos: Option<i32> = None;
    let mut select_item: Option<String> = None;
    let mut align_right = false;
    let mut switcher_mode = false;
    let mut parent_app_id: Option<String> = None;

    let args = std::env::args().skip(1).collect::<Vec<String>>();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "-p" || arg == "--prompt" {
            if i + 1 < args.len() {
                prompt = args[i + 1].clone();
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "-s" || arg == "--select" {
            if i + 1 < args.len() {
                select_item = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "-x" || arg == "--x-pos" {
            if i + 1 < args.len() {
                if let Ok(val) = args[i + 1].parse::<i32>() {
                    x_pos = Some(val);
                }
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "-y" || arg == "--y-pos" {
            if i + 1 < args.len() {
                if let Ok(val) = args[i + 1].parse::<i32>() {
                    y_pos = Some(val);
                }
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "--parent-app-id" {
            if i + 1 < args.len() {
                parent_app_id = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "--mode" {
            if i + 1 < args.len() {
                let m = &args[i + 1];
                match m.as_str() {
                    "apps" | "app" => mode = LauncherMode::Apps,
                    "path" => mode = LauncherMode::Path,
                    "dmenu" => mode = LauncherMode::Dmenu,
                    _ => eprintln!("Unknown mode: {}", m),
                }
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "--apps" || arg == "--app" {
            mode = LauncherMode::Apps;
            i += 1;
        } else if arg == "--path" {
            mode = LauncherMode::Path;
            i += 1;
        } else if arg == "--dmenu" {
            mode = LauncherMode::Dmenu;
            i += 1;
        } else if arg == "--json" || arg == "--layout" {
            mode = LauncherMode::Json;
            i += 1;
        } else if arg == "--align-right" {
            align_right = true;
            i += 1;
        } else if arg == "--switcher" {
            switcher_mode = true;
            mode = LauncherMode::Dmenu;
            i += 1;
        } else {
            i += 1;
        }
    }

    let mut json_layout_config: Option<JsonLayoutConfig> = None;
    if mode == LauncherMode::Json {
        use std::io::Read;
        let mut json_str = String::new();
        let mut stdin = std::io::stdin();
        match stdin.read_to_string(&mut json_str) {
            Ok(_) => {
                match serde_json::from_str::<JsonLayoutConfig>(&json_str) {
                    Ok(cfg) => {
                        json_layout_config = Some(cfg);
                    }
                    Err(e) => {
                        eprintln!("Failed to parse JSON layout: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read JSON layout from stdin: {}", e);
                std::process::exit(1);
            }
        }
    }

    let conn = Connection::connect_to_env().unwrap();
    let conn_clone = conn.clone();
    let (globals, mut event_queue) = registry_queue_init(&conn).unwrap();
    let qh = event_queue.handle();

    let compositor_state = CompositorState::bind(&globals, &qh).unwrap();
    let layer_shell_state = LayerShell::bind(&globals, &qh).unwrap();
    let shm_state = Shm::bind(&globals, &qh).unwrap();
    let seat_state = SeatState::new(&globals, &qh);
    let output_state = OutputState::new(&globals, &qh);
    let cce_wm = globals.bind::<cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1, _, _>(&qh, 2..=4, ()).ok();

    let (stdin_sender, stdin_channel) = calloop::channel::channel::<()>();

    let mut app = AppState {
        registry_state: RegistryState::new(&globals),
        compositor_state,
        layer_shell_state,
        shm_state,
        seat_state,
        output_state,
        seats: Vec::new(),
        pointer: None,
        keyboard: None,
        window: None,
        surface: None,
        state: None,
        exit: false,
        redraw: false,
        ctrl_pressed: false,
        super_pressed: switcher_mode,
        switcher_mode,
        fade_out: false,
        fade_start: None,
        fade_factor: 1.0,
        cce_toplevel: None,
        selected_item: None,
    };

    // Perform a roundtrip to populate output_state with active output scales
    event_queue.roundtrip(&mut app).unwrap();

    let scale = cce_ui::wayland::detect_scale_factor(&app.output_state);

    let xdg_shell_state = smithay_client_toolkit::shell::xdg::XdgShell::bind(&globals, &qh).ok();
    let use_xdg = cce_wm.is_some() && xdg_shell_state.is_some() && x_pos.is_none() && y_pos.is_none();
    log::info!("Starting launcher window: x_pos={:?}, y_pos={:?}, align_right={}, scale={}, use_xdg={}", x_pos, y_pos, align_right, scale, use_xdg);

    let (state, cce_toplevel) = pollster::block_on(State::new(
        &conn,
        &qh,
        &app.compositor_state,
        &app.layer_shell_state,
        xdg_shell_state.as_ref(),
        cce_wm.as_ref(),
        use_xdg,
        prompt,
        stdin_sender,
        mode,
        x_pos,
        y_pos,
        align_right,
        scale,
        select_item,
        switcher_mode,
        json_layout_config,
        parent_app_id,
    ));

    app.window = state.window.clone();
    app.surface = Some(state.wl_surface.clone());
    app.cce_toplevel = cce_toplevel;
    app.state = Some(state);

    let mut event_loop = calloop::EventLoop::try_new().unwrap();
    let loop_handle = event_loop.handle();

    WaylandSource::new(conn, event_queue).insert(loop_handle.clone()).unwrap();

    loop_handle.insert_source(stdin_channel, |event, _metadata, app_state: &mut AppState| {
        if let calloop::channel::Event::Msg(()) = event {
            let mut select_and_close = false;
            if let Some(st) = &mut app_state.state {
                if st.check_stdin_updates() {
                    st.update_desired_size();
                    st.apply_layout();
                    st.upload_vertices();
                    app_state.redraw = true;
                }
                if st.select_and_close_requested {
                    st.select_and_close_requested = false;
                    select_and_close = true;
                }
            }
            if select_and_close {
                app_state.trigger_select_and_close();
            }
        }
    }).unwrap();

    let mut last_tick = std::time::Instant::now();
    loop {
        if app.fade_out {
            if let Some(start) = app.fade_start {
                let elapsed = start.elapsed().as_secs_f32();
                app.fade_factor = (1.0 - elapsed / 0.15).max(0.0);
                if app.fade_factor <= 0.0 {
                    app.exit = true;
                } else {
                    app.redraw = true;
                }
            }
        }

        let timeout = if app.redraw {
            std::time::Duration::from_millis(0)
        } else {
            std::time::Duration::from_millis(16)
        };
        event_loop.dispatch(timeout, &mut app).unwrap();

        if app.exit {
            break;
        }

        let now = std::time::Instant::now();
        let mut dt = now.duration_since(last_tick).as_secs_f32();
        last_tick = now;
        if dt > 0.1 {
            dt = 0.1;
        }
        if let Some(state) = &mut app.state {
            if let Some(jl) = &mut state.json_layout {
                if jl.tick(dt, &mut state.ui_context) {
                    app.redraw = true;
                }
            }
        }

        if app.redraw {
            app.redraw = false;
            if let Some(state) = &mut app.state {
                let _ = state.render(app.fade_factor);
            }
        }
    }

    if let Some(state) = &mut app.state {
        if let Some(ref window) = state.window {
            match window {
                AppWindow::Layer(layer) => layer.set_keyboard_interactivity(KeyboardInteractivity::None),
                AppWindow::Xdg(_) => {}
            }
        }
        state.wl_surface.commit();
    }
    drop(app);
    let _ = conn_clone.roundtrip();
}

fn run_client(socket_path: &str, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Read, Write};
    let mut stream = std::os::unix::net::UnixStream::connect(socket_path)?;

    let mut needs_stdin = false;
    let mut mode_specified = false;
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--mode" {
            if i + 1 < args.len() {
                let m = &args[i + 1];
                match m.as_str() {
                    "apps" | "app" | "path" => {
                        needs_stdin = false;
                        mode_specified = true;
                    }
                    "dmenu" | "json" => {
                        needs_stdin = true;
                        mode_specified = true;
                    }
                    _ => {}
                }
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "--apps" || arg == "--app" || arg == "--path" {
            needs_stdin = false;
            mode_specified = true;
            i += 1;
        } else if arg == "--dmenu" || arg == "--json" || arg == "--layout" {
            needs_stdin = true;
            mode_specified = true;
            i += 1;
        } else if arg == "--switcher" {
            needs_stdin = true;
            mode_specified = true;
            i += 1;
        } else {
            i += 1;
        }
    }

    if !mode_specified {
        needs_stdin = !std::io::stdin().is_terminal();
    }

    let mut stdin_str = String::new();
    if needs_stdin {
        std::io::stdin().read_to_string(&mut stdin_str)?;
    }


    let payload = serde_json::json!({
        "args": args,
        "initial_stdin": stdin_str,
    });

    let payload_str = payload.to_string();
    stream.write_all(payload_str.as_bytes())?;
    stream.write_all(b"\n")?;

    let mut stream_clone = stream.try_clone()?;
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(line) = line {
                let _ = stream_clone.write_all(line.as_bytes());
                let _ = stream_clone.write_all(b"\n");
            }
        }
    });

    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    print!("{}", response);
    Ok(())
}

fn run_daemon(socket_path: &str) {
    use std::io::{Write, BufRead};
    let _ = std::fs::remove_file(socket_path);
    let listener = match std::os::unix::net::UnixListener::bind(socket_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind to socket {}: {}", socket_path, e);
            std::process::exit(1);
        }
    };

    log::info!("cce-cloud daemon started, listening on {}", socket_path);

    let conn = Connection::connect_to_env().unwrap();
    let conn_clone = conn.clone();
    let (globals, mut event_queue) = registry_queue_init(&conn).unwrap();
    let qh = event_queue.handle();

    let compositor_state = CompositorState::bind(&globals, &qh).unwrap();
    let layer_shell_state = LayerShell::bind(&globals, &qh).unwrap();
    let shm_state = Shm::bind(&globals, &qh).unwrap();
    let seat_state = SeatState::new(&globals, &qh);
    let output_state = OutputState::new(&globals, &qh);
    let cce_wm = globals.bind::<cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1, _, _>(&qh, 2..=4, ()).ok();

    let mut app = AppState {
        registry_state: RegistryState::new(&globals),
        compositor_state,
        layer_shell_state,
        shm_state,
        seat_state,
        output_state,
        seats: Vec::new(),
        pointer: None,
        keyboard: None,
        window: None,
        surface: None,
        state: None,
        exit: false,
        redraw: false,
        ctrl_pressed: false,
        super_pressed: false,
        switcher_mode: false,
        fade_out: false,
        fade_start: None,
        fade_factor: 1.0,
        cce_toplevel: None,
        selected_item: None,
    };

    event_queue.roundtrip(&mut app).unwrap();

    let scale = cce_ui::wayland::detect_scale_factor(&app.output_state);
    let xdg_shell_state = smithay_client_toolkit::shell::xdg::XdgShell::bind(&globals, &qh).ok();

    let mut event_loop = calloop::EventLoop::try_new().unwrap();
    let loop_handle = event_loop.handle();
    WaylandSource::new(conn, event_queue).insert(loop_handle.clone()).unwrap();

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        let (stdin_sender, stdin_channel) = calloop::channel::channel::<()>();

        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut initial_line = String::new();
        if reader.read_line(&mut initial_line).is_err() {
            continue;
        }

        let payload: serde_json::Value = match serde_json::from_str(&initial_line) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let client_args: Vec<String> = payload["args"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let initial_stdin = payload["initial_stdin"].as_str().unwrap_or("").to_string();

        let mut prompt = "Search: ".to_string();
        let mut mode = if !initial_stdin.is_empty() {
            LauncherMode::Dmenu
        } else {
            LauncherMode::Path
        };
        let mut x_pos: Option<i32> = None;
        let mut y_pos: Option<i32> = None;
        let mut select_item: Option<String> = None;
        let mut align_right = false;
        let mut switcher_mode = false;
        let mut parent_app_id: Option<String> = None;

        let mut i = 1;
        while i < client_args.len() {
            let arg = &client_args[i];
            if arg == "-p" || arg == "--prompt" {
                if i + 1 < client_args.len() {
                    prompt = client_args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            } else if arg == "-s" || arg == "--select" {
                if i + 1 < client_args.len() {
                    select_item = Some(client_args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            } else if arg == "-x" || arg == "--x-pos" {
                if i + 1 < client_args.len() {
                    if let Ok(val) = client_args[i + 1].parse::<i32>() {
                        x_pos = Some(val);
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else if arg == "-y" || arg == "--y-pos" {
                if i + 1 < client_args.len() {
                    if let Ok(val) = client_args[i + 1].parse::<i32>() {
                        y_pos = Some(val);
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else if arg == "--parent-app-id" {
                if i + 1 < client_args.len() {
                    parent_app_id = Some(client_args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            } else if arg == "--mode" {
                if i + 1 < client_args.len() {
                    let m = &client_args[i + 1];
                    match m.as_str() {
                        "apps" | "app" => mode = LauncherMode::Apps,
                        "path" => mode = LauncherMode::Path,
                        "dmenu" => mode = LauncherMode::Dmenu,
                        _ => eprintln!("Unknown mode: {}", m),
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else if arg == "--apps" || arg == "--app" {
                mode = LauncherMode::Apps;
                i += 1;
            } else if arg == "--path" {
                mode = LauncherMode::Path;
                i += 1;
            } else if arg == "--dmenu" {
                mode = LauncherMode::Dmenu;
                i += 1;
            } else if arg == "--json" || arg == "--layout" {
                mode = LauncherMode::Json;
                i += 1;
            } else if arg == "--align-right" {
                align_right = true;
                i += 1;
            } else if arg == "--switcher" {
                switcher_mode = true;
                mode = LauncherMode::Dmenu;
                i += 1;
            } else {
                i += 1;
            }
        }

        let mut json_layout_config: Option<JsonLayoutConfig> = None;
        if mode == LauncherMode::Json {
            match serde_json::from_str::<JsonLayoutConfig>(&initial_stdin) {
                Ok(cfg) => {
                    json_layout_config = Some(cfg);
                }
                Err(e) => {
                    let _ = stream.write_all(format!("Failed to parse JSON layout: {}\n", e).as_bytes());
                    continue;
                }
            }
        }

        let use_xdg = cce_wm.is_some() && xdg_shell_state.is_some() && x_pos.is_none() && y_pos.is_none();

        let mut initial_items = Vec::new();
        if mode == LauncherMode::Dmenu {
            for line in initial_stdin.lines() {
                initial_items.push(line.to_string());
            }
        }

        let stdin_state = Arc::new(std::sync::Mutex::new(StdinState {
            items: initial_items,
            new_data: !initial_stdin.is_empty(),
            cycle_next: 0,
            select_and_close: false,
        }));

        let stdin_state_clone = stdin_state.clone();
        let stdin_sender_clone = stdin_sender.clone();
        let mut reader_clone = stream.try_clone().unwrap();
        let thread_handle = std::thread::spawn(move || {
            let mut line = String::new();
            let mut buf_reader = std::io::BufReader::new(&mut reader_clone);
            while let Ok(n) = buf_reader.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                let trimmed = line.trim_end_matches('\n');
                if let Ok(mut lock_state) = stdin_state_clone.lock() {
                    if trimmed == "__cce_switcher_next__" {
                        lock_state.cycle_next += 1;
                        lock_state.new_data = true;
                    } else if trimmed == "__cce_switcher_select_and_close__" {
                        lock_state.select_and_close = true;
                        lock_state.new_data = true;
                    } else {
                        lock_state.items.push(trimmed.to_string());
                        lock_state.new_data = true;
                    }
                }
                let _ = stdin_sender_clone.send(());
                line.clear();
            }
        });

        let stdin_state_for_handler = stdin_state.clone();
        let registration_token = loop_handle.insert_source(stdin_channel, move |event, _metadata, app_state: &mut AppState| {
            if let calloop::channel::Event::Msg(()) = event {
                let mut select_and_close = false;
                if let Some(st) = &mut app_state.state {
                    if let (Ok(mut lock_daemon), Ok(mut lock_state)) = (stdin_state_for_handler.lock(), st.stdin_state.lock()) {
                        if lock_daemon.new_data {
                            lock_state.items = lock_daemon.items.clone();
                            lock_state.cycle_next = lock_daemon.cycle_next;
                            lock_state.select_and_close = lock_daemon.select_and_close;
                            lock_state.new_data = true;
                            lock_daemon.new_data = false;
                        }
                    }
                    if st.check_stdin_updates() {
                        st.update_desired_size();
                        st.apply_layout();
                        st.upload_vertices();
                        app_state.redraw = true;
                    }
                    if st.select_and_close_requested {
                        st.select_and_close_requested = false;
                        select_and_close = true;
                    }
                }
                if select_and_close {
                    app_state.trigger_select_and_close();
                }
            }
        }).unwrap();

        app.super_pressed = switcher_mode;
        app.switcher_mode = switcher_mode;
        app.selected_item = None;

        let (state, cce_toplevel) = pollster::block_on(State::new(
            &conn_clone,
            &qh,
            &app.compositor_state,
            &app.layer_shell_state,
            xdg_shell_state.as_ref(),
            cce_wm.as_ref(),
            use_xdg,
            prompt,
            stdin_sender.clone(),
            mode,
            x_pos,
            y_pos,
            align_right,
            scale,
            select_item,
            switcher_mode,
            json_layout_config,
            parent_app_id,
        ));

        if let Some(ref mut st) = app.state {
            st.stdin_state = stdin_state.clone();
        }
        app.window = state.window.clone();
        app.surface = Some(state.wl_surface.clone());
        app.cce_toplevel = cce_toplevel;
        app.state = Some(state);

        app.exit = false;
        app.fade_out = false;
        app.fade_start = None;
        app.fade_factor = 1.0;

        let mut last_tick = std::time::Instant::now();
        while !app.exit {
            if app.fade_out {
                if let Some(start) = app.fade_start {
                    let elapsed = start.elapsed().as_secs_f32();
                    app.fade_factor = (1.0 - elapsed / 0.15).max(0.0);
                    if app.fade_factor <= 0.0 {
                        app.exit = true;
                    } else {
                        app.redraw = true;
                    }
                }
            }

            let timeout = if app.redraw {
                std::time::Duration::from_millis(0)
            } else {
                std::time::Duration::from_millis(16)
            };
            event_loop.dispatch(timeout, &mut app).unwrap();

            if app.exit {
                break;
            }

            let now = std::time::Instant::now();
            let mut dt = now.duration_since(last_tick).as_secs_f32();
            last_tick = now;
            if dt > 0.1 {
                dt = 0.1;
            }
            if let Some(st) = &mut app.state {
                if let Some(jl) = &mut st.json_layout {
                    if jl.tick(dt, &mut st.ui_context) {
                        app.redraw = true;
                    }
                }
            }

            if app.redraw {
                app.redraw = false;
                if let Some(st) = &mut app.state {
                    let _ = st.render(app.fade_factor);
                }
            }
        }

        loop_handle.remove(registration_token);
        let _ = stream.shutdown(std::net::Shutdown::Read);
        let _ = thread_handle.join();

        let response = app.selected_item.take().unwrap_or_default();
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(b"\n");
        let _ = stream.flush();

        if let Some(st) = &mut app.state {
            if let Some(ref window) = st.window {
                match window {
                    AppWindow::Layer(layer) => layer.set_keyboard_interactivity(KeyboardInteractivity::None),
                    AppWindow::Xdg(_) => {}
                }
            }
            st.wl_surface.commit();
        }
        app.window = None;
        app.surface = None;
        app.state = None;
        app.cce_toplevel = None;
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = std::env::args().collect::<Vec<String>>();
    let is_daemon = args.iter().any(|arg| arg == "--daemon");

    let uid = unsafe { libc::getuid() };
    let socket_dir = format!("/run/user/{}", uid);
    let socket_path = if std::path::Path::new(&socket_dir).exists() {
        format!("{}/cce-cloud.socket", socket_dir)
    } else {
        format!("/tmp/cce-cloud-{}.socket", uid)
    };

    if is_daemon {
        run_daemon(&socket_path);
    } else {
        match run_client(&socket_path, &args) {
            Ok(_) => {}
            Err(e) => {
                log::warn!("Could not connect to cce-cloud daemon: {}. Running in standalone mode.", e);
                run_standalone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_layout_parsing() {
        let json_str = r#"{
            "width": 320,
            "height": 240,
            "widgets": [
                { "type": "label", "text": "Select Option:" },
                { "id": "feat_a", "type": "checkbox", "text": "Enable Feature A", "checked": true },
                { "id": "btn_ok", "type": "button", "text": "OK" }
            ]
        }"#;

        let config: JsonLayoutConfig = serde_json::from_str(json_str).expect("Failed to parse JSON");
        assert_eq!(config.width, Some(320));
        assert_eq!(config.height, Some(240));
        let widgets = config.widgets.as_ref().expect("widgets option should be Some");
        assert_eq!(widgets.len(), 3);

        assert_eq!(widgets[0].widget_type, "label");
        assert_eq!(widgets[0].text, "Select Option:");

        assert_eq!(widgets[1].widget_type, "checkbox");
        assert_eq!(widgets[1].id.as_deref(), Some("feat_a"));
        assert_eq!(widgets[1].checked, Some(true));
    }

    #[test]
    fn test_json_layout_widget_flow() {
        use cce_ui::widget::Element;

        let widgets_conf = vec![
            JsonWidgetConfig {
                widget_type: "label".to_string(),
                text: "Label 1".to_string(),
                id: None,
                checked: None,
                value: None,
                min: None,
                max: None,
                step: None,
                decimals: None,
                color: None,
                value_f32: None,
                min_f32: None,
                max_f32: None,
                target_page: None,
            },
            JsonWidgetConfig {
                widget_type: "checkbox".to_string(),
                text: "Check 1".to_string(),
                id: Some("chk".to_string()),
                checked: Some(false),
                value: None,
                min: None,
                max: None,
                step: None,
                decimals: None,
                color: None,
                value_f32: None,
                min_f32: None,
                max_f32: None,
                target_page: None,
            },
            JsonWidgetConfig {
                widget_type: "button".to_string(),
                text: "Click 1".to_string(),
                id: Some("btn".to_string()),
                checked: None,
                value: None,
                min: None,
                max: None,
                step: None,
                decimals: None,
                color: None,
                value_f32: None,
                min_f32: None,
                max_f32: None,
                target_page: None,
            },
        ];

        let config = JsonLayoutConfig {
            width: Some(300),
            height: Some(400),
            widgets: Some(widgets_conf),
            pages: None,
            justify: None,
        };

        let mut layout = JsonLayoutWidget::new(&config);
        layout.set_rect(0.0, 0.0, 300.0, 400.0);
        let mut ctx = cce_ui::context::UiContext::new();

        // Verify sub-widgets are populated and positioned correctly
        assert_eq!(layout.widgets.len(), 3);
        
        let w_label_y = layout.widgets[0].y;
        let w_label_h = layout.widgets[0].h;
        let w_label_x = layout.widgets[0].x;
        let w_label_w = layout.widgets[0].w;

        let w_check_y = layout.widgets[1].y;
        let w_check_h = layout.widgets[1].h;
        let w_check_x = layout.widgets[1].x;
        let w_check_w = layout.widgets[1].w;

        let w_btn_y = layout.widgets[2].y;
        let w_btn_h = layout.widgets[2].h;
        let w_btn_x = layout.widgets[2].x;
        let w_btn_w = layout.widgets[2].w;

        assert!(layout.widgets[0].widget.as_any().downcast_ref::<cce_ui::widget::Label>().is_some());
        assert!(layout.widgets[1].widget.as_any().downcast_ref::<cce_ui::widget::Checkbox>().is_some());
        assert!(layout.widgets[2].widget.as_any().downcast_ref::<cce_ui::widget::Button>().is_some());

        // Check vertical sequence positions
        assert_eq!(w_label_y, 16.0);
        assert_eq!(w_label_h, 18.0);

        assert_eq!(w_check_y, 16.0 + 18.0 + 12.0); // y_prev + h_prev + spacing
        assert_eq!(w_check_h, 22.0);

        assert_eq!(w_btn_y, w_check_y + 22.0 + 12.0);
        assert_eq!(w_btn_h, 24.0);

        // Check horizontal positioning (should match usable width: 300 - 2 * 16 = 268)
        assert_eq!(w_label_x, 16.0);
        assert_eq!(w_label_w, 268.0);
        assert_eq!(w_check_x, 16.0);
        assert_eq!(w_check_w, 268.0);
        assert_eq!(w_btn_x, 16.0);
        assert_eq!(w_btn_w, 268.0);

        // Verify Checkbox initial state
        assert_eq!(layout.widgets[1].widget.as_any().downcast_ref::<cce_ui::widget::Checkbox>().unwrap().checked(), false);

        // Simulate click on Checkbox row
        let changed = layout.mouse_input(
            cce_ui::widget::MouseButton::Left,
            cce_ui::widget::ElementState::Released,
            w_check_x + 5.0,
            w_check_y + 5.0,
            &mut ctx,
        );
        assert!(changed);
        assert_eq!(layout.widgets[1].widget.as_any().downcast_ref::<cce_ui::widget::Checkbox>().unwrap().checked(), true);

        // Simulate hover on button
        let changed_hover = layout.on_cursor_moved(w_btn_x + 10.0, w_btn_y + 10.0, &mut ctx);
        assert!(changed_hover);
        // Phase 5: hover state lives on the Button model (it drives the color matrix), not the base.
        assert!(layout.widgets[2].widget.as_any().downcast_ref::<cce_ui::widget::Button>().unwrap().hovered());
    }

    #[test]
    fn test_app_history_sorting() {
        let temp_dir = std::path::PathBuf::from("/tmp/cce-cloud-test-cache-dir");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);

        let orig_xdg = std::env::var("XDG_CACHE_HOME").ok();
        std::env::set_var("XDG_CACHE_HOME", &temp_dir);

        let mut apps = vec![
            AppInfo { name: "App A".to_string(), exec: "exec_a".to_string() },
            AppInfo { name: "App B".to_string(), exec: "exec_b".to_string() },
            AppInfo { name: "App C".to_string(), exec: "exec_c".to_string() },
        ];

        // Initially no history, sorted alphabetically.
        sort_apps_by_history(&mut apps);
        assert_eq!(apps[0].name, "App A");
        assert_eq!(apps[1].name, "App B");
        assert_eq!(apps[2].name, "App C");

        // Record launch for App B once, and App C twice.
        record_app_launch("App B");
        std::thread::sleep(std::time::Duration::from_millis(10));
        record_app_launch("App C");
        std::thread::sleep(std::time::Duration::from_millis(10));
        record_app_launch("App C");

        sort_apps_by_history(&mut apps);
        // App C (count 2) -> App B (count 1) -> App A (count 0)
        assert_eq!(apps[0].name, "App C");
        assert_eq!(apps[1].name, "App B");
        assert_eq!(apps[2].name, "App A");

        // Record App A launches 3 times to move it to the top.
        record_app_launch("App A");
        record_app_launch("App A");
        record_app_launch("App A");

        sort_apps_by_history(&mut apps);
        // App A (count 3) -> App C (count 2) -> App B (count 1)
        assert_eq!(apps[0].name, "App A");
        assert_eq!(apps[1].name, "App C");
        assert_eq!(apps[2].name, "App B");

        // Record App B launch once, now both App B and App C have count 2.
        // App B was launched most recently, so it should rank higher than App C.
        record_app_launch("App B");
        sort_apps_by_history(&mut apps);
        // App A (count 3) -> App B (count 2, recent) -> App C (count 2, older)
        assert_eq!(apps[0].name, "App A");
        assert_eq!(apps[1].name, "App B");
        assert_eq!(apps[2].name, "App C");

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
        if let Some(val) = orig_xdg {
            std::env::set_var("XDG_CACHE_HOME", val);
        } else {
            std::env::remove_var("XDG_CACHE_HOME");
        }
    }

    #[test]
    fn test_filter_and_sort_preserving_history() {
        let items = vec![
            "Firefox".to_string(),
            "File Manager".to_string(),
            "foo".to_string(),
        ];

        let filtered = filter_and_sort_items(&items, "f");
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0], "Firefox");
        assert_eq!(filtered[1], "File Manager");
        assert_eq!(filtered[2], "foo");

        let filtered_fi = filter_and_sort_items(&items, "fi");
        assert_eq!(filtered_fi.len(), 2);
        assert_eq!(filtered_fi[0], "Firefox");
        assert_eq!(filtered_fi[1], "File Manager");
    }
}

impl FuzzelWidget {
    fn own_labels(&self) -> Vec<TextLabel> {
        let mut labels = Vec::new();
        let pad = 15.0;
        let search_h = 35.0;

        let query_text = if self.query.is_empty() {
            format!("{}{}", self.prompt, "Type to search...")
        } else {
            format!("{}{}", self.prompt, self.query)
        };
        let query_color = if self.query.is_empty() {
            [0x66, 0x66, 0x77]
        } else {
            [0xcc, 0xff, 0xcc]
        };

        labels.push(TextLabel {
            text: query_text,
            x: self.x + pad + 10.0,
            y: self.y + pad + 9.0,
            font_size: 14.0,
            color: query_color,
        });

        let item_h = 25.0;
        for (idx, item_text) in self.filtered_items.iter().enumerate() {
            let virtual_y = idx as f32 * item_h;
            if let Some(draw_y) = self.scroll_box.get_draw_y(virtual_y, item_h) {
                let color = if idx == self.selected {
                    [0xff, 0xff, 0xff]
                } else {
                    [0xbb, 0xbb, 0xc5]
                };

                labels.push(TextLabel {
                    text: item_text.clone(),
                    x: self.x + pad + 10.0,
                    y: draw_y + 4.0,
                    font_size: 13.0,
                    color,
                });
            }
        }

        if self.filtered_items.is_empty() {
            let list_y = self.y + pad + search_h + 10.0;
            labels.push(TextLabel {
                text: "No matches found".to_string(),
                x: self.x + pad + 10.0,
                y: list_y + 4.0,
                font_size: 13.0,
                color: [0x88, 0x88, 0x99],
            });
        }

        labels
    }
}
