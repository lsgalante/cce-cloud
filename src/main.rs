use std::sync::{Arc, Mutex};
use std::io::{self, BufRead, IsTerminal};

use clear_ui::widget::{Element, TextLabel, JsonLayoutWidget, JsonLayoutConfig, JsonWidgetConfig};

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm, delegate_layer, delegate_output,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
    clip_circle: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x4,
        2 => Float32x3,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

fn quad_vertices(
    x: f32, y: f32, w: f32, h: f32,
    surface_w: f32, surface_h: f32,
    color: [f32; 4],
) -> [Vertex; 6] {
    let x0 = (x / surface_w) * 2.0 - 1.0;
    let y0 = 1.0 - (y / surface_h) * 2.0;
    let x1 = ((x + w) / surface_w) * 2.0 - 1.0;
    let y1 = 1.0 - ((y + h) / surface_h) * 2.0;

    [
        Vertex { position: [x0, y0], color, clip_circle: [0.0; 3] },
        Vertex { position: [x1, y0], color, clip_circle: [0.0; 3] },
        Vertex { position: [x0, y1], color, clip_circle: [0.0; 3] },
        Vertex { position: [x1, y0], color, clip_circle: [0.0; 3] },
        Vertex { position: [x1, y1], color, clip_circle: [0.0; 3] },
        Vertex { position: [x0, y1], color, clip_circle: [0.0; 3] },
    ]
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
    buffer.set_text(font_system, text, Attrs::new(), glyphon::Shaping::Advanced);
    buffer.shape_until_scroll(font_system, true);
    buffer
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
        .open("/tmp/clear-spawn.log")
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

fn gradient_quad_vertices(x: f32, y: f32, w: f32, h: f32, sw: f32, sh: f32, c0: [f32; 4], c1: [f32; 4]) -> [Vertex; 6] {
    let x0 = (x / sw) * 2.0 - 1.0;
    let y0 = 1.0 - (y / sh) * 2.0;
    let x1 = ((x + w) / sw) * 2.0 - 1.0;
    let y1 = 1.0 - ((y + h) / sh) * 2.0;
    [
        Vertex { position: [x0, y0], color: c0, clip_circle: [0.0; 3] },
        Vertex { position: [x1, y0], color: c1, clip_circle: [0.0; 3] },
        Vertex { position: [x0, y1], color: c0, clip_circle: [0.0; 3] },
        Vertex { position: [x1, y0], color: c1, clip_circle: [0.0; 3] },
        Vertex { position: [x1, y1], color: c1, clip_circle: [0.0; 3] },
        Vertex { position: [x0, y1], color: c0, clip_circle: [0.0; 3] },
    ]
}

fn parse_hex(hex: &str) -> Option<(f32, f32, f32)> {
    let s = hex.trim_start_matches('#');
    if s.len() == 6 {
        u32::from_str_radix(s, 16).ok().map(|v| {
            let r = ((v >> 16) & 0xFF) as f32 / 255.0;
            let g = ((v >> 8) & 0xFF) as f32 / 255.0;
            let b = (v & 0xFF) as f32 / 255.0;
            (r, g, b)
        })
    } else {
        None
    }
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let mut h = 0.0;
    let mut s = 0.0;
    let l = (max + min) / 2.0;

    if max != min {
        let d = max - min;
        s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
        if max == r {
            h = (g - b) / d + (if g < b { 6.0 } else { 0.0 });
        } else if max == g {
            h = (b - r) / d + 2.0;
        } else if max == b {
            h = (r - g) / d + 4.0;
        }
        h /= 6.0;
    }

    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 1.0 / 2.0 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    p
}

const HEADER_H: f32 = 36.0;
const SLIDER_ROW_H: f32 = 36.0;
const SLIDER_START_Y: f32 = 48.0;
const SLIDER_LABEL_X: f32 = 12.0;
const SLIDER_TRACK_X: f32 = 32.0;
const SLIDER_TRACK_W: f32 = 280.0;
const SLIDER_TRACK_H: f32 = 20.0;
const SLIDER_VALUE_X: f32 = 320.0;
const PREVIEW_X: f32 = 12.0;
const PREVIEW_Y: f32 = 276.0;
const PREVIEW_W: f32 = 160.0;
const PREVIEW_H: f32 = 72.0;
const BUTTON_Y: f32 = 360.0;
const BUTTON_H: f32 = 24.0;
const BUTTON_W: f32 = 100.0;
const BUTTON_GAP: f32 = 12.0;
const WIN_W: f32 = 380.0;
const WIN_H: f32 = 428.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragTarget { Red, Green, Blue, Hue, Saturation, Lightness }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorAction { Apply, Cancel }

struct RectWidget {
    x: f32, y: f32, w: f32, h: f32,
    color: [f32; 4],
}

struct GradientRectWidget {
    x: f32, y: f32, w: f32, h: f32,
    c0: [f32; 4],
    c1: [f32; 4],
}

struct HitButton {
    x: f32, y: f32, w: f32, h: f32,
    action: ColorAction,
}

struct ColorPickerWidget {
    _x: f32, _y: f32, _w: f32, _h: f32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub hue: f32,
    pub saturation: f32,
    pub lightness: f32,
    pub dragging: Option<DragTarget>,
    pub action_requested: Option<ColorAction>,
    cursor_x: f32,
    cursor_y: f32,
    scale_factor: f32,
    
    rects: Vec<RectWidget>,
    gradient_rects: Vec<GradientRectWidget>,
    pub labels: Vec<TextLabel>,
    action_buttons: Vec<HitButton>,
    pub needs_rebuild: bool,
}

impl ColorPickerWidget {
    fn new(red: f32, green: f32, blue: f32) -> Self {
        let (hue, saturation, lightness) = rgb_to_hsl(red, green, blue);
        Self {
            _x: 0.0, _y: 0.0, _w: 0.0, _h: 0.0,
            red, green, blue,
            hue, saturation, lightness,
            dragging: None,
            action_requested: None,
            cursor_x: 0.0, cursor_y: 0.0,
            scale_factor: 2.0,
            rects: Vec::new(),
            gradient_rects: Vec::new(),
            labels: Vec::new(),
            action_buttons: Vec::new(),
            needs_rebuild: true,
        }
    }

    fn hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}",
            (self.red * 255.0) as u8,
            (self.green * 255.0) as u8,
            (self.blue * 255.0) as u8)
    }

    fn rebuild_layout(&mut self, scale: f32) {
        self.scale_factor = scale;
        let s = scale;
        let sw = WIN_W * s;
        let sh = WIN_H * s;

        let mut rects = Vec::new();
        let mut gradient_rects = Vec::new();
        let mut labels = Vec::new();
        let mut action_buttons = Vec::new();

        rects.push(RectWidget {
            x: 0.0, y: 0.0, w: sw, h: HEADER_H * s,
            color: clear_ui::color::HEADER_BG,
        });
        
        labels.push(TextLabel {
            text: "Clear Color Interface".to_string(),
            x: 12.0 * s,
            y: 10.0 * s,
            font_size: 14.0 * s,
            color: [0xcc, 0xcc, 0xd4],
        });

        rects.push(RectWidget {
            x: 0.0, y: HEADER_H * s, w: sw, h: sh - HEADER_H * s,
            color: clear_ui::color::CONTENT_BG,
        });

        let channels = [self.red, self.green, self.blue, self.hue, self.saturation, self.lightness];
        let chan_labels = ['R', 'G', 'B', 'H', 'S', 'L'];

        let red = self.red;
        let green = self.green;
        let blue = self.blue;
        let hue = self.hue;
        let saturation = self.saturation;
        let lightness = self.lightness;

        let get_color_at = |i: usize, t: f32| -> [f32; 4] {
            match i {
                0 => [t, green, blue, 1.0],
                1 => [red, t, blue, 1.0],
                2 => [red, green, t, 1.0],
                3 => {
                    let (r, g, b) = hsl_to_rgb(t, saturation, lightness);
                    [r, g, b, 1.0]
                }
                4 => {
                    let (r, g, b) = hsl_to_rgb(hue, t, lightness);
                    [r, g, b, 1.0]
                }
                _ => {
                    let (r, g, b) = hsl_to_rgb(hue, saturation, t);
                    [r, g, b, 1.0]
                }
            }
        };

        for i in 0..6 {
            let row_y = (SLIDER_START_Y + i as f32 * SLIDER_ROW_H) * s;
            let track_y = row_y + ((SLIDER_ROW_H - SLIDER_TRACK_H) / 2.0) * s;

            rects.push(RectWidget {
                x: (SLIDER_TRACK_X - 1.0) * s,
                y: track_y - 1.0 * s,
                w: (SLIDER_TRACK_W + 2.0) * s,
                h: (SLIDER_TRACK_H + 2.0) * s,
                color: [0.08, 0.08, 0.10, 1.0],
            });

            let n_segments = if i == 3 { 30 } else { 10 };
            for j in 0..n_segments {
                let t0 = j as f32 / n_segments as f32;
                let t1 = (j + 1) as f32 / n_segments as f32;
                let c0 = get_color_at(i, t0);
                let c1 = get_color_at(i, t1);
                gradient_rects.push(GradientRectWidget {
                    x: (SLIDER_TRACK_X + t0 * SLIDER_TRACK_W) * s,
                    y: track_y,
                    w: ((t1 - t0) * SLIDER_TRACK_W) * s,
                    h: SLIDER_TRACK_H * s,
                    c0,
                    c1,
                });
            }

            let indicator_w = 4.0;
            let indicator_h = SLIDER_TRACK_H + 4.0;
            let indicator_x = SLIDER_TRACK_X + channels[i] * SLIDER_TRACK_W - indicator_w / 2.0;
            let indicator_y = (SLIDER_START_Y + i as f32 * SLIDER_ROW_H) + ((SLIDER_ROW_H - indicator_h) / 2.0);

            rects.push(RectWidget {
                x: (indicator_x - 1.0) * s,
                y: (indicator_y - 1.0) * s,
                w: (indicator_w + 2.0) * s,
                h: (indicator_h + 2.0) * s,
                color: [0.05, 0.05, 0.05, 0.95],
            });

            rects.push(RectWidget {
                x: indicator_x * s,
                y: indicator_y * s,
                w: indicator_w * s,
                h: indicator_h * s,
                color: [1.0, 1.0, 1.0, 1.0],
            });

            labels.push(TextLabel {
                text: chan_labels[i].to_string(),
                x: SLIDER_LABEL_X * s,
                y: row_y + 4.0 * s,
                font_size: 12.0 * s,
                color: [0xaa, 0xaa, 0xbb],
            });

            let val = if i < 3 {
                format!("{}", (channels[i] * 255.0) as u8)
            } else if i == 3 {
                format!("{}°", (channels[i] * 360.0).round() as u16)
            } else {
                format!("{}%", (channels[i] * 100.0).round() as u8)
            };
            
            labels.push(TextLabel {
                text: val,
                x: SLIDER_VALUE_X * s,
                y: row_y + 4.0 * s,
                font_size: 11.0 * s,
                color: [0xcc, 0xcc, 0xdd],
            });
        }

        rects.push(RectWidget {
            x: PREVIEW_X * s, y: PREVIEW_Y * s,
            w: PREVIEW_W * s, h: PREVIEW_H * s,
            color: [self.red, self.green, self.blue, 1.0],
        });

        let hex = self.hex();
        labels.push(TextLabel {
            text: hex,
            x: (PREVIEW_X + PREVIEW_W + 16.0) * s,
            y: (PREVIEW_Y + 26.0) * s,
            font_size: 16.0 * s,
            color: [0xe0, 0xe0, 0xe8],
        });

        let apply_x = PREVIEW_X;
        let cancel_x = PREVIEW_X + BUTTON_W + BUTTON_GAP;
        let btn_y = BUTTON_Y;
        let btn_bg = [0.20, 0.40, 0.65, 1.0];
        let cancel_bg = [0.40, 0.20, 0.20, 1.0];

        rects.push(RectWidget {
            x: apply_x * s, y: btn_y * s,
            w: BUTTON_W * s, h: BUTTON_H * s,
            color: btn_bg,
        });
        labels.push(TextLabel {
            text: "Apply".to_string(),
            x: (apply_x + 28.0) * s,
            y: (btn_y + 8.0) * s,
            font_size: 12.0 * s,
            color: [0xee, 0xee, 0xf0],
        });
        action_buttons.push(HitButton {
            x: apply_x * s, y: btn_y * s,
            w: BUTTON_W * s, h: BUTTON_H * s,
            action: ColorAction::Apply,
        });

        rects.push(RectWidget {
            x: cancel_x * s, y: btn_y * s,
            w: BUTTON_W * s, h: BUTTON_H * s,
            color: cancel_bg,
        });
        labels.push(TextLabel {
            text: "Cancel".to_string(),
            x: (cancel_x + 22.0) * s,
            y: (btn_y + 8.0) * s,
            font_size: 12.0 * s,
            color: [0xee, 0xee, 0xf0],
        });
        action_buttons.push(HitButton {
            x: cancel_x * s, y: btn_y * s,
            w: BUTTON_W * s, h: BUTTON_H * s,
            action: ColorAction::Cancel,
        });

        self.rects = rects;
        self.gradient_rects = gradient_rects;
        self.labels = labels;
        self.action_buttons = action_buttons;
        self.needs_rebuild = false;
    }

    fn slider_physical_rect(i: usize, s: f32) -> (f32, f32, f32, f32) {
        let row_y = (SLIDER_START_Y + i as f32 * SLIDER_ROW_H) * s;
        let track_y = row_y + ((SLIDER_ROW_H - SLIDER_TRACK_H) / 2.0) * s;
        (SLIDER_TRACK_X * s, track_y, SLIDER_TRACK_W * s, SLIDER_TRACK_H * s)
    }

    fn collect_vertices(&self, sw: f32, sh: f32) -> Vec<Vertex> {
        let mut verts = Vec::new();
        for r in &self.rects {
            verts.extend(quad_vertices(r.x, r.y, r.w, r.h, sw, sh, r.color));
        }
        for g in &self.gradient_rects {
            verts.extend(gradient_quad_vertices(g.x, g.y, g.w, g.h, sw, sh, g.c0, g.c1));
        }
        verts
    }

    fn handle_cursor_moved(&mut self, cx: f32, cy: f32) {
        self.cursor_x = cx;
        self.cursor_y = cy;
        if let Some(drag) = self.dragging {
            let i = match drag {
                DragTarget::Red => 0,
                DragTarget::Green => 1,
                DragTarget::Blue => 2,
                DragTarget::Hue => 3,
                DragTarget::Saturation => 4,
                DragTarget::Lightness => 5,
            };
            let s = self.scale_factor;
            let (tx, _, tw, _) = Self::slider_physical_rect(i, s);
            let new_val = ((self.cursor_x - tx) / tw).clamp(0.0, 1.0);
            let old = match i {
                0 => self.red,
                1 => self.green,
                2 => self.blue,
                3 => self.hue,
                4 => self.saturation,
                _ => self.lightness,
            };
            if (new_val - old).abs() > 0.002 {
                match i {
                    0 => {
                        self.red = new_val;
                        let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                        self.saturation = sat;
                        self.lightness = l;
                        if sat > 0.001 && l > 0.001 && l < 0.999 {
                            self.hue = h;
                        }
                    }
                    1 => {
                        self.green = new_val;
                        let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                        self.saturation = sat;
                        self.lightness = l;
                        if sat > 0.001 && l > 0.001 && l < 0.999 {
                            self.hue = h;
                        }
                    }
                    2 => {
                        self.blue = new_val;
                        let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                        self.saturation = sat;
                        self.lightness = l;
                        if sat > 0.001 && l > 0.001 && l < 0.999 {
                            self.hue = h;
                        }
                    }
                    3 => {
                        self.hue = new_val;
                        let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                        self.red = r;
                        self.green = g;
                        self.blue = b;
                    }
                    4 => {
                        self.saturation = new_val;
                        let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                        self.red = r;
                        self.green = g;
                        self.blue = b;
                    }
                    _ => {
                        self.lightness = new_val;
                        let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                        self.red = r;
                        self.green = g;
                        self.blue = b;
                    }
                }
                self.needs_rebuild = true;
            }
        }
    }

    fn handle_mouse_input(&mut self, state: clear_ui::widget::ElementState) {
        match state {
            clear_ui::widget::ElementState::Pressed => {
                let s = self.scale_factor;
                let (px, py) = (self.cursor_x, self.cursor_y);
                for i in 0..6 {
                    let (tx, ty, tw, th) = Self::slider_physical_rect(i, s);
                    if px >= tx && px <= tx + tw && py >= ty && py <= ty + th {
                        let val = ((px - tx) / tw).clamp(0.0, 1.0);
                        match i {
                            0 => {
                                self.red = val;
                                let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                                self.saturation = sat;
                                self.lightness = l;
                                if sat > 0.001 && l > 0.001 && l < 0.999 {
                                    self.hue = h;
                                }
                                self.dragging = Some(DragTarget::Red);
                            }
                            1 => {
                                self.green = val;
                                let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                                self.saturation = sat;
                                self.lightness = l;
                                if sat > 0.001 && l > 0.001 && l < 0.999 {
                                    self.hue = h;
                                }
                                self.dragging = Some(DragTarget::Green);
                            }
                            2 => {
                                self.blue = val;
                                let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                                self.saturation = sat;
                                self.lightness = l;
                                if sat > 0.001 && l > 0.001 && l < 0.999 {
                                    self.hue = h;
                                }
                                self.dragging = Some(DragTarget::Blue);
                            }
                            3 => {
                                self.hue = val;
                                let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                                self.red = r;
                                self.green = g;
                                self.blue = b;
                                self.dragging = Some(DragTarget::Hue);
                            }
                            4 => {
                                self.saturation = val;
                                let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                                self.red = r;
                                self.green = g;
                                self.blue = b;
                                self.dragging = Some(DragTarget::Saturation);
                            }
                            _ => {
                                self.lightness = val;
                                let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                                self.red = r;
                                self.green = g;
                                self.blue = b;
                                self.dragging = Some(DragTarget::Lightness);
                            }
                        }
                        self.needs_rebuild = true;
                        return;
                    }
                }
                for btn in &self.action_buttons {
                    if px >= btn.x && px <= btn.x + btn.w && py >= btn.y && py <= btn.y + btn.h {
                        self.action_requested = Some(btn.action);
                        self.needs_rebuild = true;
                        return;
                    }
                }
            }
            clear_ui::widget::ElementState::Released => {
                if self.dragging.is_some() {
                    self.dragging = None;
                }
            }
        }
    }

    fn handle_scroll(&mut self, scroll_amount_y: f32) {
        let s = self.scale_factor;
        let (px, py) = (self.cursor_x, self.cursor_y);
        let scroll_amount = scroll_amount_y;
        if scroll_amount.abs() > 0.0001 {
            for i in 0..6 {
                let (tx, ty, tw, th) = Self::slider_physical_rect(i, s);
                if px >= tx && px <= tx + tw && py >= ty - 4.0 * s && py <= ty + th + 4.0 * s {
                    let step = 0.02;
                    let old_val = match i {
                        0 => self.red,
                        1 => self.green,
                        2 => self.blue,
                        3 => self.hue,
                        4 => self.saturation,
                        _ => self.lightness,
                    };
                    let new_val = (old_val + scroll_amount * step).clamp(0.0, 1.0);
                    if (new_val - old_val).abs() > 0.0001 {
                        match i {
                            0 => {
                                self.red = new_val;
                                let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                                self.saturation = sat;
                                self.lightness = l;
                                if sat > 0.001 && l > 0.001 && l < 0.999 {
                                    self.hue = h;
                                }
                            }
                            1 => {
                                self.green = new_val;
                                let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                                self.saturation = sat;
                                self.lightness = l;
                                if sat > 0.001 && l > 0.001 && l < 0.999 {
                                    self.hue = h;
                                }
                            }
                            2 => {
                                self.blue = new_val;
                                let (h, sat, l) = rgb_to_hsl(self.red, self.green, self.blue);
                                self.saturation = sat;
                                self.lightness = l;
                                if sat > 0.001 && l > 0.001 && l < 0.999 {
                                    self.hue = h;
                                }
                            }
                            3 => {
                                self.hue = new_val;
                                let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                                self.red = r;
                                self.green = g;
                                self.blue = b;
                            }
                            4 => {
                                self.saturation = new_val;
                                let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                                self.red = r;
                                self.green = g;
                                self.blue = b;
                            }
                            _ => {
                                self.lightness = new_val;
                                let (r, g, b) = hsl_to_rgb(self.hue, self.saturation, self.lightness);
                                self.red = r;
                                self.green = g;
                                self.blue = b;
                            }
                        }
                        self.needs_rebuild = true;
                    }
                }
            }
        }
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
    pub scroll_box: clear_ui::widget::ScrollBox,
}

impl FuzzelWidget {
    pub fn new(prompt: String) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
            prompt,
            query: String::new(),
            all_items: Vec::new(),
            filtered_items: Vec::new(),
            selected: 0,
            scroll_box: clear_ui::widget::ScrollBox::new(),
        }
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
        self.scroll_box.update_bounds(content_h, viewport_y, viewport_h);
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

impl Element for FuzzelWidget {
    fn rect(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn set_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;

        let pad = 15.0;
        let search_h = 35.0;
        let viewport_y = y + pad + search_h + 10.0;
        let viewport_h = h - (pad + search_h + 10.0) - pad;
        self.scroll_box.set_rect(x + pad, viewport_y, w - pad * 2.0, viewport_h);
        self.update_scroll();
    }

    fn color(&self) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn extra_quads(&self) -> Vec<(f32, f32, f32, f32, [f32; 4])> {
        let mut quads = Vec::new();
        let pad = 15.0;
        let search_h = 35.0;
        
        // Search Bar Background
        quads.push((
            self.x + pad,
            self.y + pad,
            self.w - pad * 2.0,
            search_h,
            [0.10, 0.10, 0.14, 1.0],
        ));

        // Search Bar Border
        let border_color = [0.25, 0.45, 0.85, 1.0];
        let bx = self.x + pad;
        let by = self.y + pad;
        let bw = self.w - pad * 2.0;
        let bh = search_h;
        quads.push((bx, by, bw, 1.0, border_color));
        quads.push((bx, by + bh - 1.0, bw, 1.0, border_color));
        quads.push((bx, by, 1.0, bh, border_color));
        quads.push((bx + bw - 1.0, by, 1.0, bh, border_color));

        // ScrollBox quads
        quads.extend(self.scroll_box.extra_quads());

        // Selected Item Highlight
        let item_h = 25.0;
        if !self.filtered_items.is_empty() {
            let virtual_selected_y = self.selected as f32 * item_h;
            if let Some(draw_y) = self.scroll_box.get_item_draw_y(virtual_selected_y, item_h) {
                let scrollbar_w = if self.scroll_box.content_h > self.scroll_box.viewport_h { 10.0 } else { 0.0 };
                quads.push((
                    self.x + pad + 2.0,
                    draw_y,
                    self.w - pad * 2.0 - 4.0 - scrollbar_w,
                    item_h - 2.0,
                    [0.20, 0.35, 0.65, 0.9],
                ));
            }
        }

        quads
    }

    fn text_labels(&self) -> Vec<TextLabel> {
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
            if let Some(draw_y) = self.scroll_box.get_item_draw_y(virtual_y, item_h) {
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

    fn mouse_input(&mut self, button: clear_ui::widget::MouseButton, state: clear_ui::widget::ElementState, px: f32, py: f32, ctx: &mut clear_ui::context::UiContext) -> bool {
        if button == clear_ui::widget::MouseButton::Left && state == clear_ui::widget::ElementState::Pressed {
            let item_h = 25.0;
            if self.scroll_box.hit_test(px, py, ctx) {
                let click_virtual_y = py - self.scroll_box.viewport_y + self.scroll_box.scroll_y;
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
    Color,
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
}

fn read_opacity_if_configured() -> f32 {
    let config_path = "/home/lsgalante/.config/cce/config.toml";
    let content = std::fs::read_to_string(config_path).unwrap_or_default();
    
    let mut in_section = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[transparency]" {
            in_section = true;
            continue;
        }
        if trimmed.starts_with('[') && in_section {
            break;
        }
        if in_section && trimmed.starts_with("opacity") {
            if let Some(val) = trimmed.split('=').nth(1) {
                if let Ok(o) = val.trim().parse::<f32>() {
                    return o.clamp(0.0, 1.0);
                }
            }
        }
    }
    0.20 // default opacity for cce-cloud
}

struct State {
    window: LayerSurface,
    wl_surface: wl_surface::WlSurface,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,

    fuzzel: FuzzelWidget,
    color_picker: Option<ColorPickerWidget>,
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
    opacity: f32,
    fade_factor: f32,
    max_width: u32,
    max_height: u32,
    select_item: Option<String>,
    last_tick: std::time::Instant,
    ui_context: clear_ui::context::UiContext,
}

impl State {
    async fn new(
        conn: &Connection,
        qh: &QueueHandle<AppState>,
        compositor_state: &CompositorState,
        layer_shell_state: &LayerShell,
        prompt: String,
        stdin_sender: calloop::channel::Sender<()>,
        mode: LauncherMode,
        initial_hex: Option<String>,
        x_pos: Option<i32>,
        y_pos: Option<i32>,
        scale: f64,
        select_item: Option<String>,
        json_layout_config: Option<JsonLayoutConfig>,
    ) -> Self {
        clear_ui::scale::set_scale_factor(scale as f32);
        let (width, height) = if mode == LauncherMode::Color {
            (WIN_W as u32, WIN_H as u32)
        } else if mode == LauncherMode::Json {
            if let Some(ref config) = json_layout_config {
                let w = config.width.unwrap_or(300);
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
                    current_y.round() as u32
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
        let app_id = if mode == LauncherMode::Color {
            "clear-color-interface".to_string()
        } else {
            "cce-cloud".to_string()
        };
        let window = layer_shell_state.create_layer_surface(
            qh,
            wl_surface.clone(),
            Layer::Overlay,
            Some(app_id),
            None,
        );
        window.set_size(width, height);
        window.set_keyboard_interactivity(KeyboardInteractivity::OnDemand);
        if x_pos.is_some() || y_pos.is_some() {
            let x = x_pos.unwrap_or(0);
            let y = y_pos.unwrap_or(0);
            window.set_anchor(Anchor::TOP | Anchor::LEFT);
            window.set_margin(y, 0, 0, x);
        } else {
            window.set_anchor(Anchor::empty());
        }
        wl_surface.commit();

        let wayland_handle = Box::leak(Box::new(clear_ui::wayland::WaylandSurfaceHandle {
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
            source: wgpu::ShaderSource::Wgsl(clear_ui::SHADER.into()),
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

        let font_system = FontSystem::new();
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
        }));

        let mut apps = Vec::new();
        if mode == LauncherMode::Dmenu {
            let stdin_state_clone = stdin_state.clone();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    if let Ok(line) = line {
                        if let Ok(mut lock_state) = stdin_state_clone.lock() {
                            lock_state.items.push(line);
                            lock_state.new_data = true;
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

        let color_picker = if mode == LauncherMode::Color {
            let (r, g, b) = initial_hex
                .as_ref()
                .and_then(|h| parse_hex(h))
                .unwrap_or((0.5, 0.5, 0.5));
            let mut cp = ColorPickerWidget::new(r, g, b);
            cp.rebuild_layout(scale as f32);
            Some(cp)
        } else {
            None
        };

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

        let opacity = read_opacity_if_configured();

        let mut state = Self {
            window,
            wl_surface,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            vertex_count: 0,
            fuzzel,
            color_picker,
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
            opacity,
            fade_factor: 1.0,
            max_width: width,
            max_height: height,
            select_item,
            last_tick: std::time::Instant::now(),
            ui_context: clear_ui::context::UiContext::new(),
        };

        state.check_stdin_updates();
        state.update_desired_size();
        state.apply_layout();
        state.upload_vertices();
        state
    }

    fn check_stdin_updates(&mut self) -> bool {
        if let Ok(mut lock) = self.stdin_state.lock() {
            if lock.new_data {
                lock.new_data = false;
                let items = lock.items.clone();
                eprintln!("[cce-cloud debug] check_stdin_updates: items={:?}, select_item={:?}, currently selected={}", items, self.select_item, self.fuzzel.selected);
                self.fuzzel.set_items(items);
                if let Some(ref select_name) = self.select_item {
                    let select_lower = select_name.to_lowercase();
                    if let Some(idx) = self.fuzzel.filtered_items.iter().position(|item| item.to_lowercase() == select_lower) {
                        eprintln!("[cce-cloud debug] Found match for select_item {:?} at index {}, setting selected", select_name, idx);
                        self.fuzzel.selected = idx;
                        self.fuzzel.update_scroll();
                        self.fuzzel.snap_to_selected();
                        self.select_item = None;
                    } else {
                        eprintln!("[cce-cloud debug] No match found for select_item {:?} in filtered_items {:?}", select_name, self.fuzzel.filtered_items);
                    }
                }
                eprintln!("[cce-cloud debug] check_stdin_updates done: selected={}", self.fuzzel.selected);
                return true;
            }
        }
        false
    }

    fn update_desired_size(&mut self) {
        if self.mode == LauncherMode::Color || self.mode == LauncherMode::Json {
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
            self.window.set_size(target_width_u32, target_height_u32);
            self.wl_surface.commit();
            
            let pw = (target_width_u32 as f64 * self.scale) as u32;
            let ph = (target_height_u32 as f64 * self.scale) as u32;
            self.resize(pw, ph);
        }
    }

    fn apply_layout(&mut self) {
        let (w, h) = (self.width, self.height);
        if self.mode == LauncherMode::Color {
            if let Some(cp) = &mut self.color_picker {
                cp.rebuild_layout(self.scale as f32);
            }
        } else if self.mode == LauncherMode::Json {
            if let Some(jl) = &mut self.json_layout {
                clear_ui::scale::set_scale_factor(self.scale as f32);
                jl.set_rect(0.0, 0.0, w, h);
            }
        } else {
            self.fuzzel.set_rect(0.0, 0.0, w, h);
        }
    }

    fn collect_vertices(&self) -> Vec<Vertex> {
        let sw = self.width;
        let sh = self.height;
        if self.mode == LauncherMode::Color {
            if let Some(cp) = &self.color_picker {
                let pw = self.physical_width as f32;
                let ph = self.physical_height as f32;
                let mut verts = quad_vertices(0.0, 0.0, pw, ph, pw, ph, [0.05, 0.05, 0.08, self.opacity]).to_vec();
                verts.extend(cp.collect_vertices(pw, ph));
                verts
            } else {
                Vec::new()
            }
        } else if self.mode == LauncherMode::Json {
            if let Some(jl) = &self.json_layout {
                let mut verts = quad_vertices(0.0, 0.0, sw, sh, sw, sh, [0.05, 0.05, 0.08, self.opacity]).to_vec();
                verts.extend(widget_vertices(jl, sw, sh));
                verts
            } else {
                Vec::new()
            }
        } else {
            let mut verts = quad_vertices(0.0, 0.0, sw, sh, sw, sh, [0.05, 0.05, 0.08, self.opacity]).to_vec();
            verts.extend(widget_vertices(&self.fuzzel, sw, sh));
            verts
        }
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
            ref color_picker,
            ref json_layout,
            ref mode,
            ..
        } = self;

        let viewport = Resolution { width: *physical_width, height: *physical_height };
        text_viewport.update(queue, viewport);

        let scale_f32 = *scale as f32;

        let mut areas: Vec<TextArea> = Vec::new();
        let mut widget_buffers: Vec<Buffer> = Vec::new();
        let mut widget_labels: Vec<TextLabel> = Vec::new();

        let is_color_mode = *mode == LauncherMode::Color;
        let is_json_mode = *mode == LauncherMode::Json;
        if is_color_mode {
            if let Some(cp) = color_picker {
                for label in &cp.labels {
                    widget_labels.push(label.clone());
                }
            }
        } else if is_json_mode {
            if let Some(jl) = json_layout {
                for label in jl.text_labels() {
                    widget_labels.push(label);
                }
            }
        } else {
            for label in fuzzel.text_labels() {
                widget_labels.push(label);
            }
        }

        for label in &widget_labels {
            widget_buffers.push(make_text_buffer(font_system, &label.text, label.font_size));
        }

        for (buf, label) in widget_buffers.iter().zip(widget_labels.iter()) {
            let left = if is_color_mode { label.x.round() } else { (label.x * scale_f32).round() };
            let top = if is_color_mode { label.y.round() } else { (label.y * scale_f32).round() };
            let scale = if is_color_mode { 1.0 } else { scale_f32 };
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
            if self.mode == LauncherMode::Color {
                if let Some(cp) = &mut self.color_picker {
                    cp.rebuild_layout(self.scale as f32);
                }
            } else {
                self.apply_layout();
            }
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
                eprintln!("Surface error: {e:?}");
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

    window: Option<LayerSurface>,
    surface: Option<wl_surface::WlSurface>,

    state: Option<State>,
    exit: bool,
    redraw: bool,
    ctrl_pressed: bool,
    fade_out: bool,
    fade_start: Option<std::time::Instant>,
    fade_factor: f32,
}

impl AppState {
    fn trigger_close(&mut self) {
        if !self.fade_out {
            self.fade_out = true;
            self.fade_start = Some(std::time::Instant::now());
            self.redraw = true;
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
            if let Some(st) = &mut self.state {
                eprintln!("[cce-cloud pointer] Event: position={:?}, scale={}, kind={:?}", event.position, st.scale, event.kind);
                let (cx, cy) = clear_ui::wayland::scale_pointer_pos(event.position, st.scale);
                match &event.kind {
                    PointerEventKind::Motion { .. } => {
                        st.cursor_x = cx;
                        st.cursor_y = cy;
                        if st.mode == LauncherMode::Color {
                            let mut needs_rebuild = false;
                            if let Some(cp) = &mut st.color_picker {
                                cp.handle_cursor_moved(cx, cy);
                                if cp.needs_rebuild {
                                    cp.rebuild_layout(st.scale as f32);
                                    needs_rebuild = true;
                                }
                            }
                            if needs_rebuild {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        } else if st.mode == LauncherMode::Json {
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
                            if st.mode == LauncherMode::Color {
                                let mut needs_rebuild = false;
                                let mut action_requested = None;
                                let mut hex = String::new();
                                if let Some(cp) = &mut st.color_picker {
                                    cp.handle_mouse_input(clear_ui::widget::ElementState::Pressed);
                                    if cp.needs_rebuild {
                                        cp.rebuild_layout(st.scale as f32);
                                        needs_rebuild = true;
                                    }
                                    if let Some(action) = cp.action_requested {
                                        action_requested = Some(action);
                                        hex = cp.hex();
                                    }
                                }
                                if needs_rebuild {
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                                if let Some(action) = action_requested {
                                    match action {
                                        ColorAction::Apply => {
                                            println!("{}", hex);
                                            should_close = true;
                                        }
                                        ColorAction::Cancel => {
                                            should_close = true;
                                        }
                                    }
                                }
                            } else if st.mode == LauncherMode::Json {
                                let mut changed = false;
                                if let Some(jl) = &mut st.json_layout {
                                    if jl.mouse_input(
                                        clear_ui::widget::MouseButton::Left,
                                        clear_ui::widget::ElementState::Pressed,
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
                                    clear_ui::widget::MouseButton::Left,
                                    clear_ui::widget::ElementState::Pressed,
                                    event.position.0 as f32,
                                    event.position.1 as f32,
                                    &mut st.ui_context,
                                );
                                if changed {
                                    if st.fuzzel.selected == prev_selected {
                                        if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
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
                                                LauncherMode::Color => {}
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
                            if st.mode == LauncherMode::Color {
                                let mut needs_rebuild = false;
                                if let Some(cp) = &mut st.color_picker {
                                    cp.handle_mouse_input(clear_ui::widget::ElementState::Released);
                                    if cp.needs_rebuild {
                                        cp.rebuild_layout(st.scale as f32);
                                        needs_rebuild = true;
                                    }
                                }
                                if needs_rebuild {
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                            } else if st.mode == LauncherMode::Json {
                                let mut changed = false;
                                let mut clicked_btn_id = None;
                                if let Some(jl) = &mut st.json_layout {
                                    if jl.mouse_input(
                                        clear_ui::widget::MouseButton::Left,
                                        clear_ui::widget::ElementState::Released,
                                        event.position.0 as f32,
                                        event.position.1 as f32,
                                        &mut st.ui_context,
                                    ) {
                                        changed = true;
                                    }
                                    for w in &mut jl.widgets {
                                        if let Some(btn) = w.widget.as_any_mut().downcast_mut::<clear_ui::widget::Button>() {
                                            if btn.take_click() {
                                                clicked_btn_id = Some(w.id.clone());
                                                break;
                                            }
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
                                            if let Some(cb) = w.widget.as_any().downcast_ref::<clear_ui::widget::Checkbox>() {
                                                checkboxes.insert(w.id.clone(), cb.checked());
                                            } else if let Some(sb) = w.widget.as_any().downcast_ref::<clear_ui::widget::Spinbox>() {
                                                spinboxes.insert(w.id.clone(), sb.value);
                                            } else if let Some(cs) = w.widget.as_any().downcast_ref::<clear_ui::widget::ColorSelector>() {
                                                colors.insert(w.id.clone(), cs.color);
                                            } else if let Some(sl) = w.widget.as_any().downcast_ref::<clear_ui::widget::Slider>() {
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
                                    println!("{}", out_val.to_string());
                                    should_close = true;
                                }
                            }
                        }
                    }
                    PointerEventKind::Axis { horizontal, vertical, .. } => {
                        if st.mode == LauncherMode::Color {
                            let mut needs_rebuild = false;
                            if let Some(cp) = &mut st.color_picker {
                                let v_scroll = vertical.absolute as f32;
                                cp.handle_scroll(-v_scroll / 10.0);
                                if cp.needs_rebuild {
                                    cp.rebuild_layout(st.scale as f32);
                                    needs_rebuild = true;
                                }
                            }
                            if needs_rebuild {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        } else {
                            let h_scroll = horizontal.absolute as f32;
                            let v_scroll = vertical.absolute as f32;
                            let delta = clear_ui::widget::MouseScrollDelta::LineDelta(-h_scroll / 10.0, -v_scroll / 10.0);
                             if st.fuzzel.scroll_box.mouse_wheel(&delta, st.cursor_x, st.cursor_y, &mut st.ui_context) {
                                st.fuzzel.update_scroll();
                                st.upload_vertices();
                                self.redraw = true;
                            }
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
    ) {}

    fn leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _surface: &wl_surface::WlSurface,
        _serial: u32,
    ) {
        self.trigger_close();
    }

    fn press_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        event: smithay_client_toolkit::seat::keyboard::KeyEvent,
    ) {
        self.handle_key(event, clear_ui::widget::ElementState::Pressed);
    }

    fn release_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _keyboard: &wl_keyboard::WlKeyboard,
        _serial: u32,
        event: smithay_client_toolkit::seat::keyboard::KeyEvent,
    ) {
        self.handle_key(event, clear_ui::widget::ElementState::Released);
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
        self.ctrl_pressed = modifiers.ctrl;
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

impl AppState {
    fn handle_key(&mut self, event: smithay_client_toolkit::seat::keyboard::KeyEvent, state: clear_ui::widget::ElementState) {
        use clear_ui::widget::{Key, NamedKey};
        if state != clear_ui::widget::ElementState::Pressed {
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
            if st.mode == LauncherMode::Color {
                match &logical_key {
                    Key::Named(NamedKey::Escape) => {
                        should_close = true;
                    }
                    Key::Named(NamedKey::Enter) => {
                        if let Some(cp) = &st.color_picker {
                            println!("{}", cp.hex());
                        }
                        should_close = true;
                    }
                    _ => {
                        handled = false;
                    }
                }
            } else if st.mode == LauncherMode::Json {
                let mut widget_handled = false;
                let key_event = clear_ui::widget::KeyEvent {
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
                        if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
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
                                LauncherMode::Color => {}
                                LauncherMode::Json => {}
                            }
                            should_close = true;
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

fn main() {
    let mut prompt = "Search: ".to_string();
    let mut mode = if !io::stdin().is_terminal() {
        LauncherMode::Dmenu
    } else {
        LauncherMode::Path
    };
    let mut initial_hex: Option<String> = None;
    let mut x_pos: Option<i32> = None;
    let mut y_pos: Option<i32> = None;
    let mut select_item: Option<String> = None;

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
        } else if arg == "--mode" {
            if i + 1 < args.len() {
                let m = &args[i + 1];
                match m.as_str() {
                    "apps" | "app" => mode = LauncherMode::Apps,
                    "path" => mode = LauncherMode::Path,
                    "dmenu" => mode = LauncherMode::Dmenu,
                    "color" => mode = LauncherMode::Color,
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
        } else if arg == "--color" {
            mode = LauncherMode::Color;
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                initial_hex = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if arg == "--json" || arg == "--layout" {
            mode = LauncherMode::Json;
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
        redraw: true,
        ctrl_pressed: false,
        fade_out: false,
        fade_start: None,
        fade_factor: 1.0,
    };

    // Perform a roundtrip to populate output_state with active output scales
    event_queue.roundtrip(&mut app).unwrap();

    let scale = clear_ui::wayland::detect_scale_factor(&app.output_state);

    let state = pollster::block_on(State::new(
        &conn,
        &qh,
        &app.compositor_state,
        &app.layer_shell_state,
        prompt,
        stdin_sender,
        mode,
        initial_hex,
        x_pos,
        y_pos,
        scale,
        select_item,
        json_layout_config,
    ));

    app.window = Some(state.window.clone());
    app.surface = Some(state.wl_surface.clone());
    app.state = Some(state);

    let mut event_loop = calloop::EventLoop::try_new().unwrap();
    let loop_handle = event_loop.handle();

    WaylandSource::new(conn, event_queue).insert(loop_handle.clone()).unwrap();

    loop_handle.insert_source(stdin_channel, |event, _metadata, app_state: &mut AppState| {
        if let calloop::channel::Event::Msg(()) = event {
            if let Some(st) = &mut app_state.state {
                if st.check_stdin_updates() {
                    st.update_desired_size();
                    st.apply_layout();
                    st.upload_vertices();
                    app_state.redraw = true;
                }
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
        state.window.set_keyboard_interactivity(KeyboardInteractivity::None);
        state.wl_surface.commit();
    }
    drop(app);
    let _ = conn_clone.roundtrip();
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
        use clear_ui::widget::Element;

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
            },
        ];

        let config = JsonLayoutConfig {
            width: Some(300),
            height: Some(400),
            widgets: Some(widgets_conf),
            pages: None,
        };

        let mut layout = JsonLayoutWidget::new(&config);
        layout.set_rect(0.0, 0.0, 300.0, 400.0);
        let mut ctx = clear_ui::context::UiContext::new();

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

        assert!(layout.widgets[0].widget.as_any().downcast_ref::<clear_ui::widget::Label>().is_some());
        assert!(layout.widgets[1].widget.as_any().downcast_ref::<clear_ui::widget::Checkbox>().is_some());
        assert!(layout.widgets[2].widget.as_any().downcast_ref::<clear_ui::widget::Button>().is_some());

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
        assert_eq!(layout.widgets[1].widget.as_any().downcast_ref::<clear_ui::widget::Checkbox>().unwrap().checked(), false);

        // Simulate click on Checkbox row
        let changed = layout.mouse_input(
            clear_ui::widget::MouseButton::Left,
            clear_ui::widget::ElementState::Released,
            w_check_x + 5.0,
            w_check_y + 5.0,
            &mut ctx,
        );
        assert!(changed);
        assert_eq!(layout.widgets[1].widget.as_any().downcast_ref::<clear_ui::widget::Checkbox>().unwrap().checked(), true);

        // Simulate hover on button
        let changed_hover = layout.on_cursor_moved(w_btn_x + 10.0, w_btn_y + 10.0, &mut ctx);
        assert!(changed_hover);
        assert!(layout.widgets[2].widget.as_any().downcast_ref::<clear_ui::widget::Button>().unwrap().base().unwrap().hovered);
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

