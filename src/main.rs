mod scroll_region;
use scroll_region::ScrollRegion;

use std::sync::{Arc, Mutex};
use std::io::{self, BufRead, IsTerminal};

use cce_ui::widget::{WidgetHost, TextLabel};
use crate::json_layout::{JsonLayoutWidget, JsonLayoutConfig};
mod json_layout;
#[cfg(test)]
use crate::json_layout::JsonWidgetConfig;

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

use cce_ui::cosmic_text::{Attrs, Buffer, FontSystem, Metrics, SwashCache};

use cce_ui::vk::{Batch2D, Frame2D, ImageQuad, TextSpan, VkRenderer};

// Vertex is shared from the cce-ui engine.
pub(crate) use cce_ui::engine::Vertex;

/// Tessellate a display list into the flat vertex buffer plus the renderer
/// batches, converting the tessellator's logical-px `DlBatch` clips to
/// `Batch2D`'s physical px (the same mapping the engine runner applies). Going
/// through the display list — instead of the old `extra_quads` flattening —
/// is what lets bevel/recess prims survive to the tessellator.
fn tessellate(
    dl: &cce_ui::scene::paint::DisplayList,
    sw: f32, sh: f32,
    scale: f32,
) -> (Vec<Vertex>, Vec<Batch2D>, Vec<ImageQuad>, Vec<[f32; 12]>) {
    let (verts, dl_batches, dl_images, features) =
        cce_ui::backend::window_runner::tessellate_display_list(dl, sw, sh, scale);
    // Images ride a separate pipeline from the vertex batches, so they have to
    // be carried across explicitly — this return value used to be dropped and
    // `Frame2D::images` hardcoded to &[], which made `PaintCtx::image` a silent
    // no-op in this app while working fine in every engine-runner client.
    let images = dl_images
        .iter()
        .map(|di| ImageQuad {
            image: di.image,
            rect: (
                di.rect.x * scale,
                di.rect.y * scale,
                di.rect.width * scale,
                di.rect.height * scale,
            ),
            alpha: di.alpha,
            z_before: di.at,
            clip: di.clip.map(|c| {
                (
                    (c.x * scale).max(0.0) as u32,
                    (c.y * scale).max(0.0) as u32,
                    (c.width * scale) as u32,
                    (c.height * scale) as u32,
                )
            }),
        })
        .collect();
    let batches = dl_batches
        .iter()
        .map(|b| Batch2D {
            scissor: b.scissor.map(|c| {
                (
                    (c.x * scale).max(0.0) as u32,
                    (c.y * scale).max(0.0) as u32,
                    (c.width * scale) as u32,
                    (c.height * scale) as u32,
                )
            }),
            clip_rrect: b.clip_rrect.map(|c| {
                [c[0] * scale, c[1] * scale, c[2] * scale, c[3] * scale, c[4] * scale]
            }),
            start: b.start,
            end: b.end,
            plate: b.plate,
            blur_behind: b.blur_behind,
        })
        .collect();
    (verts, batches, images, features)
}

fn make_text_buffer(font_system: &mut FontSystem, text: &str, size: f32) -> Buffer {
    let metrics = Metrics::new(size, size * 1.4);
    let mut buffer = Buffer::new(font_system, metrics);
    let font_family = cce_ui::layout::control_label_font_parsed().0;
    let attrs = Attrs::new().family(cce_ui::cosmic_text::Family::Name(&font_family));
    buffer.set_text(font_system, text, attrs, cce_ui::cosmic_text::Shaping::Advanced);
    buffer.shape_until_scroll(font_system, true);
    buffer
}

/// A widget subtree's text via the paint walk (not the legacy text_labels getter),
/// reduced to the plain labels this renderer shapes: the buffer font and window bounds
/// stay exactly as before (make_text_buffer applies the control font to every label).
fn walk_text_labels(ui: &cce_ui::context::UiContext, w: &dyn WidgetHost) -> Vec<TextLabel> {
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
    let mut terminal = false;
    let mut icon = None;

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
                    "Icon" => {
                        if icon.is_none() && !value.is_empty() {
                            icon = Some(value.to_string());
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
                    "Terminal" => {
                        if value == "true" {
                            terminal = true;
                        }
                    }
                    // Hidden means "treat as deleted" — same outcome as
                    // NoDisplay for a launcher: the entry never shows.
                    "Hidden" => {
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
            return Some(AppInfo { name: n, exec: e, terminal, icon });
        }
    }
    None
}

fn scan_apps() -> Vec<AppInfo> {
    let mut apps = Vec::new();
    // XDG precedence: $XDG_DATA_HOME first, then each $XDG_DATA_DIRS entry in
    // order (defaults per the base-directory spec). Honoring XDG_DATA_DIRS is
    // what makes Flatpak/Snap exports visible.
    let mut dirs = Vec::new();
    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".local/share"))
        });
    if let Some(data_home) = data_home {
        dirs.push(data_home.join("applications"));
    }
    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "/usr/local/share:/usr/share".to_string());
    for dir in std::env::split_paths(&data_dirs) {
        if !dir.as_os_str().is_empty() {
            dirs.push(dir.join("applications"));
        }
    }

    // The first file claiming a desktop-file ID (the file stem) shadows that
    // ID in every later dir — even when the winning entry is itself
    // Hidden/NoDisplay, which is how a user entry deletes a system one.
    let mut seen_ids = std::collections::HashSet::new();
    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |ext| ext == "desktop") {
                    let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    if !seen_ids.insert(id.to_string()) {
                        continue;
                    }
                    if let Some(app) = parse_desktop_file(&path) {
                        apps.push(app);
                    }
                }
            }
        }
    }

    // Terminal=true apps need an emulator to host them; with none installed,
    // spawning them bare would fail silently, so drop the entries instead.
    if terminal_emulator().is_none() {
        apps.retain(|app| !app.terminal);
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
    spawn_detached("sh", &["-c", cmd]);
}

/// Launch an app entry; `Terminal=true` entries are hosted in a terminal
/// emulator (entries are dropped at scan time when none is installed).
fn spawn_app(app: &AppInfo) {
    if app.terminal {
        if let Some(term) = terminal_emulator() {
            spawn_detached(&term, &["sh", "-c", &format!("exec {}", app.exec)]);
            return;
        }
    }
    spawn_command(&app.exec);
}

/// Terminal used to host `Terminal=true` desktop entries: $TERMINAL if it
/// resolves on PATH (user env override), else the DE's configured default
/// (config.kdl `default_terminal`, written by the settings app's Default
/// Apps page), else foot. Callers pass the command positionally
/// (`term sh -c …`), not via `-e` — the convention every candidate must
/// accept (foot does natively; cce-terminal grew it alongside its entry).
fn terminal_emulator() -> Option<String> {
    std::env::var("TERMINAL")
        .ok()
        .filter(|t| !t.is_empty() && command_in_path(t))
        .or_else(|| {
            cce_ui::config::get_string("/default_terminal")
                .filter(|t| !t.is_empty() && command_in_path(t))
        })
        .or_else(|| command_in_path("foot").then(|| "foot".to_string()))
}

fn command_in_path(cmd: &str) -> bool {
    if cmd.contains('/') {
        return std::path::Path::new(cmd).is_file();
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| dir.join(cmd).is_file())
    })
}

fn spawn_detached(program: &str, args: &[&str]) {
    // Launched apps must outlive this daemon: process_group(0) moves them out
    // of our process group so a terminal ^C (manual daemon run) doesn't kill
    // them, and cce-cloud.service sets KillMode=process so a service restart
    // doesn't cgroup-kill them either (systemd kills by cgroup, which no
    // amount of setsid/double-fork escapes).
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(program);
    cmd.args(args).process_group(0);
    // Per-user runtime dir, not /tmp: this records every app the launcher
    // starts and captures their stdout/stderr, so a fixed /tmp path is both a
    // collision between users and a readable trace of one user's activity.
    let log_path = cce_ui::config::cce_runtime_dir().join("spawn.log");
    if let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        let mut f = file;
        use std::io::Write;
        let _ = writeln!(f, "[spawn] executing: {} {}", program, args.join(" "));
        cmd.stdout(f.try_clone().unwrap()).stderr(f);
    }
    // Through the toolkit's reaping spawn, NOT a bare `cmd.spawn()`: the daemon
    // lives for the whole session, and a dropped Child handle means every app
    // it ever launched sits in the process table as a zombie once it exits —
    // unreadable in /proc and reported "alive" by kill(pid, 0) probes.
    let _ = cce_ui::process::spawn_detached(cmd);
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
    /// Item text → `(image id, px w, px h)` for the rows that resolved an icon.
    /// Keyed by text rather than index because filtering rebuilds the index
    /// space on every keystroke while the text is what identifies a row.
    icons: std::collections::HashMap<String, (u32, u32, u32)>,
    /// Width reserved for the icon column, 0 when no row has an icon. Applied to
    /// every row, not just the ones that resolved, so a list with one missing
    /// icon keeps a straight text edge instead of ragging in and out.
    icon_gutter: f32,
    pub scroll_box: ScrollRegion,
}

/// Icon edge length inside a 25px row, and the gap between it and the label.
const ICON_PX: f32 = 17.0;
const ICON_GAP: f32 = 8.0;

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
            icons: std::collections::HashMap::new(),
            icon_gutter: 0.0,
            scroll_box: ScrollRegion::new(22.0, 0.0),
        })
    }

    pub fn set_items(&mut self, items: Vec<String>) {
        self.all_items = items;
        self.filter();
    }

    /// Give rows an icon column. Only Apps mode calls this — Dmenu/Path items
    /// are arbitrary strings with nothing to look an icon up by, and they keep
    /// the flush-left layout they have always had because the gutter stays 0.
    pub fn set_item_icons(&mut self, icons: std::collections::HashMap<String, (u32, u32, u32)>) {
        self.icon_gutter = if icons.is_empty() { 0.0 } else { ICON_PX + ICON_GAP };
        self.icons = icons;
    }

    /// The square an icon is fitted into for the row drawn at `draw_y`.
    fn icon_rect(&self, draw_y: f32, item_h: f32, w: u32, h: u32) -> cce_ui::scene::layout::Rect {
        let pad = 15.0;
        // Fit the longer side to ICON_PX so a non-square icon keeps its aspect
        // ratio and stays centered in the column.
        let (w, h) = (w.max(1) as f32, h.max(1) as f32);
        let s = ICON_PX / w.max(h);
        let (iw, ih) = (w * s, h * s);
        cce_ui::scene::layout::Rect {
            x: self.x + pad + 10.0 + (ICON_PX - iw) / 2.0,
            y: draw_y + (item_h - ih) / 2.0,
            width: iw,
            height: ih,
        }
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

        // Search bar — a well recessed into the plate, its rim lit in the
        // highlight accent (the toolkit's focused-well treatment; the query
        // line always holds keyboard focus here). Replaces the flat fill +
        // 1px border quads.
        let well = Rect { x: self.x + pad, y: self.y + pad, width: self.w - pad * 2.0, height: search_h };
        ctx.quad(well, [0.10, 0.10, 0.14, 1.0]);
        let depth = cce_ui::layout::bevel_width().min(search_h * 0.2);
        let hc = cce_ui::color::highlight_primary_color();
        ctx.recess_tinted(well, (0.0, 0.0, 0.0, 0.0), depth, [hc[0], hc[1], hc[2]]);

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
                // A raised beveled chip, not a flat tint: the selection reads
                // as sitting proud of the list the way focused panes do.
                let sel = Rect {
                    x: self.x + pad + 2.0,
                    y: draw_y,
                    width: self.w - pad * 2.0 - 4.0 - scrollbar_w,
                    height: item_h - 2.0,
                };
                let depth = cce_ui::color::plate_bevel_width().min(sel.height * 0.2);
                ctx.bevel(sel, (4.0, 4.0, 4.0, 4.0), [0.20, 0.35, 0.65, 0.9], depth);
            }
        }

        // App icons, on the same virtualization predicate as the labels below:
        // only rows `get_draw_y` places inside the viewport are emitted, so a
        // 300-app list still costs one image quad per visible row.
        if self.icon_gutter > 0.0 {
            for (idx, item_text) in self.filtered_items.iter().enumerate() {
                let Some((image, iw, ih)) = self.icons.get(item_text).copied() else { continue };
                if let Some(draw_y) = self.scroll_box.get_draw_y(idx as f32 * item_h, item_h) {
                    ctx.image(image, self.icon_rect(draw_y, item_h, iw, ih), 1.0);
                }
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
            // `hit()` spans the whole region, scrollbar strip included, and the row math
            // below accepts any y inside it — so without these two gates a press on the
            // scrollbar, or on the partially-clipped sliver at the viewport edge, resolved
            // to a row. In Dmenu/switcher mode a press commits and closes, so that emitted
            // an item the user never clicked (and, on the sliver, never even saw).
            if self.scroll_box.hit(*px, *py) && !self.scroll_box.hit_scrollbar(*px, *py) {
                let click_virtual_y = *py - self.scroll_box.viewport_y + self.scroll_box.scroll_y;
                let clicked_idx = (click_virtual_y / item_h).floor() as usize;
                // Same predicate the paint loop virtualizes on, so only a row actually
                // drawn this frame is selectable.
                if clicked_idx < self.filtered_items.len()
                    && self
                        .scroll_box
                        .get_draw_y(clicked_idx as f32 * item_h, item_h)
                        .is_some()
                {
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
    terminal: bool,
    /// The entry's `Icon=` key, resolved against the icon theme at display time.
    /// A theme name (`cce-files`), or an absolute path — both are legal per the
    /// desktop-entry spec, and `cce_ui::icon` handles the distinction.
    icon: Option<String>,
}

struct StdinState {
    items: Vec<String>,
    new_data: bool,
    cycle_next: usize,
    cycle_prev: usize,
    select_and_close: bool,
    // daemon mode: the requesting client hung up (killed/crashed) — the
    // popup has no owner left and must close, or the serial accept loop
    // in run_daemon stays wedged on it forever
    client_gone: bool,
}



#[derive(Clone)]
#[allow(dead_code)]
enum AppWindow {
    Layer(LayerSurface),
    Xdg(XdgWindow),
}

/// Logical-px breathing room kept between a positioned popup and the screen edge.
const EDGE_GAP: i32 = 8;

/// Logical geometry `(x, y, w, h)` of the output containing the point `(x, y)`, or —
/// when the point is off every output — the first one that advertises a geometry.
/// `None` if no output does (nothing to clamp against; the request is used raw).
fn output_bounds_at(output_state: &OutputState, x: i32, y: i32) -> Option<(i32, i32, i32, i32)> {
    let mut fallback = None;
    for output in output_state.outputs() {
        let Some(info) = output_state.info(&output) else { continue };
        let (Some((ox, oy)), Some((ow, oh))) = (info.logical_position, info.logical_size) else {
            continue;
        };
        if (ox..ox + ow).contains(&x) && (oy..oy + oh).contains(&y) {
            return Some((ox, oy, ow, oh));
        }
        fallback.get_or_insert((ox, oy, ow, oh));
    }
    fallback
}

/// Where a `-x/-y` popup wants to sit, and the output it must stay inside.
///
/// The requested point is a cursor position (the compositor passes the pointer
/// straight through for the desktop/window context menus), so the fit rule is the
/// usual menu one: grow away from the anchor, **flip** to the other side of it when
/// the window would overhang, and clamp only when it fits on neither side.
///
/// The flip decision latches for the life of the popup. The window auto-sizes to its
/// content continuously (`update_desired_size`), so re-deciding on every resize makes
/// a filtering list snap back and forth across the cursor.
///
/// `bounds` is the whole output, not the layer-shell *usable* area — which is why the
/// surface asks for `exclusive_zone(-1)`. Without it a panel's exclusive zone would
/// shrink the box the compositor places against while this math still used the full
/// output, and the clamp would be wrong by exactly the panel's height.
#[derive(Clone, Copy, Debug)]
struct Placement {
    /// Requested anchor, in layout (logical) coordinates.
    x: i32,
    y: i32,
    /// `--align-right`: the anchor is `x` in from the right edge and the window
    /// grows leftward from it.
    align_right: bool,
    /// The output to stay inside, as `(x, y, w, h)` in logical coords. `None` when no
    /// output advertised a logical geometry — then the raw request is used unchanged.
    bounds: Option<(i32, i32, i32, i32)>,
    flip_x: Option<bool>,
    flip_y: Option<bool>,
}

impl Placement {
    fn new(x: i32, y: i32, align_right: bool, bounds: Option<(i32, i32, i32, i32)>) -> Self {
        Self { x, y, align_right, bounds, flip_x: None, flip_y: None }
    }

    /// Anchor + `(top, right, bottom, left)` margins for a `w`x`h` logical-px layer
    /// surface. Always top-left anchored when the output is known: the margins are
    /// recomputed on every resize anyway, so a left-growing popup is expressed by
    /// moving its left edge rather than by anchoring the right one.
    fn resolve(&mut self, w: i32, h: i32) -> (Anchor, (i32, i32, i32, i32)) {
        let Some((ox, oy, ow, oh)) = self.bounds else {
            return if self.align_right {
                (Anchor::TOP | Anchor::RIGHT, (self.y, self.x, 0, 0))
            } else {
                (Anchor::TOP | Anchor::LEFT, (self.y, 0, 0, self.x))
            };
        };

        // Output-local anchor. In align-right mode `x` is measured from the right
        // edge and the window hangs to the left of the point.
        let anchor_x = if self.align_right { ow - self.x } else { self.x - ox };
        let (nat_x, alt_x) = if self.align_right {
            (anchor_x - w, anchor_x)
        } else {
            (anchor_x, anchor_x - w)
        };
        let left = Self::fit(&mut self.flip_x, nat_x, alt_x, w, ow);
        let top = Self::fit(&mut self.flip_y, self.y - oy, self.y - oy - h, h, oh);

        (Anchor::TOP | Anchor::LEFT, (top, 0, 0, left))
    }

    /// Pick between the natural and flipped edge for one axis, then clamp into
    /// `[EDGE_GAP, extent - size - EDGE_GAP]`. Latches the choice in `flip`.
    fn fit(flip: &mut Option<bool>, natural: i32, flipped: i32, size: i32, extent: i32) -> i32 {
        let fits = |start: i32| start >= EDGE_GAP && start + size <= extent - EDGE_GAP;
        let flipped_is_better = *flip.get_or_insert(!fits(natural) && fits(flipped));
        let start = if flipped_is_better { flipped } else { natural };
        // A window taller/wider than the output has no in-range clamp; pin it to the
        // near edge rather than letting max() invert the range.
        start.clamp(EDGE_GAP, (extent - size - EDGE_GAP).max(EDGE_GAP))
    }
}

struct State {
    window: Option<AppWindow>,
    wl_surface: wl_surface::WlSurface,
    // Option: dropped explicitly in Drop, before the wl_surface is destroyed
    // (the swapchain must not outlive its Wayland surface).
    renderer: Option<VkRenderer>,
    vertex_data: Vec<Vertex>,
    frame_batches: Vec<Batch2D>,
    frame_images: Vec<ImageQuad>,
    plate_features: Vec<[f32; 12]>,

    fuzzel: cce_ui::widget::Adapted<FuzzelWidget>,
    json_layout: Option<cce_ui::widget::Adapted<JsonLayoutWidget>>,
    font_system: FontSystem,
    swash_cache: SwashCache,

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
    /// `Some` for `-x/-y` popups (always layer-shell): re-applied on every resize so
    /// an auto-sizing window can't grow off the screen edge.
    placement: Option<Placement>,
}

/// Logical-px width one JSON-layout widget wants for the auto-sizing popup.
///
/// Buttons are the subtlety: `cce_ui::widget::Button` draws its label with an 8px inset on
/// each side (see `Button::paint`) in the *button* font — not the menubar font `measure_text`
/// assumes. So measure the label the way the button itself does (`measure_text_width` in the
/// button font/size) and budget the button's 16px of inset on top of the container's 16px-per-
/// side margins; otherwise a left-justified label starts 8px in and spills past the button's
/// right edge (`JsonLayoutWidget::layout_children` sets `usable_w = width - 32`).
fn json_widget_desired_width(widget_type: &str, text: &str) -> f32 {
    match widget_type {
        "button" => {
            let (family, size) = cce_ui::layout::parse_font_string(&cce_ui::layout::button_font());
            cce_ui::widget::display::measure_text_width(text, &family, size.unwrap_or(12.0)) + 48.0
        }
        "label" => cce_ui::widget::display::measure_text(text, 13.0) + 32.0,
        "checkbox" => cce_ui::widget::display::measure_text(text, 13.0) + 44.0,
        "spinbox" | "color" => cce_ui::widget::display::measure_text(text, 13.0) + 120.0,
        "slider" => cce_ui::widget::display::measure_text(text, 13.0) + 160.0,
        _ => 150.0,
    }
}

impl State {
    fn new(
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
        output_bounds: Option<(i32, i32, i32, i32)>,
        scale: f64,
        select_item: Option<String>,
        switcher_mode: bool,
        json_layout_config: Option<JsonLayoutConfig>,
        parent_app_id: Option<String>,
        fonts: Option<(FontSystem, SwashCache)>,
    ) -> (Self, Option<cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1>) {
        let t_start = std::time::Instant::now();
        cce_ui::scale::set_scale_factor(scale as f32);
        let (width, height) = if mode == LauncherMode::Json {
            if let Some(ref config) = json_layout_config {
                let w = config.width.unwrap_or_else(|| {
                    let mut max_widget_w = 120.0f32; // fallback minimum
                    if let Some(ref widgets) = config.widgets {
                        for w_conf in widgets {
                            let w_w = json_widget_desired_width(&w_conf.widget_type, &w_conf.text);
                            if w_w > max_widget_w {
                                max_widget_w = w_w;
                            }
                        }
                    } else if let Some(ref pages) = config.pages {
                        for page in pages {
                            for w_conf in &page.widgets {
                                let w_w = json_widget_desired_width(&w_conf.widget_type, &w_conf.text);
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
        log::debug!("[timing] size estimation: {:?}", t_start.elapsed());

        let t = std::time::Instant::now();
        let wl_surface = compositor_state.create_surface(qh);
        wl_surface.set_buffer_scale(scale as i32);
        let app_id = if let Some(ref parent) = parent_app_id {
            format!("cce-cloud:{}", parent)
        } else {
            "cce-cloud".to_string()
        };

        let placement = (x_pos.is_some() || y_pos.is_some()).then(|| {
            Placement::new(x_pos.unwrap_or(0), y_pos.unwrap_or(0), align_right, output_bounds)
        });

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
            if placement.is_none() {
                layer_window.set_anchor(Anchor::empty());
            } else {
                // Place against the full output, not the area left over by panels:
                // `Placement` clamps against the wl_output geometry, and the two must
                // agree on the box or the clamp is off by the panel's exclusive zone.
                layer_window.set_exclusive_zone(-1);
                // The anchor/margins are left to the first `apply_placement()`, once
                // the content has actually been measured — the size passed above is a
                // pre-layout estimate, and latching the flip decision on it would flip
                // menus that fit and clip ones that don't.
            }
            wl_surface.commit();
            AppWindow::Layer(layer_window)
        };

        log::debug!("[timing] surface/window setup: {:?}", t.elapsed());

        // Raw-Vulkan renderer on the same display/surface pointers the wgpu
        // stack used. Corner radius 0: the window background tessellates its own
        // rounded corners (rounded_rect_vertices_corners).
        let t = std::time::Instant::now();
        let renderer = unsafe {
            VkRenderer::new(
                conn.backend().display_id().as_ptr() as *mut std::ffi::c_void,
                wl_surface.id().as_ptr() as *mut std::ffi::c_void,
                pw,
                ph,
                0.0,
            )
        };
        log::debug!("[timing] VkRenderer::new: {:?}", t.elapsed());

        // Reuse the daemon's font system across popups (a rebuild re-scans the
        // fonts dir and loses the shaping caches).
        let t = std::time::Instant::now();
        let (font_system, swash_cache) =
            fonts.unwrap_or_else(|| (cce_ui::create_font_system(), SwashCache::new()));
        log::debug!("[timing] font system: {:?}", t.elapsed());

        let mut fuzzel = FuzzelWidget::new(prompt);
        fuzzel.set_rect(0.0, 0.0, lw, lh);

        let stdin_state = Arc::new(Mutex::new(StdinState {
            items: Vec::new(),
            new_data: false,
            cycle_next: 0,
            cycle_prev: 0,
            select_and_close: false,
            client_gone: false,
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
                            } else if line == "__cce_switcher_prev__" {
                                lock_state.cycle_prev += 1;
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

            // Resolve every entry's Icon= against the icon theme. Uploads are
            // per-popup by design (see cce_ui::icon::upload_themed): the daemon
            // tears its VkRenderer down between popups, so an id cached across
            // them would name freed GPU resources. Only the decode is cached, so
            // the second open of the launcher skips the disk and the rasterizer.
            let t_icons = std::time::Instant::now();
            let icons: std::collections::HashMap<String, (u32, u32, u32)> = apps
                .iter()
                .filter_map(|app| {
                    let name = app.icon.as_deref()?;
                    let img = cce_ui::icon::upload_themed(name, ICON_PX.ceil() as u32 * 2)?;
                    Some((app.name.clone(), img))
                })
                .collect();
            log::debug!(
                "[timing] app icons: {} of {} resolved in {:?}",
                icons.len(),
                apps.len(),
                t_icons.elapsed()
            );
            fuzzel.set_item_icons(icons);

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
        let window_radius = cce_ui::color::root_plate_corner_radius();

        let mut state = Self {
            window: Some(window),
            wl_surface,
            renderer: Some(renderer),
            vertex_data: Vec::new(),
            frame_batches: Vec::new(),
            frame_images: Vec::new(),
            plate_features: Vec::new(),
            fuzzel,
            json_layout,
            font_system,
            swash_cache,
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
            // For json mode the initial `height` is a crude pre-layout estimate
            // (it ignores per-widget label offsets), so it must not double as the
            // growth cap — the accurately measured page height would be clipped
            // against it. Cap at the caller's explicit height when given, else a
            // sane maximum; update_desired_size resizes to the measured content
            // within that.
            max_height: if mode == LauncherMode::Json {
                json_layout_config.as_ref().and_then(|c| c.height).unwrap_or(600)
            } else {
                height
            },
            select_item,
            switcher_mode,
            last_tick: std::time::Instant::now(),
            ui_context: cce_ui::context::UiContext::new(),
            window_rect,
            window_bg: bg_color,
            window_radius,
            select_and_close_requested: false,
            placement,
        };

        let t = std::time::Instant::now();
        state.check_stdin_updates();
        state.update_desired_size();
        // update_desired_size only re-places when the size actually moved off the
        // estimate; a popup that happened to be estimated exactly still needs its
        // first anchor.
        state.apply_placement();
        state.apply_layout();
        state.upload_vertices();
        log::debug!("[timing] initial layout/upload: {:?}", t.elapsed());
        log::debug!("[timing] State::new total: {:?}", t_start.elapsed());
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
                let cycles_back = lock.cycle_prev;
                lock.cycle_prev = 0;

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

                if (cycles > 0 || cycles_back > 0) && !self.fuzzel.filtered_items.is_empty() {
                    let len = self.fuzzel.filtered_items.len() as isize;
                    let net = cycles as isize - cycles_back as isize;
                    self.fuzzel.selected =
                        (self.fuzzel.selected as isize + net).rem_euclid(len) as usize;
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
                    let w_w = json_widget_desired_width(&w.widget_type, &w.text);
                    if w_w > max_widget_w {
                        max_widget_w = w_w;
                    }
                }
                
                let target_height = jl.page_total_heights[active_page].min(self.max_height as f32);
                let target_width = max_widget_w.clamp(120.0, self.max_width as f32);
                
                self.resize_window(target_width.round() as u32, target_height.round() as u32);
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

        self.resize_window(target_width.round() as u32, target_height.round() as u32);
    }

    /// Grow/shrink the window to `w`x`h` logical px, and re-place it so the new size
    /// still fits on screen. No-op when the size is unchanged.
    fn resize_window(&mut self, w: u32, h: u32) {
        if self.width as u32 == w && self.height as u32 == h {
            return;
        }
        if let Some(AppWindow::Layer(ref layer)) = self.window {
            layer.set_size(w, h);
        }
        let pw = (w as f64 * self.scale) as u32;
        let ph = (h as f64 * self.scale) as u32;
        self.resize(pw, ph);
        // After `resize`, so the placement sees the size it is fitting.
        self.apply_placement();
        self.wl_surface.commit();
    }

    /// Re-anchor a `-x/-y` popup for its current size. Popups without an explicit
    /// position are centered by the compositor and xdg toplevels are placed by it, so
    /// both are left alone.
    fn apply_placement(&mut self) {
        let Some(ref mut placement) = self.placement else { return };
        let Some(AppWindow::Layer(ref layer)) = self.window else { return };
        let (anchor, (top, right, bottom, left)) =
            placement.resolve(self.width.round() as i32, self.height.round() as i32);
        layer.set_anchor(anchor);
        layer.set_margin(top, right, bottom, left);
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

    fn collect_display_list(&self) -> cce_ui::scene::paint::DisplayList {
        use cce_ui::scene::layout::Rect;
        let mut pc = cce_ui::scene::paint::PaintCtx::new();

        // 1. Window background — the dissolved Backplate's emission (base color at
        // the configured backplate opacity), now as a beveled plate: the rolled
        // rim makes the popup read as a raised surface instead of a flat sheet.
        let mut bg_color = self.window_bg;
        if bg_color[3] > 0.001 {
            bg_color[3] = cce_ui::color::root_plate_opacity();
        }
        if bg_color[3] > 0.0 {
            let r = if self.window_radius > 0.1 { self.window_radius } else { 0.0 };
            let (wx, wy, ww, wh) = self.window_rect;
            pc.bevel(
                Rect { x: wx, y: wy, width: ww, height: wh },
                (r, r, r, r),
                bg_color,
                cce_ui::color::plate_bevel_width(),
            );
        }

        // 2. Child widgets, through the paint walk: bevel/recess prims reach the
        // tessellator instead of being flattened away by the legacy quad bridges.
        if self.mode == LauncherMode::Json {
            if let Some(jl) = &self.json_layout {
                jl.paint_self(&self.ui_context, &mut pc);
            }
        } else {
            self.fuzzel.paint_self(&self.ui_context, &mut pc);
        }

        pc.finish()
    }

    fn upload_vertices(&mut self) {
        let dl = self.collect_display_list();
        let (mut verts, mut batches, mut images, features) =
            tessellate(&dl, self.width, self.height, self.scale as f32);
        if self.fade_factor < 1.0 {
            for v in &mut verts {
                v.color[3] *= self.fade_factor;
            }
            // Images carry their own alpha rather than a vertex color, so they
            // need the fade applied here too or every app icon would stay fully
            // opaque over a list dissolving underneath it.
            for i in &mut images {
                i.alpha *= self.fade_factor;
            }
            // SDF-plate overlays (the recess rims) are shader-lit, not
            // vertex-alpha — drop them during the close fade rather than let
            // their shading linger at full strength over fading geometry.
            batches.retain(|b| b.plate.is_none());
        }
        // The GPU upload happens in VkRenderer::draw_frame_2d, which consumes
        // vertex_data every frame.
        self.vertex_data = verts;
        self.frame_batches = batches;
        self.frame_images = images;
        self.plate_features = features;
    }

    fn prepare_text(&mut self) {
        let scale_f32 = self.scale as f32;

        let mut widget_labels: Vec<TextLabel> = Vec::new();
        if self.mode == LauncherMode::Json {
            if let Some(jl) = &self.json_layout {
                widget_labels.extend(walk_text_labels(&self.ui_context, jl));
            }
        } else {
            widget_labels.extend(walk_text_labels(&self.ui_context, &self.fuzzel));
        }

        let mut buffers: Vec<Buffer> = Vec::with_capacity(widget_labels.len());
        for label in &widget_labels {
            buffers.push(make_text_buffer(&mut self.font_system, &label.text, label.font_size));
        }

        let alpha = self.fade_factor.clamp(0.0, 1.0);
        let spans: Vec<TextSpan> = buffers
            .iter()
            .zip(widget_labels.iter())
            .map(|(buf, label)| TextSpan {
                buffer: buf,
                left: (label.x * scale_f32).round(),
                top: (label.y * scale_f32).round(),
                // Buffers are shaped at logical size; the span scales to physical.
                scale: scale_f32,
                bounds: None,
                default_color: [
                    label.color[0] as f32 / 255.0,
                    label.color[1] as f32 / 255.0,
                    label.color[2] as f32 / 255.0,
                    alpha,
                ],
                rotation: None,
                clip_circle: [0.0; 3],
                clip_extents: [0.0; 2],
            })
            .collect();

        self.renderer.as_mut().unwrap().prepare_text(
            &mut self.font_system,
            &mut self.swash_cache,
            &spans,
        );
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.physical_width = width;
            self.physical_height = height;
            self.width = width as f32 / self.scale as f32;
            self.height = height as f32 / self.scale as f32;
            self.renderer.as_mut().unwrap().resize(width, height);
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
        self.renderer.as_mut().unwrap().draw_frame_2d(Frame2D {
            verts: &self.vertex_data,
            batches: &self.frame_batches,
            overlay_verts: &[],
            images: &self.frame_images,
            plate_features: &self.plate_features,
            clear_color: [0.0; 4],
        });
        false
    }
}

impl Drop for State {
    fn drop(&mut self) {
        // Swapchain/device teardown must precede the wl_surface's destruction
        // (daemon mode churns States, one per popup).
        self.renderer.take();
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
                                spawn_app(app);
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
                // Surface-local LOGICAL coords. This app's widget geometry is logical (the
                // window tracks `width / scale`), and every other consumer — the Json
                // dispatch and `FuzzelWidget::on_event` below — already uses the raw
                // `event.position`. `scale_pointer_pos` multiplies by the scale, so feeding
                // its result to the scroll region compared PHYSICAL cursor coords against a
                // LOGICAL rect: on a scale-2 output the wheel silently stopped working past
                // the list's midpoint (cursor at logical x=250 arrived as 500 against a rect
                // ending at 285, so `hit()` was false and nothing scrolled).
                let (cx, cy) = (event.position.0 as f32, event.position.1 as f32);
                match &event.kind {
                    PointerEventKind::Motion { .. } => {
                        st.cursor_x = cx;
                        st.cursor_y = cy;
                        if st.mode == LauncherMode::Json {
                            let mut changed = false;
                            if let Some(jl) = &mut st.json_layout {
                                // Routed dispatch (6bd shrink): one Event through the router.
                                let mv = cce_ui::widget::Event::PointerMove {
                                    x: event.position.0 as f32,
                                    y: event.position.1 as f32,
                                    local_x: event.position.0 as f32,
                                    local_y: event.position.1 as f32,
                                };
                                let root = jl.id();
                                st.ui_context.register_widget(root, jl.as_ptr_mut());
                                if st.ui_context.propagate_event(&mv, root) {
                                    changed = true;
                                }
                            }
                            if changed {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        } else if st.fuzzel.scroll_box.cursor_moved(cx, cy) {
                            // Returns true only while a thumb drag is live; it also keeps
                            // `hovered` current for the wheel/keyboard scope either way.
                            st.fuzzel.update_scroll();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    PointerEventKind::Press { button, .. } => {
                        if *button == 272 {
                            st.cursor_x = cx;
                            st.cursor_y = cy;
                            if st.mode == LauncherMode::Json {
                                let mut changed = false;
                                if let Some(jl) = &mut st.json_layout {
                                    let ev = cce_ui::widget::Event::MouseButton {
                                        button: cce_ui::widget::MouseButton::Left,
                                        state: cce_ui::widget::ElementState::Pressed,
                                        x: event.position.0 as f32,
                                        y: event.position.1 as f32,
                                        local_x: event.position.0 as f32,
                                        local_y: event.position.1 as f32,
                                    };
                                    let root = jl.id();
                                    st.ui_context.register_widget(root, jl.as_ptr_mut());
                                    if st.ui_context.propagate_event(&ev, root) {
                                        changed = true;
                                    }
                                }
                                if changed {
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                            } else if st.fuzzel.scroll_box.press(cx, cy) {
                                // Thumb grab or track jump. This must be handled here rather
                                // than inside `on_event`, because the row branch below treats
                                // ANY handled press as a selection and — in Dmenu/switcher
                                // mode — commits it and closes the popup. A scrollbar press
                                // must scroll, not choose.
                                st.fuzzel.update_scroll();
                                st.upload_vertices();
                                self.redraw = true;
                            } else {
                                let prev_selected = st.fuzzel.selected;
                                let changed = {
                                    let ev = cce_ui::widget::Event::MouseButton {
                                        button: cce_ui::widget::MouseButton::Left,
                                        state: cce_ui::widget::ElementState::Pressed,
                                        x: event.position.0 as f32,
                                        y: event.position.1 as f32,
                                        local_x: event.position.0 as f32,
                                        local_y: event.position.1 as f32,
                                    };
                                    let root = st.fuzzel.id();
                                    st.ui_context.register_widget(root, st.fuzzel.as_ptr_mut());
                                    st.ui_context.propagate_event(&ev, root)
                                };
                                if changed {
                                    // A single click launches — the fuzzel on_event only
                                    // reports presses it resolved to a really-drawn row
                                    // (scrollbar and clipped-sliver presses never get
                                    // here), so the click IS the choice, exactly as Enter.
                                    // (The old gate gated Apps/Path on `selected ==
                                    // prev_selected`, which read as the click doing
                                    // nothing.) The SWITCHER keeps two-click: its rows are
                                    // live windows, and focusing one on a stray first
                                    // click would be destructive — click to inspect the
                                    // selection, click it again to commit.
                                    let commit = !st.switcher_mode || st.fuzzel.selected == prev_selected;
                                    if commit {
                                        if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
                                            println!("{}", item);
                                            self.selected_item = Some(item.clone());
                                            match st.mode {
                                                LauncherMode::Apps => {
                                                    if let Some(app) = st.apps.iter().find(|app| &app.name == item) {
                                                        record_app_launch(&app.name);
                                                        spawn_app(app);
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
                                    let ev = cce_ui::widget::Event::MouseButton {
                                        button: cce_ui::widget::MouseButton::Left,
                                        state: cce_ui::widget::ElementState::Released,
                                        x: event.position.0 as f32,
                                        y: event.position.1 as f32,
                                        local_x: event.position.0 as f32,
                                        local_y: event.position.1 as f32,
                                    };
                                    let root = jl.id();
                                    st.ui_context.register_widget(root, jl.as_ptr_mut());
                                    if st.ui_context.propagate_event(&ev, root) {
                                        changed = true;
                                    }
                                    for w in &mut jl.widgets {
                                        // take_click is an WidgetHost method; Phase 5 Buttons are
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
                                            if let Some(cb) = w.widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Checkbox>() {
                                                checkboxes.insert(w.id.clone(), cb.checked());
                                            } else if let Some(sb) = w.widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Spinbox>() {
                                                spinboxes.insert(w.id.clone(), sb.value);
                                            } else if let Some(cs) = w.widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::ColorSelector>() {
                                                colors.insert(w.id.clone(), cs.color);
                                            } else if let Some(sl) = w.widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Slider>() {
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
                            } else if st.fuzzel.scroll_box.release() {
                                // Ends a thumb drag. Returns true only if one was live, so a
                                // plain click on a row is unaffected.
                                st.upload_vertices();
                                self.redraw = true;
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
                // Re-assert the content-derived size. The compositor's overlay
                // fresh-slot configure arrives full-height (cce-cloud app_ids are
                // mode-forced to Overlay); obeying it verbatim left --json popups
                // as a monitor-tall strip. Dmenu mode always recovered because
                // every stdin batch re-runs this — json got sized exactly once,
                // before the configure. The commit below updates box_geom, which
                // the compositor's stored-geometry path then respects.
                state.update_desired_size();
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

        // Shift+Tab arrives as ISO_Left_Tab: cycle the highlight backwards
        // with wrap, mirroring Tab's forward cycle below.
        if event.keysym == xkeysym::Keysym::ISO_Left_Tab {
            if let Some(st) = &mut self.state {
                if st.mode != LauncherMode::Json && !st.fuzzel.filtered_items.is_empty() {
                    let len = st.fuzzel.filtered_items.len();
                    st.fuzzel.selected = (st.fuzzel.selected + len - 1) % len;
                    st.fuzzel.update_scroll();
                    st.fuzzel.snap_to_selected();
                    st.upload_vertices();
                    self.redraw = true;
                }
            }
            return;
        }

        let (select_next, select_prev) = *nav_keys();
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
            sym if nav_matches(select_next, self.ctrl_pressed, sym) => Key::Named(NamedKey::ArrowDown),
            sym if nav_matches(select_prev, self.ctrl_pressed, sym) => Key::Named(NamedKey::ArrowUp),
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
                    alt: false,
                };
                if let Some(jl) = &mut st.json_layout {
                    let kev = cce_ui::widget::Event::KeyInput(key_event.clone());
                    let root = jl.id();
                    st.ui_context.register_widget(root, jl.as_ptr_mut());
                    if st.ui_context.propagate_event(&kev, root) {
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

    let (state, cce_toplevel) = State::new(
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
        output_bounds_at(&app.output_state, x_pos.unwrap_or(0), y_pos.unwrap_or(0)),
        scale,
        select_item,
        switcher_mode,
        json_layout_config,
        parent_app_id,
        None,
    );

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
            // The compositor holds the pipe open to stream __cce_switcher_next__
            // cycle lines after the item list, so there is no EOF to wait for —
            // skip the blocking initial read and let the forwarding thread
            // below stream everything (items included) to the daemon.
            needs_stdin = false;
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

    // Pay the window-independent startup costs now (login time), not on the
    // first popup: Vulkan driver + shader compiles, and the fonts-dir scan.
    // The font system is then reused across popups (each State hands it back).
    let t_prewarm = std::time::Instant::now();
    cce_ui::vk::prewarm();
    let mut fonts_slot: Option<(FontSystem, SwashCache)> =
        Some((cce_ui::create_font_system(), SwashCache::new()));
    let _ = cce_ui::widget::get_font_db(); // measure_text's resvg fontdb (system-font scan)
    log::info!("[timing] daemon prewarm: {:?}", t_prewarm.elapsed());

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

    let mut pending: Option<std::os::unix::net::UnixStream> = None;
    loop {
        let mut stream = match pending.take() {
            Some(s) => s,
            None => {
                let _ = listener.set_nonblocking(false);
                match listener.accept() {
                    Ok((s, _)) => s,
                    Err(_) => continue,
                }
            }
        };

        let (stdin_sender, stdin_channel) = calloop::channel::channel::<()>();

        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut initial_line = String::new();
        if reader.read_line(&mut initial_line).is_err() {
            continue;
        }
        let t_request = std::time::Instant::now();

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
            cycle_prev: 0,
            select_and_close: false,
            client_gone: false,
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
                    } else if trimmed == "__cce_switcher_prev__" {
                        lock_state.cycle_prev += 1;
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
            // EOF/error: the client hung up. Tell the event loop so the
            // popup closes instead of wedging the accept loop. (The normal
            // service-end path also lands here via shutdown(Read); by then
            // the channel source is already removed, so the send is inert.)
            if let Ok(mut lock_state) = stdin_state_clone.lock() {
                lock_state.client_gone = true;
            }
            let _ = stdin_sender_clone.send(());
        });

        let stdin_state_for_handler = stdin_state.clone();
        let registration_token = loop_handle.insert_source(stdin_channel, move |event, _metadata, app_state: &mut AppState| {
            if let calloop::channel::Event::Msg(()) = event {
                if stdin_state_for_handler
                    .lock()
                    .map(|s| s.client_gone)
                    .unwrap_or(false)
                {
                    log::info!("client disconnected, closing popup");
                    app_state.exit = true;
                    return;
                }
                let mut select_and_close = false;
                if let Some(st) = &mut app_state.state {
                    if let (Ok(mut lock_daemon), Ok(mut lock_state)) = (stdin_state_for_handler.lock(), st.stdin_state.lock()) {
                        if lock_daemon.new_data {
                            lock_state.items = lock_daemon.items.clone();
                            // Drain (not copy) the cycle counters: the daemon-side
                            // counts are never consumed elsewhere, so leaving them
                            // would re-apply every past cycle on each transfer.
                            lock_state.cycle_next += lock_daemon.cycle_next;
                            lock_daemon.cycle_next = 0;
                            lock_state.cycle_prev += lock_daemon.cycle_prev;
                            lock_daemon.cycle_prev = 0;
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

        let (state, cce_toplevel) = State::new(
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
            output_bounds_at(&app.output_state, x_pos.unwrap_or(0), y_pos.unwrap_or(0)),
            scale,
            select_item,
            switcher_mode,
            json_layout_config,
            parent_app_id,
            fonts_slot.take(),
        );

        app.window = state.window.clone();
        app.surface = Some(state.wl_surface.clone());
        app.cce_toplevel = cce_toplevel;
        app.state = Some(state);

        app.exit = false;
        app.fade_out = false;
        app.fade_start = None;
        app.fade_factor = 1.0;

        // The reader thread only signals for lines that arrive after this
        // point; the initial_stdin items are already sitting in stdin_state,
        // so fire one signal to make the channel handler ingest them.
        let _ = stdin_sender.send(());

        // Watch for preempting connections while the popup is open.
        let _ = listener.set_nonblocking(true);

        log::debug!("[timing] request -> popup ready: {:?}", t_request.elapsed());
        let mut first_frame_logged = false;
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

            // A new client preempts the current popup: global single-popup
            // semantics, even when the two popups are owned by different
            // bar module processes that can't see each other's state.
            if let Ok((s, _)) = listener.accept() {
                pending = Some(s);
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
                    if !first_frame_logged {
                        first_frame_logged = true;
                        log::info!("[timing] request -> first frame: {:?}", t_request.elapsed());
                    }
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
            // Take the font system back for the next popup (State has a Drop
            // impl, so swap rather than move; the placeholder is never used).
            let fs = std::mem::replace(
                &mut st.font_system,
                FontSystem::new_with_locale_and_db(
                    "en-US".to_string(),
                    cce_ui::cosmic_text::fontdb::Database::new(),
                ),
            );
            let sc = std::mem::replace(&mut st.swash_cache, SwashCache::new());
            fonts_slot = Some((fs, sc));
        }
        app.window = None;
        app.surface = None;
        app.state = None;
        app.cce_toplevel = None;

        // Dropping the State only queues wl_surface.destroy() on the
        // connection; the blocking accept() below would leave it unsent and
        // the compositor would keep showing the dead popup until the next
        // client connects.
        let _ = conn_clone.flush();
    }
}

/// A launcher list-nav chord parsed to (needs_ctrl, key char). This app
/// handles keys at the raw keysym layer, so only single-character keys
/// (optionally with ctrl) are supported here.
fn chord_ctrl_char(chord: &str) -> Option<(bool, char)> {
    let mut ctrl = false;
    let mut segs = chord.split('+').map(str::trim);
    let key = segs.next_back()?;
    for seg in segs {
        match seg.to_lowercase().as_str() {
            "ctrl" | "control" => ctrl = true,
            _ => return None,
        }
    }
    let mut chars = key.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    Some((ctrl, c.to_ascii_lowercase()))
}

/// input.kdl `cce-cloud` domain: select_next / select_prev (emacs-style
/// ctrl+n / ctrl+p defaults), resolved once per process.
fn nav_keys() -> &'static (Option<(bool, char)>, Option<(bool, char)>) {
    static KEYS: std::sync::OnceLock<(Option<(bool, char)>, Option<(bool, char)>)> = std::sync::OnceLock::new();
    KEYS.get_or_init(|| {
        (
            chord_ctrl_char(&cce_ui::input::app_chord("select_next", "ctrl+n")),
            chord_ctrl_char(&cce_ui::input::app_chord("select_prev", "ctrl+p")),
        )
    })
}

fn nav_matches(spec: Option<(bool, char)>, ctrl_pressed: bool, sym: xkeysym::Keysym) -> bool {
    match spec {
        Some((need_ctrl, c)) => {
            ctrl_pressed == need_ctrl && sym.key_char().map(|k| k.to_ascii_lowercase()) == Some(c)
        }
        None => false,
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
    // Key the socket by display so a nested/second compositor session gets its
    // own daemon instead of hijacking (or being hijacked by) another session's.
    let display = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "wayland-0".to_string());
    let socket_path = if std::path::Path::new(&socket_dir).exists() {
        format!("{}/cce-cloud-{}.socket", socket_dir, display)
    } else {
        format!("/tmp/cce-cloud-{}-{}.socket", uid, display)
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
mod placement_tests {
    use super::*;

    /// A 1920x1200 logical output at the layout origin.
    const SCREEN: Option<(i32, i32, i32, i32)> = Some((0, 0, 1920, 1200));

    /// `(left, top)` of a `w`x`h` popup anchored at `(x, y)`.
    fn place(x: i32, y: i32, w: i32, h: i32) -> (i32, i32) {
        let (_, (top, _, _, left)) = Placement::new(x, y, false, SCREEN).resolve(w, h);
        (left, top)
    }

    #[test]
    fn interior_anchor_is_used_verbatim() {
        assert_eq!(place(400, 300, 200, 250), (400, 300));
    }

    #[test]
    fn overhanging_popup_flips_to_the_other_side_of_the_cursor() {
        // The desktop context menu at (1800, 1050) with a 200x250 body: it ran off
        // both edges before, and now hangs up-and-left of the cursor instead.
        assert_eq!(place(1800, 1050, 200, 250), (1600, 800));
        // One axis at a time.
        assert_eq!(place(1850, 300, 200, 250), (1650, 300));
        assert_eq!(place(400, 1150, 200, 250), (400, 900));
    }

    #[test]
    fn a_popup_that_fits_on_neither_side_clamps_to_the_edge_gap() {
        // Anchored in the far corner, so flipping alone still leaves it off-screen.
        assert_eq!(place(1919, 1199, 200, 250), (1920 - 200 - EDGE_GAP, 1200 - 250 - EDGE_GAP));
        // Bigger than the output on both axes: pin to the near edge rather than
        // letting the clamp range invert.
        assert_eq!(place(500, 500, 3000, 3000), (EDGE_GAP, EDGE_GAP));
    }

    #[test]
    fn align_right_measures_the_anchor_from_the_right_edge() {
        // `x` in from the right, growing leftward: right edge at 1920-100, so left
        // edge at 1620.
        let mut p = Placement::new(100, 300, true, SCREEN);
        let (_, (top, _, _, left)) = p.resolve(200, 250);
        assert_eq!((left, top), (1620, 300));
        // Wider than the room to its left, so it flips and grows rightward instead.
        let mut p = Placement::new(1850, 300, true, SCREEN);
        let (_, (_, _, _, left)) = p.resolve(200, 250);
        assert_eq!(left, 1920 - 1850);
    }

    #[test]
    fn the_flip_decision_latches_across_resizes() {
        // The popup auto-sizes as its list filters. Once flipped it stays flipped:
        // unlatching would snap the window back across the cursor mid-typing.
        let mut p = Placement::new(400, 1150, false, SCREEN);
        assert_eq!(p.resolve(200, 250).1 .0, 900); // flips up
        assert_eq!(p.resolve(200, 100).1 .0, 1050); // shrinks upward, still flipped
    }

    #[test]
    fn the_anchor_is_output_local_on_a_secondary_output() {
        // Layer-shell margins are relative to the output, but `-x/-y` are layout
        // coordinates — the output origin has to come back off.
        let mut p = Placement::new(2320, 300, false, Some((1920, 0, 1920, 1200)));
        assert_eq!(p.resolve(200, 250).1 .3, 400);
    }

    #[test]
    fn the_window_switcher_geometry_is_untouched() {
        // window_manager.rs centers the 600-wide switcher horizontally and drops it
        // 80px down. It already fits, so placement must be a no-op for it.
        assert_eq!(place((1920 - 600) / 2, 80, 600, 800), (660, 80));
    }

    #[test]
    fn an_unknown_output_falls_back_to_the_raw_request() {
        let (anchor, margins) = Placement::new(1800, 1050, false, None).resolve(200, 250);
        assert_eq!(anchor, Anchor::TOP | Anchor::LEFT);
        assert_eq!(margins, (1050, 0, 0, 1800));

        let (anchor, margins) = Placement::new(1800, 1050, true, None).resolve(200, 250);
        assert_eq!(anchor, Anchor::TOP | Anchor::RIGHT);
        assert_eq!(margins, (1050, 1800, 0, 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_width_covers_label_inset() {
        // Regression: a left-justified Button draws its label 8px in from its own left edge
        // (Button::paint) in the button font. layout_children gives the button
        // `usable_w = popup_width - 32`. If the popup width doesn't budget the button's
        // 8px-per-side inset, the label spills past the button's right edge — the desktop
        // context-menu bug. Every desktop-menu label must fit within usable_w with the inset.
        let (family, size) = cce_ui::layout::parse_font_string(&cce_ui::layout::button_font());
        let size = size.unwrap_or(12.0);
        for label in [
            "Terminal", "Files", "Data Editor", "Applications",
            "System Settings", "Expose Windows", "Reload Config", "Logout",
        ] {
            let popup_w = json_widget_desired_width("button", label);
            let usable_w = popup_w - 32.0; // container margins, per layout_children
            let label_w = cce_ui::widget::display::measure_text_width(label, &family, size);
            // 8px left inset + label + 8px right breathing room must fit the button.
            assert!(
                usable_w >= label_w + 16.0,
                "button '{label}': usable_w {usable_w} < label {label_w} + 16 inset",
            );
        }
    }

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
        use cce_ui::widget::WidgetHost;

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

        assert!(layout.widgets[0].widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Label>().is_some());
        assert!(layout.widgets[1].widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Checkbox>().is_some());
        assert!(layout.widgets[2].widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Button>().is_some());

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
        assert_eq!(layout.widgets[1].widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Checkbox>().unwrap().checked(), false);

        // Simulate click on Checkbox row
        let changed = layout.mouse_input(
            cce_ui::widget::MouseButton::Left,
            cce_ui::widget::ElementState::Released,
            w_check_x + 5.0,
            w_check_y + 5.0,
            &mut ctx,
        );
        assert!(changed);
        assert_eq!(layout.widgets[1].widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Checkbox>().unwrap().checked(), true);

        // Simulate hover on button
        let changed_hover = layout.on_cursor_moved(w_btn_x + 10.0, w_btn_y + 10.0, &mut ctx);
        assert!(changed_hover);
        // Phase 5: hover state lives on the Button model (it drives the color matrix), not the base.
        assert!(layout.widgets[2].widget.as_dyn().as_any().downcast_ref::<cce_ui::widget::Button>().unwrap().hovered());
    }

    /// Hover must be exclusive: a pointer over one button leaves the others
    /// unhovered (the desktop context menu regression — every button lit up
    /// once the paint walk started rendering the Button model's hover state).
    #[test]
    fn test_json_button_hover_is_exclusive() {
        let mk_btn = |id: &str, text: &str| JsonWidgetConfig {
            widget_type: "button".to_string(),
            text: text.to_string(),
            id: Some(id.to_string()),
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
        };
        let config = JsonLayoutConfig {
            width: Some(300),
            height: Some(400),
            widgets: Some(vec![mk_btn("a", "Alpha"), mk_btn("b", "Beta"), mk_btn("c", "Gamma")]),
            pages: None,
            justify: None,
        };
        let mut layout = JsonLayoutWidget::new(&config);
        layout.set_rect(0.0, 0.0, 300.0, 400.0);
        let mut ctx = cce_ui::context::UiContext::new();

        let hovered = |layout: &cce_ui::widget::Adapted<JsonLayoutWidget>, i: usize| {
            layout.widgets[i]
                .widget
                .as_dyn()
                .as_any()
                .downcast_ref::<cce_ui::widget::Button>()
                .unwrap()
                .hovered()
        };

        // Pointer over the first button only.
        let (x0, y0) = (layout.widgets[0].x, layout.widgets[0].y);
        layout.on_cursor_moved(x0 + 10.0, y0 + 5.0, &mut ctx);
        assert!(hovered(&layout, 0), "hovered button must be hovered");
        assert!(!hovered(&layout, 1), "second button must not be hovered");
        assert!(!hovered(&layout, 2), "third button must not be hovered");

        // Move to the third button: hover follows, first clears.
        let (x2, y2) = (layout.widgets[2].x, layout.widgets[2].y);
        layout.on_cursor_moved(x2 + 10.0, y2 + 5.0, &mut ctx);
        assert!(!hovered(&layout, 0), "old hover must clear");
        assert!(!hovered(&layout, 1));
        assert!(hovered(&layout, 2), "new hover must set");

        // Pointer inside the panel but on no button: everything clears.
        layout.on_cursor_moved(150.0, 395.0, &mut ctx);
        assert!(!hovered(&layout, 0));
        assert!(!hovered(&layout, 1));
        assert!(!hovered(&layout, 2));

        // The live path: routed dispatch through the UiContext, and crucially a
        // SECOND move that changes no child hover. The first (consumed) move
        // skips the adapter's base-hover bookkeeping; the unconsumed second one
        // runs it, synthesizing a MouseEnter for the panel — which route_event
        // must NOT broadcast to the children (the desktop-menu regression: every
        // button lit up on the first stationary wiggle).
        let (x1, y1) = (layout.widgets[1].x, layout.widgets[1].y);
        let root = layout.id();
        ctx.register_widget(root, layout.as_ptr_mut());
        let mv = |x: f32, y: f32| cce_ui::widget::Event::PointerMove { x, y, local_x: x, local_y: y };
        ctx.propagate_event(&mv(x1 + 10.0, y1 + 5.0), root);
        ctx.propagate_event(&mv(x1 + 12.0, y1 + 5.0), root);
        assert!(!hovered(&layout, 0), "unconsumed move must not hover-broadcast");
        assert!(hovered(&layout, 1));
        assert!(!hovered(&layout, 2), "unconsumed move must not hover-broadcast");
    }

    #[test]
    fn parse_desktop_terminal_flag() {
        let dir = std::path::PathBuf::from("/tmp/cce-cloud-test-desktop-dir");
        let _ = std::fs::create_dir_all(&dir);

        let path = dir.join("htop.desktop");
        std::fs::write(
            &path,
            "[Desktop Entry]\nType=Application\nName=htop\nExec=htop\nTerminal=true\n",
        )
        .unwrap();
        let app = parse_desktop_file(&path).unwrap();
        assert!(app.terminal);

        let path = dir.join("gui.desktop");
        std::fs::write(
            &path,
            "[Desktop Entry]\nType=Application\nName=Gui\nExec=gui %U\n",
        )
        .unwrap();
        let app = parse_desktop_file(&path).unwrap();
        assert!(!app.terminal);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_apps_xdg_precedence() {
        let base = std::path::PathBuf::from("/tmp/cce-cloud-test-xdg");
        let _ = std::fs::remove_dir_all(&base);
        let user = base.join("home/applications");
        let sys = base.join("sys/applications");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(&sys).unwrap();

        std::fs::write(
            sys.join("editor.desktop"),
            "[Desktop Entry]\nType=Application\nName=Editor\nExec=editor-sys\n",
        )
        .unwrap();
        std::fs::write(
            sys.join("player.desktop"),
            "[Desktop Entry]\nType=Application\nName=Player\nExec=player\n",
        )
        .unwrap();
        // User dir: renames editor (same ID must still shadow the system
        // entry) and deletes player via Hidden.
        std::fs::write(
            user.join("editor.desktop"),
            "[Desktop Entry]\nType=Application\nName=My Editor\nExec=editor-user\n",
        )
        .unwrap();
        std::fs::write(
            user.join("player.desktop"),
            "[Desktop Entry]\nType=Application\nName=Player\nExec=player\nHidden=true\n",
        )
        .unwrap();

        let orig_home = std::env::var("XDG_DATA_HOME").ok();
        let orig_dirs = std::env::var("XDG_DATA_DIRS").ok();
        std::env::set_var("XDG_DATA_HOME", base.join("home"));
        std::env::set_var("XDG_DATA_DIRS", base.join("sys"));

        let apps = scan_apps();

        match orig_home {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        match orig_dirs {
            Some(v) => std::env::set_var("XDG_DATA_DIRS", v),
            None => std::env::remove_var("XDG_DATA_DIRS"),
        }

        assert_eq!(apps.len(), 1);
        assert_eq!(apps[0].name, "My Editor");
        assert_eq!(apps[0].exec, "editor-user");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_app_history_sorting() {
        let temp_dir = std::path::PathBuf::from("/tmp/cce-cloud-test-cache-dir");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);

        let orig_xdg = std::env::var("XDG_CACHE_HOME").ok();
        std::env::set_var("XDG_CACHE_HOME", &temp_dir);

        let mut apps = vec![
            AppInfo { name: "App A".to_string(), exec: "exec_a".to_string(), terminal: false },
            AppInfo { name: "App B".to_string(), exec: "exec_b".to_string(), terminal: false },
            AppInfo { name: "App C".to_string(), exec: "exec_c".to_string(), terminal: false },
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
                    // Indented past the icon column whether or not THIS row
                    // resolved an icon — see `icon_gutter`.
                    x: self.x + pad + 10.0 + self.icon_gutter,
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
