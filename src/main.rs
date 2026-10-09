use cce_ui::widget::Handle;
use cce_ui::widget::ScrollRegion;

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
/// Each label rides with its merged clip bounds (logical `[l, t, r, b]`, from
/// the paint walk's clip ∩ the prim's own bounds — `append_widget_text` merges
/// them): prepare_text turns them into the span's physical clip so text cut by
/// a clip (a partially visible list row) is cut at the glyph pass too, not
/// drawn whole.
fn walk_text_labels(
    ui: &cce_ui::context::UiContext,
    w: &dyn WidgetHost,
) -> Vec<(TextLabel, Option<[f32; 4]>)> {
    let mut pc = cce_ui::scene::paint::PaintCtx::new();
    cce_ui::scene::painter::append_widget_text(ui, w, &mut pc);
    pc.finish()
        .items
        .into_iter()
        .filter_map(|item| match item.prim {
            cce_ui::scene::paint::Prim::Text { text, x, y, font_size, color, bounds, .. } => {
                Some((TextLabel { text, x, y, font_size, color }, bounds))
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
            let icon = path.file_stem().and_then(|s| s.to_str()).and_then(icon_override).or(icon);
            return Some(AppInfo { name: n, exec: e, terminal, icon });
        }
    }
    None
}

/// The `applications/` dirs `.desktop` files are read from, in XDG precedence:
/// $XDG_DATA_HOME first, then each $XDG_DATA_DIRS entry in order (defaults per
/// the base-directory spec). Honoring XDG_DATA_DIRS is what makes Flatpak/Snap
/// exports visible.
fn application_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Some(data_home) = data_home() {
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
    dirs
}

/// `$XDG_DATA_HOME`, defaulting per the base-directory spec.
fn data_home() -> Option<std::path::PathBuf> {
    std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".local/share"))
        })
}

/// cce's own icon for the desktop entry `id` (its file stem), when cce-icons
/// ships one: `hicolor/scalable/apps/<id>.svg` under `$XDG_DATA_HOME/icons`,
/// where `ccebuild install` puts that tree. Returned as an absolute path, so
/// it bypasses the theme search entirely.
///
/// This is what lets cce draw its own artwork for apps it does not own. An
/// override named after the entry's `Icon=` value already wins without help
/// (the user's data dir is searched first), but that cannot reach an entry
/// whose `Icon=` is an absolute path (Houdini's PNG), is missing (Raindrop),
/// or is a generic name several apps share (`network-wired` for all three
/// Avahi browsers) — those are overridden by desktop-file ID instead. One stat
/// per entry: the theme walk `cce_ui::icon::lookup` does would cost ~200 per
/// miss, for every app, on every launcher open.
fn icon_override(id: &str) -> Option<String> {
    icon_override_in(&data_home()?.join("icons/hicolor/scalable/apps"), id)
}

fn icon_override_in(dir: &std::path::Path, id: &str) -> Option<String> {
    let path = dir.join(format!("{id}.svg"));
    path.is_file().then(|| path.to_string_lossy().into_owned())
}

/// `Name=` and `Icon=` of the desktop entry with ID `id` (its file stem),
/// from the first applications dir that has it — NoDisplay entries
/// included: a handler the portal offers is a valid choice whether or not
/// the launcher lists it. The icon goes through [`icon_override`] like the
/// launcher's. `None` when no dir has the entry.
fn desktop_entry_label(id: &str, dirs: &[std::path::PathBuf]) -> Option<(String, Option<String>)> {
    let path = dirs.iter().map(|d| d.join(format!("{id}.desktop"))).find(|p| p.is_file())?;
    let text = std::fs::read_to_string(&path).ok()?;
    let (mut name, mut icon) = (None, None);
    let mut in_entry = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_entry = line == "[Desktop Entry]";
            continue;
        }
        if !in_entry {
            continue;
        }
        if let Some(v) = line.strip_prefix("Name=") {
            name.get_or_insert_with(|| v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Icon=") {
            if !v.trim().is_empty() {
                icon.get_or_insert_with(|| v.trim().to_string());
            }
        }
    }
    Some((name.unwrap_or_else(|| id.to_string()), icon_override(id).or(icon)))
}

/// `--choose`: the app chooser the desktop portal opens ("Open with…").
/// Dmenu mode with a different feed and a different answer: stdin carries
/// desktop-file IDs, the rows show each one's name and icon, and the choice
/// is printed back as the ID. Rows are labels, so the label → ID map is
/// what turns a pick back into the answer.
#[derive(Default)]
struct Chooser {
    /// The IDs last ingested, so an unchanged feed is not re-resolved on
    /// every poll (the rows hold labels, which never equal the IDs).
    fed: Vec<String>,
    by_label: std::collections::HashMap<String, String>,
}

impl Chooser {
    /// Labels for `ids`, in order. A name two entries share (two "Firefox"
    /// builds) is told apart by the ID, so every label maps back to one app.
    fn labels(ids: &[String], dirs: &[std::path::PathBuf]) -> Vec<(String, String, Option<String>)> {
        let entries: Vec<(String, String, Option<String>)> = ids
            .iter()
            .map(|id| {
                let (name, icon) = desktop_entry_label(id, dirs).unwrap_or_else(|| (id.clone(), None));
                (id.clone(), name, icon)
            })
            .collect();
        entries
            .iter()
            .map(|(id, name, icon)| {
                let shared = entries.iter().filter(|(_, n, _)| n == name).count() > 1;
                let label = if shared { format!("{name} ({id})") } else { name.clone() };
                (id.clone(), label, icon.clone())
            })
            .collect()
    }

    /// What a picked row answers: its ID, or the row itself if unknown.
    fn answer(&self, row: &str) -> String {
        self.by_label.get(row).cloned().unwrap_or_else(|| row.to_string())
    }
}

fn scan_apps() -> Vec<AppInfo> {
    let mut apps = Vec::new();
    let dirs = application_dirs();

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

/// A Super-Tab switcher row split into `(title, app_id)`. The compositor
/// (`launch_window_switcher` in cce-compositor's window_manager.rs) writes each
/// window as `Title (app_id)`, or the bare app_id when the title is empty —
/// which is then both halves. It maps the echoed row back to a window by its
/// whole text, so the row itself is left as sent: the title is only what is
/// drawn, and the id is only what the icon is looked up by. The LAST
/// parenthesised group is the id: a title may carry parentheses of its own.
fn split_switcher_row(item: &str) -> (&str, &str) {
    item.strip_suffix(')')
        .and_then(|rest| rest.rfind(" (").map(|i| (&rest[..i], &rest[i + 2..])))
        .unwrap_or((item, item))
}

/// app_id → `Icon=` value, from every `.desktop` file on the search path.
///
/// A window's app_id is not an icon name, but by convention it names its
/// desktop entry: the file stem (`org.gnome.Nautilus`), or `StartupWMClass`
/// for the apps whose id doesn't match their file. Keys are lowercased — ids in
/// the wild disagree with their entries on case (`firefox` / `Firefox`) — and
/// the last reverse-DNS component is indexed too, so `org.gnome.Nautilus`
/// matches a window that reports plain `nautilus`. Unlike [`scan_apps`] this
/// keeps NoDisplay entries: a helper window is still a window with an icon.
/// A cce icon override for the entry ([`icon_override`]) replaces its `Icon=`.
fn desktop_icon_index(dirs: &[std::path::PathBuf]) -> std::collections::HashMap<String, String> {
    let mut index = std::collections::HashMap::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(true, |ext| ext != "desktop") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let mut in_entry = false;
            let mut icon = None;
            let mut wm_class = None;
            for line in text.lines() {
                let line = line.trim();
                if line.starts_with('[') {
                    in_entry = line == "[Desktop Entry]";
                } else if in_entry {
                    if let Some(v) = line.strip_prefix("Icon=") {
                        icon.get_or_insert(v.trim().to_string());
                    } else if let Some(v) = line.strip_prefix("StartupWMClass=") {
                        wm_class.get_or_insert(v.trim().to_lowercase());
                    }
                }
            }
            // A cce override for the entry wins, as in [`parse_desktop_file`].
            let Some(icon) = icon_override(stem).or(icon.filter(|i| !i.is_empty())) else { continue };
            let stem = stem.to_lowercase();
            let short = stem.rsplit('.').next().map(str::to_string);
            // First claim wins, in XDG precedence — the same shadowing
            // [`scan_apps`] applies to desktop-file IDs.
            for key in [Some(stem), wm_class, short].into_iter().flatten() {
                index.entry(key).or_insert_with(|| icon.clone());
            }
        }
    }
    index
}

/// The icon name to draw for a window with this app_id: its desktop entry's
/// `Icon=`, else the app_id itself, which is what apps without an entry (and
/// cce's own clients, whose icons are installed under their app_id) name their
/// icon after.
fn icon_name_for_app_id(index: &std::collections::HashMap<String, String>, app_id: &str) -> String {
    index
        .get(&app_id.to_lowercase())
        .cloned()
        .unwrap_or_else(|| app_id.to_string())
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

/// Resolve a cce binary installed beside this one.
///
/// The launcher runs as a systemd user service, whose PATH is
/// `/usr/local/bin:/usr/bin` — `~/.local/bin`, where every cce binary lives,
/// is not on it, so spawning one by bare name fails with ENOENT under systemd
/// while working fine from a shell.
fn de_bin(name: &str) -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(name);
            if beside.exists() {
                return beside;
            }
        }
    }
    std::path::PathBuf::from(name)
}

/// Open the launched window at the grid square this launcher was invoked at,
/// keeping its remembered size and growing away from its neighbours.
///
/// Only when there IS an invocation point: the desktop menu passes one
/// through, a launcher summoned by keyboard does not, and in that case the app
/// keeps its remembered place — there is no "here" to mean.
fn place_next_at(exec: &str, invoked_at: Option<(i32, i32)>) {
    let Some((x, y)) = invoked_at else { return };
    let first = exec.split_whitespace().next().unwrap_or("");
    let prog = first.rsplit('/').next().unwrap_or(first);
    if prog.is_empty() {
        return;
    }
    let _ = std::process::Command::new(de_bin("ccectl"))
        .args(["place-next-cell", prog, &x.to_string(), &y.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
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

/// Whether holding this key should repeat it: text, deletion and cursor or
/// list movement. Not Return, Escape or Tab — a held Return would launch the
/// selection again and again, a held Escape has nothing left to close.
fn key_repeats(keysym: xkeysym::Keysym, utf8: Option<&str>) -> bool {
    use xkeysym::Keysym as K;
    match keysym {
        K::BackSpace | K::Delete | K::KP_Delete | K::Left | K::Right | K::Up | K::Down
        | K::KP_Left | K::KP_Right | K::KP_Up | K::KP_Down | K::Page_Up | K::Page_Down => true,
        K::Return | K::KP_Enter | K::Escape | K::Tab | K::ISO_Left_Tab => false,
        _ => utf8.is_some_and(|t| !t.is_empty() && !t.chars().any(char::is_control)),
    }
}

/// The session target a launched app's scope is `PartOf`: startcce starts it
/// once the compositor is up and stops it when the compositor exits (see
/// cce-cloud.service).
const SESSION_TARGET: &str = "cce-session.target";

/// A systemd unit-name fragment for `program`: its file name, reduced to the
/// characters a unit name may hold.
fn scope_name_part(program: &str) -> String {
    let base = program.rsplit('/').next().unwrap_or(program);
    let part: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .take(40)
        .collect();
    if part.is_empty() { "app".to_string() } else { part }
}

/// The program a launch actually runs, for naming its scope. Every desktop
/// entry goes through `sh -c <Exec>` (a terminal-hosted one through
/// `<terminal> sh -c "exec <Exec>"`), so naming the scope after `program`
/// called every one of them `sh`: the first word of the shell command is
/// the app, past a leading `exec`, `env` and `VAR=value` assignments.
fn launch_name(program: &str, args: &[&str]) -> String {
    let shell_cmd = args
        .windows(2)
        .position(|w| w[0] == "-c")
        .filter(|&i| i == 0 || matches!(scope_name_part(args[i - 1]).as_str(), "sh" | "bash"))
        .filter(|&i| i > 0 || matches!(scope_name_part(program).as_str(), "sh" | "bash"))
        .map(|i| args[i + 1]);
    let Some(cmd) = shell_cmd else {
        return program.to_string();
    };
    cmd.split_whitespace()
        .map(|w| w.trim_matches(|c| c == '\'' || c == '"'))
        .find(|w| !w.is_empty() && *w != "exec" && *w != "env" && !w.contains('='))
        .unwrap_or(program)
        .to_string()
}

/// `systemd-run` arguments that start `program args` in its own transient
/// scope, `app-cce\x2dcloud-<app>-<n>.scope` in app.slice, `PartOf` the
/// session target, named after the app it runs (`launch_name`). `n` only
/// has to make the name unique.
fn scope_argv(program: &str, args: &[&str], n: u128) -> Vec<String> {
    let mut argv = vec![
        "--user".to_string(),
        "--scope".to_string(),
        "--collect".to_string(),
        "--slice=app.slice".to_string(),
        format!("--unit=app-cce\\x2dcloud-{}-{n}", scope_name_part(&launch_name(program, args))),
        format!("--property=PartOf={SESSION_TARGET}"),
        "--".to_string(),
        program.to_string(),
    ];
    argv.extend(args.iter().map(|a| a.to_string()));
    argv
}

fn spawn_detached(program: &str, args: &[&str]) {
    // Launched apps must outlive this daemon but not the session. Each gets
    // its own transient scope (`scope_argv`): a service restart signals only
    // the daemon (cce-cloud.service's KillMode=process) and never reaches
    // another unit's cgroup, while the scope's PartOf= stops the app with
    // the session. Until 2026-09-26 apps stayed in this service's cgroup,
    // where KillMode=process left them running after logout: a Proton
    // launch queued behind a still-running game started in the NEXT
    // session before its Xwayland existed, ran with no display, and held
    // every later launch of it behind itself. process_group(0) still keeps
    // a terminal ^C (manual daemon run) away from them.
    use std::os::unix::process::CommandExt;
    let mut cmd = if command_in_path("systemd-run") {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut c = std::process::Command::new("systemd-run");
        c.args(scope_argv(program, args, n));
        c
    } else {
        let mut c = std::process::Command::new(program);
        c.args(args);
        c
    };
    cmd.process_group(0);
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
    // Through the reaping spawn below, NOT a bare `cmd.spawn()`: the daemon
    // lives for the whole session, and a dropped Child handle means every app
    // it ever launched sits in the process table as a zombie once it exits —
    // unreadable in /proc and reported "alive" by kill(pid, 0) probes.
    let _ = spawn_reaped(cmd);
}

/// Spawn `cmd` and reap it on a background thread. This was
/// `cce_ui::process::spawn_detached` until the toolkit dropped that module
/// (cce-ui 4e94236) as caller-less — `spawn_detached` above was a caller.
fn spawn_reaped(mut cmd: std::process::Command) -> std::io::Result<()> {
    let mut child = cmd.spawn()?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}




/// One row of the System tab: the label the list shows and the command that
/// label runs.
struct SystemCommand {
    name: &'static str,
    program: &'static str,
    args: &'static [&'static str],
}

/// The System tab's rows — window-manager verbs driven through `ccectl` (the
/// DE's control CLI already exposes every one of them, so there is nothing to
/// reimplement here) plus the session and power commands that are not the
/// compositor's to run.
///
/// The window verbs act on the window BEHIND this popup, not on the popup:
/// the compositor's `focused_window` skips overlay UI and names `cce-cloud`
/// explicitly among it, falling back to the most recent real window. That is
/// what makes "Close Window" from a launcher mean anything at all.
///
/// Hardcoded rather than config-driven: these are the DE's own verbs, and a
/// row naming a command `ccectl` does not have is a row that silently does
/// nothing.
const SYSTEM_COMMANDS: &[SystemCommand] = &[
    SystemCommand { name: "Close Window", program: "ccectl", args: &["close"] },
    SystemCommand { name: "Minimize Window", program: "ccectl", args: &["minimize"] },
    SystemCommand { name: "Toggle Fullscreen", program: "ccectl", args: &["fullscreen"] },
    SystemCommand { name: "Center Window", program: "ccectl", args: &["center-window"] },
    SystemCommand { name: "Overlay Window Left", program: "ccectl", args: &["overlay-left"] },
    SystemCommand { name: "Overlay Window Right", program: "ccectl", args: &["overlay-right"] },
    SystemCommand { name: "Next Tiling Mode", program: "ccectl", args: &["mode-next"] },
    SystemCommand { name: "Next Tiling Mode (Shared)", program: "ccectl", args: &["mode-next-shared"] },
    SystemCommand { name: "Retile Windows", program: "ccectl", args: &["retile"] },
    SystemCommand { name: "Toggle Overview", program: "ccectl", args: &["overview"] },
    SystemCommand { name: "Zoom In", program: "ccectl", args: &["zoom-in"] },
    SystemCommand { name: "Zoom Out", program: "ccectl", args: &["zoom-out"] },
    SystemCommand { name: "Reset Zoom", program: "ccectl", args: &["zoom-reset"] },
    SystemCommand { name: "Take Screenshot", program: "ccectl", args: &["screenshot"] },
    SystemCommand { name: "Reload Configuration", program: "ccectl", args: &["reload"] },
    SystemCommand { name: "Restart Compositor", program: "ccectl", args: &["restart-compositor"] },
    SystemCommand { name: "Log Out", program: "ccectl", args: &["exit"] },
    SystemCommand { name: "Turn Off Display", program: "ccectl", args: &["idle", "display", "off"] },
    SystemCommand { name: "Suspend", program: "systemctl", args: &["suspend"] },
    SystemCommand { name: "Reboot", program: "systemctl", args: &["reboot"] },
    SystemCommand { name: "Power Off", program: "systemctl", args: &["poweroff"] },
];

/// The titles the tabbed launcher shows, in strip order. Tab 0 is the mode's
/// own list; tab 1 is [`SYSTEM_COMMANDS`].
const SYSTEM_TAB_TITLE: &str = "System";

/// Run `item` if it is a System-tab row, and say whether it was. These rows
/// are the DE's own verbs rather than apps: they go straight to `ccectl` (or
/// systemd), with none of the desktop-entry or place-next handling an app
/// launch gets.
fn run_system_item(fuzzel: &FuzzelWidget, item: &str) -> bool {
    if fuzzel.active_tab == 0 {
        return false;
    }
    let Some(cmd) = SYSTEM_COMMANDS.iter().find(|c| c.name == item) else {
        return false;
    };
    spawn_detached(cmd.program, cmd.args);
    true
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
    /// Rows are the window switcher's `Title (app_id)` lines: draw only the
    /// title (see [`split_switcher_row`]). The item text — what filtering
    /// matches and what a selection echoes back — keeps the suffix. Also
    /// makes pointer motion select (see [`Self::pointer_moved`]).
    pub switcher_rows: bool,
    /// Row under the pointer, by filtered index — the hover wash and the
    /// brighter label. Distinct from `selected`: hovering never moves the
    /// keyboard selection, only a click does — except in the switcher, where
    /// MOVING onto a row selects it ([`Self::pointer_moved`]).
    hovered: Option<usize>,
    /// The row the switcher's last pointer motion selected, so the pointer
    /// only takes the selection when it moves onto a NEW row: Tab can still
    /// move the chip away from a resting (or jiggling) pointer. Cleared when
    /// the pointer leaves, so the first motion after it enters selects.
    motion_row: Option<usize>,
    /// Last pointer position seen over the surface, so the hovered row can be
    /// re-derived when the rows move under a STATIONARY pointer — a wheel
    /// glide, a keystroke refiltering the list, a keyboard snap.
    cursor: Option<(f32, f32)>,
    pub scroll_box: ScrollRegion,
    /// The pages the list is split into. Fewer than two means no tab strip and
    /// no chrome height for one, which is what keeps Dmenu, Path and the
    /// Super-Tab window switcher laid out exactly as they were.
    ///
    /// The ACTIVE page's items and query live in `all_items` / `query`, not in
    /// its `TabPage` — every existing caller reads them there, and only
    /// [`Self::switch_tab`] moves them across. A page's own copies are
    /// therefore stale for as long as it is the active one.
    pub tabs: Vec<TabPage>,
    pub active_tab: usize,
    /// Tab under the pointer, mirroring `hovered` for the rows.
    tab_hovered: Option<usize>,
}

/// One page of the tabbed list. See [`FuzzelWidget::tabs`] for which copy of
/// `items` / `query` is the authoritative one.
pub struct TabPage {
    title: String,
    items: Vec<String>,
    query: String,
}

/// Icon edge length inside an [`ICON_ITEM_H`] row. The gap between it and the
/// label is the toolkit's control text inset, the same standoff the label
/// keeps from the selection chip's edge.
const ICON_PX: f32 = 26.0;

/// The pixel size row icons are rasterized at: twice [`ICON_PX`], for a
/// scale-2 output.
const ICON_RASTER_PX: u32 = ICON_PX as u32 * 2;

/// A row icon's image, uploaded once per renderer: `(id, width, height)`, or
/// `None` when the theme has no such icon. The daemon keeps one renderer for
/// its whole life, so the launcher's icons are uploaded on its first open (or
/// by the startup warm-up, `run_daemon`) and every later open draws the same
/// images. Uploading per popup cost a synchronous GPU copy per icon at the
/// first frame, plus a device-idle wait per free at the next one: ~90 stalls
/// for 46 icons, most of the launcher's time to first frame.
///
/// The cache follows [`cce_ui::vk::renderer_epoch`]: a different renderer
/// means none of the ids name anything, so they are dropped and uploaded
/// again. Thread-safe, because the warm-up runs off the main thread.
fn icon_image(name: &str) -> Option<(u32, u32, u32)> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    type Cache = (u32, HashMap<String, Option<(u32, u32, u32)>>);
    static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

    let epoch = cce_ui::vk::renderer_epoch();
    let mut guard = CACHE.lock().unwrap();
    let (cached_epoch, ids) = guard.get_or_insert_with(|| (epoch, HashMap::new()));
    if *cached_epoch != epoch {
        *cached_epoch = epoch;
        ids.clear();
    }
    if let Some(hit) = ids.get(name) {
        return *hit;
    }
    let img = cce_ui::icon::upload_themed(name, ICON_RASTER_PX);
    ids.insert(name.to_string(), img);
    img
}

/// The list chrome's metrics. The spacing around them — the inset from the
/// popup edge, the gap under the tab strip and under the search well, the
/// text inset inside a well or a row — is the toolkit's ladder
/// (`cce_ui::layout::root_plate_inset` / `root_plate_gap` /
/// `CONTROL_TEXT_INSET`), read where it is used; these were repeated as bare
/// `let pad = 15.0;` locals in every one of the paint, scroll and hit-test
/// paths. The tab strip shifts the whole list down by its own height, so the
/// offset has to be derived in one place or the rows, the clip and the click
/// go out of step. The search well and the tab run are the toolkit's
/// textbox and button heights (`cce_ui::layout::textbox_height` /
/// `button_height`), read where they are used.
const ITEM_H: f32 = 25.0;
/// Row height once the list carries icons (Apps mode, the window switcher):
/// the icon plus a 5px standoff above and below. Text-only lists — dmenu,
/// Path, the System tab — keep the tighter [`ITEM_H`].
const ICON_ITEM_H: f32 = 36.0;
/// style: deliberate — the hairline the selection chip (and the hover wash on
/// its footprint) stands in from the list viewport on each side, so the chip's
/// roll clears the clip. A standoff, not a rung of the spacing ladder.
const CHIP_STANDOFF: f32 = 2.0;
/// Tab-title size — a step under the row labels, as a control label is.
const TAB_FONT_PX: f32 = 12.0;

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
            switcher_rows: false,
            hovered: None,
            motion_row: None,
            cursor: None,
            // Designer raise/sink treatment: the bar idles sunk under the
            // list's translucent bg (dimly visible through it) and raises over
            // the rows on scroll. The region already sits inset from the popup
            // edge, so the stock 4px edge inset reads right here.
            scroll_box: ScrollRegion::new(22.0, 0.0).with_sink_behind(true),
            tabs: Vec::new(),
            active_tab: 0,
            tab_hovered: None,
        })
    }

    pub fn set_items(&mut self, items: Vec<String>) {
        self.all_items = items;
        self.recompute_icon_gutter();
        self.filter();
    }

    /// Split the list into tabs. Tab 0 is the mode's own list — its items keep
    /// arriving through [`Self::set_items`] — and every later tab carries the
    /// items it is given here. A single tab (or none) draws no strip.
    pub fn set_tabs(&mut self, tabs: Vec<(String, Vec<String>)>) {
        self.tabs = tabs
            .into_iter()
            .map(|(title, items)| TabPage { title, items, query: String::new() })
            .collect();
        self.active_tab = 0;
        if let Some(first) = self.tabs.first_mut() {
            self.all_items = std::mem::take(&mut first.items);
        }
        self.recompute_icon_gutter();
        self.filter();
    }

    /// Replace tab `idx`'s items wherever they are parked. The stdin/socket
    /// feed always addresses tab 0 through this, never `set_items` directly:
    /// the ingest compares against the items it last pushed, and on any other
    /// tab that comparison would differ every time and clobber the list the
    /// user is reading.
    pub fn set_tab_items(&mut self, idx: usize, items: Vec<String>) {
        if idx == self.active_tab {
            self.set_items(items);
        } else if let Some(page) = self.tabs.get_mut(idx) {
            page.items = items;
        }
    }

    /// The items tab `idx` holds right now — from `all_items` when it is the
    /// active tab, from its parked page otherwise.
    pub fn tab_items(&self, idx: usize) -> &[String] {
        if idx == self.active_tab {
            &self.all_items
        } else {
            self.tabs.get(idx).map(|p| p.items.as_slice()).unwrap_or(&[])
        }
    }

    /// Move to tab `idx`, parking the current tab's items and query in its
    /// page and unpacking the target's — so switching back lands on the same
    /// query and the same rows. False when nothing moved.
    pub fn switch_tab(&mut self, idx: usize) -> bool {
        if idx >= self.tabs.len() || idx == self.active_tab {
            return false;
        }
        self.tabs[self.active_tab].items = std::mem::take(&mut self.all_items);
        self.tabs[self.active_tab].query = std::mem::take(&mut self.query);
        self.active_tab = idx;
        self.all_items = std::mem::take(&mut self.tabs[idx].items);
        self.query = std::mem::take(&mut self.tabs[idx].query);
        self.selected = 0;
        self.scroll_box.scroll_y = 0.0;
        self.recompute_icon_gutter();
        self.filter();
        true
    }

    /// Step one tab forward (or back) with wrap — what Tab and Shift+Tab do
    /// once the list has more than one. False when there is nothing to step
    /// through, which is the signal for those keys to fall back to their old
    /// job of cycling the highlight.
    pub fn cycle_tab(&mut self, forward: bool) -> bool {
        let n = self.tabs.len();
        if n < 2 {
            return false;
        }
        let idx = if forward { (self.active_tab + 1) % n } else { (self.active_tab + n - 1) % n };
        self.switch_tab(idx)
    }

    /// Height the tab strip takes off the top of the popup — the run (a
    /// button's height) and the gap between it and the search well; 0 below
    /// two tabs.
    pub fn tab_strip_h(&self) -> f32 {
        if self.tabs.len() > 1 { cce_ui::layout::button_height() + cce_ui::layout::root_plate_gap() } else { 0.0 }
    }

    /// The segmented run itself, inset from the popup edge like the search
    /// well under it. `None` when no strip is drawn.
    fn tab_strip_rect(&self) -> Option<cce_ui::scene::layout::Rect> {
        let inset = cce_ui::layout::root_plate_inset();
        (self.tabs.len() > 1).then(|| cce_ui::scene::layout::Rect {
            x: self.x + inset,
            y: self.y + inset,
            width: self.w - inset * 2.0,
            height: cce_ui::layout::button_height(),
        })
    }

    /// Segment `i` of the run — equal shares of its width.
    fn tab_rect(&self, i: usize) -> Option<cce_ui::scene::layout::Rect> {
        let strip = self.tab_strip_rect()?;
        let seg_w = strip.width / self.tabs.len() as f32;
        Some(cce_ui::scene::layout::Rect {
            x: strip.x + i as f32 * seg_w,
            y: strip.y,
            width: seg_w,
            height: strip.height,
        })
    }

    /// The tab under `(px, py)` — the one predicate the strip's hover wash and
    /// its click share, as `row_at` is for the rows.
    pub fn tab_at(&self, px: f32, py: f32) -> Option<usize> {
        let strip = self.tab_strip_rect()?;
        if px < strip.x || px >= strip.x + strip.width || py < strip.y || py >= strip.y + strip.height {
            return None;
        }
        let n = self.tabs.len();
        Some((((px - strip.x) / (strip.width / n as f32)).floor() as usize).min(n - 1))
    }

    /// Y of the search well's top edge: under the tab strip, where there is one.
    fn search_y(&self) -> f32 {
        self.y + cce_ui::layout::root_plate_inset() + self.tab_strip_h()
    }

    /// Y of the list viewport's top edge — the number the scroll math, the row
    /// hit-test, the clip and the empty-state label all have to agree on.
    fn list_y(&self) -> f32 {
        self.search_y() + cce_ui::layout::textbox_height() + cce_ui::layout::root_plate_gap()
    }

    /// Height of the list viewport: everything left between it and the bottom
    /// inset.
    fn list_h(&self) -> f32 {
        (self.y + self.h - cce_ui::layout::root_plate_inset()) - self.list_y()
    }

    /// X of the text in a row (and of the query line in the search well): the
    /// control text inset past the selection chip's edge, which itself stands
    /// a hairline in from the list. Icons start here too.
    fn text_x(&self) -> f32 {
        self.x + cce_ui::layout::root_plate_inset() + CHIP_STANDOFF + cce_ui::layout::CONTROL_TEXT_INSET
    }

    /// Vertical chrome around the list: the inset above and below, the search
    /// well and the gap under it, and the tab strip when there is one. What
    /// the popup's height is over its rows.
    pub fn chrome_h(&self) -> f32 {
        2.0 * cce_ui::layout::root_plate_inset() + self.tab_strip_h() + cce_ui::layout::textbox_height() + cce_ui::layout::root_plate_gap()
    }

    /// Horizontal chrome around a row's text: the text inset on both sides.
    /// What the popup's width is over its widest label.
    pub fn chrome_w(&self) -> f32 {
        2.0 * (self.text_x() - self.x)
    }

    /// Reserve the icon column only when some item on the ACTIVE tab resolved
    /// an icon, so the System tab's rows sit flush left while the Apps tab
    /// keeps its gutter. Within a tab the gutter still applies to every row
    /// (see [`Self::set_item_icons`]).
    fn recompute_icon_gutter(&mut self) {
        let any = self.all_items.iter().any(|t| self.icons.contains_key(t));
        let gutter = if any { ICON_PX + cce_ui::layout::CONTROL_TEXT_INSET } else { 0.0 };
        if gutter != self.icon_gutter {
            self.icon_gutter = gutter;
            // The row height follows the gutter (see `item_h`), so the
            // content height the scroll bounds hold just changed.
            self.update_scroll();
        }
    }

    /// Height of one row: taller when the list has an icon column. Every
    /// path that turns an index into a y — paint, hit-test, scroll bounds,
    /// keyboard snap, the popup's own height — reads it here.
    pub fn item_h(&self) -> f32 {
        if self.icon_gutter > 0.0 { ICON_ITEM_H } else { ITEM_H }
    }

    /// Give rows an icon column. Apps mode sets its whole map here; the
    /// window switcher adds to it through [`Self::extend_item_icons`]. Other
    /// Dmenu and Path items are arbitrary strings with nothing to look an icon
    /// up by, and they keep the flush-left layout they have always had because
    /// the gutter stays 0.
    pub fn set_item_icons(&mut self, icons: std::collections::HashMap<String, (u32, u32, u32)>) {
        self.icons = icons;
        self.recompute_icon_gutter();
    }

    /// Add icons for rows that arrived after the map was set — the window
    /// switcher's rows stream in over stdin.
    pub fn extend_item_icons(&mut self, icons: impl IntoIterator<Item = (String, (u32, u32, u32))>) {
        self.icons.extend(icons);
        self.recompute_icon_gutter();
    }

    pub fn has_item_icon(&self, item: &str) -> bool {
        self.icons.contains_key(item)
    }

    /// What a row draws for `item` — the item itself, except for the window
    /// switcher's rows (see `switcher_rows`).
    pub fn row_label<'a>(&self, item: &'a str) -> &'a str {
        if self.switcher_rows {
            split_switcher_row(item).0
        } else {
            item
        }
    }

    /// The square an icon is fitted into for the row drawn at `draw_y`.
    fn icon_rect(&self, draw_y: f32, item_h: f32, w: u32, h: u32) -> cce_ui::scene::layout::Rect {
        // Fit the longer side to ICON_PX so a non-square icon keeps its aspect
        // ratio and stays centered in the column.
        let (w, h) = (w.max(1) as f32, h.max(1) as f32);
        let s = ICON_PX / w.max(h);
        let (iw, ih) = (w * s, h * s);
        cce_ui::scene::layout::Rect {
            x: self.text_x() + (ICON_PX - iw) / 2.0,
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
        let content_h = self.filtered_items.len() as f32 * self.item_h();
        self.scroll_box.update_bounds_raw(content_h, self.list_y(), self.list_h());
        self.refresh_hover();
    }

    /// Filtered index of the row drawn at `(px, py)` — the ONE predicate the
    /// click and the hover share, so what lights up is what a press picks:
    /// inside the list, off the scrollbar strip (`hit()` spans it, and a
    /// press there once resolved to a row and committed it in dmenu mode),
    /// and a row `get_draw_y` places in the viewport — partially visible
    /// rows included, drawn cut by the clip, so an edge sliver counts.
    fn row_at(&self, px: f32, py: f32) -> Option<usize> {
        let item_h = self.item_h();
        if !self.scroll_box.hit(px, py) || self.scroll_box.hit_scrollbar(px, py) {
            return None;
        }
        let virtual_y = py - self.scroll_box.viewport_y + self.scroll_box.scroll_y;
        if virtual_y < 0.0 {
            return None;
        }
        let idx = (virtual_y / item_h).floor() as usize;
        (idx < self.filtered_items.len()
            && self.scroll_box.get_draw_y(idx as f32 * item_h, item_h).is_some())
            .then_some(idx)
    }

    /// The pointer moved to `(px, py)`; true when the hovered row changed.
    pub fn hover_at(&mut self, px: f32, py: f32) -> bool {
        self.cursor = Some((px, py));
        self.refresh_hover()
    }

    /// The pointer MOVED to `(px, py)` — a Motion, unlike the Enter that a
    /// popup mapping under a resting pointer also sends. In the switcher,
    /// moving onto a row selects it, so releasing the hold modifier switches
    /// to the window under the pointer. Only motion does this: rows shifting
    /// under a still pointer (a refilter, a scroll, the popup mapping where
    /// it rests) must not steal the selection Super+Tab just advanced. True
    /// when anything drawn changed.
    pub fn pointer_moved(&mut self, px: f32, py: f32) -> bool {
        let moved = self.cursor != Some((px, py));
        let mut changed = self.hover_at(px, py);
        if self.switcher_rows && moved && self.hovered != self.motion_row {
            self.motion_row = self.hovered;
            // No `snap_to_selected`: the row is already drawn (`row_at`), and
            // scrolling a cut edge row fully in would slide the next row under
            // the pointer and select that one too.
            if let Some(idx) = self.hovered.filter(|&i| i != self.selected) {
                self.selected = idx;
                changed = true;
            }
        }
        changed
    }

    /// The pointer left the surface; true when a row was lit.
    pub fn clear_hover(&mut self) -> bool {
        self.cursor = None;
        self.motion_row = None;
        self.refresh_hover()
    }

    /// Re-derive the hovered row from the last pointer position — the rows
    /// move under a stationary pointer on every scroll and refilter. True on
    /// change, so callers can skip the re-upload when nothing moved.
    pub fn refresh_hover(&mut self) -> bool {
        let now = self.cursor.and_then(|(px, py)| self.row_at(px, py));
        let tab_now = self.cursor.and_then(|(px, py)| self.tab_at(px, py));
        let changed = now != self.hovered || tab_now != self.tab_hovered;
        self.hovered = now;
        self.tab_hovered = tab_now;
        changed
    }

    pub fn snap_to_selected(&mut self) {
        let item_h = self.item_h();
        let viewport_h = self.list_h();
        let content_h = self.filtered_items.len() as f32 * item_h;

        if self.filtered_items.is_empty() {
            return;
        }

        let virtual_selected_y = self.selected as f32 * item_h;
        let old_scroll = self.scroll_box.scroll_y;
        if virtual_selected_y + item_h > self.scroll_box.scroll_y + viewport_h {
            self.scroll_box.scroll_y = virtual_selected_y + item_h - viewport_h;
        } else if virtual_selected_y < self.scroll_box.scroll_y {
            self.scroll_box.scroll_y = virtual_selected_y;
        }

        let max_scroll = (content_h - viewport_h).max(0.0);
        self.scroll_box.scroll_y = self.scroll_box.scroll_y.clamp(0.0, max_scroll);
        // Keyboard navigation scrolls the list without touching the wheel
        // path — raise the sink-behind bar for it too.
        if (self.scroll_box.scroll_y - old_scroll).abs() > 0.01 {
            self.scroll_box.notify_scrolled();
        }
        self.refresh_hover();
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

        let inset = cce_ui::layout::root_plate_inset();
        self.scroll_box.set_rect(self.x + inset, self.list_y(), self.w - inset * 2.0, self.list_h());
        self.update_scroll();
    }
}

impl cce_ui::widget::Paint for FuzzelWidget {
    fn color(&self) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn paint(&self, _rect: cce_ui::scene::layout::Rect, ctx: &mut cce_ui::scene::paint::PaintCtx) {
        use cce_ui::scene::layout::Rect;
        let pad = cce_ui::layout::root_plate_inset();

        // Tab strip — one well carved into the window plate with the segments
        // butting together on its floor and the active one raised back out of
        // it, which is the toolkit's recessed ButtonStrip treatment rendered
        // by hand (this widget paints straight onto the PaintCtx; nesting a
        // real ButtonStrip would need a child layout pass it does not have).
        if let Some(strip) = self.tab_strip_rect() {
            let radius = cce_ui::layout::button_corner_radius();
            let depth = cce_ui::layout::bevel_width().min(strip.height * 0.2);
            let (floor, radii) =
                cce_ui::layout::carve_inside(strip, (radius, radius, radius, radius), depth);
            ctx.recess(floor, radii, depth);
            let inset = depth * 0.5;
            let seg_r = (radius - inset).max(0.0);
            for i in 0..self.tabs.len() {
                let Some(r) = self.tab_rect(i) else { continue };
                let seg = Rect {
                    x: r.x + inset,
                    y: r.y + inset,
                    width: (r.width - 2.0 * inset).max(0.0),
                    height: (r.height - 2.0 * inset).max(0.0),
                };
                if i == self.active_tab {
                    // Faceless on purpose: the floor shows through the raised
                    // plate, so the active tab reads as part of the strip
                    // rather than a chip dropped on it.
                    ctx.control_plate(
                        &cce_ui::widget::ControlPlate::control(
                            seg,
                            seg_r,
                            cce_ui::widget::PlateStance::Raised,
                            None,
                        )
                        .with_depth(depth),
                    );
                } else if self.tab_hovered == Some(i) {
                    ctx.rounded_rect(seg, seg_r, (true, true, true, true), cce_ui::colors::PANEL_MENU_HOVER);
                }
            }
        }

        // Search bar — a well recessed into the plate, its rim lit in the
        // highlight accent (the toolkit's focused-well treatment; the query
        // line always holds keyboard focus here). Replaces the flat fill +
        // 1px border quads.
        let search_h = cce_ui::layout::textbox_height();
        let well = Rect { x: self.x + pad, y: self.search_y(), width: self.w - pad * 2.0, height: search_h };
        ctx.quad(well, [0.10, 0.10, 0.14, 1.0]);
        let depth = cce_ui::layout::bevel_width().min(search_h * 0.2);
        let hc = cce_ui::color::highlight_primary_color();
        ctx.recess_tinted(well, (0.0, 0.0, 0.0, 0.0), depth, [hc[0], hc[1], hc[2]]);

        // The list's scrollbar idles UNDER its translucent bg fill, every
        // frame: down the list's centre line (the toolkit centres a
        // sink-behind bar), dimly seen through the fill, taking no press.
        // The fore copy fades in over the rows below while a scroll holds it.
        self.scroll_box.push_scrollbar_prims(ctx);
        let sb = &self.scroll_box;
        ctx.quad(Rect { x: sb.x, y: sb.y, width: sb.w, height: sb.h }, cce_ui::color::list_bg_color());

        // The list content — selection chip, icons, row labels — under the
        // list-viewport clip: `get_draw_y` returns PARTIALLY visible rows (the
        // toolkit ScrollRegion's intersection contract), so an edge row
        // renders cut by the clip instead of vanishing. Row text carries the
        // clip as bounds through walk_text_labels → prepare_text.
        let viewport = Rect {
            x: self.scroll_box.x,
            y: self.scroll_box.viewport_y,
            width: self.scroll_box.w,
            height: self.scroll_box.viewport_h,
        };
        ctx.clip(viewport, |ctx| {
            // Selected Item Highlight
            let item_h = self.item_h();
            if !self.filtered_items.is_empty() {
                let virtual_selected_y = self.selected as f32 * item_h;
                if let Some(draw_y) = self.scroll_box.get_draw_y(virtual_selected_y, item_h) {
                    // A raised beveled chip, not a flat tint: the selection reads
                    // as sitting proud of the list the way focused panes do. No
                    // width reserved for the scrollbar anymore — the sink-behind
                    // bar idles under the list bg and rides OVER the rows while
                    // raised, so the chip keeps its full width either way.
                    let sel = Rect {
                        x: self.x + pad + CHIP_STANDOFF,
                        y: draw_y,
                        width: self.w - pad * 2.0 - 2.0 * CHIP_STANDOFF,
                        height: item_h - CHIP_STANDOFF,
                    };
                    let depth = cce_ui::color::plate_bevel_width().min(sel.height * 0.2);
                    ctx.bevel(sel, (4.0, 4.0, 4.0, 4.0), &cce_ui::scene::Material::from_fill([0.20, 0.35, 0.65, 0.9]), depth);
                }
            }

            // Hover wash — a flat, translucent pass of the selection colour
            // on the chip's footprint under the pointer. Flat on purpose: the
            // bevelled chip says "this is what Enter picks", the wash only
            // "this is what a click would pick". Never on the selected row,
            // which already wears the chip.
            if let Some(idx) = self.hovered.filter(|&i| i != self.selected) {
                if let Some(draw_y) = self.scroll_box.get_draw_y(idx as f32 * item_h, item_h) {
                    let hov = Rect {
                        x: self.x + pad + CHIP_STANDOFF,
                        y: draw_y,
                        width: self.w - pad * 2.0 - 2.0 * CHIP_STANDOFF,
                        height: item_h - CHIP_STANDOFF,
                    };
                    ctx.quad(hov, [0.20, 0.35, 0.65, 0.35]);
                }
            }

            // App icons, on the same virtualization predicate as the labels:
            // only rows `get_draw_y` places in the viewport are emitted, so a
            // 300-app list still costs one image quad per visible row.
            if self.icon_gutter > 0.0 {
                let item_h = self.item_h();
                for (idx, item_text) in self.filtered_items.iter().enumerate() {
                    let Some((image, iw, ih)) = self.icons.get(item_text).copied() else { continue };
                    if let Some(draw_y) = self.scroll_box.get_draw_y(idx as f32 * item_h, item_h) {
                        ctx.image(image, self.icon_rect(draw_y, item_h, iw, ih), 1.0);
                    }
                }
            }

            // Visible item labels.
            for l in self.row_labels() {
                ctx.text(l.text, l.x, l.y, l.font_size, l.color);
            }
        });

        // The fore copy rides over the rows at the fade, so the raise and the
        // sink are a fade rather than a flip.
        self.scroll_box.push_scrollbar_fore(ctx);

        // Prompt/query line and the empty-state notice — outside the list clip.
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
            // `row_at` is the same predicate the hover and the paint loop use
            // (visible ⇒ clickable, culled ⇒ not), so a press picks the row
            // that is lit under the pointer.
            if let Some(idx) = self.row_at(*px, *py) {
                self.selected = idx;
                return true;
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
/// style: deliberate — a placement clearance against the OUTPUT edge, not an
/// inset on any plate; the spacing ladder has no rung for it.
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

    /// Where this popup was invoked, in layout px, when it was given a
    /// position at all (the desktop menu passes its click through; a
    /// keyboard-summoned launcher has no "here" to mean). Used to place the
    /// launched window on that grid square.
    invoked_at: Option<(i32, i32)>,

    fuzzel: Handle<cce_ui::widget::Adapted<FuzzelWidget>>,
    json_layout: Option<Handle<cce_ui::widget::Adapted<JsonLayoutWidget>>>,
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

    max_width: u32,
    max_height: u32,
    select_item: Option<String>,
    switcher_mode: bool,
    /// `Some` in `--choose` mode ([`Chooser`]).
    chooser: Option<Chooser>,
    /// app_id → icon name for the switcher's rows ([`desktop_icon_index`]),
    /// built on the first row that needs it and kept for the popup's life.
    switcher_icon_index: Option<std::collections::HashMap<String, String>>,
    /// When `State::new` began, and whether the first frame that shows any
    /// rows has been logged since. A streamed list (Dmenu, the switcher)
    /// arrives after the popup opens, so the first frame alone can be an
    /// empty list; this is the one that shows the user something to pick.
    opened_at: std::time::Instant,
    rows_frame_logged: bool,
    /// Whether the compositor has configured the surface yet. Nothing may be
    /// drawn before: a buffer attached ahead of a layer surface's first
    /// configure is a protocol error, and the compositor disconnects the
    /// whole client, which for the daemon is every popup after it. Until
    /// 2026-10-05 the renderer took long enough to build that the configure
    /// always won; with the daemon's kept renderer a popup is ready in
    /// microseconds, and the switcher's streamed rows asked for a frame first.
    configured: bool,
    last_tick: std::time::Instant,
    ui_context: cce_ui::context::UiContext,
    /// Dissolved root plate container (Phase 6as): the plate was a pure value-holder for the
    /// window background — color (at root plate opacity), radius, rect. No border, no
    /// children, no events.
    window_rect: (f32, f32, f32, f32),
    window_bg: [f32; 4],
    select_and_close_requested: bool,
    /// `Some` for `-x/-y` popups (always layer-shell): re-applied on every resize so
    /// an auto-sizing window can't grow off the screen edge.
    placement: Option<Placement>,
}

/// Logical-px width one JSON-layout widget wants for the auto-sizing popup.
///
/// Buttons are the subtlety: `cce_ui::widget::Button` draws its label with the control text
/// inset on each side (see `Button::paint`) in the *button* font — not the menubar font
/// `measure_text` assumes. So measure the label the way the button itself does
/// (`measure_text_width` in the button font/size) and budget the button's two insets on top
/// of the container's root-plate inset per side; otherwise a left-justified label starts
/// an inset in and spills past the button's right edge (`JsonLayoutWidget::layout_children`
/// sets `usable_w = width - 2 * root_plate_inset()`).
fn json_widget_desired_width(widget_type: &str, text: &str) -> f32 {
    let margins = 2.0 * cce_ui::layout::root_plate_inset();
    match widget_type {
        "button" => {
            let (family, size) = cce_ui::layout::parse_font_string(&cce_ui::layout::button_font());
            cce_ui::widget::display::measure_text_width(text, &family, size.unwrap_or(12.0))
                + margins
                + 2.0 * cce_ui::layout::CONTROL_TEXT_INSET
        }
        "label" => cce_ui::widget::display::measure_text(text, 13.0) + margins,
        // The remainders are each control's own width beside its label (the
        // checkbox's toggle-height box and the 10px before its label, the
        // spinbox's field, the slider's track), not spacing.
        "checkbox" => cce_ui::widget::display::measure_text(text, 13.0) + margins + cce_ui::layout::toggle_height() + 10.0,
        "spinbox" | "color" => cce_ui::widget::display::measure_text(text, 13.0) + margins + 88.0,
        "slider" => cce_ui::widget::display::measure_text(text, 13.0) + margins + 128.0,
        _ => 150.0,
    }
}

/// [`json_widget_desired_width`] for a widget as configured on page
/// `page_idx` of `page` (its widgets): a button's width also holds its
/// glyphs (`json_layout::button_glyphs`), and its text is measured without
/// the mark the glyph replaces.
fn json_conf_desired_width(w_conf: &crate::json_layout::JsonWidgetConfig, page: &[crate::json_layout::JsonWidgetConfig], page_idx: usize) -> f32 {
    use crate::json_layout::{button_font_size, button_glyphs, glyph_room, page_has_lead};
    if w_conf.widget_type != "button" {
        return json_widget_desired_width(&w_conf.widget_type, &w_conf.text);
    }
    let (_, text, trail) = button_glyphs(&w_conf.text, page_idx, w_conf.target_page);
    json_widget_desired_width("button", text)
        + glyph_room(page_has_lead(page, page_idx), trail.is_some(), button_font_size())
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
        chooser_mode: bool,
        json_layout_config: Option<JsonLayoutConfig>,
        parent_app_id: Option<String>,
        fonts: Option<(FontSystem, SwashCache)>,
        renderer: Option<VkRenderer>,
    ) -> Result<
        (Self, Option<cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1>),
        cce_ui::vk::SurfaceLost,
    > {
        let t_start = std::time::Instant::now();
        cce_ui::scale::set_scale_factor(scale as f32);
        let (width, height) = if mode == LauncherMode::Json {
            if let Some(ref config) = json_layout_config {
                let w = config.width.unwrap_or_else(|| {
                    let mut max_widget_w = 120.0f32; // fallback minimum
                    if let Some(ref widgets) = config.widgets {
                        for w_conf in widgets {
                            let w_w = json_conf_desired_width(w_conf, widgets, 0);
                            if w_w > max_widget_w {
                                max_widget_w = w_w;
                            }
                        }
                    } else if let Some(ref pages) = config.pages {
                        for (page_idx, page) in pages.iter().enumerate() {
                            for w_conf in &page.widgets {
                                let w_w = json_conf_desired_width(w_conf, &page.widgets, page_idx);
                                if w_w > max_widget_w {
                                    max_widget_w = w_w;
                                }
                            }
                        }
                    }
                    max_widget_w.round() as u32
                });
                let h = config.height.unwrap_or_else(|| {
                    // The same walk as `JsonLayoutWidget::layout_children`:
                    // the root-plate inset above, a root-plate gap after each
                    // widget, and the trailing gap traded for the inset below.
                    let inset = cce_ui::layout::root_plate_inset();
                    let gap = cce_ui::layout::root_plate_gap();
                    let mut current_y = inset;
                    if let Some(ref widgets) = config.widgets {
                        for w_conf in widgets {
                            let h = match w_conf.widget_type.as_str() {
                                "label" => 18.0,
                                "checkbox" => cce_ui::layout::toggle_height(),
                                "button" => cce_ui::widget::context_menu::ROW_H,
                                "spinbox" => cce_ui::layout::spinbox_height(),
                                "color" => cce_ui::layout::color_selector_height(),
                                _ => 20.0,
                            };
                            current_y += h + gap;
                        }
                    } else if let Some(ref pages) = config.pages {
                        let mut max_page_y = inset;
                        for page in pages {
                            let mut page_y = inset;
                            for w_conf in &page.widgets {
                                let h = match w_conf.widget_type.as_str() {
                                    "label" => 18.0,
                                    "checkbox" => cce_ui::layout::toggle_height(),
                                    "button" => cce_ui::widget::context_menu::ROW_H,
                                    "spinbox" => cce_ui::layout::spinbox_height(),
                                    "color" => cce_ui::layout::color_selector_height(),
                                    _ => 20.0,
                                };
                                page_y += h + gap;
                            }
                            if page_y > max_page_y {
                                max_page_y = page_y;
                            }
                        }
                        current_y = max_page_y;
                    }
                    current_y = (current_y - gap).max(inset) + inset;
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
        //
        // The daemon hands in the renderer it kept from the last popup, and
        // moving it onto this surface costs one swapchain. Building a new one
        // costs a device and every pipeline: 70-100 ms of a ~100 ms popup on
        // an idle machine, several hundred under load. Both fail only when
        // the connection is already dead under the surface.
        let t = std::time::Instant::now();
        let display_ptr = conn.backend().display_id().as_ptr() as *mut std::ffi::c_void;
        let surface_ptr = wl_surface.id().as_ptr() as *mut std::ffi::c_void;
        let renderer = match renderer {
            Some(mut kept) => {
                unsafe { kept.attach_surface(display_ptr, surface_ptr, pw, ph) }?;
                log::debug!("[timing] VkRenderer::attach_surface: {:?}", t.elapsed());
                kept
            }
            None => {
                let made = unsafe { VkRenderer::try_new(display_ptr, surface_ptr, pw, ph, 0.0) }?;
                log::debug!("[timing] VkRenderer::try_new: {:?}", t.elapsed());
                made
            }
        };

        // Reuse the daemon's font system across popups (a rebuild re-scans the
        // fonts dir and loses the shaping caches).
        let t = std::time::Instant::now();
        let (font_system, swash_cache) =
            fonts.unwrap_or_else(|| (cce_ui::create_font_system(), SwashCache::new()));
        log::debug!("[timing] font system: {:?}", t.elapsed());

        let mut fuzzel = FuzzelWidget::new(prompt);
        fuzzel.set_rect(0.0, 0.0, lw, lh);
        fuzzel.switcher_rows = switcher_mode;

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

            // Resolve every entry's Icon= against the icon theme. Each icon is
            // uploaded once for the daemon's life (see `icon_image`), so after
            // the first open this is a map lookup per entry.
            let t_icons = std::time::Instant::now();
            let icons: std::collections::HashMap<String, (u32, u32, u32)> = apps
                .iter()
                .filter_map(|app| {
                    let img = icon_image(app.icon.as_deref()?)?;
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

            // Apps is the only tabbed mode. Dmenu carries arbitrary caller
            // items (and the Super-Tab window switcher, whose Tab key must
            // keep cycling the highlight), Path is a raw $PATH dump, and Json
            // is not a list at all.
            fuzzel.set_tabs(vec![
                ("Apps".to_string(), Vec::new()),
                (
                    SYSTEM_TAB_TITLE.to_string(),
                    SYSTEM_COMMANDS.iter().map(|c| c.name.to_string()).collect(),
                ),
            ]);

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

        let invoked_at = match (x_pos, y_pos) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        };
        // The context owns the widgets; the app keeps their handles.
        let mut ui_context = cce_ui::context::UiContext::new();
        let mut state = Self {
            invoked_at,
            window: Some(window),
            wl_surface,
            renderer: Some(renderer),
            vertex_data: Vec::new(),
            frame_batches: Vec::new(),
            frame_images: Vec::new(),
            plate_features: Vec::new(),
            fuzzel: ui_context.insert(fuzzel),
            json_layout: json_layout.map(|w| ui_context.insert(w)),
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
            chooser: chooser_mode.then(Chooser::default),
            switcher_icon_index: None,
            opened_at: t_start,
            rows_frame_logged: false,
            configured: false,
            last_tick: std::time::Instant::now(),
            ui_context,
            window_rect,
            window_bg: bg_color,
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
        Ok((state, cce_toplevel))
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
                let mut items = lock.items.clone();
                // The chooser's feed is IDs and its rows are labels: resolve a
                // new feed once, and leave the rows alone on an unchanged one.
                if let Some(chooser) = self.chooser.as_mut() {
                    if chooser.fed == items {
                        items = self.ui_context[self.fuzzel].tab_items(0).to_vec();
                    } else {
                        let entries = Chooser::labels(&items, &application_dirs());
                        chooser.fed = items;
                        chooser.by_label = entries.iter().map(|(id, label, _)| (label.clone(), id.clone())).collect();
                        // `-s` names the portal's last choice by ID; the row
                        // it selects is that ID's label.
                        if let Some(sel) = self.select_item.as_ref() {
                            if let Some((_, label, _)) = entries.iter().find(|(id, _, _)| id == sel) {
                                self.select_item = Some(label.clone());
                            }
                        }
                        self.ui_context[self.fuzzel].set_item_icons(
                            entries
                                .iter()
                                .filter_map(|(_, label, icon)| Some((label.clone(), icon_image(icon.as_deref()?)?)))
                                .collect(),
                        );
                        items = entries.into_iter().map(|(_, label, _)| label).collect();
                    }
                }
                // Against tab 0's items, not the active tab's: the feed only
                // ever fills the mode's own list, and comparing against
                // whatever tab the user is reading would differ every time.
                if self.ui_context[self.fuzzel].tab_items(0) != items.as_slice() {
                    if self.switcher_mode {
                        // Fields, not `self`: the stdin lock guard borrows it.
                        Self::resolve_switcher_icons(&mut self.ui_context[self.fuzzel], &mut self.switcher_icon_index, &items);
                    }
                    self.ui_context[self.fuzzel].set_tab_items(0, items);
                    changed = true;
                    if let Some(ref select_name) = self.select_item {
                        let select_lower = select_name.to_lowercase();
                        if let Some(idx) = self.ui_context[self.fuzzel].filtered_items.iter().position(|item| item.to_lowercase() == select_lower) {
                            self.ui_context[self.fuzzel].selected = idx;
                            self.ui_context[self.fuzzel].update_scroll();
                            self.ui_context[self.fuzzel].snap_to_selected();
                            self.select_item = None;
                        }
                    } else if self.switcher_mode && self.ui_context[self.fuzzel].filtered_items.len() > 1 {
                        self.ui_context[self.fuzzel].selected = 1;
                        self.ui_context[self.fuzzel].update_scroll();
                        self.ui_context[self.fuzzel].snap_to_selected();
                    }
                }

                if (cycles > 0 || cycles_back > 0) && !self.ui_context[self.fuzzel].filtered_items.is_empty() {
                    let len = self.ui_context[self.fuzzel].filtered_items.len() as isize;
                    let net = cycles as isize - cycles_back as isize;
                    self.ui_context[self.fuzzel].selected =
                        (self.ui_context[self.fuzzel].selected as isize + net).rem_euclid(len) as usize;
                    self.ui_context[self.fuzzel].update_scroll();
                    self.ui_context[self.fuzzel].snap_to_selected();
                    changed = true;
                }

                return changed;
            }
        }
        false
    }

    /// Give the switcher's window rows their app's icon. Rows stream in over
    /// stdin, so this runs per ingest and only resolves the rows that don't
    /// have an icon yet. Each icon is uploaded once and shared with the
    /// launcher (`icon_image`). `index` is `State::switcher_icon_index`.
    fn resolve_switcher_icons(
        fuzzel: &mut FuzzelWidget,
        index: &mut Option<std::collections::HashMap<String, String>>,
        items: &[String],
    ) {
        let new: Vec<&String> = items.iter().filter(|i| !fuzzel.has_item_icon(i)).collect();
        if new.is_empty() {
            return;
        }
        let t = std::time::Instant::now();
        let index = index.get_or_insert_with(|| desktop_icon_index(&application_dirs()));
        log::debug!("[timing] switcher icon index: {:?}", t.elapsed());
        let icons: Vec<(String, (u32, u32, u32))> = new
            .into_iter()
            .filter_map(|item| {
                let name = icon_name_for_app_id(index, split_switcher_row(item).1);
                let img = icon_image(&name)?;
                Some((item.clone(), img))
            })
            .collect();
        fuzzel.extend_item_icons(icons);
    }

    fn update_desired_size(&mut self) {
        if self.mode == LauncherMode::Json {
            if let Some(jl) = self.json_layout.and_then(|h| self.ui_context.get(h)) {
                let mut max_widget_w = 120.0f32; // fallback minimum
                let active_page = jl.active_page;
                
                for w in &jl.widgets {
                    if w.page_idx != active_page {
                        continue;
                    }
                    // `w.text` is a button's label without its mark; the
                    // glyphs take their own room beside it.
                    let mut w_w = json_widget_desired_width(&w.widget_type, &w.text);
                    if w.widget_type == "button" {
                        w_w += crate::json_layout::glyph_room(
                            w.lead_column,
                            w.trail.is_some(),
                            crate::json_layout::button_font_size(),
                        );
                    }
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
        let num_items = self.ui_context[self.fuzzel].filtered_items.len();
        let item_count = if num_items == 0 { 1 } else { num_items };
        let needed_height = self.ui_context[self.fuzzel].chrome_h() + (item_count as f32) * self.ui_context[self.fuzzel].item_h();
        let target_height = needed_height.min(self.max_height as f32);

        // Calculate max text width
        let mut max_text_w: f32 = 0.0;
        
        let query_text = if self.ui_context[self.fuzzel].query.is_empty() {
            format!("{}{}", self.ui_context[self.fuzzel].prompt, "Type to search...")
        } else {
            format!("{}{}", self.ui_context[self.fuzzel].prompt, self.ui_context[self.fuzzel].query)
        };
        let buf = make_text_buffer(&mut self.font_system, &query_text, 14.0);
        let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
        if tw > max_text_w {
            max_text_w = tw;
        }

        if self.ui_context[self.fuzzel].filtered_items.is_empty() {
            let buf = make_text_buffer(&mut self.font_system, "No matches found", 13.0);
            let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
            if tw > max_text_w {
                max_text_w = tw;
            }
        } else {
            for item in &self.ui_context[self.fuzzel].filtered_items {
                let label = self.ui_context[self.fuzzel].row_label(item);
                let buf = make_text_buffer(&mut self.font_system, label, 13.0);
                let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
                if tw > max_text_w {
                    max_text_w = tw;
                }
            }
        }

        // The strip's segments are equal shares of the run, so the whole run
        // has to hold its widest title n times over or the narrowest tab
        // clips. (Today it fits inside the 300px floor below; it is measured
        // rather than assumed so adding a third tab cannot quietly break it.)
        let titles: Vec<String> = self.ui_context[self.fuzzel].tabs.iter().map(|t| t.title.clone()).collect();
        if titles.len() > 1 {
            let mut widest = 0.0f32;
            for title in &titles {
                let buf = make_text_buffer(&mut self.font_system, title, TAB_FONT_PX);
                let tw = buf.layout_runs().next().map(|r| r.line_w).unwrap_or(0.0);
                widest = widest.max(tw);
            }
            // Each title gets the control text inset on both sides, as a
            // button label does.
            let strip_w = (widest + 2.0 * cce_ui::layout::CONTROL_TEXT_INSET) * titles.len() as f32;
            if strip_w > max_text_w {
                max_text_w = strip_w;
            }
        }

        let scrollbar_w = if needed_height > self.max_height as f32 { 10.0 } else { 0.0 };
        let needed_width = max_text_w + self.ui_context[self.fuzzel].chrome_w() + self.ui_context[self.fuzzel].icon_gutter + scrollbar_w;
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
            if let Some(jh) = self.json_layout {
                cce_ui::scale::set_scale_factor(self.scale as f32);
                self.ui_context[jh].set_rect(0.0, 0.0, w, h);
            }
        } else {
            self.ui_context[self.fuzzel].set_rect(0.0, 0.0, w, h);
        }
    }

    fn collect_display_list(&self) -> cce_ui::scene::paint::DisplayList {
        use cce_ui::scene::layout::Rect;
        let mut pc = cce_ui::scene::paint::PaintCtx::new();

        // 1. Window background — the dissolved root plate container's emission (base color at
        // the configured root plate opacity), now as a beveled plate: the rolled
        // rim makes the popup read as a raised surface instead of a flat sheet.
        let mut bg_color = self.window_bg;
        if bg_color[3] > 0.001 {
            bg_color[3] = cce_ui::color::root_plate_opacity();
        }
        if bg_color[3] > 0.0 {
            // PlateSpec (cce-ui RFC 7b, closing 7b-2's cce-cloud question): the
            // overlay SHARES the decorated-window silhouette. The compositor
            // never clips layer surfaces (layer_shell.rs passes blur radius 0;
            // corner rounding is the app's), so what this draws IS the
            // silhouette — and a launcher-sized panel wearing the nominal
            // widget-scale radius reads nearly square next to the windows
            // around it. All four corners are window corners; the spec snaps
            // them to the shared curve. Depth stays this app's shallower
            // plate_bevel_width, not the window default.
            let (wx, wy, ww, wh) = self.window_rect;
            pc.plate_spec(
                &cce_ui::scene::paint::PlateSpec::root_at(Rect { x: wx, y: wy, width: ww, height: wh })
                    .with_material(cce_ui::scene::Material::opaque(bg_color))
                    .with_depth(cce_ui::color::plate_bevel_width()),
            );
        }

        // 2. Child widgets, through the paint walk: bevel/recess prims reach the
        // tessellator instead of being flattened away by the legacy quad bridges.
        if self.mode == LauncherMode::Json {
            if let Some(jl) = self.json_layout.and_then(|h| self.ui_context.get(h)) {
                jl.paint_self(&self.ui_context, &mut pc);
            }
        } else {
            self.ui_context[self.fuzzel].paint_self(&self.ui_context, &mut pc);
        }

        pc.finish()
    }

    fn upload_vertices(&mut self) {
        let dl = self.collect_display_list();
        let (verts, batches, images, features) =
            tessellate(&dl, self.width, self.height, self.scale as f32);
        // No close fade is applied here any more, and deliberately so. This
        // used to multiply every vertex and image alpha by a fade factor and
        // then DROP the SDF-plate batches outright — which took the window's
        // whole background plate with them, since a plate batch IS its cover
        // quad, leaving the rows and text dissolving over nothing. The fade is
        // the compositor's now (`cce_ui::ipc::request_close_fade`): it ramps
        // this surface's scene-node opacity, which fades the backdrop blur
        // behind the popup along with it.
        // The GPU upload happens in VkRenderer::draw_frame_2d, which consumes
        // vertex_data every frame.
        self.vertex_data = verts;
        self.frame_batches = batches;
        self.frame_images = images;
        self.plate_features = features;
    }

    fn prepare_text(&mut self) {
        let scale_f32 = self.scale as f32;

        let mut widget_labels: Vec<(TextLabel, Option<[f32; 4]>)> = Vec::new();
        if self.mode == LauncherMode::Json {
            if let Some(jl) = self.json_layout.and_then(|h| self.ui_context.get(h)) {
                widget_labels.extend(walk_text_labels(&self.ui_context, jl));
            }
        } else {
            widget_labels.extend(walk_text_labels(&self.ui_context, &self.ui_context[self.fuzzel]));
        }

        let mut buffers: Vec<Buffer> = Vec::with_capacity(widget_labels.len());
        for (label, _) in &widget_labels {
            buffers.push(make_text_buffer(&mut self.font_system, &label.text, label.font_size));
        }

        let spans: Vec<TextSpan> = buffers
            .iter()
            .zip(widget_labels.iter())
            .map(|(buf, (label, bounds))| TextSpan {
                buffer: buf,
                left: (label.x * scale_f32).round(),
                top: (label.y * scale_f32).round(),
                // Buffers are shaped at logical size; the span scales to physical.
                scale: scale_f32,
                // Logical merged clip (walk clip ∩ prim bounds) → physical px,
                // so a partially visible row's text is cut at the viewport.
                bounds: bounds.map(|b| {
                    [
                        (b[0] * scale_f32).floor() as i32,
                        (b[1] * scale_f32).floor() as i32,
                        (b[2] * scale_f32).ceil() as i32,
                        (b[3] * scale_f32).ceil() as i32,
                    ]
                }),
                default_color: [
                    label.color[0] as f32 / 255.0,
                    label.color[1] as f32 / 255.0,
                    label.color[2] as f32 / 255.0,
                    // TextLabel carries RGB only; labels are opaque.
                    1.0,
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

    /// Draw a frame, or nothing before the surface's first configure (see
    /// `State::configured`): true if a frame was drawn. The configure
    /// handlers ask for a redraw, so a skipped frame is drawn right after it.
    fn render(&mut self) -> bool {
        if !self.configured {
            return false;
        }
        let now = std::time::Instant::now();
        self.last_tick = now;

        self.upload_vertices();
        self.prepare_text();
        self.renderer.as_mut().unwrap().draw_frame_2d(Frame2D {
            verts: &self.vertex_data,
            batches: &self.frame_batches,
            overlay_verts: &[],
            images: &self.frame_images,
            plate_features: &self.plate_features,
            clear_color: [0.0; 4],
            // Always a full frame: the popup is small and repaints whole.
            damage: None,
        });
        if !self.rows_frame_logged && !self.ui_context[self.fuzzel].filtered_items.is_empty() {
            self.rows_frame_logged = true;
            log::info!("[timing] open -> first frame with rows: {:?}", self.opened_at.elapsed());
        }
        true
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
    /// The switcher's hold modifier — Super or Alt, whichever chord opened
    /// it — is still down. Starts true in switcher mode (the chord that
    /// opened it is held); its release commits the highlighted row.
    switch_held: bool,
    switcher_mode: bool,
    /// When the compositor's close dissolve ends and this popup may go, set
    /// by `trigger_close`. `None` while the popup is live. The surface has to
    /// stay mapped until then — the fade is the compositor ramping this
    /// surface's scene-node opacity, and a destroyed surface cuts it off.
    fade_until: Option<std::time::Instant>,
    cce_toplevel: Option<cce_ui::protocol::cce_window_management_v1::zcce_toplevel_v1::ZcceToplevelV1>,
    selected_item: Option<String>,
    /// The event loop, for the keyboard's repeat timer. Set before the first
    /// roundtrip, since that is when the seat announces its keyboard.
    loop_handle: calloop::LoopHandle<'static, AppState>,
}

impl AppState {
    fn trigger_close(&mut self) {
        if self.fade_until.is_some() {
            return;
        }
        // The compositor owns the dissolve and its duration; all this side
        // does is hold the surface open for as long as it asks. A zero
        // answer — fading configured off, or no compositor — means go now.
        let fade = cce_ui::ipc::request_close_fade();
        if fade.is_zero() {
            self.exit = true;
        } else {
            self.fade_until = Some(std::time::Instant::now() + fade);
        }
    }

    fn trigger_select_and_close(&mut self) {
        let mut should_close = false;
        if let Some(st) = &mut self.state {
            if !st.ui_context[st.fuzzel].filtered_items.is_empty() {
                if let Some(item) = st.ui_context[st.fuzzel].filtered_items.get(st.ui_context[st.fuzzel].selected) {
                    let answer = st.chooser.as_ref().map_or_else(|| item.clone(), |c| c.answer(item));
                    println!("{}", answer);
                    self.selected_item = Some(answer);
                    if !run_system_item(&st.ui_context[st.fuzzel], item) {
                        match st.mode {
                            LauncherMode::Apps => {
                                if let Some(app) = st.apps.iter().find(|app| &app.name == item) {
                                    record_app_launch(&app.name);
                                    place_next_at(&app.exec, st.invoked_at);
                                    spawn_app(app);
                                }
                            }
                            LauncherMode::Path => {
                                place_next_at(item, st.invoked_at);
                                spawn_command(item);
                            }
                            LauncherMode::Dmenu => {}
                            LauncherMode::Json => {}
                        }
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
            // With repeat: held keys re-fire at the compositor's
            // repeat_info rate (see `repeat_key`). Plain `get_keyboard` has
            // no repeat at all, which is how holding Backspace in the search
            // field deleted one character (until 2026-09-26).
            let keyboard = self
                .seat_state
                .get_keyboard_with_repeat(
                    qh,
                    &seat,
                    None,
                    self.loop_handle.clone(),
                    Box::new(|state: &mut AppState, _keyboard, event| state.repeat_key(event)),
                )
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
                            if let Some(jh) = st.json_layout {
                                // Routed dispatch (6bd shrink): one Event through the router.
                                let mv = cce_ui::widget::Event::PointerMove {
                                    x: event.position.0 as f32,
                                    y: event.position.1 as f32,
                                    local_x: event.position.0 as f32,
                                    local_y: event.position.1 as f32,
                                };
                                let root = jh.id();
                                if st.ui_context.propagate_event(&mv, root) {
                                    changed = true;
                                }
                            }
                            if changed {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        } else {
                            // Returns true only while a thumb drag is live; it also keeps
                            // the region's `hovered` current for the wheel/keyboard scope
                            // either way. A drag moves the rows under the pointer, so the
                            // hover row is re-derived after it (update_scroll does that).
                            let dragged = st.ui_context[st.fuzzel].scroll_box.cursor_moved(cx, cy);
                            if dragged {
                                st.ui_context[st.fuzzel].update_scroll();
                            }
                            if st.ui_context[st.fuzzel].pointer_moved(cx, cy) || dragged {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        }
                    }
                    // The entry into the surface arrives as Enter with the
                    // position, not as a Motion — a pointer that crosses onto
                    // the list from outside lands ON a row and must light it.
                    PointerEventKind::Enter { .. } => {
                        st.cursor_x = cx;
                        st.cursor_y = cy;
                        if st.mode != LauncherMode::Json && st.ui_context[st.fuzzel].hover_at(cx, cy) {
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    PointerEventKind::Leave { .. } => {
                        if st.mode != LauncherMode::Json && st.ui_context[st.fuzzel].clear_hover() {
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
                                if let Some(jh) = st.json_layout {
                                    let ev = cce_ui::widget::Event::MouseButton {
                                        button: cce_ui::widget::MouseButton::Left,
                                        state: cce_ui::widget::ElementState::Pressed,
                                        x: event.position.0 as f32,
                                        y: event.position.1 as f32,
                                        local_x: event.position.0 as f32,
                                        local_y: event.position.1 as f32,
                                    };
                                    let root = jh.id();
                                    if st.ui_context.propagate_event(&ev, root) {
                                        changed = true;
                                    }
                                }
                                if changed {
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                            } else if let Some(idx) = st.ui_context[st.fuzzel].tab_at(cx, cy) {
                                // Ahead of the row branch for the same reason
                                // the scrollbar is: ANY press `on_event`
                                // resolves is treated there as a selection and
                                // — in Dmenu/switcher mode — committed. A tab
                                // click must switch tabs, not choose a row.
                                if st.ui_context[st.fuzzel].switch_tab(idx) {
                                    st.update_desired_size();
                                    st.upload_vertices();
                                    self.redraw = true;
                                }
                            } else if st.ui_context[st.fuzzel].scroll_box.press(cx, cy) {
                                // Thumb grab or track jump. This must be handled here rather
                                // than inside `on_event`, because the row branch below treats
                                // ANY handled press as a selection and — in Dmenu/switcher
                                // mode — commits it and closes the popup. A scrollbar press
                                // must scroll, not choose.
                                st.ui_context[st.fuzzel].update_scroll();
                                st.upload_vertices();
                                self.redraw = true;
                            } else {
                                let prev_selected = st.ui_context[st.fuzzel].selected;
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
                                    st.ui_context.propagate_event(&ev, root)
                                };
                                if changed {
                                    // A single click launches — the fuzzel on_event only
                                    // reports presses it resolved to a really-drawn row
                                    // (scrollbar and clipped-sliver presses never get
                                    // here), so the click IS the choice, exactly as Enter.
                                    // (The old gate gated Apps/Path on `selected ==
                                    // prev_selected`, which read as the click doing
                                    // nothing.) The SWITCHER commits only a click on the
                                    // row already selected — which, since moving onto a
                                    // row selects it (`pointer_moved`), is any row the
                                    // pointer reached by motion. A click on a row it only
                                    // rests on (the popup mapped under it) selects first.
                                    let commit = !st.switcher_mode || st.ui_context[st.fuzzel].selected == prev_selected;
                                    if commit {
                                        if let Some(item) = st.ui_context[st.fuzzel].filtered_items.get(st.ui_context[st.fuzzel].selected) {
                                            let answer = st.chooser.as_ref().map_or_else(|| item.clone(), |c| c.answer(item));
                                            println!("{}", answer);
                                            self.selected_item = Some(answer);
                                            if !run_system_item(&st.ui_context[st.fuzzel], item) {
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
                                // A `target_page` button switches pages inside
                                // propagate_event (JsonLayoutWidget takes its
                                // click there, so it never reaches the
                                // clicked_btn_id scan below). The popup is
                                // sized per page — a submenu page with more
                                // rows than the first was clipped to the first
                                // page's height until this resize.
                                let mut page_switched = false;
                                if let Some(jh) = st.json_layout {
                                    let page_before = st.ui_context[jh].active_page;
                                    let ev = cce_ui::widget::Event::MouseButton {
                                        button: cce_ui::widget::MouseButton::Left,
                                        state: cce_ui::widget::ElementState::Released,
                                        x: event.position.0 as f32,
                                        y: event.position.1 as f32,
                                        local_x: event.position.0 as f32,
                                        local_y: event.position.1 as f32,
                                    };
                                    let root = jh.id();
                                    if st.ui_context.propagate_event(&ev, root) {
                                        changed = true;
                                    }
                                    page_switched = st.ui_context[jh].active_page != page_before;
                                    for w in &mut st.ui_context[jh].widgets {
                                        // take_click is an WidgetHost method; Phase 5 Buttons are
                                        // Adapted, so ask the box directly.
                                        if w.widget_type == "button" && w.widget.take_click() {
                                            clicked_btn_id = Some(w.id.clone());
                                            break;
                                        }
                                    }
                                }
                                if page_switched {
                                    st.update_desired_size();
                                    changed = true;
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
                                    if let Some(jl) = st.json_layout.and_then(|h| st.ui_context.get(h)) {
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
                            } else if st.ui_context[st.fuzzel].scroll_box.release() {
                                // Ends a thumb drag. Returns true only if one was live, so a
                                // plain click on a row is unaffected.
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        }
                    }
                    PointerEventKind::Axis { horizontal, vertical, source, .. } => {
                        // Synthesized the way cce-ui's runner does it
                        // (window_runner.rs, `axis_stop`): discrete steps are
                        // wheel notches (LineDelta), anything else is pixels
                        // 1:1 (PixelDelta), and a bare stop is the finger
                        // lift. The phase is published before the dispatch so
                        // the ScrollMotion under each scroll host knows whether
                        // to glide a notch, track a finger, or fling.
                        use cce_ui::widget::scroll_motion::{set_scroll_phase, ScrollPhase};
                        let factors = cce_ui::input::scroll_factors();
                        let discrete = horizontal.discrete != 0 || vertical.discrete != 0;
                        let no_delta = !discrete && horizontal.absolute == 0.0 && vertical.absolute == 0.0;
                        let stop = horizontal.stop || vertical.stop;
                        let phase = if stop && no_delta {
                            ScrollPhase::FingerEnd
                        } else if !discrete
                            && matches!(
                                source,
                                None | Some(wl_pointer::AxisSource::Finger) | Some(wl_pointer::AxisSource::Continuous)
                            )
                        {
                            ScrollPhase::Finger
                        } else {
                            ScrollPhase::Wheel
                        };
                        set_scroll_phase(phase);
                        let delta = if discrete {
                            let h = if horizontal.discrete != 0 { horizontal.discrete as f32 } else { horizontal.absolute as f32 / 10.0 };
                            let v = if vertical.discrete != 0 { vertical.discrete as f32 } else { vertical.absolute as f32 / 10.0 };
                            cce_ui::widget::MouseScrollDelta::LineDelta(-h * factors.mouse as f32, -v * factors.mouse as f32)
                        } else {
                            cce_ui::widget::MouseScrollDelta::PixelDelta(cce_ui::widget::Position {
                                x: -horizontal.absolute * factors.trackpad,
                                y: -vertical.absolute * factors.trackpad,
                            })
                        };
                        if st.mode == LauncherMode::Json {
                            // The JSON layout's page scroll never received the
                            // wheel (only the fuzzel list did, and it is not the
                            // surface shown in this mode): route it the way the
                            // PointerMove above is routed.
                            let mut changed = false;
                            if let Some(jh) = st.json_layout {
                                let ev = cce_ui::widget::Event::MouseWheel { delta, x: cx, y: cy, local_x: cx, local_y: cy };
                                let root = jh.id();
                                if st.ui_context.propagate_event(&ev, root) {
                                    changed = true;
                                }
                            }
                            if changed {
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        } else if st.ui_context[st.fuzzel].scroll_box.wheel(&delta, st.cursor_x, st.cursor_y) {
                            st.ui_context[st.fuzzel].update_scroll();
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
        let prev_held = self.switch_held;
        self.ctrl_pressed = modifiers.ctrl;
        // Super+Tab or Alt+Tab: either can be bound to the switcher, and it
        // cannot tell which opened it, so it commits once neither is held.
        self.switch_held = modifiers.logo || modifiers.alt;
        log::debug!("update_modifiers: logo={}, alt={}, prev_held={}", modifiers.logo, modifiers.alt, prev_held);

        if self.switcher_mode && prev_held && !self.switch_held {
            log::info!("Switcher modifier (Super/Alt) released, selecting currently highlighted item");
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
            state.configured = true;
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
        if let Some(state) = &mut self.state {
            state.configured = true;
        }
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
    /// A held key's repeat: fed back in as another press, but only for keys
    /// where a repeat means more of the same (`key_repeats`).
    fn repeat_key(&mut self, event: smithay_client_toolkit::seat::keyboard::KeyEvent) {
        if key_repeats(event.keysym, event.utf8.as_deref()) {
            self.handle_key(event, cce_ui::widget::ElementState::Pressed);
        }
    }

    fn handle_key(&mut self, event: smithay_client_toolkit::seat::keyboard::KeyEvent, state: cce_ui::widget::ElementState) {
        use cce_ui::widget::{Key, NamedKey};
        if state != cce_ui::widget::ElementState::Pressed {
            return;
        }

        // Shift+Tab arrives as ISO_Left_Tab: step back through the tabs where
        // the list has them, else cycle the highlight backwards with wrap —
        // mirroring Tab's forward step below in both halves.
        if event.keysym == xkeysym::Keysym::ISO_Left_Tab {
            if let Some(st) = &mut self.state {
                if st.mode != LauncherMode::Json {
                    let mut changed = false;
                    if st.ui_context[st.fuzzel].cycle_tab(false) {
                        st.update_desired_size();
                        changed = true;
                    } else if !st.ui_context[st.fuzzel].filtered_items.is_empty() {
                        let len = st.ui_context[st.fuzzel].filtered_items.len();
                        st.ui_context[st.fuzzel].selected = (st.ui_context[st.fuzzel].selected + len - 1) % len;
                        st.ui_context[st.fuzzel].update_scroll();
                        st.ui_context[st.fuzzel].snap_to_selected();
                        changed = true;
                    }
                    if changed {
                        st.upload_vertices();
                        self.redraw = true;
                    }
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
                if let Some(jh) = st.json_layout {
                    let kev = cce_ui::widget::Event::KeyInput(key_event.clone());
                    let root = jh.id();
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
                        // Tab switches tabs where there are tabs — the Apps
                        // mode's Apps/System split. It keeps its old job of
                        // cycling the highlight with wrap in the single-tab
                        // modes, which is what the Super-Tab window switcher
                        // (Dmenu, one tab) rides on.
                        if st.ui_context[st.fuzzel].cycle_tab(true) {
                            st.update_desired_size();
                            st.upload_vertices();
                            self.redraw = true;
                        } else if !st.ui_context[st.fuzzel].filtered_items.is_empty() {
                            st.ui_context[st.fuzzel].selected = (st.ui_context[st.fuzzel].selected + 1) % st.ui_context[st.fuzzel].filtered_items.len();
                            st.ui_context[st.fuzzel].update_scroll();
                            st.ui_context[st.fuzzel].snap_to_selected();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if !st.ui_context[st.fuzzel].filtered_items.is_empty() {
                            st.ui_context[st.fuzzel].selected = (st.ui_context[st.fuzzel].selected + 1).min(st.ui_context[st.fuzzel].filtered_items.len() - 1);
                            st.ui_context[st.fuzzel].update_scroll();
                            st.ui_context[st.fuzzel].snap_to_selected();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        if st.ui_context[st.fuzzel].selected > 0 {
                            st.ui_context[st.fuzzel].selected -= 1;
                            st.ui_context[st.fuzzel].update_scroll();
                            st.ui_context[st.fuzzel].snap_to_selected();
                            st.upload_vertices();
                            self.redraw = true;
                        }
                    }
                    Key::Named(NamedKey::Backspace) => {
                        st.ui_context[st.fuzzel].query.pop();
                        st.ui_context[st.fuzzel].filter();
                        st.update_desired_size();
                        st.upload_vertices();
                        self.redraw = true;
                    }
                    _ => {
                        if let Some(text) = &event.utf8 {
                            for ch in text.chars().filter(|c| !c.is_control()) {
                                st.ui_context[st.fuzzel].query.push(ch);
                            }
                            st.ui_context[st.fuzzel].filter();
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
    let mut chooser_mode = false;
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
        } else if arg == "--choose" {
            chooser_mode = true;
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

    // Before the app state: the roundtrip below delivers the seat's keyboard,
    // and binding it with key repeat needs the loop.
    let mut event_loop = calloop::EventLoop::try_new().unwrap();

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
        switch_held: switcher_mode,
        switcher_mode,
        fade_until: None,
        cce_toplevel: None,
        selected_item: None,
        loop_handle: event_loop.handle(),
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
        chooser_mode,
        json_layout_config,
        parent_app_id,
        None,
        None,
    )
    .unwrap_or_else(|lost| {
        log::error!("cannot open the window: {lost}");
        std::process::exit(1);
    });

    app.window = state.window.clone();
    app.surface = Some(state.wl_surface.clone());
    app.cce_toplevel = cce_toplevel;
    app.state = Some(state);

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
    // A popup opens animating (its expand); the first ticks say when it stops.
    let mut animating = true;
    loop {
        // The compositor is dissolving the popup out; hold the surface open
        // until its deadline, then go. Nothing to redraw in the meantime —
        // the pixels stay put and the scene node's opacity does the work.
        if let Some(until) = app.fade_until {
            if std::time::Instant::now() >= until {
                app.exit = true;
            }
        }

        // Frame rate only while something moved last turn (a layout
        // animation, the scrollbar's hold, a wheel glide); otherwise sleep
        // until an event — input, the client's lines, a new request on the
        // daemon socket all wake the loop — or the fade's deadline. Until
        // 2026-10-06 an open popup woke every 16 ms with nothing moving.
        let timeout = if app.redraw {
            Some(std::time::Duration::from_millis(0))
        } else if animating {
            Some(std::time::Duration::from_millis(16))
        } else {
            app.fade_until.map(|t| t.saturating_duration_since(std::time::Instant::now()))
        };
        if let Err(e) = event_loop.dispatch(timeout, &mut app) {
            log::error!("compositor connection lost: {e}");
            std::process::exit(1);
        }

        if app.exit {
            break;
        }

        let now = std::time::Instant::now();
        let mut dt = now.duration_since(last_tick).as_secs_f32();
        last_tick = now;
        if dt > 0.1 {
            dt = 0.1;
        }
        animating = false;
        if let Some(state) = &mut app.state {
            if let Some(jh) = state.json_layout {
                if state.ui_context.lend_h(jh, |jl, ctx| jl.tick(dt, ctx)).unwrap_or(false) {
                    app.redraw = true;
                    animating = true;
                }
            }
            // Raise/sink upkeep for the list scrollbar (true while the
            // post-scroll hold runs or on the depth flip).
            if state.ui_context[state.fuzzel].scroll_box.tick(dt) {
                app.redraw = true;
                animating = true;
            }
            // A wheel glide moves the rows under a stationary pointer.
            if state.ui_context[state.fuzzel].refresh_hover() {
                state.upload_vertices();
                app.redraw = true;
                animating = true;
            }
        }

        if app.redraw {
            app.redraw = false;
            if let Some(state) = &mut app.state {
                let _ = state.render();
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
        } else if arg == "--dmenu" || arg == "--json" || arg == "--layout" || arg == "--choose" {
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

/// The compositor this daemon serves has gone: exit, as a Wayland client whose
/// display went away does.
///
/// The daemon holds ONE Wayland connection for its whole life, so it has to
/// notice when that connection dies. Until 2026-09-25 it did not: between
/// popups it sat in a blocking `accept()`, never reading the Wayland socket,
/// and a daemon that outlived a logout kept a connection to a compositor that
/// no longer existed. The next session's first popup was then built on that
/// dead connection and the renderer panicked (`No surface formats:
/// ERROR_SURFACE_LOST_KHR`). The idle wait now dispatches the Wayland source
/// alongside the listener, so a dead connection surfaces as a dispatch error
/// the moment the compositor goes; a surface that is lost anyway (the death
/// raced a request) lands here too.
///
/// Status 0, so `Restart=on-failure` does not relaunch it into a session with
/// no compositor: the unit is bound to `cce-session.target`, which startcce
/// starts with each compositor, and that start brings up a fresh daemon.
fn compositor_gone(why: impl std::fmt::Display) -> ! {
    log::warn!("compositor is gone ({why}); exiting");
    std::process::exit(0);
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

    let conn = match Connection::connect_to_env() {
        Ok(conn) => conn,
        Err(e) => {
            // A failure status, so systemd's Restart=on-failure tries again
            // if the session is still starting up.
            log::error!("cannot connect to the compositor: {e}");
            std::process::exit(1);
        }
    };
    let conn_clone = conn.clone();
    let (globals, mut event_queue) = registry_queue_init(&conn).unwrap();
    let qh = event_queue.handle();

    let compositor_state = CompositorState::bind(&globals, &qh).unwrap();
    let layer_shell_state = LayerShell::bind(&globals, &qh).unwrap();
    let shm_state = Shm::bind(&globals, &qh).unwrap();
    let seat_state = SeatState::new(&globals, &qh);
    let output_state = OutputState::new(&globals, &qh);
    let cce_wm = globals.bind::<cce_ui::protocol::cce_window_management_v1::zcce_window_manager_v1::ZcceWindowManagerV1, _, _>(&qh, 2..=4, ()).ok();

    // Before the app state: the roundtrip below delivers the seat's keyboard,
    // and binding it with key repeat needs the loop.
    let mut event_loop = calloop::EventLoop::try_new().unwrap();

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
        switch_held: false,
        switcher_mode: false,
        fade_until: None,
        cce_toplevel: None,
        selected_item: None,
        loop_handle: event_loop.handle(),
    };

    event_queue.roundtrip(&mut app).unwrap();

    // The renderer every popup draws with, built now on a surface that is
    // never mapped and then detached from it, so the first popup only
    // attaches it as every later one does. Each popup hands it back when it
    // closes (see the end of the loop). If this fails the first popup builds
    // its own, as all of them used to.
    let t_renderer = std::time::Instant::now();
    let mut renderer_slot: Option<VkRenderer> = {
        let scratch = app.compositor_state.create_surface(&qh);
        let made = unsafe {
            VkRenderer::try_new(
                conn_clone.backend().display_id().as_ptr() as *mut std::ffi::c_void,
                scratch.id().as_ptr() as *mut std::ffi::c_void,
                1,
                1,
                0.0,
            )
        };
        let kept = match made {
            Ok(mut r) => {
                r.detach_surface();
                Some(r)
            }
            Err(e) => {
                log::warn!("could not prewarm the popup renderer: {e}");
                None
            }
        };
        scratch.destroy();
        let _ = conn_clone.flush();
        kept
    };
    log::info!("[timing] daemon renderer prewarm: {:?}", t_renderer.elapsed());

    // Decode and queue the launcher's icons now, off the main thread, so its
    // first open does not read and rasterize every one (~350 ms). The uploads
    // drain into the renderer at the first popup's first frame. An open that
    // starts before this finishes simply shares the cache with it.
    if renderer_slot.is_some() {
        std::thread::spawn(|| {
            let t = std::time::Instant::now();
            let apps = scan_apps();
            let resolved = apps
                .iter()
                .filter(|app| app.icon.as_deref().and_then(icon_image).is_some())
                .count();
            log::info!(
                "[timing] launcher icon prewarm: {resolved} of {} in {:?}",
                apps.len(),
                t.elapsed()
            );
        });
    }

    let scale = cce_ui::wayland::detect_scale_factor(&app.output_state);
    let xdg_shell_state = smithay_client_toolkit::shell::xdg::XdgShell::bind(&globals, &qh).ok();

    let loop_handle = event_loop.handle();
    WaylandSource::new(conn, event_queue).insert(loop_handle.clone()).unwrap();
    // The listener wakes the loop too, so waiting for the next request also
    // reads the Wayland connection — see `compositor_gone`. The callback does
    // nothing: the accept after each dispatch takes the connection.
    let listener_wake = listener.try_clone().expect("dup the daemon socket");
    loop_handle
        .insert_source(
            calloop::generic::Generic::new(
                listener_wake,
                calloop::Interest::READ,
                calloop::Mode::Level,
            ),
            |_, _, _| Ok(calloop::PostAction::Continue),
        )
        .unwrap();
    let _ = listener.set_nonblocking(true);

    let mut pending: Option<std::os::unix::net::UnixStream> = None;
    loop {
        let mut stream = match pending.take() {
            Some(s) => s,
            None => match listener.accept() {
                Ok((s, _)) => s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if let Err(e) = event_loop.dispatch(None, &mut app) {
                        compositor_gone(e);
                    }
                    continue;
                }
                Err(_) => continue,
            },
        };

        let (stdin_sender, stdin_channel) = calloop::channel::channel::<()>();

        // Bounded: this runs on the launcher's main loop, so a client that
        // connected and said nothing used to freeze the launcher outright.
        // The line carries a dmenu list inline, hence the generous size.
        let Some(initial_line) = cce_ui::ipc::read_request_line(&stream, 16 * 1024 * 1024, std::time::Duration::from_secs(3)) else {
            continue;
        };
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
        let mut chooser_mode = false;
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
            } else if arg == "--choose" {
                chooser_mode = true;
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

        app.switch_held = switcher_mode;
        app.switcher_mode = switcher_mode;
        app.selected_item = None;

        let (state, cce_toplevel) = match State::new(
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
            chooser_mode,
            json_layout_config,
            parent_app_id,
            fonts_slot.take(),
            renderer_slot.take(),
        ) {
            Ok(made) => made,
            Err(lost) => {
                let _ = stream.write_all(b"\n");
                compositor_gone(lost);
            }
        };

        app.window = state.window.clone();
        app.surface = Some(state.wl_surface.clone());
        app.cce_toplevel = cce_toplevel;
        app.state = Some(state);

        app.exit = false;
        app.fade_until = None;

        // The reader thread only signals for lines that arrive after this
        // point; the initial_stdin items are already sitting in stdin_state,
        // so fire one signal to make the channel handler ingest them.
        let _ = stdin_sender.send(());

        log::debug!("[timing] request -> popup ready: {:?}", t_request.elapsed());
        let mut first_frame_logged = false;
        let mut last_tick = std::time::Instant::now();
        let mut animating = true;
        while !app.exit {
            // The compositor is dissolving the popup out; hold the surface open
            // until its deadline, then go. Nothing to redraw in the meantime —
            // the pixels stay put and the scene node's opacity does the work.
            if let Some(until) = app.fade_until {
                if std::time::Instant::now() >= until {
                    app.exit = true;
                }
            }

            // Frame rate only while something moved last turn (a layout
            // animation, the scrollbar's hold, a wheel glide); otherwise sleep
            // until an event — input, the client's lines, a new request on the
            // daemon socket all wake the loop — or the fade's deadline. Until
            // 2026-10-06 an open popup woke every 16 ms with nothing moving.
            let timeout = if app.redraw {
                Some(std::time::Duration::from_millis(0))
            } else if animating {
                Some(std::time::Duration::from_millis(16))
            } else {
                app.fade_until.map(|t| t.saturating_duration_since(std::time::Instant::now()))
            };
            if let Err(e) = event_loop.dispatch(timeout, &mut app) {
                let _ = stream.write_all(b"\n");
                compositor_gone(e);
            }

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
            animating = false;
            if let Some(st) = &mut app.state {
                if let Some(jh) = st.json_layout {
                    if st.ui_context.lend_h(jh, |jl, ctx| jl.tick(dt, ctx)).unwrap_or(false) {
                        app.redraw = true;
                        animating = true;
                    }
                }
                // Raise/sink upkeep for the list scrollbar (true while the
                // post-scroll hold runs or on the depth flip).
                if st.ui_context[st.fuzzel].scroll_box.tick(dt) {
                    app.redraw = true;
                    animating = true;
                }
                // A wheel glide moves the rows under a stationary pointer.
                if st.ui_context[st.fuzzel].refresh_hover() {
                    st.upload_vertices();
                    app.redraw = true;
                    animating = true;
                }
            }

            if app.redraw {
                app.redraw = false;
                if let Some(st) = &mut app.state {
                    if st.render() && !first_frame_logged {
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
            // Keep the renderer for the next popup, detached now: the State's
            // drop below destroys this wl_surface, and the swapchain must not
            // outlive it. The row icons stay uploaded with it (`icon_image`).
            if let Some(mut r) = st.renderer.take() {
                r.detach_surface();
                renderer_slot = Some(r);
            }
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
        // Regression: a left-justified Button draws its label the control text inset in
        // from its own left edge (Button::paint) in the button font. layout_children gives
        // the button `usable_w = popup_width - 2 * root_plate_inset()`. If the popup width
        // doesn't budget the button's inset per side, the label spills past the button's
        // right edge — the desktop context-menu bug. Every desktop-menu label must fit
        // within usable_w with the inset.
        let (family, size) = cce_ui::layout::parse_font_string(&cce_ui::layout::button_font());
        let size = size.unwrap_or(12.0);
        let text_inset = cce_ui::layout::CONTROL_TEXT_INSET;
        for label in [
            "Terminal", "Files", "Data Editor", "Applications",
            "System Settings", "Expose Windows", "Reload Config", "Logout",
        ] {
            let popup_w = json_widget_desired_width("button", label);
            // container margins, per layout_children
            let usable_w = popup_w - 2.0 * cce_ui::layout::root_plate_inset();
            let label_w = cce_ui::widget::display::measure_text_width(label, &family, size);
            // left inset + label + right breathing room must fit the button.
            assert!(
                usable_w >= label_w + 2.0 * text_inset,
                "button '{label}': usable_w {usable_w} < label {label_w} + {} inset",
                2.0 * text_inset,
            );
        }
    }

    /// A JSON menu's marks and page turns are cce-icons glyphs, by the
    /// toolkit's context-menu conventions: the label's leading mark is
    /// drawn as its glyph and not as text, a button turning to a later page
    /// ends in chevron-right, one turning back starts with chevron-left, and
    /// the labels of a page with a left glyph share one edge.
    #[test]
    fn json_menu_marks_and_page_turns_are_glyphs() {
        use crate::json_layout::button_glyphs;
        assert_eq!(button_glyphs("✓ Wrap", 0, None), (Some("check"), "Wrap", None));
        assert_eq!(button_glyphs("● Tiled", 1, None), (Some("circle"), "Tiled", None));
        assert_eq!(button_glyphs("○ Floating", 1, None), (Some("circle-outline"), "Floating", None));
        assert_eq!(button_glyphs("Window Mode", 0, Some(1)), (None, "Window Mode", Some("chevron-right")));
        assert_eq!(button_glyphs("Back", 1, Some(0)), (Some("chevron-left"), "Back", None));
        assert_eq!(button_glyphs("Terminal", 0, None), (None, "Terminal", None));

        let json = r#"{"pages": [
            {"title": "menu", "justify": "left", "widgets": [
                {"type": "button", "text": "Window Mode", "id": "mode_page", "target_page": 1},
                {"type": "button", "text": "Close Window", "id": "close"}
            ]},
            {"title": "Window Mode", "justify": "left", "widgets": [
                {"type": "button", "text": "○ Floating", "id": "floating"},
                {"type": "button", "text": "● Tiled", "id": "tiled"},
                {"type": "button", "text": "Back", "id": "back", "target_page": 0}
            ]}
        ]}"#;
        let config: JsonLayoutConfig = serde_json::from_str(json).unwrap();
        let mut layout = JsonLayoutWidget::new(&config);
        layout.set_rect(0.0, 0.0, 240.0, 300.0);
        let ctx = cce_ui::context::UiContext::new();

        let (labels, glyphs) = layout.labels_and_glyphs(&ctx);
        let texts: Vec<&str> = labels.iter().map(|(l, _)| l.text.as_str()).collect();
        assert_eq!(texts, ["Window Mode", "Close Window"]);
        let names: Vec<&str> = glyphs.iter().map(|g| g.0).collect();
        assert_eq!(names, ["chevron-right"]);
        // The chevron stands at the row's right end.
        let row = &layout.widgets[0];
        assert!(glyphs[0].1.x > row.x + row.w / 2.0);

        layout.active_page = 1;
        layout.layout_children();
        let (labels, glyphs) = layout.labels_and_glyphs(&ctx);
        let texts: Vec<&str> = labels.iter().map(|(l, _)| l.text.as_str()).collect();
        assert_eq!(texts, ["Floating", "Tiled", "Back"]);
        let names: Vec<&str> = glyphs.iter().map(|g| g.0).collect();
        assert_eq!(names, ["circle-outline", "circle", "chevron-left"]);
        // Every label starts after its glyph, on one edge.
        let edge = labels[0].0.x;
        for ((l, _), g) in labels.iter().zip(&glyphs) {
            assert!((l.x - edge).abs() < 0.01, "{} is not on the page's text edge", l.text);
            assert!(g.1.x + g.1.width <= l.x, "the {} glyph overlaps its label", g.0);
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

        // Check vertical sequence positions: the root-plate inset above, a
        // root-plate gap after each widget.
        let inset = cce_ui::layout::root_plate_inset();
        let gap = cce_ui::layout::root_plate_gap();
        assert_eq!(w_label_y, inset);
        assert_eq!(w_label_h, 18.0);

        assert_eq!(w_check_y, inset + 18.0 + gap); // y_prev + h_prev + spacing
        assert_eq!(w_check_h, cce_ui::layout::toggle_height());

        assert_eq!(w_btn_y, w_check_y + w_check_h + gap);
        assert_eq!(w_btn_h, cce_ui::widget::context_menu::ROW_H);

        // Check horizontal positioning (should match usable width: 300 - 2 * inset)
        let usable_w = 300.0 - 2.0 * inset;
        assert_eq!(w_label_x, inset);
        assert_eq!(w_label_w, usable_w);
        assert_eq!(w_check_x, inset);
        assert_eq!(w_check_w, usable_w);
        assert_eq!(w_btn_x, inset);
        assert_eq!(w_btn_w, usable_w);

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
        ctx.register_host(&mut layout);
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
    fn switcher_rows_split_into_title_and_app_id() {
        assert_eq!(split_switcher_row("~/src - Terminal (cce-terminal)"), ("~/src - Terminal", "cce-terminal"));
        // The last group is the id; the title's own parentheses are not.
        assert_eq!(
            split_switcher_row("Inbox (3) - Mail (org.gnome.Evolution)"),
            ("Inbox (3) - Mail", "org.gnome.Evolution")
        );
        // An untitled window is sent as the bare app_id, which is both halves.
        assert_eq!(split_switcher_row("firefox"), ("firefox", "firefox"));
    }

    #[test]
    fn app_ids_resolve_through_their_desktop_entry() {
        let base = std::path::PathBuf::from("/tmp/cce-cloud-test-icon-index");
        let _ = std::fs::remove_dir_all(&base);
        let user = base.join("user");
        let sys = base.join("sys");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(&sys).unwrap();
        std::fs::write(
            sys.join("org.gnome.Nautilus.desktop"),
            "[Desktop Entry]\nName=Files\nIcon=org.gnome.Nautilus\n",
        )
        .unwrap();
        std::fs::write(
            sys.join("code-oss.desktop"),
            "[Desktop Entry]\nName=Code\nIcon=com.visualstudio.code.oss\nStartupWMClass=Code - OSS\n",
        )
        .unwrap();
        // NoDisplay helpers still name an icon for their windows.
        std::fs::write(
            sys.join("helper.desktop"),
            "[Desktop Entry]\nName=Helper\nIcon=helper-icon\nNoDisplay=true\n\n[Desktop Action x]\nIcon=wrong\n",
        )
        .unwrap();
        // The user dir shadows the system entry's icon.
        std::fs::write(sys.join("editor.desktop"), "[Desktop Entry]\nIcon=editor-sys\n").unwrap();
        std::fs::write(user.join("editor.desktop"), "[Desktop Entry]\nIcon=editor-user\n").unwrap();

        let index = desktop_icon_index(&[user, sys]);
        let icon = |id| icon_name_for_app_id(&index, id);
        assert_eq!(icon("org.gnome.Nautilus"), "org.gnome.Nautilus");
        assert_eq!(icon("nautilus"), "org.gnome.Nautilus");
        assert_eq!(icon("Code - OSS"), "com.visualstudio.code.oss");
        assert_eq!(icon("helper"), "helper-icon");
        assert_eq!(icon("editor"), "editor-user");
        // No entry: the app_id is the icon name, as it is for cce's own apps.
        assert_eq!(icon("cce-terminal"), "cce-terminal");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn icon_override_is_keyed_by_desktop_id() {
        let dir = std::path::PathBuf::from("/tmp/cce-cloud-test-icon-override");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("houdini.svg"), "<svg/>").unwrap();

        // The ID's own file, as an absolute path - an entry whose Icon= is
        // an absolute path or missing still gets cce's artwork.
        assert_eq!(
            icon_override_in(&dir, "houdini").as_deref(),
            Some(dir.join("houdini.svg").to_str().unwrap())
        );
        // No override: the caller falls back to the entry's own Icon=.
        assert_eq!(icon_override_in(&dir, "raindropio"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chooser_labels_resolve_ids_and_tell_twins_apart() {
        let base = std::env::temp_dir().join(format!("cce-cloud-test-choose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (user, sys) = (base.join("user/applications"), base.join("sys/applications"));
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(&sys).unwrap();
        let entry = |name: &str, extra: &str| format!("[Desktop Entry]\nType=Application\nName={name}\nExec=x\n{extra}");
        std::fs::write(sys.join("org.gnome.Evince.desktop"), entry("Document Viewer", "Icon=evince\n")).unwrap();
        // NoDisplay: the launcher hides it, the chooser must not.
        std::fs::write(sys.join("helper.desktop"), entry("Helper", "NoDisplay=true\n")).unwrap();
        std::fs::write(sys.join("firefox.desktop"), entry("Firefox", "")).unwrap();
        std::fs::write(sys.join("firefox-nightly.desktop"), entry("Firefox", "")).unwrap();
        // The user dir shadows the system entry of the same ID.
        std::fs::write(user.join("helper.desktop"), entry("My Helper", "")).unwrap();
        // A localized name after the plain one does not replace it.
        std::fs::write(sys.join("loc.desktop"), entry("Plain", "Name[de]=Lokal\n")).unwrap();

        let dirs = vec![user.clone(), sys.clone()];
        let ids: Vec<String> = ["org.gnome.Evince", "helper", "firefox", "firefox-nightly", "missing", "loc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let got = Chooser::labels(&ids, &dirs);
        let labels: Vec<&str> = got.iter().map(|(_, l, _)| l.as_str()).collect();
        assert_eq!(
            labels,
            ["Document Viewer", "My Helper", "Firefox (firefox)", "Firefox (firefox-nightly)", "missing", "Plain"]
        );
        assert_eq!(got[0].2.as_deref(), Some("evince"));

        let chooser = Chooser { fed: ids.clone(), by_label: got.iter().map(|(id, l, _)| (l.clone(), id.clone())).collect() };
        assert_eq!(chooser.answer("Firefox (firefox-nightly)"), "firefox-nightly");
        assert_eq!(chooser.answer("Document Viewer"), "org.gnome.Evince");
        assert_eq!(chooser.answer("typed text"), "typed text", "an unknown row answers itself");
        let _ = std::fs::remove_dir_all(&base);
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
    fn a_tab_keeps_its_own_query_and_items() {
        let mut f = FuzzelWidget::new("Search: ".to_string());
        f.set_tabs(vec![
            ("Apps".to_string(), vec!["Firefox".to_string(), "Files".to_string()]),
            ("System".to_string(), vec!["Suspend".to_string(), "Reboot".to_string()]),
        ]);
        f.query.push_str("fi");
        f.filter();
        assert_eq!(f.filtered_items, vec!["Firefox".to_string(), "Files".to_string()]);

        assert!(f.cycle_tab(true));
        assert_eq!(f.active_tab, 1);
        // The System tab opens on its own (empty) query, not the Apps one.
        assert_eq!(f.query, "");
        assert_eq!(f.filtered_items, vec!["Suspend".to_string(), "Reboot".to_string()]);

        // ...and coming back lands on the query that was left behind.
        assert!(f.cycle_tab(true), "two tabs wrap");
        assert_eq!(f.active_tab, 0);
        assert_eq!(f.query, "fi");
        assert_eq!(f.filtered_items, vec!["Firefox".to_string(), "Files".to_string()]);
    }

    #[test]
    fn the_feed_fills_tab_zero_from_any_tab() {
        let mut f = FuzzelWidget::new("Search: ".to_string());
        f.set_tabs(vec![
            ("Apps".to_string(), Vec::new()),
            ("System".to_string(), vec!["Suspend".to_string()]),
        ]);
        f.switch_tab(1);
        // The stdin/socket ingest addresses tab 0 while the user reads tab 1:
        // the rows on screen must not move, and the items must still land.
        f.set_tab_items(0, vec!["Firefox".to_string()]);
        assert_eq!(f.filtered_items, vec!["Suspend".to_string()]);
        assert_eq!(f.tab_items(0), ["Firefox".to_string()]);
        f.switch_tab(0);
        assert_eq!(f.filtered_items, vec!["Firefox".to_string()]);
    }

    #[test]
    fn an_untabbed_list_takes_no_chrome_and_does_not_cycle() {
        let mut f = FuzzelWidget::new("Search: ".to_string());
        f.set_items(vec!["a".to_string(), "b".to_string()]);
        // What keeps Dmenu, Path and the Super-Tab window switcher laid out
        // and keyed exactly as they were.
        assert_eq!(f.tab_strip_h(), 0.0);
        assert!(!f.cycle_tab(true));
        assert!(f.tab_at(20.0, 20.0).is_none());
    }

    #[test]
    fn switcher_motion_onto_a_row_selects_it() {
        let mut f = FuzzelWidget::new("Search: ".to_string());
        f.set_rect(0.0, 0.0, 600.0, 400.0);
        f.switcher_rows = true;
        f.set_items((0..4).map(|i| format!("Window {i} (app{i})")).collect());
        f.selected = 1; // Super+Tab advanced past the focused window
        let x = 100.0;
        let (list_y, item_h) = (f.list_y(), f.item_h());
        let row_y = |i: usize| list_y + item_h * (i as f32 + 0.5);

        // The popup mapping under a resting pointer is an Enter, not motion:
        // it lights the row but leaves the selection where Super+Tab put it.
        f.hover_at(x, row_y(3));
        assert_eq!(f.selected, 1);

        // Any motion over a row selects it...
        assert!(f.pointer_moved(x, row_y(3) + 1.0));
        assert_eq!(f.selected, 3);
        assert!(f.pointer_moved(x, row_y(2)));
        assert_eq!(f.selected, 2);

        // ...but Tab can still move the chip off a pointer jiggling in place.
        f.selected = 0;
        f.pointer_moved(x, row_y(2) + 1.0);
        assert_eq!(f.selected, 0);

        // Outside the switcher, hovering never moves the selection.
        let mut g = FuzzelWidget::new("Search: ".to_string());
        g.set_rect(0.0, 0.0, 600.0, 400.0);
        g.set_items(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let (list_y, item_h) = (g.list_y(), g.item_h());
        g.pointer_moved(x, list_y + item_h * 2.5);
        assert_eq!(g.selected, 0);
    }

    #[test]
    fn every_system_row_runs_something() {
        // A row whose label no longer matches its command silently does
        // nothing when picked, so the lookup the commit path makes is the
        // thing to pin down.
        let mut f = FuzzelWidget::new("Search: ".to_string());
        f.set_tabs(vec![
            ("Apps".to_string(), Vec::new()),
            (
                SYSTEM_TAB_TITLE.to_string(),
                SYSTEM_COMMANDS.iter().map(|c| c.name.to_string()).collect(),
            ),
        ]);
        f.switch_tab(1);
        assert_eq!(f.filtered_items.len(), SYSTEM_COMMANDS.len());
        for item in &f.filtered_items {
            assert!(
                SYSTEM_COMMANDS.iter().any(|c| c.name == item),
                "System row {:?} resolves to no command",
                item
            );
        }
    }

    #[test]
    fn test_app_history_sorting() {
        let temp_dir = std::path::PathBuf::from("/tmp/cce-cloud-test-cache-dir");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);

        let orig_xdg = std::env::var("XDG_CACHE_HOME").ok();
        std::env::set_var("XDG_CACHE_HOME", &temp_dir);

        let mut apps = vec![
            AppInfo { name: "App A".to_string(), exec: "exec_a".to_string(), terminal: false, icon: None },
            AppInfo { name: "App B".to_string(), exec: "exec_b".to_string(), terminal: false, icon: None },
            AppInfo { name: "App C".to_string(), exec: "exec_c".to_string(), terminal: false, icon: None },
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

        // Tab titles, centred on their segments. Outside the list clip, like
        // the rest of the chrome.
        for i in 0..self.tabs.len() {
            let Some(r) = self.tab_rect(i) else { continue };
            let title = &self.tabs[i].title;
            let tw = cce_ui::widget::display::measure_text(title, TAB_FONT_PX);
            labels.push(TextLabel {
                text: title.clone(),
                x: r.x + (r.width - tw) / 2.0,
                y: r.y + (r.height - TAB_FONT_PX) / 2.0 - 1.0,
                font_size: TAB_FONT_PX,
                color: if i == self.active_tab {
                    [0xff, 0xff, 0xff]
                } else if self.tab_hovered == Some(i) {
                    [0xe6, 0xe6, 0xee]
                } else {
                    [0x99, 0x99, 0xa6]
                },
            });
        }

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
            x: self.text_x(),
            // The 14px line centred in the well.
            y: self.search_y() + (cce_ui::layout::textbox_height() - 17.0) / 2.0,
            font_size: 14.0,
            color: query_color,
        });

        if self.filtered_items.is_empty() {
            labels.push(TextLabel {
                text: "No matches found".to_string(),
                x: self.text_x(),
                y: self.list_y() + 4.0,
                font_size: 13.0,
                color: [0x88, 0x88, 0x99],
            });
        }

        labels
    }

    /// Visible item labels — one per row `get_draw_y` places in (or partially
    /// in) the viewport. Emitted under the paint walk's list clip, separately
    /// from [`Self::own_labels`], which draws chrome outside it.
    fn row_labels(&self) -> Vec<TextLabel> {
        let item_h = self.item_h();
        let mut labels = Vec::new();
        for (idx, item_text) in self.filtered_items.iter().enumerate() {
            let virtual_y = idx as f32 * item_h;
            if let Some(draw_y) = self.scroll_box.get_draw_y(virtual_y, item_h) {
                let color = if idx == self.selected {
                    [0xff, 0xff, 0xff]
                } else if self.hovered == Some(idx) {
                    // A step toward the selected white, over the hover wash.
                    [0xe6, 0xe6, 0xee]
                } else {
                    [0xbb, 0xbb, 0xc5]
                };

                labels.push(TextLabel {
                    text: self.row_label(item_text).to_string(),
                    // Indented past the icon column whether or not THIS row
                    // resolved an icon — see `icon_gutter`.
                    x: self.text_x() + self.icon_gutter,
                    // 4px down in a text-only row; a taller icon row
                    // centres the same line box.
                    y: draw_y + (item_h - ITEM_H) / 2.0 + 4.0,
                    font_size: 13.0,
                    color,
                });
            }
        }
        labels
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn launches_run_in_a_session_bound_scope() {
        let argv = scope_argv("/usr/bin/sh", &["-c", "exec cce-files"], 7);
        assert_eq!(
            argv,
            [
                "--user",
                "--scope",
                "--collect",
                "--slice=app.slice",
                "--unit=app-cce\\x2dcloud-cce-files-7",
                "--property=PartOf=cce-session.target",
                "--",
                "/usr/bin/sh",
                "-c",
                "exec cce-files",
            ]
        );
    }

    #[test]
    fn a_scope_is_named_after_the_app_not_the_shell() {
        assert_eq!(launch_name("sh", &["-c", "cce-mail"]), "cce-mail");
        assert_eq!(launch_name("sh", &["-c", "/opt/google/chrome/chrome %U"]), "/opt/google/chrome/chrome");
        assert_eq!(launch_name("sh", &["-c", "env GDK_SCALE=1 inkscape"]), "inkscape");
        assert_eq!(launch_name("sh", &["-c", "FOO=1 exec 'cce-files'"]), "cce-files");
        // A terminal-hosted entry: the app inside the terminal.
        assert_eq!(launch_name("foot", &["sh", "-c", "exec htop"]), "htop");
        // Not a shell command: the program itself.
        assert_eq!(launch_name("ccectl", &["close"]), "ccectl");
        assert_eq!(scope_name_part(&launch_name("sh", &["-c", "/opt/1Password/1password"])), "1password");
    }

    #[test]
    fn unit_names_keep_only_legal_characters() {
        assert_eq!(scope_name_part("google-chrome-stable"), "google-chrome-stable");
        assert_eq!(scope_name_part("/opt/1Password/1password"), "1password");
        assert_eq!(scope_name_part("my app.bin"), "my_app_bin");
        assert_eq!(scope_name_part(""), "app");
    }
}

#[cfg(test)]
mod repeat_tests {
    use super::key_repeats;
    use xkeysym::Keysym as K;

    #[test]
    fn text_deletion_and_movement_repeat_but_commits_do_not() {
        assert!(key_repeats(K::a, Some("a")));
        assert!(key_repeats(K::BackSpace, Some("\u{8}")));
        assert!(key_repeats(K::Down, None));
        assert!(!key_repeats(K::Return, Some("\r")));
        assert!(!key_repeats(K::Escape, Some("\u{1b}")));
        assert!(!key_repeats(K::Tab, Some("\t")));
        // A modifier or dead key carries no text: nothing to repeat.
        assert!(!key_repeats(K::Shift_L, None));
    }
}
