# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cce-cloud` is the **launcher / popup app** of the cce Wayland desktop environment: a
fuzzel-style fuzzy launcher, a dmenu replacement, a Super-Tab window switcher, and a generic
JSON-defined popup panel — all in one binary. It is one crate of the cce multi-repo
workspace; workspace-wide conventions (multi-repo layout, no `[workspace.dependencies]`,
shared `../target/`) live in **`../cce-compositor/WORKSPACE.md`** — read that too.

Two source files, all logic in `src/main.rs` (~4.6k lines):

- `src/main.rs` — CLI parsing, daemon/client/standalone entry points, the Wayland
  surface and event loop, `FuzzelWidget` (the list UI: keyboard selection chip,
  pointer hover wash, icons, row labels — hover and click share one `row_at`
  predicate, so the lit row is the row a press picks), all input handling,
  app/PATH scanning. (Rendering itself is cce-ui's — this crate declares no
  graphics dependency of its own.)
- `src/json_layout.rs` — `JsonLayoutWidget`: the JSON-config popup panel host
  (labels, checkboxes, buttons, spinboxes, color selectors, sliders, multi-page).
  App-owned copy of a dissolved cce-ui type; cloud is its only consumer.

The list's scrollbar/viewport math is the toolkit's `cce_ui::widget::ScrollRegion`
(`cce-ui/src/widget/scroll_region.rs`). This crate carried its own copy in
`src/scroll_region.rs` until 74f9ad4 (2026-09-01); the shared one is what gives
`get_draw_y` its partially-visible-rows contract, which paint, click and hover all
virtualize on.

## Build, test, run

```sh
cargo build                      # standalone (this repo is its own workspace root)
cargo build -p cce-cloud         # from the workspace root (avoids compositor rebuild)
cargo test                       # unit tests live at the bottom of main.rs
cargo test test_json_layout      # single test
make install                     # release build, then `ccebuild install --no-build cce-cloud`
```

Running requires a Wayland session (layer-shell). Quick manual checks:

```sh
cce-cloud --path                         # launcher over $PATH executables
printf "a\nb\nc\n" | cce-cloud --dmenu   # dmenu mode; selection prints to stdout
echo '{"widgets":[{"type":"button","id":"ok","text":"OK"}]}' | cce-cloud --json
cce-cloud --daemon                       # run the daemon (normally started by the DE)
```

## Architecture

### Daemon / client / standalone

`main()` (bottom of `src/main.rs`) picks one of three roles:

- `--daemon`: `run_daemon()` binds `/run/user/$UID/cce-cloud.socket` (fallback
  `/tmp/cce-cloud-$UID.socket`), holds one long-lived Wayland connection, and serves
  **one popup at a time, serially**. A new client connection **preempts** the currently
  open popup (global single-popup semantics); a client hangup closes the popup
  (`StdinState.client_gone`).
- No `--daemon`: `run_client()` connects to that socket, sends one JSON line
  `{"args": [...], "initial_stdin": "..."}`, streams any further stdin lines over the
  socket, and blocks until the daemon writes the selection back — which it prints to
  stdout. So callers get dmenu semantics whether or not the daemon is running.
- If the socket connect fails, it falls back to `run_standalone()`: the same UI
  in-process, selection printed directly to stdout.

**The daemon's life is one compositor's.** It holds one Wayland connection for good,
so it has to notice that connection dying: the wait for the next request dispatches the
Wayland source alongside the socket listener (a calloop `Generic` on a dup of it), and a
dispatch error — the compositor gone — is `compositor_gone`, a clean exit with status 0.
The unit is `PartOf=` / `WantedBy=cce-session.target`, which startcce starts and stops
with each compositor, so the next session brings up a fresh daemon. Until 2026-09-25 the
wait was a blocking `accept()` that never read the Wayland socket, and the unit hung off
`graphical-session.target`, whose stop systemd skipped at one logout (a queued dropbox
start made the transaction "destructive"): the daemon survived into the next session and
panicked building its first popup on the dead connection (`No surface formats:
ERROR_SURFACE_LOST_KHR`). A unit's `[Install]` change needs `systemctl --user reenable
cce-cloud.service` — `ccebuild install` copies units but does not re-enable them.

The daemon re-parses the forwarded args with the same flag loop as standalone — **flag
changes must be made in both `run_standalone()` and `run_daemon()`** (and, for the
needs-stdin decision, in `run_client()`).

### Modes (`LauncherMode`)

- `Dmenu` — items from stdin, selection echoed out. Magic stdin lines
  `__cce_switcher_next__` / `__cce_switcher_select_and_close__` drive the compositor's
  window switcher (`--switcher` starts in Dmenu with Super held).
- `Path` — executables scanned from `$PATH`.
- `Apps` — `.desktop` files from the standard application dirs, sorted by launch
  frecency persisted in `~/.cache/cce-cloud-apps.json`; selecting spawns the app's
  `Exec` (spawn output logged to `$XDG_RUNTIME_DIR/cce/spawn.log`, via
  `cce_ui::config::cce_runtime_dir()` — it was `/tmp/cce-spawn.log` before
  2026-08-22). Each entry's `Icon=` is
  resolved through `cce_ui::icon` and drawn in a gutter left of the label — the
  gutter is applied to every row, so one unresolvable icon doesn't rag the text
  edge. Apps and the Super-Tab switcher are the only lists with icons: other
  Dmenu/Path items are arbitrary strings with nothing to look up, and their
  gutter stays 0. The switcher's rows are the compositor's `Title (app_id)`
  lines; `switcher_app_id` reads the id back out of the row (the row itself is
  left as sent, since the compositor maps the echoed text back to a window) and
  `desktop_icon_index` maps it to a `.desktop` entry's `Icon=` by file stem,
  `StartupWMClass`, or last reverse-DNS component, falling back to the app_id
  itself, which is the name cce's own apps install their icons under. The rows
  stream in over stdin, so `resolve_switcher_icons` runs on each ingest.
  Icons are uploaded per popup on purpose (`cce_ui::icon::upload_themed` caches
  the decode, not the image id) because `Drop` destroys this app's `VkRenderer`
  between popups and an id cached across them would name freed GPU resources.
  Apps is also the one **tabbed** mode: the list carries an *Apps* page and a
  *System* page of DE verbs (`SYSTEM_COMMANDS` — window-manager actions through
  `ccectl`, plus session/power commands), and **Tab / Shift+Tab step between
  them**. See "Tabs" below.
- `Json` — a `JsonLayoutConfig` read from stdin builds a widget panel; clicking a
  button prints one JSON object with the button id and every control's state
  (`{"button", "checkboxes", "spinboxes", "colors", "sliders"}`) and closes.

Key flags: `-p/--prompt`, `-s/--select <item>`, `-x/-y` (position → forces layer-shell
anchoring), `--align-right`, `--parent-app-id` (app_id becomes `cce-cloud:<parent>`),
`--switcher`, and mode flags `--apps|--path|--dmenu|--json` (or `--mode <m>`).

### Tabs

`FuzzelWidget::tabs` splits the list into `TabPage`s. Fewer than two draws no
strip and claims no height (`tab_strip_h()` is 0), which is what leaves Dmenu,
Path and the Super-Tab window switcher laid out and keyed exactly as they were —
**Tab only switches tabs where tabs exist**, and falls back to its old job of
cycling the highlight everywhere else. That fallback is load-bearing: the
switcher is Dmenu mode, and its Tab is the key the whole feature is named for.

Three things to know before touching it:

- **The active page's items and query live in `all_items` / `query`, not in its
  `TabPage`.** Every pre-existing caller reads them there, and only
  `switch_tab` moves them across — so a page's own copies are stale for as long
  as it is the active one. Parking the query is what makes switching back land
  on the same filtered rows.
- **The stdin/socket feed addresses tab 0, through `set_tab_items`,** never
  `set_items` directly. `check_stdin_updates` skips the ingest when the items
  match what it last pushed; compared against the *active* tab that test would
  differ on every poll and clobber the page the user is reading.
- **A tab click is handled in the pointer branch, ahead of the row branch** —
  next to the scrollbar press, and for the same reason. Any press
  `FuzzelWidget::on_event` resolves is treated there as a row selection and (in
  Dmenu/switcher mode) committed, so a tab click routed through it would choose
  a row and close the popup.

The list geometry is derived from `search_y()` / `list_y()` / `list_h()` rather
than the `let pad = 15.0; let search_h = 35.0;` locals that used to be repeated
in each of the paint, scroll and hit-test paths: the strip shifts the whole list
down by its own height, and a path that missed the shift would put the rows, the
clip and the click out of step. The icon gutter is per-tab
(`recompute_icon_gutter`), so the System page's rows sit flush left while the
Apps page keeps its column.

The strip is painted by hand in the toolkit's recessed `ButtonStrip` idiom — one
well carved into the window plate, segments on its floor, the active one a
raised `control_plate` — because this widget paints straight onto the `PaintCtx`
and has no child layout pass to host a real `ButtonStrip`.

`SYSTEM_COMMANDS` is hardcoded, not config-driven: the rows are the DE's own
verbs, and a row naming a command `ccectl` does not have is one that silently
does nothing when picked (`every_system_row_runs_something` pins the lookup the
commit path makes). The window verbs act on the window *behind* the popup — the
compositor's `focused_window` skips overlay UI and names `cce-cloud` among it,
falling back to the most recent real window — which is the only reason "Close
Window" from a launcher means anything.

### Rendering: hand-rolled loop on `cce_ui::vk`, not the cce-ui engine runner

Unlike most cce clients, this app does **not** implement the `Application` trait or use
`cce_ui::engine::run`. It owns its event loop directly (smithay-client-toolkit handlers
+ calloop) and renders through **`cce_ui::vk::VkRenderer`** (the toolkit's raw-Vulkan
backend): `collect_display_list` builds a `PaintCtx` (beveled window plate, then the
active widget's `paint_self` walk — bevel/recess prims included), `upload_vertices`
runs it through `tessellate_display_list` into `State::vertex_data` + `frame_batches`
(`Batch2D`, physical-px scissors) + `plate_features`, `prepare_text` builds `TextSpan`s
from the paint walk (buffers shaped at logical size, spans scaled to physical), and
`render` is `draw_frame_2d(Frame2D { .. })`. Nothing in this path knows about the
open/close fade any more — see "Fading in and out" below. `tessellate` must carry the tessellator's image
list across to `Frame2D::images`: images ride a separate pipeline from the vertex
batches, and that return value was dropped (with `images: &[]` hardcoded) until
2026-08-16, which made `PaintCtx::image` a silent no-op *in this app only* while
it worked in every engine-runner client. `State::renderer` is an `Option`
solely so `Drop` can tear the swapchain down before destroying the `wl_surface` (daemon
mode churns one `State` per popup). It still reuses cce-ui pieces à la carte: the
narrow widget traits (`Layout`/`Paint`/`Input` via `Adapted<T>`), the scene paint walk
(`append_widget_text`) for text extraction, `color`/`layout`/`scale` getters, and the
`zcce_window_manager_v1` protocol. Follow existing cce-ui conventions when touching
widget code, but don't try to "port" this app onto the engine runner.

Surface choice: an XDG toplevel flagged as popup via the cce window-management
protocol when the compositor global is present and no `-x/-y` was given; otherwise a
layer-shell **Overlay** surface with exclusive keyboard. The window continuously
auto-sizes to its content (`update_desired_size` — list rows measured with
`cce_ui::cosmic_text` buffers, the JSON panel's widgets with
`cce_ui::widget::display::measure_text`)
and closes through the DE-wide dissolve (`trigger_close`, below).

### Fading in and out

The popup dissolves in when it maps and back out when it closes, and **neither
is drawn by this app** — both are the compositor ramping the opacity of the
surface's scene node, which carries the backdrop blur behind the popup down
with it. All this side does on close is ask and then wait:
`trigger_close` calls `cce_ui::ipc::request_close_fade()` (one `fade-out` line
on the compositor's control socket, answered with a duration in ms), stores
the deadline in `fade_until`, and keeps the surface mapped until the loop
sees it pass. Nothing is redrawn in between — the pixels stay put while the
scene node fades under them.

This replaced a hand-rolled fade that multiplied every vertex and image alpha
by a factor and then dropped the SDF-plate batches outright. A plate batch
**is** its cover quad, and the window background is a plate, so the first
frame of every close fade deleted the entire background and left the rows and
text dissolving over nothing. That is the failure mode to remember: per-element
alpha cannot express a window fade, because a client's surface stays fully
present to the compositor no matter how transparent it draws itself — the blur
behind it does not fade, and shader-lit output (plate rims, specular) never
honoured the vertex alpha in the first place.

`-x/-y` is a *cursor* position (the compositor hands the raw pointer to the desktop
and window-border context menus), not a final window origin — `Placement` fits it to
the output: grow away from the anchor, flip to its other side when the window would
overhang, clamp only when it fits on neither side, with the flip latched for the
popup's life so an auto-sizing list can't snap back and forth across the cursor.
Because the size is not known until the content is measured, the anchor/margins are
set by `apply_placement()` after the first `update_desired_size()` and re-applied by
`resize_window()` on every subsequent resize — a one-shot placement at surface
creation would use the pre-layout estimate and clip. The surface asks for
`exclusive_zone(-1)` so that the box it is clamped against (the wl_output logical
geometry) is the same one the compositor places it in.

### State flow

Stdin/socket items land in a shared `Arc<Mutex<StdinState>>` written by a reader
thread; a calloop channel pings the main loop, which ingests via
`check_stdin_updates()` → refilter → resize → re-upload vertices. Rendering is
demand-driven off a `redraw` flag with a 16ms dispatch timeout.
