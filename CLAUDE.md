# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cce-cloud` is the **launcher / popup app** of the cce Wayland desktop environment: a
fuzzel-style fuzzy launcher, a dmenu replacement, a Super-Tab window switcher, and a
generic JSON-defined popup panel — all in one binary. It is one crate of the cce
multi-repo workspace; workspace-wide conventions (multi-repo layout, no
`[workspace.dependencies]`, shared `../target/`) live in **`../CLAUDE.md`** — read that
too.

Three source files, all logic in `src/main.rs` (~3.3k lines):

- `src/main.rs` — CLI parsing, daemon/client/standalone entry points, the Wayland +
  wgpu stack, `FuzzelWidget` (the list UI), all input handling, app/PATH scanning.
- `src/json_layout.rs` — `JsonLayoutWidget`: the JSON-config popup panel host
  (labels, checkboxes, buttons, spinboxes, color selectors, sliders, multi-page).
  App-owned copy of a dissolved cce-ui type; cloud is its only consumer.
- `src/scroll_region.rs` — `ScrollRegion`: scrollbar/viewport math for the item list.

## Build, test, run

```sh
cargo build                      # standalone (this repo is its own workspace root)
cargo build -p cce-cloud         # from the workspace root (avoids compositor rebuild)
cargo test                       # unit tests live at the bottom of main.rs
cargo test test_json_layout      # single test
make install                     # installs ../target/release/cce-cloud → ~/.local/bin
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

`main()` picks one of three roles (`src/main.rs:3021`):

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
  `Exec` (spawn output logged to `/tmp/cce-spawn.log`).
- `Json` — a `JsonLayoutConfig` read from stdin builds a widget panel; clicking a
  button prints one JSON object with the button id and every control's state
  (`{"button", "checkboxes", "spinboxes", "colors", "sliders"}`) and closes.

Key flags: `-p/--prompt`, `-s/--select <item>`, `-x/-y` (position → forces layer-shell
anchoring), `--align-right`, `--parent-app-id` (app_id becomes `cce-cloud:<parent>`),
`--switcher`, and mode flags `--apps|--path|--dmenu|--json` (or `--mode <m>`).

### Rendering: hand-rolled loop on `cce_ui::vk`, not the cce-ui engine runner

Unlike most cce clients, this app does **not** implement the `Application` trait or use
`cce_ui::engine::run`. It owns its event loop directly (smithay-client-toolkit handlers
+ calloop) and renders through **`cce_ui::vk::VkRenderer`** (the toolkit's raw-Vulkan
backend): `upload_vertices` rebuilds `State::vertex_data`, `prepare_text` builds
`TextSpan`s from the paint walk (buffers shaped at logical size, spans scaled to
physical; fade alpha rides `default_color`), and `render` is
`draw_frame(&vertex_data)`. `State::renderer` is an `Option` solely so `Drop` can tear
the swapchain down before destroying the `wl_surface` (daemon mode churns one `State`
per popup). It still reuses cce-ui pieces à la carte: `Vertex`/`quad_vertices`, the
narrow widget traits (`Layout`/`Paint`/`Input` via `Adapted<T>`), the scene paint walk
(`append_widget_text`) for text extraction, `color`/`layout`/`scale` getters, and the
`zcce_window_manager_v1` protocol. Follow existing cce-ui conventions when touching
widget code, but don't try to "port" this app onto the engine runner.

Surface choice: an XDG toplevel flagged as popup via the cce window-management
protocol when the compositor global is present and no `-x/-y` was given; otherwise a
layer-shell **Overlay** surface with exclusive keyboard. The window continuously
auto-sizes to its content (`update_desired_size`, measured via glyphon text buffers)
and closes through a ~150ms fade-out (`trigger_close` → `fade_factor`).

### State flow

Stdin/socket items land in a shared `Arc<Mutex<StdinState>>` written by a reader
thread; a calloop channel pings the main loop, which ingests via
`check_stdin_updates()` → refilter → resize → re-upload vertices. Rendering is
demand-driven off a `redraw` flag with a 16ms dispatch timeout.
