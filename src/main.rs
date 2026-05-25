use std::sync::{Arc, Mutex};
use std::io::{self, BufRead, IsTerminal};

use clear_ui::widget::{Widget, TextLabel};

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
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x4,
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
        Vertex { position: [x0, y0], color },
        Vertex { position: [x1, y0], color },
        Vertex { position: [x0, y1], color },
        Vertex { position: [x1, y0], color },
        Vertex { position: [x1, y1], color },
        Vertex { position: [x0, y1], color },
    ]
}

fn widget_vertices(w: &dyn Widget, sw: f32, sh: f32) -> Vec<Vertex> {
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
    
    let mut scored: Vec<(i32, &String)> = items
        .iter()
        .filter_map(|item| {
            let item_lower = item.to_lowercase();
            if item_lower == query_lower {
                Some((100, item))
            } else if item_lower.starts_with(&query_lower) {
                Some((80 - (item.len() as i32), item))
            } else if item_lower.contains(&query_lower) {
                Some((50 - (item.len() as i32), item))
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
                    Some((10 - (item.len() as i32), item))
                } else {
                    None
                }
            }
        })
        .collect();
        
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().map(|(_, item)| item.clone()).collect()
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
    scroll_offset: usize,
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
            scroll_offset: 0,
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
    }

    pub fn update_scroll(&mut self) {
        let visible_items = 13;
        if self.selected >= self.scroll_offset + visible_items {
            self.scroll_offset = self.selected - visible_items + 1;
        } else if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
    }
}

impl Widget for FuzzelWidget {
    fn rect(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.w, self.h)
    }

    fn set_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
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

        // Items list background highlight
        let list_y = self.y + pad + search_h + 10.0;
        let row_h = 25.0;
        let visible_items = 13;

        if !self.filtered_items.is_empty() && self.selected >= self.scroll_offset {
            let relative_selected = self.selected - self.scroll_offset;
            if relative_selected < visible_items {
                let y = list_y + relative_selected as f32 * row_h;
                quads.push((
                    self.x + pad,
                    y,
                    self.w - pad * 2.0,
                    row_h - 2.0,
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

        let list_y = self.y + pad + search_h + 10.0;
        let row_h = 25.0;
        let visible_items = 13;

        let start = self.scroll_offset;
        let end = self.filtered_items.len().min(start + visible_items);

        for (i, idx) in (start..end).enumerate() {
            let item_text = &self.filtered_items[idx];
            let y = list_y + i as f32 * row_h + 4.0;
            let color = if idx == self.selected {
                [0xff, 0xff, 0xff]
            } else {
                [0xbb, 0xbb, 0xc5]
            };

            labels.push(TextLabel {
                text: item_text.clone(),
                x: self.x + pad + 10.0,
                y,
                font_size: 13.0,
                color,
            });
        }

        if self.filtered_items.is_empty() {
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

    fn mouse_input(&mut self, button: clear_ui::widget::MouseButton, state: clear_ui::widget::ElementState, px: f32, py: f32) -> bool {
        if button == clear_ui::widget::MouseButton::Left && state == clear_ui::widget::ElementState::Pressed {
            let pad = 15.0;
            let search_h = 35.0;
            let list_y = self.y + pad + search_h + 10.0;
            let row_h = 25.0;
            let visible_items = 13;

            if px >= self.x + pad && px <= self.x + self.w - pad {
                if py >= list_y && py < list_y + visible_items as f32 * row_h {
                    let clicked_row = ((py - list_y) / row_h).floor() as usize;
                    let target_idx = self.scroll_offset + clicked_row;
                    if target_idx < self.filtered_items.len() {
                        self.selected = target_idx;
                        return true;
                    }
                }
            }
        }
        false
    }
}

struct StdinState {
    items: Vec<String>,
    new_data: bool,
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
}

impl State {
    async fn new(
        conn: &Connection,
        qh: &QueueHandle<AppState>,
        compositor_state: &CompositorState,
        layer_shell_state: &LayerShell,
        prompt: String,
        stdin_sender: calloop::channel::Sender<()>,
    ) -> Self {
        let scale = 2.0f64;
        let (width, height) = (600, 400);
        let pw = (width as f64 * scale) as u32;
        let ph = (height as f64 * scale) as u32;
        let lw = width as f32;
        let lh = height as f32;

        let wl_surface = compositor_state.create_surface(qh);
        wl_surface.set_buffer_scale(scale as i32);
        let window = layer_shell_state.create_layer_surface(
            qh,
            wl_surface.clone(),
            Layer::Overlay,
            Some("clear-cloud".to_string()),
            None,
        );
        window.set_size(width, height);
        window.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
        window.set_anchor(Anchor::empty());
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
                power_preference: wgpu::PowerPreference::HighPerformance,
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

        if !io::stdin().is_terminal() {
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
        } else {
            let path_items = scan_path();
            if let Ok(mut lock_state) = stdin_state.lock() {
                lock_state.items = path_items;
                lock_state.new_data = true;
            }
        }

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
        };

        state.check_stdin_updates();
        state.apply_layout();
        state.upload_vertices();
        state
    }

    fn check_stdin_updates(&mut self) -> bool {
        if let Ok(mut lock) = self.stdin_state.lock() {
            if lock.new_data {
                lock.new_data = false;
                let items = lock.items.clone();
                self.fuzzel.set_items(items);
                return true;
            }
        }
        false
    }

    fn apply_layout(&mut self) {
        let (w, h) = (self.width, self.height);
        self.fuzzel.set_rect(0.0, 0.0, w, h);
    }

    fn collect_vertices(&self) -> Vec<Vertex> {
        let sw = self.width;
        let sh = self.height;
        widget_vertices(&self.fuzzel, sw, sh)
    }

    fn upload_vertices(&mut self) {
        let verts = self.collect_vertices();
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
            ..
        } = self;

        let viewport = Resolution { width: *physical_width, height: *physical_height };
        text_viewport.update(queue, viewport);

        let scale_f32 = *scale as f32;

        let mut areas: Vec<TextArea> = Vec::new();
        let mut widget_buffers: Vec<Buffer> = Vec::new();
        let mut widget_labels: Vec<TextLabel> = Vec::new();

        for label in fuzzel.text_labels() {
            widget_buffers.push(make_text_buffer(font_system, &label.text, label.font_size));
            widget_labels.push(label);
        }

        for (buf, label) in widget_buffers.iter().zip(widget_labels.iter()) {
            areas.push(TextArea {
                buffer: buf,
                left: label.x * scale_f32,
                top: label.y * scale_f32,
                scale: scale_f32,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: *physical_width as i32,
                    bottom: *physical_height as i32,
                },
                default_color: glyphon::Color::rgb(label.color[0], label.color[1], label.color[2]),
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

    fn render(&mut self) -> bool {
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
                            r: 0.05, g: 0.05, b: 0.08, a: 0.92,
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

    window: LayerSurface,
    surface: wl_surface::WlSurface,

    state: Option<State>,
    exit: bool,
    redraw: bool,
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
        for event in events {
            let (x, y) = event.position;
            if let Some(st) = &mut self.state {
                let scale = st.scale;
                let cx = (x * scale) as f32;
                let cy = (y * scale) as f32;
                match &event.kind {
                    PointerEventKind::Motion { .. } => {
                        st.cursor_x = cx;
                        st.cursor_y = cy;
                    }
                    PointerEventKind::Press { button, .. } => {
                        if *button == 272 {
                            st.cursor_x = cx;
                            st.cursor_y = cy;
                            let prev_selected = st.fuzzel.selected;
                            let changed = st.fuzzel.mouse_input(
                                clear_ui::widget::MouseButton::Left,
                                clear_ui::widget::ElementState::Pressed,
                                st.cursor_x,
                                st.cursor_y,
                            );
                            if changed {
                                if st.fuzzel.selected == prev_selected {
                                    if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
                                        println!("{}", item);
                                        self.exit = true;
                                    }
                                }
                                st.upload_vertices();
                                self.redraw = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
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
    ) {}

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
        _modifiers: smithay_client_toolkit::seat::keyboard::Modifiers,
        _layout: u32,
    ) {}
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
            state.resize(width, height);
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
            _ => {
                if let Some(ref text) = event.utf8 {
                    Key::Character(text.clone())
                } else {
                    return;
                }
            }
        };

        if let Some(st) = &mut self.state {
            let mut handled = true;
            match &logical_key {
                Key::Named(NamedKey::Escape) => {
                    self.exit = true;
                }
                Key::Named(NamedKey::Enter) => {
                    if let Some(item) = st.fuzzel.filtered_items.get(st.fuzzel.selected) {
                        println!("{}", item);
                        self.exit = true;
                    }
                }
                Key::Named(NamedKey::ArrowDown) => {
                    if !st.fuzzel.filtered_items.is_empty() {
                        st.fuzzel.selected = (st.fuzzel.selected + 1).min(st.fuzzel.filtered_items.len() - 1);
                        st.fuzzel.update_scroll();
                        st.upload_vertices();
                        self.redraw = true;
                    }
                }
                Key::Named(NamedKey::ArrowUp) => {
                    if st.fuzzel.selected > 0 {
                        st.fuzzel.selected -= 1;
                        st.fuzzel.update_scroll();
                        st.upload_vertices();
                        self.redraw = true;
                    }
                }
                Key::Named(NamedKey::Backspace) => {
                    st.fuzzel.query.pop();
                    st.fuzzel.filter();
                    st.upload_vertices();
                    self.redraw = true;
                }
                _ => {
                    if let Some(text) = &event.utf8 {
                        for ch in text.chars().filter(|c| !c.is_control()) {
                            st.fuzzel.query.push(ch);
                        }
                        st.fuzzel.filter();
                        st.upload_vertices();
                        self.redraw = true;
                    } else {
                        handled = false;
                    }
                }
            }
            if handled {
                self.redraw = true;
            }
        }
    }
}

fn main() {
    let mut prompt = "Search: ".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "-p" || arg == "--prompt" {
            if let Some(p) = args.next() {
                prompt = p;
            }
        }
    }

    let conn = Connection::connect_to_env().unwrap();
    let conn_clone = conn.clone();
    let (globals, event_queue) = registry_queue_init(&conn).unwrap();
    let qh = event_queue.handle();

    let compositor_state = CompositorState::bind(&globals, &qh).unwrap();
    let layer_shell_state = LayerShell::bind(&globals, &qh).unwrap();
    let shm_state = Shm::bind(&globals, &qh).unwrap();
    let seat_state = SeatState::new(&globals, &qh);
    let output_state = OutputState::new(&globals, &qh);

    let (stdin_sender, stdin_channel) = calloop::channel::channel::<()>();

    let state = pollster::block_on(State::new(
        &conn,
        &qh,
        &compositor_state,
        &layer_shell_state,
        prompt,
        stdin_sender,
    ));

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
        window: state.window.clone(),
        surface: state.wl_surface.clone(),
        state: Some(state),
        exit: false,
        redraw: true,
    };

    let mut event_loop = calloop::EventLoop::try_new().unwrap();
    let loop_handle = event_loop.handle();

    WaylandSource::new(conn, event_queue).insert(loop_handle.clone()).unwrap();

    loop_handle.insert_source(stdin_channel, |event, _metadata, app_state: &mut AppState| {
        if let calloop::channel::Event::Msg(()) = event {
            if let Some(st) = &mut app_state.state {
                if st.check_stdin_updates() {
                    st.apply_layout();
                    st.upload_vertices();
                    app_state.redraw = true;
                }
            }
        }
    }).unwrap();

    loop {
        let timeout = if app.redraw {
            std::time::Duration::from_millis(0)
        } else {
            std::time::Duration::from_millis(16)
        };
        event_loop.dispatch(timeout, &mut app).unwrap();

        if app.exit {
            break;
        }

        if app.redraw {
            app.redraw = false;
            if let Some(state) = &mut app.state {
                if state.render() {
                    app.redraw = true;
                }
            }
        }
    }

    drop(app);
    let _ = conn_clone.flush();
}
