use std::sync::{Arc, Mutex};
use std::io::{self, BufRead, IsTerminal};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes};
use winit::keyboard::{Key, NamedKey};
use winit::dpi::{PhysicalSize, LogicalSize, LogicalPosition};

use clear_ui::widget::{Widget, TextLabel};

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

    fn mouse_input(&mut self, button: winit::event::MouseButton, state: ElementState, px: f32, py: f32) -> bool {
        if button == winit::event::MouseButton::Left && state == ElementState::Pressed {
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
    window: Arc<Window>,
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
    async fn new(window: Arc<Window>, prompt: String) -> Self {
        let scale = window.scale_factor();
        let physical_size = window.inner_size();
        let pw = physical_size.width.max(1);
        let ph = physical_size.height.max(1);
        let lw = pw as f32 / scale as f32;
        let lh = ph as f32 / scale as f32;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
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
            let window_clone = window.clone();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    if let Ok(line) = line {
                        if let Ok(mut lock_state) = stdin_state_clone.lock() {
                            lock_state.items.push(line);
                            lock_state.new_data = true;
                        }
                        window_clone.request_redraw();
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

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.physical_width = new_size.width;
            self.physical_height = new_size.height;
            self.width = new_size.width as f32 / self.scale as f32;
            self.height = new_size.height as f32 / self.scale as f32;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.apply_layout();
            self.upload_vertices();
        }
    }

    fn render(&mut self) {
        self.prepare_text();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(e) => {
                eprintln!("Surface error: {e:?}");
                return;
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
        self.window.pre_present_notify();
        output.present();
    }
}

struct App {
    state: Option<State>,
    prompt: String,
}

impl App {
    fn new(prompt: String) -> Self {
        Self { state: None, prompt }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let monitor = event_loop.primary_monitor().or_else(|| event_loop.available_monitors().next());
        let (width, height) = (600, 400);
        let mut window_attributes = WindowAttributes::default()
            .with_title("clear-cloud")
            .with_decorations(false)
            .with_inner_size(LogicalSize::new(width, height))
            .with_transparent(true);

        if let Some(monitor) = monitor {
            let monitor_size = monitor.size();
            let scale_factor = monitor.scale_factor();
            let monitor_w = monitor_size.width as f64 / scale_factor;
            let monitor_h = monitor_size.height as f64 / scale_factor;
            let x = (monitor_w - width as f64) / 2.0;
            let y = (monitor_h - height as f64) / 2.0;
            window_attributes = window_attributes.with_position(LogicalPosition::new(x, y));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let state = pollster::block_on(State::new(window, self.prompt.clone()));
        self.state = Some(state);
        self.state.as_ref().unwrap().window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if self.state.is_none() { return; }
        let state = self.state.as_mut().unwrap();

        let needs_redraw = match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                true
            }
            WindowEvent::Resized(size) => {
                state.resize(size);
                true
            }
            WindowEvent::RedrawRequested => {
                if state.check_stdin_updates() {
                    state.apply_layout();
                    state.upload_vertices();
                }
                state.render();
                false
            }
            WindowEvent::ScaleFactorChanged { scale_factor, mut inner_size_writer } => {
                let new_physical = winit::dpi::PhysicalSize::new(
                    (state.width as f64 * scale_factor) as u32,
                    (state.height as f64 * scale_factor) as u32,
                );
                let _ = inner_size_writer.request_inner_size(new_physical);
                state.scale = scale_factor;
                state.resize(new_physical);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_x = position.x as f32 / state.scale as f32;
                state.cursor_y = position.y as f32 / state.scale as f32;
                false
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if button == winit::event::MouseButton::Left && btn_state == ElementState::Pressed {
                    let prev_selected = state.fuzzel.selected;
                    let changed = state.fuzzel.mouse_input(button, btn_state, state.cursor_x, state.cursor_y);
                    if changed {
                        if state.fuzzel.selected == prev_selected {
                            if let Some(item) = state.fuzzel.filtered_items.get(state.fuzzel.selected) {
                                println!("{}", item);
                                std::process::exit(0);
                            }
                        }
                        state.upload_vertices();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    let mut handled = true;
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            std::process::exit(0);
                        }
                        Key::Named(NamedKey::Enter) => {
                            if let Some(item) = state.fuzzel.filtered_items.get(state.fuzzel.selected) {
                                println!("{}", item);
                                std::process::exit(0);
                            }
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            if !state.fuzzel.filtered_items.is_empty() {
                                state.fuzzel.selected = (state.fuzzel.selected + 1).min(state.fuzzel.filtered_items.len() - 1);
                                state.fuzzel.update_scroll();
                                state.upload_vertices();
                            }
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            if state.fuzzel.selected > 0 {
                                state.fuzzel.selected -= 1;
                                state.fuzzel.update_scroll();
                                state.upload_vertices();
                            }
                        }
                        Key::Named(NamedKey::Backspace) => {
                            state.fuzzel.query.pop();
                            state.fuzzel.filter();
                            state.upload_vertices();
                        }
                        _ => {
                            if let Some(text) = &event.text {
                                for ch in text.chars().filter(|c| !c.is_control()) {
                                    state.fuzzel.query.push(ch);
                                }
                                state.fuzzel.filter();
                                state.upload_vertices();
                            } else {
                                handled = false;
                            }
                        }
                    }
                    handled
                } else {
                    false
                }
            }
            _ => false,
        };

        if needs_redraw {
            state.window.request_redraw();
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

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(prompt);
    event_loop.run_app(&mut app).unwrap();
}
