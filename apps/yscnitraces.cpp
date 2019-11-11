//
// LICENSE:
//
// Copyright (c) 2016 -- 2019 Fabio Pellacini
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "../yocto/yocto_commonio.h"
#include "../yocto/yocto_sceneio.h"
#include "../yocto/yocto_shape.h"
#include "../yocto/yocto_trace.h"
#include "yocto_opengl.h"
using namespace yocto;

#include <future>

// Application state
struct app_state {
  // loading options
  string filename  = "app.yaml";
  string imagename = "out.png";
  string name      = "";

  // options
  trace_params params = {};
  int          pratio = 8;

  // scene
  trace_scene scene      = {};
  bool        add_skyenv = false;

  // rendering state
  trace_state  state    = {};
  image<vec4f> render   = {};
  image<vec4f> display  = {};
  float        exposure = 0;

  // view scene
  opengl_image        glimage  = {};
  draw_glimage_params glparams = {};

  // editing
  pair<string, int> selection = {"camera", 0};

  // computation
  int               render_sample  = 0;
  std::atomic<bool> render_stop    = {};
  std::future<void> render_future  = {};
  int               render_counter = 0;

  ~app_state() {
    render_stop = true;
    if (render_future.valid()) render_future.get();
  }
};

// construct a scene from io
trace_scene make_scene(const sceneio_model& ioscene) {
  auto scene = trace_scene{};

  for (auto& iocamera : ioscene.cameras) {
    auto& camera = scene.cameras.emplace_back();
    camera.frame = iocamera.frame;
    camera.film  = iocamera.aspect >= 1
                      ? vec2f{iocamera.film, iocamera.film / iocamera.aspect}
                      : vec2f{iocamera.film * iocamera.aspect, iocamera.film};
    camera.lens     = iocamera.lens;
    camera.focus    = iocamera.focus;
    camera.aperture = iocamera.aperture;
  }

  for (auto& iotexture : ioscene.textures) {
    auto& texture = scene.textures.emplace_back();
    texture.hdr   = iotexture.hdr;
    texture.ldr   = iotexture.ldr;
  }

  for (auto& iomaterial : ioscene.materials) {
    auto& material            = scene.materials.emplace_back();
    material.emission         = iomaterial.emission;
    material.diffuse          = iomaterial.diffuse;
    material.specular         = iomaterial.specular;
    material.transmission     = iomaterial.transmission;
    material.roughness        = iomaterial.roughness;
    material.opacity          = iomaterial.opacity;
    material.refract          = iomaterial.refract;
    material.volemission      = iomaterial.volemission;
    material.voltransmission  = iomaterial.voltransmission;
    material.volmeanfreepath  = iomaterial.volmeanfreepath;
    material.volscatter       = iomaterial.volscatter;
    material.volscale         = iomaterial.volscale;
    material.volanisotropy    = iomaterial.volanisotropy;
    material.emission_tex     = iomaterial.emission_tex;
    material.diffuse_tex      = iomaterial.diffuse_tex;
    material.specular_tex     = iomaterial.specular_tex;
    material.transmission_tex = iomaterial.transmission_tex;
    material.roughness_tex    = iomaterial.roughness_tex;
    material.opacity_tex      = iomaterial.opacity_tex;
    material.subsurface_tex   = iomaterial.subsurface_tex;
    material.normal_tex       = iomaterial.normal_tex;
  }

  for (auto& ioshape_ : ioscene.shapes) {
    auto& ioshape = (needs_tesselation(ioscene, ioshape_))
                        ? tesselate_shape(ioscene, ioshape_)
                        : ioshape_;
    auto& shape         = scene.shapes.emplace_back();
    shape.points        = ioshape.points;
    shape.lines         = ioshape.lines;
    shape.triangles     = ioshape.triangles;
    shape.quads         = ioshape.quads;
    shape.quadspos      = ioshape.quadspos;
    shape.quadsnorm     = ioshape.quadsnorm;
    shape.quadstexcoord = ioshape.quadstexcoord;
    shape.positions     = ioshape.positions;
    shape.normals       = ioshape.normals;
    shape.texcoords     = ioshape.texcoords;
    shape.colors        = ioshape.colors;
    shape.radius        = ioshape.radius;
    shape.tangents      = ioshape.tangents;
  }

  for (auto& ioinstance : ioscene.instances) {
    auto& instance    = scene.instances.emplace_back();
    instance.frame    = ioinstance.frame;
    instance.shape    = ioinstance.shape;
    instance.material = ioinstance.material;
  }

  for (auto& ioenvironment : ioscene.environments) {
    auto& environment        = scene.environments.emplace_back();
    environment.frame        = ioenvironment.frame;
    environment.emission     = ioenvironment.emission;
    environment.emission_tex = ioenvironment.emission_tex;
  }

  return scene;
}

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes the integer index.
template <typename Func>
inline void parallel_for(const vec2i& size, Func&& func) {
  auto             futures  = vector<std::future<void>>{};
  auto             nthreads = std::thread::hardware_concurrency();
  std::atomic<int> next_idx(0);
  for (auto thread_id = 0; thread_id < nthreads; thread_id++) {
    futures.emplace_back(
        std::async(std::launch::async, [&func, &next_idx, size]() {
          while (true) {
            auto j = next_idx.fetch_add(1);
            if (j >= size.y) break;
            for (auto i = 0; i < size.x; i++) func({i, j});
          }
        }));
  }
  for (auto& f : futures) f.get();
}

void reset_display(app_state& app) {
  // stop render
  app.render_stop = true;
  if (app.render_future.valid()) app.render_future.get();

  // reset state
  app.state = make_state(app.scene, app.params);
  app.render.resize(app.state.size());
  app.display.resize(app.state.size());

  // render preview
  auto preview_prms = app.params;
  preview_prms.resolution /= app.pratio;
  preview_prms.samples = 1;
  auto preview         = trace_image(app.scene, preview_prms);
  preview              = tonemap_image(preview, app.exposure);
  for (auto j = 0; j < app.display.size().y; j++) {
    for (auto i = 0; i < app.display.size().x; i++) {
      auto pi             = clamp(i / app.pratio, 0, preview.size().x - 1),
           pj             = clamp(j / app.pratio, 0, preview.size().y - 1);
      app.display[{i, j}] = preview[{pi, pj}];
    }
  }

  // start renderer
  app.render_counter = 0;
  app.render_stop    = false;
  app.render_future  = std::async(std::launch::async, [&app]() {
    for (auto sample = 0; sample < app.params.samples; sample++) {
      if (app.render_stop) return;
      parallel_for(app.render.size(), [&app](const vec2i& ij) {
        if (app.render_stop) return;
        app.render[ij]  = trace_sample(app.state, app.scene, ij, app.params);
        app.display[ij] = tonemap(app.render[ij], app.exposure);
      });
    }
  });
}

void draw(const opengl_window& win) {
  auto& app = *(app_state*)get_gluser_pointer(win);
  clear_glframebuffer(vec4f{0.15f, 0.15f, 0.15f, 1.0f});
  if (!app.glimage || app.glimage.size() != app.display.size() ||
      !app.render_counter) {
    update_glimage(app.glimage, app.display, false, false);
  }
  app.glparams.window      = get_glwindow_size(win);
  app.glparams.framebuffer = get_glframebuffer_viewport(win);
  update_imview(app.glparams.center, app.glparams.scale, app.display.size(),
      app.glparams.window, app.glparams.fit);
  draw_glimage(app.glimage, app.glparams);
  swap_glbuffers(win);
  app.render_counter++;
  if (app.render_counter > 10) app.render_counter = 0;
}

// run ui loop
void run_ui(app_state& app) {
  // window
  auto win = opengl_window();
  init_glwindow(win, {1280 + 320, 720}, "yscnitrace", &app, draw);

  // loop
  auto mouse_pos = zero2f, last_pos = zero2f;
  while (!should_glwindow_close(win)) {
    last_pos         = mouse_pos;
    mouse_pos        = get_glmouse_pos(win);
    auto mouse_left  = get_glmouse_left(win);
    auto mouse_right = get_glmouse_right(win);
    auto alt_down    = get_glalt_key(win);
    auto shift_down  = get_glshift_key(win);

    // handle mouse and keyboard for navigation
    if ((mouse_left || mouse_right) && !alt_down) {
      auto& camera = app.scene.cameras.at(app.params.camera);
      auto  dolly  = 0.0f;
      auto  pan    = zero2f;
      auto  rotate = zero2f;
      if (mouse_left && !shift_down) rotate = (mouse_pos - last_pos) / 100.0f;
      if (mouse_right) dolly = (mouse_pos.x - last_pos.x) / 100.0f;
      if (mouse_left && shift_down)
        pan = (mouse_pos - last_pos) * camera.focus / 200.0f;
      pan.x = -pan.x;
      update_turntable(camera.frame, camera.focus, rotate, dolly, pan);
      reset_display(app);
    }

    // draw
    draw(win);

    // event hadling
    process_glevents(win);
  }

  // clear
  delete_glwindow(win);
}

int main(int argc, const char* argv[]) {
  // application
  app_state app{};

  // parse command line
  auto cli = make_cli("yscnitrace", "progressive path tracing");
  add_cli_option(cli, "--camera", app.params.camera, "Camera index.");
  add_cli_option(
      cli, "--resolution,-r", app.params.resolution, "Image resolution.");
  add_cli_option(cli, "--samples,-s", app.params.samples, "Number of samples.");
  add_cli_option(cli, "--tracer,-t", (int&)app.params.sampler, "Tracer type.",
      trace_sampler_names);
  add_cli_option(cli, "--falsecolor,-F", (int&)app.params.falsecolor,
      "Tracer false color type.", trace_falsecolor_names);
  add_cli_option(
      cli, "--bounces", app.params.bounces, "Maximum number of bounces.");
  add_cli_option(cli, "--clamp", app.params.clamp, "Final pixel clamping.");
  add_cli_option(cli, "--filter", app.params.tentfilter, "Filter image.");
  add_cli_option(cli, "--env-hidden/--no-env-hidden", app.params.envhidden,
      "Environments are hidden in renderer");
  add_cli_option(
      cli, "--bvh", (int&)app.params.bvh, "Bvh type", trace_bvh_names);
  add_cli_option(cli, "--add-skyenv", app.add_skyenv, "Add sky envmap");
  add_cli_option(cli, "--output,-o", app.imagename, "Image output", false);
  add_cli_option(cli, "scene", app.filename, "Scene filename", true);
  if (!parse_cli(cli, argc, argv)) exit(1);

  // scene loading
  auto ioscene    = sceneio_model{};
  auto load_timer = print_timed("loading scene");
  if (auto ret = load_scene(app.filename, ioscene); !ret) {
    print_fatal(ret.error);
  }
  print_elapsed(load_timer);

  // conversion
  auto convert_timer = print_timed("converting");
  app.scene          = make_scene(ioscene);
  print_elapsed(convert_timer);

  // build bvh
  auto bvh_timer = print_timed("building bvh");
  init_bvh(app.scene, app.params);
  print_elapsed(bvh_timer);

  // init renderer
  auto lights_timer = print_timed("building lights");
  init_lights(app.scene);
  print_elapsed(lights_timer);

  // fix renderer type if no lights
  if (app.scene.lights.empty() && is_sampler_lit(app.params)) {
    print_info("no lights presents, switching to eyelight shader");
    app.params.sampler = trace_sampler_type::eyelight;
  }

  // allocate buffers
  app.state   = make_state(app.scene, app.params);
  app.render  = image{app.state.size(), zero4f};
  app.display = app.render;
  reset_display(app);

  // run interactive
  run_ui(app);

  // done
  return 0;
}
