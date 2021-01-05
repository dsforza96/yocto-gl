//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
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

#include <yocto/yocto_commonio.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_image.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sceneio.h>
#include <yocto/yocto_shape.h>
#include <yocto_gui/yocto_imgui.h>
#include <yocto_gui/yocto_shade.h>
using namespace yocto;

#include "yshade_scene.h"

using namespace std::string_literals;

#ifdef _WIN32
#undef near
#undef far
#endif

namespace yocto::sceneio {
void print_obj_camera(sceneio_camera* camera);
};

// Application state
struct shade_scene_state {
  // loading parameters
  string filename  = "scene.json";
  string imagename = "out.png";
  string outname   = "scene.json";
  string name      = "";

  // options
  shade_params drawgl_prms = {};

  // scene
  sceneio_scene ioscene  = sceneio_scene{};
  camera_handle iocamera = invalid_handle;

  // rendering state
  shade_scene glscene = {};

  // editing
  int selected_camera      = -1;
  int selected_instance    = -1;
  int selected_shape       = -1;
  int selected_material    = -1;
  int selected_environment = -1;
  int selected_texture     = -1;

  // loading status
  std::atomic<bool> ok           = false;
  std::future<void> loader       = {};
  string            status       = "";
  string            error        = "";
  std::atomic<int>  current      = 0;
  std::atomic<int>  total        = 0;
  string            loader_error = "";
};

static void init_glscene(shade_scene& glscene, const sceneio_scene& ioscene,
    progress_callback progress_cb) {
  // handle progress
  auto progress = vec2i{
      0, (int)ioscene.cameras.size() + (int)ioscene.materials.size() +
             (int)ioscene.textures.size() + (int)ioscene.shapes.size() +
             (int)ioscene.instances.size()};

  // init scene
  init_scene(glscene);

  // camera
  for (auto& iocamera : ioscene.cameras) {
    if (progress_cb) progress_cb("convert camera", progress.x++, progress.y);
    auto& camera = glscene.cameras.at(add_camera(glscene));
    set_frame(camera, iocamera.frame);
    set_lens(camera, iocamera.lens, iocamera.aspect, iocamera.film);
    set_nearfar(camera, 0.001, 10000);
  }

  // textures
  for (auto& iotexture : ioscene.textures) {
    if (progress_cb) progress_cb("convert texture", progress.x++, progress.y);
    auto  handle    = add_texture(glscene);
    auto& gltexture = glscene.textures[handle];
    if (!iotexture.hdr.empty()) {
      set_texture(gltexture, iotexture.hdr);
    } else if (!iotexture.ldr.empty()) {
      set_texture(gltexture, iotexture.ldr);
    }
  }

  // material
  for (auto& iomaterial : ioscene.materials) {
    if (progress_cb) progress_cb("convert material", progress.x++, progress.y);
    auto  handle     = add_material(glscene);
    auto& glmaterial = glscene.materials[handle];
    set_emission(glmaterial, iomaterial.emission, iomaterial.emission_tex);
    set_color(glmaterial, (1 - iomaterial.transmission) * iomaterial.color,
        iomaterial.color_tex);
    set_specular(glmaterial,
        (1 - iomaterial.transmission) * iomaterial.specular,
        iomaterial.specular_tex);
    set_metallic(glmaterial,
        (1 - iomaterial.transmission) * iomaterial.metallic,
        iomaterial.metallic_tex);
    set_roughness(glmaterial, iomaterial.roughness, iomaterial.roughness_tex);
    set_opacity(glmaterial, iomaterial.opacity, iomaterial.opacity_tex);
    set_normalmap(glmaterial, iomaterial.normal_tex);
  }

  // shapes
  for (auto& ioshape : ioscene.shapes) {
    if (progress_cb) progress_cb("convert shape", progress.x++, progress.y);
    add_shape(glscene, ioshape.points, ioshape.lines, ioshape.triangles,
        ioshape.quads, ioshape.positions, ioshape.normals, ioshape.texcoords,
        ioshape.colors);
  }

  // shapes
  for (auto& ioinstance : ioscene.instances) {
    if (progress_cb) progress_cb("convert instance", progress.x++, progress.y);
    auto  handle     = add_instance(glscene);
    auto& glinstance = glscene.instances[handle];
    set_frame(glinstance, ioinstance.frame);
    set_shape(glinstance, ioinstance.shape);
    set_material(glinstance, ioinstance.material);
  }

  // environments
  for (auto& ioenvironment : ioscene.environments) {
    auto  handle        = add_environment(glscene);
    auto& glenvironment = glscene.environments[handle];
    set_frame(glenvironment, ioenvironment.frame);
    set_emission(
        glenvironment, ioenvironment.emission, ioenvironment.emission_tex);
  }

  // init environments
  init_environments(glscene);

  // done
  if (progress_cb) progress_cb("convert done", progress.x++, progress.y);
}

int run_shade_scene(const shade_scene_params& params) {
  // initialize app
  auto app_guard = std::make_unique<shade_scene_state>();
  auto app       = app_guard.get();

  // copy command line
  app->filename = params.scene;

  // loading scene
  auto ioerror = ""s;
  if (!load_scene(app->filename, app->ioscene, ioerror, print_progress))
    print_fatal(ioerror);

  // get camera
  app->iocamera = get_camera_handle(app->ioscene, "");

  // tesselation
  tesselate_shapes(app->ioscene, print_progress);

  // callbacks
  auto callbacks    = gui_callbacks{};
  callbacks.init_cb = [app](gui_window* win, const gui_input& input) {
    init_glscene(app->glscene, app->ioscene,
        [app](const string& message, int current, int total) {
          app->status  = "init scene";
          app->current = current;
          app->total   = total;
        });
  };
  callbacks.clear_cb = [app](gui_window* win, const gui_input& input) {
    clear_scene(app->glscene);
  };
  callbacks.draw_cb = [app](gui_window* win, const gui_input& input) {
    draw_scene(app->glscene, app->glscene.cameras.at(0),
        input.framebuffer_viewport, app->drawgl_prms);
  };
  callbacks.widgets_cb = [app](gui_window* win, const gui_input& input) {
    draw_progressbar(win, app->status.c_str(), app->current, app->total);
    // if (draw_combobox(
    //         win, "camera", app->iocamera, app->ioscene.camera_names)) {
    //   for (auto idx = 0; idx < app->ioscene.cameras.size(); idx++) {
    //     if (app->ioscene.cameras[idx] == app->iocamera)
    //       app->glcamera = app->glscene->cameras[idx];
    //   }
    // }
    auto& params = app->drawgl_prms;
    draw_slider(win, "resolution", params.resolution, 0, 4096);
    draw_checkbox(win, "wireframe", params.wireframe);
    continue_line(win);
    draw_checkbox(win, "faceted", params.faceted);
    continue_line(win);
    draw_checkbox(win, "double sided", params.double_sided);
    draw_combobox(win, "lighting", (int&)params.lighting, shade_lighting_names);
    // draw_checkbox(win, "edges", params.edges);
    draw_slider(win, "exposure", params.exposure, -10, 10);
    draw_slider(win, "gamma", params.gamma, 0.1f, 4);
    draw_slider(win, "near", params.near, 0.01f, 1.0f);
    draw_slider(win, "far", params.far, 1000.0f, 10000.0f);
  };
  callbacks.update_cb = [](gui_window* win, const gui_input& input) {
    // update(win, apps);
  };
  callbacks.uiupdate_cb = [app](gui_window* win, const gui_input& input) {
    // handle mouse and keyboard for navigation
    if ((input.mouse_left || input.mouse_right) && !input.modifier_alt &&
        !input.widgets_active) {
      auto dolly  = 0.0f;
      auto pan    = zero2f;
      auto rotate = zero2f;
      if (input.mouse_left && !input.modifier_shift)
        rotate = (input.mouse_pos - input.mouse_last) / 100.0f;
      if (input.mouse_right)
        dolly = (input.mouse_pos.x - input.mouse_last.x) / 100.0f;
      if (input.mouse_left && input.modifier_shift)
        pan = (input.mouse_pos - input.mouse_last) / 100.0f;
      auto& camera = app->ioscene.cameras.at(app->iocamera);
      std::tie(camera.frame, camera.focus) = camera_turntable(
          camera.frame, camera.focus, rotate, dolly, pan);
      set_frame(app->glscene.cameras.at(app->iocamera), camera.frame);
    }
  };

  // run ui
  run_ui({1280 + 320, 720}, "ysceneviews", callbacks);

  // done
  return 0;
}
