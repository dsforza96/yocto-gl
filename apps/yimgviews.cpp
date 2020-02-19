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
#include "../yocto/yocto_image.h"
#include "yocto_opengl.h"
using namespace yocto::math;
using namespace yocto::image;
using namespace yocto::commonio;
using namespace yocto::opengl;

#include <future>

struct app_state {
  // original data
  string filename = "image.png";
  string outname  = "out.png";

  // image data
  image<vec4f> source = {};

  // diplay data
  image<vec4f>      display    = {};
  float             exposure   = 0;
  bool              filmic     = false;
  colorgrade_params params     = {};
  bool              colorgrade = false;

  // viewing properties
  opengl_image*       glimage  = new opengl_image{};
  draw_glimage_params glparams = {};

  ~app_state() {
    if (glimage) delete glimage;
  }
};

void update_display(app_state* app) {
  if (app->display.size() != app->source.size()) app->display = app->source;
  if (app->colorgrade) {
    colorgrade_image_mt(app->display, app->source, true, app->params);
  } else {
    tonemap_image_mt(app->display, app->source, app->exposure, app->filmic);
  }
}

int main(int argc, const char* argv[]) {
  // prepare application
  auto app_guard = make_unique<app_state>();
  auto app       = app_guard.get();
  auto filenames = vector<string>{};

  // command line options
  auto cli = make_cli("yimgviews", "view images");
  add_option(cli, "--output,-o", app->outname, "image output");
  add_option(cli, "image", app->filename, "image filename", true);
  parse_cli(cli, argc, argv);

  // load image
  auto ioerror = ""s;
  if (!load_image(app->filename, app->source, ioerror)) {
    print_fatal(ioerror);
    return 1;
  }

  // update display
  update_display(app);

  // create window
  auto win_guard = make_glwindow({1280, 720}, "yimgviews", false);
  auto win       = win_guard.get();

  // set callbacks
  set_draw_glcallback(
      win, [app](opengl_window* win, const opengl_input& input) {
        app->glparams.window      = input.window_size;
        app->glparams.framebuffer = input.framebuffer_viewport;
        if (!is_initialized(app->glimage)) {
          init_glimage(app->glimage);
          set_glimage(app->glimage, app->display, false, false);
        }
        update_imview(app->glparams.center, app->glparams.scale,
            app->display.size(), app->glparams.window, app->glparams.fit);
        draw_glimage(app->glimage, app->glparams);
      });
  set_uiupdate_glcallback(
      win, [app](opengl_window* win, const opengl_input& input) {
        // handle mouse
        if (input.mouse_left) {
          app->glparams.center += input.mouse_pos - input.mouse_last;
        }
        if (input.mouse_right) {
          app->glparams.scale *= powf(
              2, (input.mouse_pos.x - input.mouse_last.x) * 0.001f);
        }
      });

  // run ui
  run_ui(win);

  // cleanup
  clear_glwindow(win);

  // done
  return 0;
}
