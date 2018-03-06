//
// LICENSE:
//
// Copyright (c) 2016 -- 2017 Fabio Pellacini
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

#include "../yocto/yocto_gl.h"
using namespace std::literals;

// Application state
struct app_state {
    ygl::scene* scn = nullptr;
    ygl::camera* view = nullptr;
    ygl::camera* cam = nullptr;
    std::string filename;
    std::string imfilename;
    std::string outfilename;
    ygl::gl_stdsurface_params params = {};
    ygl::tonemap_params tmparams = {};
    ygl::gl_stdsurface_program prog;
    std::unordered_map<ygl::texture*, ygl::gl_texture> textures;
    std::unordered_map<ygl::shape*, ygl::gl_shape> shapes;
    ygl::gl_lights lights;
    bool navigation_fps = false;
    ygl::scene_selection selection = {};
    std::vector<ygl::scene_selection> update_list;
    ygl::proc_scene* pscn = nullptr;
    float time = 0;
    ygl::vec2f time_range = ygl::zero2f;
    bool animate = false;
    bool quiet = false;
    bool screenshot_and_exit = false;
    bool no_widgets = false;
    std::unordered_map<std::string, std::string> inspector_highlights;

    ~app_state() {
        if (scn) delete scn;
        if (view) delete view;
        if (pscn) delete pscn;
    }
};

// draw with shading
inline void draw(ygl::gl_window* win, app_state* app) {
    auto framebuffer_size = get_framebuffer_size(win);
    app->params.resolution = framebuffer_size.y;

    static auto last_time = 0.0f;
    for (auto& sel : app->update_list) {
        if (sel.is<ygl::texture>()) {
            ygl::update_gl_texture(sel.get<ygl::texture>(),
                app->textures[sel.get<ygl::texture>()]);
        }
        if (sel.is<ygl::shape>()) {
            ygl::update_gl_shape(
                sel.get<ygl::shape>(), app->shapes[sel.get<ygl::shape>()]);
        }
        if (sel.is<ygl::node>() || sel.is<ygl::animation>() ||
            sel.is<ygl::animation_group>() || app->time != last_time) {
            ygl::update_transforms(app->scn, app->time);
            last_time = app->time;
        }
        if (sel.is<ygl::shape>() || sel.is<ygl::material>() ||
            sel.is<ygl::node>()) {
            app->lights = ygl::make_gl_lights(app->scn);
            if (app->lights.pos.empty()) app->params.eyelight = true;
        }
    }
    app->update_list.clear();

    ygl::gl_clear_buffers(app->params.background);
    ygl::gl_enable_depth_test(true);
    ygl::gl_enable_culling(app->params.cull_backface);
    ygl::draw_stdsurface_scene(app->scn, app->cam, app->prog, app->shapes,
        app->textures, app->lights, framebuffer_size,
        app->selection.get_untyped(), app->params, app->tmparams);

    if (app->no_widgets) {
        ygl::swap_buffers(win);
        return;
    }

    if (ygl::begin_widgets(win, "yview")) {
        if (ygl::draw_header_widget(win, "view")) {
            ygl::draw_groupid_widget_push(win, app);
            ygl::draw_value_widget(win, "scene", app->filename);
            if (ygl::draw_button_widget(win, "new")) {
                delete app->pscn;
                delete app->scn;
                ygl::clear_gl_textures(app->textures);
                ygl::clear_gl_shapes(app->shapes);
                app->pscn = new ygl::proc_scene();
                app->scn = new ygl::scene();
                ygl::update_proc_elems(app->scn, app->pscn);
                app->textures = ygl::make_gl_textures(app->scn);
                app->shapes = ygl::make_gl_shapes(app->scn);
            }
            ygl::draw_continue_widget(win);
            if (ygl::draw_button_widget(win, "load")) {
                delete app->pscn;
                delete app->scn;
                app->scn = ygl::load_scene(app->filename, {});
                app->pscn = new ygl::proc_scene();
                ygl::clear_gl_textures(app->textures);
                ygl::clear_gl_shapes(app->shapes);
                app->textures = ygl::make_gl_textures(app->scn);
                app->shapes = ygl::make_gl_shapes(app->scn);
            }
            ygl::draw_continue_widget(win);
            if (ygl::draw_button_widget(win, "save")) {
                ygl::save_scene(app->filename, app->scn, {});
            }
            ygl::draw_continue_widget(win);
            if (draw_button_widget(win, "save proc")) {
                ygl::save_proc_scene(
                    ygl::replace_path_extension(app->filename, ".json"),
                    app->pscn);
            }
            ygl::draw_camera_selection_widget(
                win, "camera", app->cam, app->scn, app->view);
            ygl::draw_camera_widgets(win, "camera", app->cam);
            ygl::draw_value_widget(win, "fps", app->navigation_fps);
            if (app->time_range != ygl::zero2f) {
                ygl::draw_value_widget(win, "time", app->time,
                    app->time_range.x, app->time_range.y);
                ygl::draw_value_widget(win, "animate", app->animate);
            }
            if (ygl::draw_button_widget(win, "print stats"))
                std::cout << ygl::compute_stats(app->scn);
            ygl::draw_groupid_widget_pop(win);
        }
        if (ygl::draw_header_widget(win, "params")) {
            ygl::draw_params_widgets(win, "", app->params);
            ygl::draw_params_widgets(win, "", app->tmparams);
        }
        if (ygl::draw_header_widget(win, "scene")) {
            ygl::draw_scene_widgets(win, "", app->scn, app->selection,
                app->update_list, app->textures, app->pscn,
                app->inspector_highlights);
        }
    }
    ygl::end_widgets(win);

    ygl::swap_buffers(win);
}

// run ui loop
void run_ui(app_state* app) {
    // window
    auto win = ygl::make_window(
        (int)std::round(app->cam->aspect * app->params.resolution),
        app->params.resolution, "yview | " + app->filename);
    ygl::set_window_callbacks(
        win, nullptr, nullptr, [win, app]() { draw(win, app); });

    // load textures and vbos
    app->prog = ygl::make_stdsurface_program();
    app->textures = ygl::make_gl_textures(app->scn);
    app->shapes = ygl::make_gl_shapes(app->scn);
    ygl::update_transforms(app->scn, app->time);
    app->lights = ygl::make_gl_lights(app->scn);
    if (app->lights.pos.empty()) app->params.eyelight = true;

    // init widget
    ygl::init_widgets(win);

    // loop
    while (!should_close(win)) {
        // handle mouse and keyboard for navigation
        if (app->cam == app->view) {
            ygl::handle_camera_navigation(win, app->view, app->navigation_fps);
        }

        // animation
        if (app->animate) {
            app->time += 1 / 60.0f;
            if (app->time < app->time_range.x || app->time > app->time_range.y)
                app->time = app->time_range.x;
        }

        // draw
        draw(win, app);

        // check if exiting is needed
        if (app->screenshot_and_exit) {
            ygl::log_info("taking screenshot and exiting...");
            ygl::save_screenshot(win, app->imfilename);
            break;
        }

        // event hadling
        ygl::poll_events(win);
    }

    // cleanup
    delete win;
}

int main(int argc, char* argv[]) {
    // create empty scene
    auto app = new app_state();

    // parse command line
    auto parser =
        ygl::make_parser(argc, argv, "yview", "views scenes inteactively");
    app->params = ygl::parse_params(parser, "", app->params);
    app->tmparams = ygl::parse_params(parser, "", app->tmparams);
    auto preserve_quads = ygl::parse_flag(
        parser, "--preserve-quads", "", "Preserve quads on load");
    auto preserve_facevarying = ygl::parse_flag(
        parser, "--preserve-facevarying", "", "Preserve facevarying on load");
    app->quiet =
        ygl::parse_flag(parser, "--quiet", "-q", "Print only errors messages");
    app->screenshot_and_exit = ygl::parse_flag(
        parser, "--screenshot-and-exit", "", "Take a screenshot and exit");
    app->no_widgets =
        ygl::parse_flag(parser, "--no-widgets", "", "Disable widgets");
    auto highlight_filename =
        ygl::parse_opt(parser, "--highlights", "", "Highlight filename", ""s);
    app->imfilename = ygl::parse_opt(
        parser, "--output-image", "-o", "Image filename", "out.png"s);
    auto filenames = ygl::parse_args(
        parser, "scenes", "Scene filenames", std::vector<std::string>());
    if (ygl::should_exit(parser)) {
        printf("%s\n", get_usage(parser).c_str());
        exit(1);
    }

    // setup logger
    if (app->quiet) ygl::get_default_logger()->verbose = false;

    // fix hilights
    if (!highlight_filename.empty()) {
        try {
            app->inspector_highlights =
                ygl::load_ini(highlight_filename).at("");
        } catch (std::exception e) {
            ygl::log_fatal("cannot load highlihgt file {}", highlight_filename);
        }
    }

    // scene loading
    app->scn = new ygl::scene();
    for (auto filename : filenames) {
        try {
            ygl::log_info("loading scene {}", filename);
            auto opts = ygl::load_options();
            opts.preserve_quads = preserve_quads;
            opts.preserve_facevarying = preserve_facevarying;
            auto scn = load_scene(filename, opts);
            ygl::merge_into(app->scn, scn);
            delete scn;
        } catch (std::exception e) {
            ygl::log_fatal("cannot load scene {}", filename);
        }
    }
    app->filename = filenames.front();

    // tesselate input shapes
    ygl::tesselate_shapes(app->scn, true, !preserve_facevarying,
        !preserve_quads && !preserve_facevarying, false);

    // add missing data
    ygl::add_elements(app->scn);

    // fix double sided materials
    if (app->params.double_sided) {
        for (auto m : app->scn->materials) m->double_sided = true;
    }

    // view camera
    app->view = ygl::make_view_camera(app->scn, 0);
    app->cam = app->view;

    // animation
    app->time_range = ygl::compute_animation_range(app->scn);
    app->time = app->time_range.x;

    // lights
    app->lights = ygl::make_gl_lights(app->scn);

    // run ui
    run_ui(app);

    // cleanup
    delete app;

    // done
    return 0;
}
