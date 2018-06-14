//
// LICENSE:
//
// Copyright (c) 2016 -- 2018 Fabio Pellacini
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
#include "../yocto/yocto_glio.h"
#include "CLI11.hpp"
using namespace std::literals;

struct app_state {
    std::shared_ptr<ygl::scene> scn;
    std::shared_ptr<ygl::camera> cam;
    ygl::image4f img;
    ygl::image<ygl::rng_state> rngs;

    std::string filename = "scene.json"s;
    std::string imfilename = "out.hdr"s;
    int resolution = 512;
    int nsamples = 256;
    std::string tracer = "pathtrace"s;
    ygl::trace_func tracef = &ygl::trace_path;
    int nbounces = 8;
    float pixel_clamp = 100.0f;
    bool noparallel = false;
    int seed = 7;
    int batch_size = 16;
    bool save_batch = false;
    float exposure = 0.0f;
    bool double_sided = false;
    bool add_skyenv = false;
    bool quiet = false;
};

auto tracer_names = std::unordered_map<std::string, ygl::trace_func>{
    {"pathtrace", ygl::trace_path},
    {"direct", ygl::trace_direct},
    {"environment", ygl::trace_environment},
    {"eyelight", ygl::trace_eyelight},
    {"pathtrace-nomis", ygl::trace_path_nomis},
    {"pathtrace-naive", ygl::trace_path_naive},
    {"direct-nomis", ygl::trace_direct_nomis},
    {"debug-normal", ygl::trace_debug_normal},
    {"debug-albedo", ygl::trace_debug_albedo},
    {"debug-texcoord", ygl::trace_debug_texcoord},
    {"debug-frontfacing", ygl::trace_debug_frontfacing},
};

int main(int argc, char* argv[]) {
    // create empty scene
    auto app = std::make_shared<app_state>();

    // parse command line
    CLI::App parser("Offline path tracing", "ytrace");
    parser.add_option(
        "--resolution,-r", app->resolution, "Image vertical resolution.");
    parser.add_option("--nsamples,-s", app->nsamples, "Number of samples.");
    parser.add_option("--tracer,-t", app->tracer, "Trace type.")
        ->check([](const std::string& s) -> std::string {
            if (tracer_names.find(s) == tracer_names.end())
                throw CLI::ValidationError("unknown tracer name");
            return s;
        });
    parser.add_option(
        "--nbounces", app->nbounces, "Maximum number of bounces.");
    parser.add_option(
        "--pixel-clamp", app->pixel_clamp, "Final pixel clamping.");
    parser.add_flag(
        "--noparallel", app->noparallel, "Disable parallel execution.");
    parser.add_option(
        "--seed", app->seed, "Seed for the random number generators.");
    parser.add_option("--batch-size", app->batch_size, "Sample batch size.");
    parser.add_flag(
        "--save-batch", app->save_batch, "Save images progressively");
    parser.add_option("--exposure,-e", app->exposure, "Hdr exposure");
    parser.add_flag(
        "--double-sided,-D", app->double_sided, "Double-sided rendering.");
    parser.add_flag("--add-skyenv,-E", app->add_skyenv, "add missing env map");
    parser.add_flag("--quiet,-q", app->quiet, "Print only errors messages");
    parser.add_option("--output-image,-o", app->imfilename, "Image filename");
    parser.add_option("scene", app->filename, "Scene filename")->required(true);
    try {
        parser.parse(argc, argv);
    } catch (const CLI::ParseError& e) { return parser.exit(e); }
    app->tracef = tracer_names.at(app->tracer);

    // scene loading
    if (!app->quiet) std::cout << "loading scene" << app->filename << "\n";
    auto load_start = ygl::get_time();
    try {
        app->scn = ygl::load_scene(app->filename);
    } catch (const std::exception& e) {
        std::cout << "cannot load scene " << app->filename << "\n";
        std::cout << "error: " << e.what() << "\n";
        exit(1);
    }
    if (!app->quiet)
        std::cout << "loading in "
                  << ygl::format_duration(ygl::get_time() - load_start) << "\n";

    // tesselate
    if (!app->quiet) std::cout << "tesselating scene elements\n";
    ygl::update_tesselation(app->scn);

    // update bbox and transforms
    ygl::update_transforms(app->scn);
    ygl::update_bbox(app->scn);

    // add components
    if (!app->quiet) std::cout << "adding scene elements\n";
    if (app->add_skyenv && app->scn->environments.empty()) {
        app->scn->environments.push_back(ygl::make_sky_environment("sky"));
        app->scn->textures.push_back(app->scn->environments.back()->ke_txt);
    }
    if (app->double_sided)
        for (auto mat : app->scn->materials) mat->double_sided = true;
    if (app->scn->cameras.empty())
        app->scn->cameras.push_back(
            ygl::make_bbox_camera("<view>", app->scn->bbox));
    app->cam = app->scn->cameras[0];
    ygl::add_missing_names(app->scn);
    for (auto err : ygl::validate(app->scn))
        std::cout << "warning: " << err << "\n";

    // build bvh
    if (!app->quiet) std::cout << "building bvh\n";
    auto bvh_start = ygl::get_time();
    ygl::update_bvh(app->scn);
    if (!app->quiet)
        std::cout << "building bvh in "
                  << ygl::format_duration(ygl::get_time() - bvh_start) << "\n";

    // init renderer
    if (!app->quiet) std::cout << "initializing lights\n";
    ygl::update_lights(app->scn);

    // initialize rendering objects
    if (!app->quiet) std::cout << "initializing tracer data\n";
    app->img = ygl::image4f{
        (int)round(app->resolution * app->cam->width / app->cam->height),
        app->resolution};
    app->rngs =
        ygl::make_trace_rngs(app->img.width(), app->img.height(), app->seed);

    // render
    if (!app->quiet) std::cout << "rendering image\n";
    auto render_start = ygl::get_time();
    for (auto sample = 0; sample < app->nsamples; sample += app->batch_size) {
        if (app->save_batch && sample) {
            auto filename = ygl::replace_extension(
                app->imfilename, std::to_string(sample) + "." +
                                     ygl::get_extension(app->imfilename));
            if (!app->quiet) std::cout << "saving image " << filename << "\n";
            ygl::save_image(
                filename, ygl::expose_image(app->img, app->exposure));
        }
        if (!app->quiet)
            std::cout << "rendering sample " << sample << "/" << app->nsamples
                      << "\n";
        auto block_start = ygl::get_time();
        if (app->noparallel) {
            ygl::trace_samples(app->scn, app->cam, app->img, app->rngs, sample,
                std::min(app->batch_size, app->nsamples - sample), app->tracef,
                app->nbounces, app->pixel_clamp);
        } else {
            ygl::trace_samples_mt(app->scn, app->cam, app->img, app->rngs,
                sample, std::min(app->batch_size, app->nsamples - sample),
                app->tracef, app->nbounces, app->pixel_clamp);
        }
        if (!app->quiet)
            std::cout << "rendering block in "
                      << ygl::format_duration(ygl::get_time() - block_start)
                      << "\n";
    }
    if (!app->quiet)
        std::cout << "rendering image in "
                  << ygl::format_duration(ygl::get_time() - render_start)
                  << "\n";

    // save image
    if (!app->quiet) std::cout << "saving image " << app->imfilename << "\n";
    ygl::save_image(
        app->imfilename, ygl::expose_image(app->img, app->exposure));

    // done
    return 0;
}
