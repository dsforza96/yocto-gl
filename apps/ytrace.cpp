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

#include "../yocto/yocto_scene.h"
#include "../yocto/yocto_sceneio.h"
#include "../yocto/yocto_trace.h"
#include "../yocto/yocto_utils.h"
using namespace yocto;

#include "ext/CLI11.hpp"

#include <map>

int main(int argc, char* argv[]) {
    // options
    auto load_options  = load_scene_options{};
    auto bvh_options   = bvh_build_options{};
    auto trace_options = trace_image_options{};
    auto all_cameras   = false;
    auto no_parallel   = false;
    auto save_batch    = false;
    auto exposure      = 0.0f;
    auto filmic        = false;
    auto srgb          = true;
    auto add_skyenv    = false;
    auto validate      = false;
    auto logo          = true;
    auto imfilename    = "out.hdr"s;
    auto filename      = "scene.json"s;

    // names for enums
    auto trace_sampler_type_namemap = std::map<string, trace_sampler_type>{};
    for (auto type = 0; type < trace_sampler_type_names.size(); type++) {
        trace_sampler_type_namemap[trace_sampler_type_names[type]] =
            (trace_sampler_type)type;
    }
    auto trace_falsecolor_type_namemap =
        std::map<string, trace_falsecolor_type>{};
    for (auto type = 0; type < trace_falsecolor_type_names.size(); type++) {
        trace_falsecolor_type_namemap[trace_falsecolor_type_names[type]] =
            (trace_falsecolor_type)type;
    }

    // parse command line
    auto parser = CLI::App{"Offline path tracing"};
    parser.add_option("--camera", trace_options.camera_id, "Camera index.");
    parser.add_flag(
        "--all-cameras,!--no-all-cameras", all_cameras, "Render all cameras.");
    parser.add_option("--hres,-R", trace_options.image_size.x,
        "Image horizontal resolution.");
    parser.add_option(
        "--vres,-r", trace_options.image_size.y, "Image vertical resolution.");
    parser.add_option(
        "--nsamples,-s", trace_options.num_samples, "Number of samples.");
    parser.add_option("--tracer,-t", trace_options.sampler_type, "Trace type.")
        ->transform(CLI::IsMember(trace_sampler_type_namemap));
    parser
        .add_option("--falsecolor,-F", trace_options.falsecolor_type,
            "Tracer false color type.")
        ->transform(CLI::IsMember(trace_falsecolor_type_namemap));
    parser.add_option(
        "--nbounces", trace_options.max_bounces, "Maximum number of bounces.");
    parser.add_option(
        "--pixel-clamp", trace_options.pixel_clamp, "Final pixel clamping.");
    parser.add_flag("--parallel,!--no-parallel", no_parallel,
        "Disable parallel execution.");
    parser.add_option("--seed", trace_options.random_seed,
        "Seed for the random number generators.");
    parser.add_option(
        "--nbatch,-b", trace_options.samples_per_batch, "Samples per batch.");
    parser.add_flag("--env-hidden,!--no-env-hidden",
        trace_options.environments_hidden,
        "Environments are hidden in renderer");
    parser.add_flag("--double-sided,!--no-double-sided,-D",
        trace_options.double_sided, "Double-sided rendering.");
    parser.add_option("--save-batch", save_batch, "Save images progressively");
    parser.add_option("--exposure,-e", exposure, "Hdr exposure");
    parser.add_flag("--filmic,!--no-filmic", filmic, "Hdr filmic");
    parser.add_flag("--srgb,!--no-srgb", srgb, "Hdr srgb");
    parser.add_flag("--bvh-high-quality,!--no-bvh-high-quality",
        bvh_options.high_quality, "Use high quality bvh mode");
#if YOCTO_EMBREE
    parser.add_flag("--bvh-embree,!--no-bvh-embree", bvh_options.use_embree,
        "Use Embree ratracer");
    parser.add_flag("--bvh-embree-flatten,!--no-bvh-fembree-latten",
        bvh_options.embree_flatten, "Flatten BVH scene");
    parser.add_flag("--bvh-embree-compact,!--no-bvh-embree-compact",
        bvh_options.embree_compact, "Embree runs in compact memory");
#endif
    parser.add_flag(
        "--add-skyenv,!--no-add-skyenv", add_skyenv, "Add sky envmap");
    parser.add_option("--output-image,-o", imfilename, "Image filename");
    parser.add_flag("--validate,!--no-validate", validate, "Validate scene");
    parser.add_flag("--logo,!--no-logo", logo, "Whether to append a logo");
    parser.add_option("scene", filename, "Scene filename", true);
    try {
        parser.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return parser.exit(e);
    }

    // fix parallel code
    if (no_parallel) {
        bvh_options.run_serially  = true;
        load_options.run_serially = true;
    }

    // scene loading
    auto scene = yocto_scene{};
    try {
        auto timer = print_timed("loading scene");
        load_scene(filename, scene, load_options);
    } catch (const std::exception& e) {
        print_fatal(e.what());
    }

    // tesselate
    {
        auto timer = print_timed("tesselating scene");
        tesselate_subdivs(scene);
    }

    // add components
    if (validate) {
        auto timer = print_timed("validating scene");
        print_validation_errors(scene);
    }

    // add sky
    if (add_skyenv) add_sky_environment(scene);

    // build bvh
    auto bvh = bvh_scene{};
    {
        auto timer = print_timed("building bvh");
        build_scene_bvh(scene, bvh, bvh_options);
    }

    // init renderer
    auto lights = trace_lights{};
    {
        auto timer = print_timed("initializing lights");
        init_trace_lights(lights, scene);
    }

    // fix renderer type if no lights
    if ((lights.instances.empty() && lights.environments.empty()) &&
        is_trace_sampler_lit(trace_options)) {
        print_info("no lights presents, switching to eyelight shader");
        trace_options.sampler_type = trace_sampler_type::eyelight;
    }

    // cameras to render from
    auto selected_cameras = vector<int>{};
    if (all_cameras) {
        for (auto i = 0; i < scene.cameras.size(); i++) {
            selected_cameras.push_back(i);
        }
    } else {
        selected_cameras.push_back(trace_options.camera_id);
    }

    // render all selected cameras
    for (auto camera_id : selected_cameras) {
        // set camera
        trace_options.camera_id = camera_id;

        // allocate buffers
        auto image_size = get_camera_image_size(
            scene.cameras[trace_options.camera_id], trace_options.image_size);
        auto render = image{image_size, zero4f};
        auto state  = trace_state{};
        init_trace_state(state, image_size, trace_options.random_seed);

        // render
        for (auto sample = 0; sample < trace_options.num_samples;
             sample += trace_options.samples_per_batch) {
            auto nsamples = min(trace_options.samples_per_batch,
                trace_options.num_samples - sample);
            {
                auto timer = print_timed("rendering camera {} [{}/{}]",
                    trace_options.camera_id, sample, trace_options.num_samples);
                trace_image_samples(
                    render, state, scene, bvh, lights, sample, trace_options);
            }
            if (save_batch) {
                auto outfilename = format("{}.cam{}.s{}.{}",
                    get_noextension(imfilename), trace_options.camera_id,
                    sample + nsamples, get_extension(imfilename));
                try {
                    if (logo) {
                        save_tonemapped_image_with_logo(
                            outfilename, render, exposure, filmic, srgb);
                    } else {
                        save_tonemapped_image(
                            outfilename, render, exposure, filmic, srgb);
                    }
                } catch (const std::exception& e) {
                    print_fatal(e.what());
                }
            }
        }

        // save image
        try {
            auto outfilename = imfilename;
            if (all_cameras) {
                outfilename = format("{}.cam{}.{}", get_noextension(imfilename),
                    trace_options.camera_id, get_extension(imfilename));
            }
            auto timer = print_timed("saving image {}", outfilename);
            if (logo) {
                save_tonemapped_image_with_logo(
                    outfilename, render, exposure, filmic, srgb);
            } else {
                save_tonemapped_image(
                    outfilename, render, exposure, filmic, srgb);
            }
        } catch (const std::exception& e) {
            print_fatal(e.what());
        }
    }

    // done
    return 0;
}
