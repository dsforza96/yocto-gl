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

#include "../yocto/ygl.h"
#include "../yocto/yglio.h"
using namespace ygl;

void mkdir(const std::string& dir) {
    if (dir == "" || dir == "." || dir == ".." || dir == "./" || dir == "../")
        return;
#ifndef _MSC_VER
    system(("mkdir -p " + dir).c_str());
#else
    system(("mkdir " + dir).c_str());
#endif
}

int main(int argc, char** argv) {
    // parse command line
    auto parser = make_cmdline_parser(argc, argv, "Process scene", "yscnproc");
    auto notextures = parse_arg(
        parser, "--notextures", false, "Disable textures.");
    auto uniform_txt = parse_arg(
        parser, "--uniform-txt", false, "uniform texture formats");
    auto build_bvh = parse_arg(parser, "--build-bvh", false, "build bvh");
    auto output    = parse_arg(
        parser, "--output,-o", "out.json", "output scene", true);
    auto filename = parse_arg(
        parser, "scene", "scene.json", "input scene", true);
    check_cmdline(parser);

    // load scene
    auto scn = (scene*)nullptr;
    try {
        scn = load_scene(filename, !notextures);
    } catch (const std::exception& e) {
        printf("cannot load scene %s\n", filename.c_str());
        printf("error: %s\n", e.what());
        exit(1);
    }

    // change texture names
    if (uniform_txt) {
        for (auto txt : scn->textures) {
            auto ext = get_extension(txt->path);
            if (is_hdr_filename(txt->path)) {
                if (ext == "hdr" || ext == "exr") continue;
                if (ext == "pfm")
                    replace_extension(filename, "hdr");
                else
                    printf("unknown texture format %s\n", ext.c_str());
            } else {
                if (ext == "png" || ext == "jpg") continue;
                if (ext == "tga" || ext == "bmp")
                    replace_extension(filename, "png");
                else
                    printf("unknown texture format %s\n", ext.c_str());
            }
        }
    }

    // build bvh
    if (build_bvh) ygl::build_bvh(scn, false);

    // make a directory if needed
    try {
        mkdir(get_dirname(output));
    } catch (const std::exception& e) {
        printf("cannot create directory %s\n", get_dirname(output).c_str());
        printf("error: %s\n", e.what());
        exit(1);
    }
    // save scene
    try {
        save_scene(output, scn);
    } catch (const std::exception& e) {
        printf("cannot save scene %s\n", output.c_str());
        printf("error: %s\n", e.what());
        exit(1);
    }

    // done
    return 0;
}
