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

#if 0
template <typename Image>
Image make_image_grid(const std::vector<Image>& imgs, int tilex) {
    auto nimgs = (int)imgs.size();
    auto width = imgs[0].size().x * tilex;
    auto height = imgs[0].size().y * (nimgs / tilex + ((nimgs % tilex) ? 1 : 0));
    auto ret = make_image4f(width, height, (bool)imgs[0].hdr);
    auto img_idx = 0;
    for (auto& img : imgs) {
        if (img.size != imgs[0].size) {
            log_fatal("images of different sizes are not accepted");
        }
        auto ox = (img_idx % tilex) * img.size().x,
             oy = (img_idx / tilex) * img.size().y;
        if (ret.hdr) {
            for (auto j = 0; j < img.size().y; j++) {
                for (auto i = 0; i < img.size().x; i++) {
                    ret.hdr[{i + ox, j + oy}] = img.hdr[{i, j}];
                }
            }
        } else {
            for (auto j = 0; j < img.size().y; j++) {
                for (auto i = 0; i < img.size().x; i++) {
                    ret.ldr[{i + ox, j + oy}] = img.ldr[{i, j}];
                }
            }
        }
    }
    return ret;
}
#endif

image<vec4f> filter_bilateral(const image<vec4f>& img, float spatial_sigma,
    float range_sigma, const std::vector<image<vec4f>>& features,
    const std::vector<float>& features_sigma) {
    auto filtered     = image<vec4f>{img.size()};
    auto filter_width = (int)ceil(2.57f * spatial_sigma);
    auto sw           = 1 / (2.0f * spatial_sigma * spatial_sigma);
    auto rw           = 1 / (2.0f * range_sigma * range_sigma);
    auto fw           = std::vector<float>();
    for (auto feature_sigma : features_sigma)
        fw.push_back(1 / (2.0f * feature_sigma * feature_sigma));
    for (auto j = 0; j < img.size().y; j++) {
        for (auto i = 0; i < img.size().x; i++) {
            auto av = zero4f;
            auto aw = 0.0f;
            for (auto fj = -filter_width; fj <= filter_width; fj++) {
                for (auto fi = -filter_width; fi <= filter_width; fi++) {
                    auto ii = i + fi, jj = j + fj;
                    if (ii < 0 || jj < 0) continue;
                    if (ii >= img.size().x || jj >= img.size().y) continue;
                    auto uv  = vec2f{float(i - ii), float(j - jj)};
                    auto rgb = img[{i, j}] - img[{ii, jj}];
                    auto w   = (float)std::exp(-dot(uv, uv) * sw) *
                             (float)std::exp(-dot(rgb, rgb) * rw);
                    for (auto fi = 0; fi < features.size(); fi++) {
                        auto feat = features[fi][{i, j}] -
                                    features[fi][{ii, jj}];
                        w *= exp(-dot(feat, feat) * fw[fi]);
                    }
                    av += w * img[{ii, jj}];
                    aw += w;
                }
            }
            filtered[{i, j}] = av / aw;
        }
    }
    return filtered;
}

image<vec4f> filter_bilateral(
    const image<vec4f>& img, float spatial_sigma, float range_sigma) {
    auto filtered = image<vec4f>{img.size()};
    auto fwidth   = (int)ceil(2.57f * spatial_sigma);
    auto sw       = 1 / (2.0f * spatial_sigma * spatial_sigma);
    auto rw       = 1 / (2.0f * range_sigma * range_sigma);
    for (auto j = 0; j < img.size().y; j++) {
        for (auto i = 0; i < img.size().x; i++) {
            auto av = zero4f;
            auto aw = 0.0f;
            for (auto fj = -fwidth; fj <= fwidth; fj++) {
                for (auto fi = -fwidth; fi <= fwidth; fi++) {
                    auto ii = i + fi, jj = j + fj;
                    if (ii < 0 || jj < 0) continue;
                    if (ii >= img.size().x || jj >= img.size().y) continue;
                    auto uv  = vec2f{float(i - ii), float(j - jj)};
                    auto rgb = img[{i, j}] - img[{ii, jj}];
                    auto w   = std::exp(-dot(uv, uv) * sw) *
                             std::exp(-dot(rgb, rgb) * rw);
                    av += w * img[{ii, jj}];
                    aw += w;
                }
            }
            filtered[{i, j}] = av / aw;
        }
    }
    return filtered;
}

int main(int argc, char* argv[]) {
    // parse command line
    auto parser  = make_cmdline_parser(argc, argv, "Process images", "yimproc");
    auto tonemap = parse_arg(parser, "--tonemap,-t", false, "Tonemap image");
    auto exposure = parse_arg(parser, "--exposure,-e", 0.0f, "Tonemap exposure");
    auto gamma    = parse_arg(parser, "--gamma,-g", 2.2f, "Tonemap gamma.");
    auto filmic   = parse_arg(
        parser, "--filmic,-f", false, "Tonemap uses filmic curve");
    auto res_width = parse_arg(
        parser, "--res-width", 0, "resize width (0 to maintain aspect)");
    auto res_height = parse_arg(
        parser, "--res-height", 0, "resize height (0 to maintain aspect)");
    auto spatial_sigma = parse_arg(
        parser, "--spatial-sigma", 0.0f, "blur spatial sigma");
    auto range_sigma = parse_arg(
        parser, "--range-sigma", 0.0f, "bilateral blur range sigma");
    auto alpha_filename = parse_arg(
        parser, "--set-alpha", "", "set alpha as this image alpha");
    auto coloralpha_filename = parse_arg(
        parser, "--set-color-as-alpha", "", "set alpha as this image color");
    auto output = parse_arg(
        parser, "--output,-o", "out.png", "output image filename", true);
    auto filename = parse_arg(
        parser, "filename", "img.hdr", "input image filename", true);
    check_cmdline(parser);

    // load
    auto img = image<vec4f>();
    try {
        img = load_image4f(filename);
    } catch (std::exception& e) {
        printf("cannot load image %s\n", filename.c_str());
        printf("error: %s\n", e.what());
        exit(1);
    }

    // set alpha
    if (alpha_filename != "") {
        auto alpha = load_image4f(alpha_filename);
        if (img.size().x != alpha.size().x || img.size().y != alpha.size().y) {
            printf("bad image size\n");
            exit(1);
        }
        for (auto j = 0; j < img.size().y; j++)
            for (auto i = 0; i < img.size().x; i++)
                img[{i, j}].w = alpha[{i, j}].w;
    }

    // set alpha
    if (coloralpha_filename != "") {
        auto alpha = load_image4f(coloralpha_filename);
        if (img.size().x != alpha.size().x || img.size().y != alpha.size().y) {
            printf("bad image size\n");
            exit(1);
        }
        for (auto j = 0; j < img.size().y; j++)
            for (auto i = 0; i < img.size().x; i++)
                img[{i, j}].w = luminance(alpha[{i, j}]);
    }

    // resize
    if (res_width || res_height) {
        img = resize_image(img, {res_width, res_height});
    }

    // bilateral
    if (spatial_sigma && range_sigma) {
        img = filter_bilateral(img, spatial_sigma, range_sigma, {}, {});
    }

    // hdr correction
    if (tonemap) img = tonemap_exposuregamma(img, exposure, gamma, filmic);

    // save
    try {
        save_image4f(output, img);
    } catch (std::exception& e) {
        printf("cannot save image %s\n", output.c_str());
        printf("error: %s\n", e.what());
        exit(1);
    }

    // done
    return 0;
}
