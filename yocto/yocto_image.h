//
// # Yocto/Image: Tiny C++ Library of Image Utilities
//
// Yocto/Image is a collection of utilities for image loading and saving,
// example image creation, tonempping, image resizing, and sunsky procedural
// images. The library contains only color conversion utilities.
// Yocto/Image is written to support the need of a minimal, but
// fully-featured, global illumination renderer, rather than the need of
// generic image editing. Image IO and resizing are wrappers of `stb_image`
// and `tinyexr` libraries.
//
// Yocto/Img support only two image types, `image4b` for low-dynamic range
// images and `image4f` for high-dynamic range images. Pixels are represented
// as RGBA values of either 8-bytes ints (LDRs) or float (HDRs) types.
//
// ## Usage
//
// 1. create images with `make_image4b()` and `make_image4f()`
// 1. load images with `load_image4b()` or `load_image4f()`
// 2. save images with `save_image4b()` or `save_image4f()`
// 3. resize images with `resize_image()`
// 4. tonemap images with `tonemap_image()`
// 5. make various image examples with the `make_XXX_image()` functions
// 6. create procedural sun-sky images with `make_sunsky_image()`
// 7. we also support low level image IO with `load_XXX()` and `save_XXX()`
//    functions.
// 8. see list below for the color conversion utilities supported
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2018 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#ifndef _YGL_IMAGE_H_
#define _YGL_IMAGE_H_

// -----------------------------------------------------------------------------
// COMPILATION OPTIONS AND INCLUDES
// -----------------------------------------------------------------------------

// enable image io
#ifndef YGL_IMAGEIO
#define YGL_IMAGEIO 1
#endif

#include "yocto_math.h"

#include <vector>

// -----------------------------------------------------------------------------
// COLOR CONVERSION UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Element-wise float to byte conversion.
inline vec4b float_to_byte(const vec4f& a) {
    return {(byte)max(0, min(int(a.x * 256), 255)),
        (byte)max(0, min(int(a.y * 256), 255)),
        (byte)max(0, min(int(a.z * 256), 255)),
        (byte)max(0, min(int(a.w * 256), 255))};
}
inline vec4f byte_to_float(const vec4b& a) {
    return {a.x / 255.0f, a.y / 255.0f, a.z / 255.0f, a.w / 255.0f};
}

// Approximate conversion from srgb.
inline vec4f srgb_to_linear(const vec4b& srgb) {
    return {pow(srgb.x / 255.0f, 2.2f), pow(srgb.y / 255.0f, 2.2f),
        pow(srgb.z / 255.0f, 2.2f), srgb.w / 255.0f};
}
// Approximate conversion to srgb.
inline vec4b linear_to_srgb(const vec4f& lin) {
    return float_to_byte({pow(lin.x, 1 / 2.2f), pow(lin.y, 1 / 2.2f),
        pow(lin.z, 1 / 2.2f), lin.w});
}

// Convert between CIE XYZ and xyY
inline vec3f xyz_to_xyY(const vec3f& xyz) {
    if (xyz == zero3f) return zero3f;
    return {xyz.x / (xyz.x + xyz.y + xyz.z), xyz.y / (xyz.x + xyz.y + xyz.z),
        xyz.y};
}
// Convert between CIE XYZ and xyY
inline vec3f xyY_to_xyz(const vec3f& xyY) {
    if (xyY.y == 0) return zero3f;
    return {xyY.x * xyY.z / xyY.y, xyY.z, (1 - xyY.x - xyY.y) * xyY.z / xyY.y};
}
// Convert between CIE XYZ and RGB
inline vec3f xyz_to_rgb(const vec3f& xyz) {
    // from http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    if (xyz == zero3f) return zero3f;
    return {+3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
        -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
        +0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z};
}
// Convert between CIE XYZ and RGB
inline vec3f rgb_to_xyz(const vec3f& rgb) {
    // from http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    if (rgb == zero3f) return zero3f;
    return {0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z,
        0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z,
        0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z};
}

// Converts HSV to RGB.
vec4b hsv_to_rgb(const vec4b& hsv);

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMAGE TYPE AND UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Ldr image of (r,g,b,a) pixels. Access pixels with at().
struct image4b {
    std::vector<vec4b> pixels;  // image pixels
    int width = 0;              // image width
    int height = 0;             // image height

    // pixel access
    vec4b& at(int i, int j) { return pixels.at(j * width + i); }
    const vec4b& at(int i, int j) const { return pixels.at(j * width + i); }
};

// Hdr image container. Access pixels with at().
struct image4f {
    std::vector<vec4f> pixels;  // image pixels
    int width = 0;              // image width
    int height = 0;             // image height

    // pixel access
    vec4f& at(int i, int j) { return pixels.at(j * width + i); }
    const vec4f& at(int i, int j) const { return pixels.at(j * width + i); }
};

// Initializes empty images
inline image4b make_image4b(
    int width, int height, const vec4b& val = {0, 0, 0, 0}) {
    auto img = image4b();
    img.width = width;
    img.height = height;
    img.pixels.assign(width * height, val);
    return img;
}
inline image4f make_image4f(
    int width, int height, const vec4f& val = {0, 0, 0, 0}) {
    auto img = image4f();
    img.width = width;
    img.height = height;
    img.pixels.assign(width * height, val);
    return img;
}

// Create an image with values stored in an array in scanline order.
inline image4b make_image4b(int width, int height, const vec4b* vals) {
    auto img = image4b();
    img.width = width;
    img.height = height;
    img.pixels.assign(vals, vals + width * height);
    return img;
}
inline image4f make_image4f(int width, int height, const vec4f* vals) {
    auto img = image4f();
    img.width = width;
    img.height = height;
    img.pixels.assign(vals, vals + width * height);
    return img;
}

// Create a 4 channel image with the given number of channels
inline image4f make_image4f(
    int w, int h, int nc, const float* vals, const vec4f& def) {
    auto img = make_image4f(w, h);
    for (auto j = 0; j < h; j++) {
        for (auto i = 0; i < w; i++) {
            auto pixel = vals + (j * w + i) * nc;
            img.at(i, j) = def;
            auto img_pixel = &img.at(i, j).x;
            for (auto c = 0; c < nc; c++) img_pixel[c] = pixel[c];
        }
    }
    return img;
}
inline image4b make_image4b(
    int w, int h, int nc, const byte* vals, const vec4b& def) {
    auto img = make_image4b(w, h);
    for (auto j = 0; j < h; j++) {
        for (auto i = 0; i < w; i++) {
            auto pixel = vals + (j * w + i) * nc;
            img.at(i, j) = def;
            auto img_pixel = &img.at(i, j).x;
            for (auto c = 0; c < nc; c++) img_pixel[c] = pixel[c];
        }
    }
    return img;
}

// Check if a pixel is inside an image.
inline bool contains(const image4b& img, int i, int j) {
    return i >= 0 && i < img.width && j >= 0 && j < img.height;
}
inline bool contains(const image4f& img, int i, int j) {
    return i >= 0 && i < img.width && j >= 0 && j < img.height;
}

// Image over operator.
void image_over(vec4f* img, int width, int height, int nlayers, vec4f** layers);
void image_over(vec4b* img, int width, int height, int nlayers, vec4b** layers);

// Approximate conversion from srgb.
inline image4f srgb_to_linear(const image4b& srgb) {
    auto lin = make_image4f(srgb.width, srgb.height);
    for (auto j = 0; j < srgb.height; j++)
        for (auto i = 0; i < srgb.width; i++)
            lin.at(i, j) = srgb_to_linear(srgb.at(i, j));
    return lin;
}
// Approximate conversion to srgb.
inline image4b linear_to_srgb(const image4f& lin) {
    auto srgb = make_image4b(lin.width, lin.height);
    for (auto j = 0; j < srgb.height; j++)
        for (auto i = 0; i < srgb.width; i++)
            srgb.at(i, j) = linear_to_srgb(lin.at(i, j));
    return srgb;
}

// Tone mapping type.
enum struct tonemap_type {
    linear = 0,
    gamma = 1,
    srgb = 2,
    filmic1 = 3,
    filmic2 = 4,
    filmic3 = 5
};

// Tone mapping HDR to LDR images.
image4b tonemap_image(const image4f& hdr, tonemap_type type, float exposure);

// Names of enum values.
inline const std::map<tonemap_type, std::string>& tonemap_type_names() {
    static auto names = std::map<tonemap_type, std::string>{
        {tonemap_type::linear, "linear"},
        {tonemap_type::gamma, "gamma"},
        {tonemap_type::srgb, "srgb"},
        {tonemap_type::filmic1, "filmic1"},
        {tonemap_type::filmic2, "filmic2"},
        {tonemap_type::filmic3, "filmic3"},
    };
    return names;
}

// Make example images.
image4b make_grid_image(int width, int height, int tile = 64,
    const vec4b& c0 = {64, 64, 64, 255},
    const vec4b& c1 = {128, 128, 128, 255});
image4b make_checker_image(int width, int height, int tile = 64,
    const vec4b& c0 = {64, 64, 64, 255},
    const vec4b& c1 = {128, 128, 128, 255});
image4b make_bumpdimple_image(int width, int height, int tile = 64);
image4b make_ramp_image(
    int width, int height, const vec4b& c0, const vec4b& c1, bool srgb = false);
image4b make_gammaramp_image(int width, int height);
image4f make_gammaramp_imagef(int width, int height);
image4b make_uv_image(int width, int height);
image4b make_uvgrid_image(
    int width, int height, int tile = 64, bool colored = true);
image4b make_recuvgrid_image(
    int width, int height, int tile = 64, bool colored = true);

// Comvert a bump map to a normal map.
image4b bump_to_normal_map(const image4b& img, float scale = 1);

// Make a sunsky HDR model with sun at theta elevation in [0,pi/2], turbidity
// in [1.7,10] with or without sun.
image4f make_sunsky_image(int res, float thetaSun, float turbidity = 3,
    bool has_sun = false, bool has_ground = true);

// Make a noise image. Wrap works only if both resx and resy are powers of two.
image4b make_noise_image(int resx, int resy, float scale = 1, bool wrap = true);
image4b make_fbm_image(int resx, int resy, float scale = 1,
    float lacunarity = 2, float gain = 0.5f, int octaves = 6, bool wrap = true);
image4b make_ridge_image(int resx, int resy, float scale = 1,
    float lacunarity = 2, float gain = 0.5f, float offset = 1.0f,
    int octaves = 6, bool wrap = true);
image4b make_turbulence_image(int resx, int resy, float scale = 1,
    float lacunarity = 2, float gain = 0.5f, int octaves = 6, bool wrap = true);

#if YGL_IMAGEIO

// Check if an image is HDR based on filename.
bool is_hdr_filename(const std::string& filename);

// Loads/saves a 4 channel ldr/hdr image.
image4b load_image4b(const std::string& filename);
image4f load_image4f(const std::string& filename);
bool save_image4b(const std::string& filename, const image4b& img);
bool save_image4f(const std::string& filename, const image4f& img);
// Save a 4 channel HDR or LDR image with tonemapping based on filename.
bool save_image(const std::string& filename, const image4f& hdr,
    tonemap_type tonemapper, float exposure);

// Loads.saves an image with variable number of channels.
std::vector<float> load_imagef(
    const std::string& filename, int& width, int& height, int& ncomp);
std::vector<byte> load_imageb(
    const std::string& filename, int& width, int& height, int& ncomp);
std::vector<float> load_imagef_from_memory(const std::string& filename,
    const byte* data, int length, int& width, int& height, int& ncomp);
std::vector<byte> load_imageb_from_memory(const std::string& filename,
    const byte* data, int length, int& width, int& height, int& ncomp);
bool save_imagef(const std::string& filename, int width, int height, int ncomp,
    const float* hdr);
bool save_imageb(const std::string& filename, int width, int height, int ncomp,
    const byte* ldr);

// Filter type and edge mode for resizing.
enum struct resize_filter {
    def,
    box,
    triangle,
    cubic_spline,
    catmull_rom,
    mitchell
};
enum struct resize_edge { def, clamp, reflect, wrap, zero };

// Resize an image.
void resize_image(const image4f& img, image4f& res_img,
    resize_filter filter = resize_filter::def,
    resize_edge edge = resize_edge::def, bool premultiplied_alpha = false);
void resize_image(const image4b& img, image4b& res_img,
    resize_filter filter = resize_filter::def,
    resize_edge edge = resize_edge::def, bool premultiplied_alpha = false);

#endif

}  // namespace ygl

#endif
