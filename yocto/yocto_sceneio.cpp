//
// Implementation for Yocto/GL Input and Output functions.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2019 Fabio Pellacini
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
//
// LICENSE OF INCLUDED CODE FOR BASE64 (base64.h, base64.cpp)
//
// Copyright (C) 2004-2008 René Nyffenegger
//
// This source code is provided 'as-is', without any express or implied
// warranty. In no event will the author be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this source code must not be misrepresented; you must not
// claim that you wrote the original source code. If you use this source code
// in a product, an acknowledgment in the product documentation would be
// appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original source code.
//
// 3. This notice may not be removed or altered from any source distribution.
//
// René Nyffenegger rene.nyffenegger@adp-gmbh.ch
//
//

#include "yocto_sceneio.h"
#include "yocto_imageio.h"
#include "yocto_json.h"
#include "yocto_random.h"
#include "yocto_shape.h"
#include "yocto_utils.h"

#include <array>
#include <climits>
#include <cstdlib>
#include <regex>

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF FAST PARSING
// -----------------------------------------------------------------------------
namespace yocto {

// normalize obj line for simpler parsing
inline void normalize_obj_line(char* str, char comment_char = '#') {
    auto has_content = false;
    auto start       = str;
    while (*str) {
        if (*str == comment_char) {
            *str = 0;
            break;
        } else if (*str == ' ' || *str == '\t' || *str == '\r' ||
                   *str == '\n') {
            *str = ' ';
        } else {
            has_content = true;
        }
        str++;
    }
    if (!has_content) *start = 0;
}

// Parse values from a string
inline void parse_value(char*& str, int& value) {
    char* end = nullptr;
    value     = (int)strtol(str, &end, 10);
    if (str == end) throw io_error("cannot parse value");
    str = end;
}
inline void parse_value(char*& str, bool& value) {
    auto valuei = 0;
    parse_value(str, valuei);
    value = (bool)valuei;
}
inline void parse_value(char*& str, float& value) {
    char* end = nullptr;
    value     = strtof(str, &end);
    if (str == end) throw io_error("cannot parse value");
    str = end;
}
inline void parse_value(char*& str, string& value, bool ok_if_empty = false) {
    value = "";
    while (*str == ' ') str++;
    if (!*str && !ok_if_empty) {
        throw io_error("cannot parse value");
    }
    while (*str && *str != ' ') {
        value += *str;
        str++;
    }
}
template <typename T, int N>
inline void parse_value(char*& str, vec<T, N>& value) {
    for (auto i = 0; i < N; i++) parse_value(str, value[i]);
}
template <typename T, int N>
inline void parse_value(char*& str, frame<T, N>& value) {
    for (auto i = 0; i < N + 1; i++) parse_value(str, value[i]);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF CONVERSION TO/FROM JSON
// -----------------------------------------------------------------------------
namespace yocto {

template <typename T>
inline void to_json(json& js, const image<T>& value) {
    js           = json::object();
    js["width"]  = value.width;
    js["height"] = value.height;
    js["pixels"] = value.pixels;
}
template <typename T>
inline void from_json(const json& js, image<T>& value) {
    auto width  = js.at("width").get<int>();
    auto height = js.at("height").get<int>();
    auto pixels = js.at("pixels").get<vector<T>>();
    value       = image{width, height, (const T*)data(pixels)};
}
template <typename T>
inline void to_json(json& js, const volume<T>& value) {
    js           = json::object();
    js["width"]  = value.width;
    js["height"] = value.height;
    js["depth"]  = value.depth;
    js["voxels"] = value.voxels;
}
template <typename T>
inline void from_json(const json& js, volume<T>& value) {
    auto width  = js.at("width").get<int>();
    auto height = js.at("height").get<int>();
    auto depth  = js.at("depth").get<int>();
    auto voxels = js.at("voxels").get<vector<T>>();
    value       = volume{width, height, depth, (const T*)data(voxels)};
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// GENERIC SCENE LOADING
// -----------------------------------------------------------------------------
namespace yocto {

// Load a scene
void load_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    auto ext = get_extension(filename);
    if (ext == "json" || ext == "JSON") {
        load_json_scene(filename, scene, options);
    } else if (ext == "obj" || ext == "OBJ") {
        load_obj_scene(filename, scene, options);
    } else if (ext == "gltf" || ext == "GLTF") {
        load_gltf_scene(filename, scene, options);
    } else if (ext == "pbrt" || ext == "PBRT") {
        load_pbrt_scene(filename, scene, options);
    } else if (ext == "ybin" || ext == "YBIN") {
        load_ybin_scene(filename, scene, options);
    } else if (ext == "ply" || ext == "PLY") {
        load_ply_scene(filename, scene, options);
    } else {
        scene = {};
        throw io_error("unsupported scene format " + ext);
    }
}

// Save a scene
void save_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    auto ext = get_extension(filename);
    if (ext == "json" || ext == "JSON") {
        save_json_scene(filename, scene, options);
    } else if (ext == "obj" || ext == "OBJ") {
        save_obj_scene(filename, scene, options);
    } else if (ext == "gltf" || ext == "GLTF") {
        save_gltf_scene(filename, scene, options);
    } else if (ext == "pbrt" || ext == "PBRT") {
        save_pbrt_scene(filename, scene, options);
    } else if (ext == "ybin" || ext == "YBIN") {
        save_ybin_scene(filename, scene, options);
    } else {
        throw io_error("unsupported scene format " + ext);
    }
}

void load_scene_textures(yocto_scene& scene, const string& dirname,
    const load_scene_options& options) {
    if (options.skip_textures) return;

    // load images
    parallel_foreach(
        scene.textures,
        [&dirname](yocto_texture& texture) {
            if (texture.filename == "" || !empty(texture.hdr_image) ||
                !empty(texture.ldr_image))
                return;
            auto filename = normalize_path(dirname + texture.filename);
            if (is_hdr_filename(filename)) {
                load_image(filename, texture.hdr_image);
            } else {
                load_image(filename, texture.ldr_image);
            }
        },
        options.cancel_flag, options.run_serially);

    // load volumes
    parallel_foreach(
        scene.voltextures,
        [&dirname](yocto_voltexture& texture) {
            if (texture.filename == "" || !empty(texture.volume_data)) return;
            auto filename = normalize_path(dirname + texture.filename);
            load_volume(filename, texture.volume_data);
        },
        options.cancel_flag, options.run_serially);

    // assign opacity texture if needed
    if (options.assign_texture_opacity) {
        auto has_opacity = vector<int>(scene.textures.size());
        parallel_for((int)scene.textures.size(),
            [&scene, &has_opacity](int texture_id) {
                auto& texture           = scene.textures[texture_id];
                has_opacity[texture_id] = 0;
                for (auto& p : texture.hdr_image) {
                    if (p.w < 0.999f) {
                        has_opacity[texture_id] = 1;
                        break;
                    }
                }
                for (auto& p : texture.ldr_image) {
                    if (p.w < 255) {
                        has_opacity[texture_id] = 1;
                        break;
                    }
                }
            },
            options.cancel_flag, options.run_serially);
        for (auto& material : scene.materials) {
            if (material.diffuse_texture >= 0 && material.opacity_texture < 0 &&
                has_opacity[material.diffuse_texture])
                material.opacity_texture = material.diffuse_texture;
        }
    }
}

// helper to save textures
void save_scene_textures(const yocto_scene& scene, const string& dirname,
    const save_scene_options& options) {
    if (options.skip_textures) return;

    // save images
    parallel_foreach(
        scene.textures,
        [&dirname](const yocto_texture& texture) {
            if (empty(texture.hdr_image) && empty(texture.ldr_image)) return;
            auto filename = normalize_path(dirname + texture.filename);
            if (is_hdr_filename(filename)) {
                save_image(filename, texture.hdr_image);
            } else {
                save_image(filename, texture.ldr_image);
            }
        },
        options.cancel_flag, options.run_serially);

    // save volumes
    parallel_foreach(
        scene.voltextures,
        [&dirname](const yocto_voltexture& texture) {
            if (empty(texture.volume_data)) return;
            auto filename = normalize_path(dirname + texture.filename);
            save_volume(filename, texture.volume_data);
        },
        options.cancel_flag, options.run_serially);
}

// merge quads and triangles
void merge_triangles_and_quads(
    vector<vec3i>& triangles, vector<vec4i>& quads, bool force_triangles) {
    if (empty(quads)) return;
    if (force_triangles) {
        auto qtriangles = vector<vec3i>{};
        convert_quads_to_triangles(qtriangles, quads);
        triangles.insert(triangles.end(), qtriangles.begin(), qtriangles.end());
        quads = {};
    } else {
        auto tquads = vector<vec4i>{};
        convert_triangles_to_quads(tquads, triangles);
        quads.insert(quads.end(), tquads.begin(), tquads.end());
        triangles = {};
    }
}

// check if it is really face varying
bool is_face_varying(const vector<vec4i>& quads_positions,
    const vector<vec4i>& quads_normals, const vector<vec4i>& quads_texcoords) {
    if (empty(quads_positions)) return false;
    if (!empty(quads_normals)) {
        for (auto i = 0; i < quads_positions.size(); i++)
            if (quads_positions[i] != quads_normals[i]) return true;
    }
    if (!empty(quads_texcoords)) {
        for (auto i = 0; i < quads_positions.size(); i++)
            if (quads_positions[i] != quads_texcoords[i]) return true;
    }
    return false;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IO UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Encode in base64
string base64_encode(
    unsigned char const* bytes_to_encode, unsigned int in_len) {
    static const string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    string        ret;
    int           i = 0;
    int           j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                              ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                              ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++) ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                          ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                          ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++) ret += base64_chars[char_array_4[j]];

        while ((i++ < 3)) ret += '=';
    }

    return ret;
}

// Decode from base64
string base64_decode(string const& encoded_string) {
    static const string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    auto is_base64 = [](unsigned char c) -> bool {
        return (isalnum(c) || (c == '+') || (c == '/'));
    };

    int           in_len = (int)encoded_string.size();
    int           i      = 0;
    int           j      = 0;
    int           in_    = 0;
    unsigned char char_array_4[4], char_array_3[3];
    string        ret;

    while (in_len-- && (encoded_string[in_] != '=') &&
           is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) +
                              ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) +
                              ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++) ret += char_array_3[i];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) +
                          ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) +
                          ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }

    return ret;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// BUILTIN JSON FORMAT
// -----------------------------------------------------------------------------
namespace yocto {

// Dumps a json value
template <typename T>
json ref_to_json(int value, const vector<T>& refs) {
    return value >= 0 ? refs[value].name : ""s;
}
template <typename T>
int ref_from_json(const json& js, const vector<T>& refs) {
    auto value = -1;
    auto name  = js.get<string>();
    if (name == "") return -1;
    for (auto index = 0; index < refs.size(); index++) {
        if (refs[index].name == name) {
            value = index;
            break;
        }
    }
    if (value < 0) throw std::runtime_error("invalid object reference");
    return value;
}
template <typename T>
json refs_to_json(const vector<int>& values, const vector<T>& refs) {
    auto js = json::array_t{};
    for (auto value : values) {
        js.push_back({});
        js.back() = ref_to_json(value, refs);
    }
    return js;
}
template <typename T>
vector<int> refs_from_json(const json& js, const vector<T>& refs) {
    auto values = vector<int>{};
    for (auto& js_ : js) {
        values.push_back(-1);
        values.back() = ref_from_json(js_, refs);
    }
    return values;
}

// Procedural commands for cameras
void from_json_procedural(
    const json& js, yocto_camera& value, yocto_scene& scene) {
    if (js.count("from") || js.count("to")) {
        auto from            = js.value("from", zero3f);
        auto to              = js.value("to", zero3f);
        auto up              = js.value("up", vec3f{0, 1, 0});
        value.frame          = make_lookat_frame(from, to, up);
        value.focus_distance = length(from - to);
    }
}

// Serialize struct
void to_json(json& js, const yocto_camera& value, const yocto_scene& scene) {
    static const auto def = yocto_camera();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.frame != def.frame) js["frame"] = value.frame;
    if (value.orthographic != def.orthographic)
        js["orthographic"] = value.orthographic;
    if (value.film_width != def.film_width) js["film_width"] = value.film_width;
    if (value.film_height != def.film_height)
        js["film_height"] = value.film_height;
    if (value.focal_length != def.focal_length)
        js["focal_length"] = value.focal_length;
    if (value.focus_distance != def.focus_distance)
        js["focus_distance"] = value.focus_distance;
    if (value.lens_aperture != def.lens_aperture)
        js["lens_aperture"] = value.lens_aperture;
}
void from_json(const json& js, yocto_camera& value, yocto_scene& scene) {
    static const auto def = yocto_camera();
    value.name            = js.value("name", def.name);
    value.frame           = js.value("frame", def.frame);
    value.orthographic    = js.value("orthographic", def.orthographic);
    value.film_width      = js.value("film_width", def.film_width);
    value.film_height     = js.value("film_height", def.film_height);
    value.focal_length    = js.value("focal_length", def.focal_length);
    value.focus_distance  = js.value("focus_distance", def.focus_distance);
    value.lens_aperture   = js.value("lens_aperture", def.lens_aperture);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for textures
void from_json_procedural(
    const json& js, yocto_texture& value, yocto_scene& scene) {
    auto type = js.value("type", ""s);
    if (type == "")
        throw std::invalid_argument("unknown procedural type " + type);
    auto is_hdr = false;
    auto width  = js.value("width", 1024);
    auto height = js.value("height", 1024);
    if (type == "sky" && width < height * 2) width = height * 2;
    value.hdr_image.resize(width, height);
    if (type == "grid") {
        make_grid_image(value.hdr_image, js.value("tile", 8),
            js.value("c0", vec4f{0.2f, 0.2f, 0.2f, 1}),
            js.value("c1", vec4f{0.5f, 0.5f, 0.5f, 1}));
    } else if (type == "checker") {
        make_checker_image(value.hdr_image, js.value("tile", 8),
            js.value("c0", vec4f{0.2f, 0.2f, 0.2f, 1}),
            js.value("c1", vec4f{0.5f, 0.5f, 0.5f, 1}));
    } else if (type == "bump") {
        make_bumpdimple_image(value.hdr_image, js.value("tile", 8));
    } else if (type == "uvramp") {
        make_uvramp_image(value.hdr_image);
    } else if (type == "gammaramp") {
        make_gammaramp_image(value.hdr_image);
    } else if (type == "blackbodyramp") {
        make_blackbodyramp_image(value.hdr_image);
    } else if (type == "uvgrid") {
        make_uvgrid_image(value.hdr_image);
    } else if (type == "sky") {
        make_sunsky_image(value.hdr_image, js.value("sun_angle", pif / 4),
            js.value("turbidity", 3.0f), js.value("has_sun", false),
            js.value("sun_intensity", 1.0f), js.value("sun_temperature", 0.0f),
            js.value("ground_albedo", vec3f{0.7f, 0.7f, 0.7f}));
        is_hdr = true;
    } else if (type == "noise") {
        make_noise_image(
            value.hdr_image, js.value("scale", 1.0f), js.value("wrap", true));
    } else if (type == "fbm") {
        make_fbm_image(value.hdr_image, js.value("scale", 1.0f),
            js.value("lacunarity", 2.0f), js.value("gain", 0.5f),
            js.value("octaves", 6), js.value("wrap", true));
    } else if (type == "ridge") {
        make_ridge_image(value.hdr_image, js.value("scale", 1.0f),
            js.value("lacunarity", 2.0f), js.value("gain", 0.5f),
            js.value("offset", 1.0f), js.value("octaves", 6),
            js.value("wrap", true));
    } else if (type == "turbulence") {
        make_turbulence_image(value.hdr_image, js.value("scale", 1.0f),
            js.value("lacunarity", 2.0f), js.value("gain", 0.5f),
            js.value("octaves", 6), js.value("wrap", true));
    } else {
        throw std::invalid_argument("unknown procedural type " + type);
    }
    if (js.value("border", false)) {
        add_image_border(value.hdr_image, js.value("border_width", 2),
            js.value("border_color", vec4f{0, 0, 0, 1}));
    }
    if (js.value("bump_to_normal", false)) {
        auto buffer = value.hdr_image;
        bump_to_normal_map(
            value.hdr_image, buffer, js.value("bump_scale", 1.0f));
        value.ldr_as_linear = true;
    }
    if (!is_hdr) {
        if (!value.ldr_as_linear) {
            value.ldr_image = float_to_byte(linear_to_srgb(value.hdr_image));
        } else {
            value.ldr_image = float_to_byte(value.hdr_image);
        }
        value.hdr_image = {};
    }
    if (value.filename == "") {
        auto ext       = (is_hdr) ? string("hdr") : string("png");
        value.filename = "textures/" + value.name + "." + ext;
    }
}

// Serialize struct
void to_json(json& js, const yocto_texture& value, const yocto_scene& scene) {
    static const auto def = yocto_texture();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.filename != def.filename) js["filename"] = value.filename;
    if (value.clamp_to_edge != def.clamp_to_edge)
        js["clamp_to_edge"] = value.clamp_to_edge;
    if (value.height_scale != def.height_scale)
        js["height_scale"] = value.height_scale;
    if (value.no_interpolation != def.no_interpolation)
        js["no_interpolation"] = value.no_interpolation;
    if (value.ldr_as_linear != def.ldr_as_linear)
        js["ldr_as_linear"] = value.ldr_as_linear;
    if (value.filename == "") {
        if (value.hdr_image != def.hdr_image) js["hdr_image"] = value.hdr_image;
        if (value.ldr_image != def.ldr_image) js["ldr_image"] = value.ldr_image;
    }
}
void from_json(const json& js, yocto_texture& value, yocto_scene& scene) {
    static const auto def  = yocto_texture();
    value.name             = js.value("name", def.name);
    value.filename         = js.value("filename", def.filename);
    value.clamp_to_edge    = js.value("clamp_to_edge", def.clamp_to_edge);
    value.height_scale     = js.value("height_scale", def.height_scale);
    value.no_interpolation = js.value("no_interpolation", def.no_interpolation);
    value.ldr_as_linear    = js.value("ldr_as_linear", def.ldr_as_linear);
    value.hdr_image        = js.value("hdr_image", def.hdr_image);
    value.ldr_image        = js.value("ldr_image", def.ldr_image);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for textures
void from_json_procedural(
    const json& js, yocto_voltexture& value, yocto_scene& scene) {
    auto type = js.value("type", ""s);
    if (type == "")
        throw std::invalid_argument("unknown procedural type " + type);
    auto width  = js.value("width", 512);
    auto height = js.value("height", 512);
    auto depth  = js.value("depth", 512);
    value.volume_data.resize(width, height, depth);
    if (type == "test_volume") {
        make_test_volume(value.volume_data, js.value("scale", 10.0f),
            js.value("exponent", 6.0f));
    } else {
        throw std::invalid_argument("unknown procedural type " + type);
    }
    if (value.filename == "") {
        auto ext       = string("vol");
        value.filename = "textures/" + value.name + "." + ext;
    }
}

// Serialize struct
void to_json(
    json& js, const yocto_voltexture& value, const yocto_scene& scene) {
    static const auto def = yocto_voltexture();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.filename != def.filename) js["filename"] = value.filename;
    if (value.clamp_to_edge != def.clamp_to_edge)
        js["clamp_to_edge"] = value.clamp_to_edge;
    if (value.no_interpolation != def.no_interpolation)
        js["no_interpolation"] = value.no_interpolation;
    if (value.filename == "") {
        if (value.volume_data != def.volume_data)
            js["volume_data"] = value.volume_data;
    }
}
void from_json(const json& js, yocto_voltexture& value, yocto_scene& scene) {
    static const auto def  = yocto_voltexture();
    value.name             = js.value("name", def.name);
    value.filename         = js.value("filename", def.filename);
    value.clamp_to_edge    = js.value("clamp_to_edge", def.clamp_to_edge);
    value.no_interpolation = js.value("no_interpolation", def.no_interpolation);
    value.volume_data      = js.value("volume_data", def.volume_data);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for materials
void from_json_procedural(
    const json& js, yocto_material& value, yocto_scene& scene) {}

// Serialize struct
void to_json(json& js, const yocto_material& value, const yocto_scene& scene) {
    static const auto def = yocto_material();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.base_metallic != def.base_metallic)
        js["base_metallic"] = value.base_metallic;
    if (value.gltf_textures != def.gltf_textures)
        js["gltf_textures"] = value.gltf_textures;
    if (value.emission != def.emission) js["emission"] = value.emission;
    if (value.diffuse != def.diffuse) js["diffuse"] = value.diffuse;
    if (value.specular != def.specular) js["specular"] = value.specular;
    if (value.transmission != def.transmission)
        js["transmission"] = value.transmission;
    if (value.roughness != def.roughness) js["roughness"] = value.roughness;
    if (value.opacity != def.opacity) js["opacity"] = value.opacity;
    if (value.fresnel != def.fresnel) js["fresnel"] = value.fresnel;
    if (value.refract != def.refract) js["refract"] = value.refract;
    if (value.emission_texture != def.emission_texture)
        js["emission_texture"] = ref_to_json(
            value.emission_texture, scene.textures);
    if (value.diffuse_texture != def.diffuse_texture)
        js["diffuse_texture"] = ref_to_json(
            value.diffuse_texture, scene.textures);
    if (value.specular_texture != def.specular_texture)
        js["specular_texture"] = ref_to_json(
            value.specular_texture, scene.textures);
    if (value.transmission_texture != def.transmission_texture)
        js["transmission_texture"] = ref_to_json(
            value.transmission_texture, scene.textures);
    if (value.roughness_texture != def.roughness_texture)
        js["roughness_texture"] = ref_to_json(
            value.roughness_texture, scene.textures);
    if (value.opacity_texture != def.opacity_texture)
        js["opacity_texture"] = ref_to_json(
            value.opacity_texture, scene.textures);
    if (value.occlusion_texture != def.occlusion_texture)
        js["occlusion_texture"] = ref_to_json(
            value.occlusion_texture, scene.textures);
    if (value.bump_texture != def.bump_texture)
        js["bump_texture"] = ref_to_json(value.bump_texture, scene.textures);
    if (value.displacement_texture != def.displacement_texture)
        js["displacement_texture"] = ref_to_json(
            value.displacement_texture, scene.textures);
    if (value.normal_texture != def.normal_texture)
        js["normal_texture"] = ref_to_json(
            value.normal_texture, scene.textures);
    if (value.volume_density_texture != def.volume_density_texture)
        js["volume_density_texture"] = ref_to_json(
            value.volume_density_texture, scene.voltextures);
}
void from_json(const json& js, yocto_material& value, yocto_scene& scene) {
    static const auto def  = yocto_material();
    value.name             = js.value("name", def.name);
    value.base_metallic    = js.value("base_metallic", def.base_metallic);
    value.gltf_textures    = js.value("gltf_textures", def.gltf_textures);
    value.emission         = js.value("emission", def.emission);
    value.diffuse          = js.value("diffuse", def.diffuse);
    value.specular         = js.value("specular", def.specular);
    value.transmission     = js.value("transmission", def.transmission);
    value.roughness        = js.value("roughness", def.roughness);
    value.opacity          = js.value("opacity", def.opacity);
    value.fresnel          = js.value("fresnel", def.fresnel);
    value.refract          = js.value("refract", def.refract);
    value.emission_texture = ref_from_json(
        js.value("emission_texture", ""s), scene.textures);
    value.diffuse_texture = ref_from_json(
        js.value("diffuse_texture", ""s), scene.textures);
    value.specular_texture = ref_from_json(
        js.value("specular_texture", ""s), scene.textures);
    value.transmission_texture = ref_from_json(
        js.value("transmission_texture", ""s), scene.textures);
    value.roughness_texture = ref_from_json(
        js.value("roughness_texture", ""s), scene.textures);
    value.opacity_texture = ref_from_json(
        js.value("opacity_texture", ""s), scene.textures);
    value.occlusion_texture = ref_from_json(
        js.value("occlusion_texture", ""s), scene.textures);
    value.bump_texture = ref_from_json(
        js.value("bump_texture", ""s), scene.textures);
    value.displacement_texture = ref_from_json(
        js.value("displacement_texture", ""s), scene.textures);
    value.normal_texture = ref_from_json(
        js.value("normal_texture", ""s), scene.textures);
    value.volume_density_texture = ref_from_json(
        js.value("volume_density_texture", ""s), scene.voltextures);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for materials
void from_json_procedural(
    const json& js, yocto_shape& value, yocto_scene& scene) {
    auto type = js.value("type", ""s);
    if (type == "")
        throw std::invalid_argument("unknown procedural type " + type);
    value.points        = {};
    value.lines         = {};
    value.triangles     = {};
    value.quads         = {};
    value.positions     = {};
    value.normals       = {};
    value.texturecoords = {};
    value.radius        = {};
    if (type == "quad") {
        make_quad_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{2, 2}), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "quady") {
        make_quad_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{2, 2}), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "quad_stack") {
        make_quad_stack_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "cube") {
        make_cube_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else if (type == "cube_rounded") {
        make_cube_rounded_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec3i{32, 32, 32}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}), js.value("radius", 0.3f));
    } else if (type == "uvsphere") {
        make_uvsphere_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{64, 32}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "sphere") {
        make_sphere_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f));
    } else if (type == "uvsphere_flipcap") {
        make_uvsphere_flipcap_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{64, 32}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}),
            js.value("zflip", vec2f{-0.75f, +0.75f}));
    } else if (type == "uvdisk") {
        make_uvdisk_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{32, 16}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "disk") {
        make_disk_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f));
    } else if (type == "disk_bulged") {
        make_disk_bulged_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f), js.value("height", 0.25f));
    } else if (type == "quad_bulged") {
        make_quad_bulged_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f), js.value("height", 0.25f));
    } else if (type == "uvcylinder") {
        make_uvcylinder_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec3i{64, 32, 16}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else if (type == "uvcylinder_rounded") {
        make_uvcylinder_rounded_shape(value.quads, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec3i{64, 32, 16}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec3f{1, 1, 1}), js.value("radius", 0.15f));
    } else if (type == "sphere_geodesic") {
        make_geodesic_sphere_shape(value.triangles, value.positions,
            value.normals, js.value("tesselation", 4), js.value("size", 2.0f));
    } else if (type == "floor") {
        make_floor_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{40, 40}), js.value("uvsize", vec2f{20, 20}));
    } else if (type == "floor_bent") {
        make_floor_bent_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{1, 40}),
            js.value("size", vec2f{40, 40}), js.value("uvsize", vec2f{20, 20}),
            js.value("radius", 10.0f));
    } else if (type == "matball") {
        make_sphere_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f));
    } else if (type == "hairball") {
        auto base_quads         = vector<vec4i>{};
        auto base_positions     = vector<vec3f>{};
        auto base_normals       = vector<vec3f>{};
        auto base_texturecoords = vector<vec2f>{};
        make_sphere_shape(base_quads, base_positions, base_normals,
            base_texturecoords, 32, js.value("size", 2.0f) * 0.8f, 1);
        make_hair_shape(value.lines, value.positions, value.normals,
            value.texturecoords, value.radius,
            js.value("steps", vec2i{4, 65536}), {}, base_quads, base_positions,
            base_normals, base_texturecoords,
            js.value("length", vec2f{0.2f, 0.2f}),
            js.value("radius", vec2f{0.001f, 0.001f}),
            js.value("noise", vec2f{0, 0}), js.value("clump", vec2f{0, 0}));
    } else if (type == "hairball_interior") {
        make_sphere_shape(value.quads, value.positions, value.normals,
            value.texturecoords, 32, js.value("size", 2.0f) * 0.8f, 1);
    } else if (type == "suzanne") {
        make_suzanne_shape(
            value.quads, value.positions, js.value("size", 2.0f));
    } else if (type == "cube_posonly") {
        auto ignore1 = vector<vec4i>{};
        auto ignore2 = vector<vec4i>{};
        auto ignore3 = vector<vec3f>{};
        auto ignore4 = vector<vec2f>{};
        make_cube_fvshape(value.quads, ignore1, ignore2, value.positions,
            ignore3, ignore4, js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else {
        throw std::invalid_argument("unknown procedural type " + type);
    }
    if (!value.quads.empty() && js.value("shell_thickness", 0.0f) > 0) {
        make_shell_shape(value.quads, value.positions, value.normals,
            value.texturecoords, js.value("shell_thickness", 0.0f));
    }
    if (!value.quads.empty() && js.value("as_triangles", false)) {
        convert_quads_to_triangles(value.triangles, value.quads);
        value.quads = {};
    }
    if (js.value("flipyz", false)) {
        for (auto& p : value.positions) p = {p.x, p.z, p.y};
        for (auto& n : value.normals) n = {n.x, n.z, n.y};
    }
}

// Serialize struct
void to_json(json& js, const yocto_shape& value, const yocto_scene& scene) {
    static const auto def = yocto_shape();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.filename != def.filename) js["filename"] = value.filename;
    if (value.material != def.material)
        js["material"] = ref_to_json(value.material, scene.materials);
    if (value.subdivision_level != def.subdivision_level)
        js["subdivision_level"] = value.subdivision_level;
    if (value.catmull_clark != def.catmull_clark)
        js["catmull_clark"] = value.catmull_clark;
    if (value.compute_vertex_normals != def.compute_vertex_normals)
        js["compute_vertex_normals"] = value.compute_vertex_normals;
    if (value.filename == "") {
        if (value.points != def.points) js["points"] = value.points;
        if (value.lines != def.lines) js["lines"] = value.lines;
        if (value.triangles != def.triangles) js["triangles"] = value.triangles;
        if (value.quads != def.quads) js["quads"] = value.quads;
        if (value.positions != def.positions) js["positions"] = value.positions;
        if (value.normals != def.normals) js["normals"] = value.normals;
        if (value.texturecoords != def.texturecoords)
            js["texturecoords"] = value.texturecoords;
        if (value.colors != def.colors) js["colors"] = value.colors;
        if (value.radius != def.radius) js["radius"] = value.radius;
        if (value.tangentspaces != def.tangentspaces)
            js["tangentspaces"] = value.tangentspaces;
    }
}
void from_json(const json& js, yocto_shape& value, yocto_scene& scene) {
    static const auto def = yocto_shape();
    value.name            = js.value("name", def.name);
    value.filename        = js.value("filename", def.filename);
    value.material = ref_from_json(js.value("material", ""s), scene.materials);
    value.subdivision_level = js.value(
        "subdivision_level", def.subdivision_level);
    value.catmull_clark          = js.value("catmull_clark", def.catmull_clark);
    value.compute_vertex_normals = js.value(
        "compute_vertex_normals", def.compute_vertex_normals);
    value.points        = js.value("points", def.points);
    value.lines         = js.value("lines", def.lines);
    value.triangles     = js.value("triangles", def.triangles);
    value.quads         = js.value("quads", def.quads);
    value.positions     = js.value("positions", def.positions);
    value.normals       = js.value("normals", def.normals);
    value.texturecoords = js.value("texturecoords", def.texturecoords);
    value.colors        = js.value("colors", def.colors);
    value.radius        = js.value("radius", def.radius);
    value.tangentspaces = js.value("tangentspaces", def.tangentspaces);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for materials
void from_json_procedural(
    const json& js, yocto_surface& value, yocto_scene& scene) {
    auto type = js.value("type", ""s);
    if (type == "")
        throw std::invalid_argument("unknown procedural type " + type);
    value.quads_positions     = {};
    value.quads_normals       = {};
    value.quads_texturecoords = {};
    value.quads_materials     = {};
    value.positions           = {};
    value.normals             = {};
    value.texturecoords       = {};
    if (type == "quad") {
        make_quad_shape(value.quads_positions, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{2, 2}), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "quad_stack") {
        make_quad_stack_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec3i{1, 1, 1}), js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec2f{1, 1}));
    } else if (type == "cube") {
        make_cube_fvshape(value.quads_positions, value.quads_normals,
            value.quads_texturecoords, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else if (type == "cube_rounded") {
        make_cube_rounded_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec3i{32, 32, 32}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}), js.value("radius", 0.3f));
    } else if (type == "uvsphere") {
        make_uvsphere_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec2i{64, 32}), js.value("size", 2.0f),
            js.value("uvsize", vec2f{1, 1}));
    } else if (type == "sphere") {
        make_sphere_fvshape(value.quads_positions, value.quads_normals,
            value.quads_texturecoords, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f));
    } else if (type == "uvsphere_flipcap") {
        make_uvsphere_flipcap_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec2i{64, 32}), js.value("size", 2.0f),
            js.value("uvsize", vec2f{1, 1}),
            js.value("zflip", vec2f{-0.75f, +0.75f}));
    } else if (type == "uvdisk") {
        make_uvdisk_shape(value.quads_positions, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{32, 16}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}));
    } else if (type == "disk") {
        make_disk_shape(value.quads_positions, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f));
    } else if (type == "disk_bulged") {
        make_disk_bulged_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords, js.value("steps", 32),
            js.value("size", 2.0f), js.value("uvsize", 1.0f),
            js.value("height", 0.25f));
    } else if (type == "quad_bulged") {
        make_quad_bulged_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords, js.value("steps", 32),
            js.value("size", 2.0f), js.value("uvsize", 1.0f),
            js.value("height", 0.25f));
    } else if (type == "uvcylinder") {
        make_uvcylinder_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec3i{64, 32, 16}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else if (type == "uvcylinder_rounded") {
        make_uvcylinder_rounded_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords,
            js.value("steps", vec3i{64, 32, 16}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec3f{1, 1, 1}), js.value("radius", 0.15f));
    } else if (type == "floor") {
        make_floor_shape(value.quads_positions, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{40, 40}), js.value("uvsize", vec2f{20, 20}));
    } else if (type == "floor_bent") {
        make_floor_bent_shape(value.quads_positions, value.positions,
            value.normals, value.texturecoords, js.value("steps", vec2i{1, 40}),
            js.value("size", vec2f{40, 40}), js.value("uvsize", vec2f{20, 20}),
            js.value("radius", 10.0f));
    } else if (type == "matball") {
        make_sphere_shape(value.quads_positions, value.positions, value.normals,
            value.texturecoords, js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f));
    } else if (type == "hairball_interior") {
        make_sphere_shape(value.quads_positions, value.positions, value.normals,
            value.texturecoords, 32, js.value("size", 2.0f) * 0.8f, 1);
    } else if (type == "suzanne") {
        make_suzanne_shape(
            value.quads_positions, value.positions, js.value("size", 2.0f));
    } else if (type == "cube_multiplematerials") {
        make_cube_fvshape(value.quads_positions, value.quads_normals,
            value.quads_texturecoords, value.positions, value.normals,
            value.texturecoords, js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
        for (auto& q : value.quads_normals) {
            auto n  = value.normals[q.x];
            auto nd = vec3f{
                dot(n, {1, 0, 0}), dot(n, {0, 1, 0}), dot(n, {0, 0, 1})};
            if (nd.z > 0.5f)
                value.quads_materials.push_back(0);
            else if (nd.z < -0.5f)
                value.quads_materials.push_back(1);
            else if (nd.x > 0.5f)
                value.quads_materials.push_back(2);
            else if (nd.x < -0.5f)
                value.quads_materials.push_back(3);
            else if (nd.y > 0.5f)
                value.quads_materials.push_back(4);
            else if (nd.y < -0.5f)
                value.quads_materials.push_back(5);
        }
    } else if (type == "cube_posonly") {
        auto ignore1 = vector<vec4i>{};
        auto ignore2 = vector<vec4i>{};
        make_cube_fvshape(value.quads_positions, ignore1, ignore2,
            value.positions, value.normals, value.texturecoords,
            js.value("steps", vec3i{1, 1, 1}), js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else {
        throw std::invalid_argument("unknown procedural type " + type);
    }
    if (empty(value.quads_normals) && !empty(value.normals))
        value.quads_normals = value.quads_positions;
    if (empty(value.quads_texturecoords) && !empty(value.texturecoords))
        value.quads_texturecoords = value.quads_positions;
    if (js.value("flipyz", false)) {
        for (auto& p : value.positions) p = {p.x, p.z, p.y};
        for (auto& n : value.normals) n = {n.x, n.z, n.y};
    }
}

// Serialize struct
void to_json(json& js, const yocto_surface& value, const yocto_scene& scene) {
    static const auto def = yocto_surface();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.filename != def.filename) js["filename"] = value.filename;
    if (value.materials != def.materials)
        js["materials"] = refs_to_json(value.materials, scene.materials);
    if (value.subdivision_level != def.subdivision_level)
        js["subdivision_level"] = value.subdivision_level;
    if (value.catmull_clark != def.catmull_clark)
        js["catmull_clark"] = value.catmull_clark;
    if (value.compute_vertex_normals != def.compute_vertex_normals)
        js["compute_vertex_normals"] = value.compute_vertex_normals;
    if (value.filename == "") {
        if (value.quads_positions != def.quads_positions)
            js["quads_positions"] = value.quads_positions;
        if (value.quads_normals != def.quads_normals)
            js["quads_normals"] = value.quads_normals;
        if (value.quads_texturecoords != def.quads_texturecoords)
            js["quads_texturecoords"] = value.quads_texturecoords;
        if (value.quads_materials != def.quads_materials)
            js["quads_materials"] = value.quads_materials;
        if (value.positions != def.positions) js["positions"] = value.positions;
        if (value.normals != def.normals) js["normals"] = value.normals;
        if (value.texturecoords != def.texturecoords)
            js["texturecoords"] = value.texturecoords;
    }
}
void from_json(const json& js, yocto_surface& value, yocto_scene& scene) {
    static const auto def = yocto_surface();
    value.name            = js.value("name", def.name);
    value.filename        = js.value("filename", def.filename);
    value.materials       = refs_from_json(
        js.value("materials", vector<string>{}), scene.materials);
    value.subdivision_level = js.value(
        "subdivision_level", def.subdivision_level);
    value.catmull_clark          = js.value("catmull_clark", def.catmull_clark);
    value.compute_vertex_normals = js.value(
        "compute_vertex_normals", def.compute_vertex_normals);
    value.quads_positions = js.value("quads_positions", def.quads_positions);
    value.quads_normals   = js.value("quads_normals", def.quads_normals);
    value.quads_texturecoords = js.value(
        "quads_texturecoords", def.quads_texturecoords);
    value.quads_materials = js.value("quads_materials", def.quads_materials);
    value.positions       = js.value("positions", def.positions);
    value.normals         = js.value("normals", def.normals);
    value.texturecoords   = js.value("texturecoords", def.texturecoords);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for instances
void from_json_procedural(
    const json& js, yocto_instance& value, yocto_scene& scene) {
    if (js.count("from")) {
        auto from   = js.value("from", zero3f);
        auto to     = js.value("to", zero3f);
        auto up     = js.value("up", vec3f{0, 1, 0});
        value.frame = make_lookat_frame(from, to, up, true);
    }
    if (js.count("translation") || js.count("rotation") || js.count("scale")) {
        auto translation = js.value("translation", zero3f);
        auto rotation    = js.value("rotation", zero4f);
        auto scaling     = js.value("scale", vec3f{1, 1, 1});
        value.frame      = make_translation_frame(translation) *
                      make_scaling_frame(scaling) *
                      make_rotation_frame(xyz(rotation), rotation.w);
    }
}

// Serialize struct
void to_json(json& js, const yocto_instance& value, const yocto_scene& scene) {
    static const auto def = yocto_instance();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.frame != def.frame) js["frame"] = value.frame;
    if (value.shape != def.shape)
        js["shape"] = ref_to_json(value.shape, scene.shapes);
    if (value.surface != def.surface)
        js["surface"] = ref_to_json(value.surface, scene.surfaces);
}
void from_json(const json& js, yocto_instance& value, yocto_scene& scene) {
    static const auto def = yocto_instance();
    value.name            = js.value("name", def.name);
    value.frame           = js.value("frame", def.frame);
    value.shape           = ref_from_json(js.value("shape", ""s), scene.shapes);
    value.surface = ref_from_json(js.value("surface", ""s), scene.surfaces);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for materials
void from_json_procedural(
    const json& js, yocto_environment& value, yocto_scene& scene) {
    if (js.count("rotation")) {
        auto rotation = js.value("rotation", zero4f);
        value.frame   = make_rotation_frame(xyz(rotation), rotation.w);
    }
}

// Serialize struct
void to_json(
    json& js, const yocto_environment& value, const yocto_scene& scene) {
    static const auto def = yocto_environment();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.frame != def.frame) js["frame"] = value.frame;
    if (value.emission != def.emission) js["emission"] = value.emission;
    if (value.emission_texture != def.emission_texture)
        js["emission_texture"] = ref_to_json(
            value.emission_texture, scene.textures);
}
void from_json(const json& js, yocto_environment& value, yocto_scene& scene) {
    static const auto def  = yocto_environment();
    value.name             = js.value("name", def.name);
    value.frame            = js.value("frame", def.frame);
    value.emission         = js.value("emission", def.emission);
    value.emission_texture = ref_from_json(
        js.value("emission_texture", ""s), scene.textures);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for nodes
void from_json_procedural(
    const json& js, yocto_scene_node& value, yocto_scene& scene) {
    if (js.count("from")) {
        auto from   = js.value("from", zero3f);
        auto to     = js.value("to", zero3f);
        auto up     = js.value("up", vec3f{0, 1, 0});
        value.local = make_lookat_frame(from, to, up, true);
    }
}

// Serialize struct
void to_json(
    json& js, const yocto_scene_node& value, const yocto_scene& scene) {
    static const auto def = yocto_scene_node();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.local != def.local) js["local"] = value.local;
    if (value.translation != def.translation)
        js["translation"] = value.translation;
    if (value.rotation != def.rotation) js["rotation"] = value.rotation;
    if (value.scale != def.scale) js["scale"] = value.scale;
    if (value.weights != def.weights) js["weights"] = value.weights;
    if (value.parent != def.parent)
        js["parent"] = ref_to_json(value.parent, scene.nodes);
    if (value.camera != def.camera)
        js["camera"] = ref_to_json(value.camera, scene.cameras);
    if (value.instance != def.instance)
        js["instance"] = ref_to_json(value.instance, scene.instances);
    if (value.environment != def.environment)
        js["environment"] = ref_to_json(value.environment, scene.environments);
}
void from_json(const json& js, yocto_scene_node& value, yocto_scene& scene) {
    static const auto def = yocto_scene_node();
    value.name            = js.value("name", def.name);
    value.local           = js.value("local", def.local);
    value.translation     = js.value("translation", def.translation);
    value.rotation        = js.value("rotation", def.rotation);
    value.scale           = js.value("scale", def.scale);
    value.weights         = js.value("weights", def.weights);
    value.parent          = ref_from_json(js.value("parent", ""s), scene.nodes);
    value.camera   = ref_from_json(js.value("camera", ""s), scene.cameras);
    value.instance = ref_from_json(js.value("instance", ""s), scene.instances);
    value.environment = ref_from_json(
        js.value("environment", ""s), scene.environments);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Serialize enum
void to_json(json& js, const yocto_interpolation_type& value) {
    static auto names = unordered_map<int, string>{
        {(int)yocto_interpolation_type::linear, "linear"},
        {(int)yocto_interpolation_type::step, "step"},
        {(int)yocto_interpolation_type::bezier, "bezier"},
    };
    js = names.at((int)value);
}
void from_json(const json& js, yocto_interpolation_type& value) {
    static auto names = unordered_map<string, int>{
        {"linear", (int)yocto_interpolation_type::linear},
        {"step", (int)yocto_interpolation_type::step},
        {"bezier", (int)yocto_interpolation_type::bezier},
    };
    value = (yocto_interpolation_type)names.at(js.get<string>());
}

// Procedural commands for animations
void from_json_procedural(
    const json& js, yocto_animation& value, yocto_scene& scene) {
    if (js.count("make_rotation_axisangle")) {
        for (auto& j : js.at("make_rotation_axisangle")) {
            value.rotation_keyframes.push_back(
                make_rotation_quat(j.get<vec4f>()));
        }
    }
}

// Serialize struct
void to_json(json& js, const yocto_animation& value, const yocto_scene& scene) {
    static const auto def = yocto_animation();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (value.filename != def.filename) js["filename"] = value.filename;
    if (value.animation_group != def.animation_group)
        js["animation_group"] = value.animation_group;
    if (value.interpolation_type != def.interpolation_type)
        js["interpolation_type"] = value.interpolation_type;
    if (value.filename == "") {
        if (value.keyframes_times != def.keyframes_times)
            js["keyframes_times"] = value.keyframes_times;
        if (value.translation_keyframes != def.translation_keyframes)
            js["translation_keyframes"] = value.translation_keyframes;
        if (value.rotation_keyframes != def.rotation_keyframes)
            js["rotation_keyframes"] = value.rotation_keyframes;
        if (value.scale_keyframes != def.scale_keyframes)
            js["scale_keyframes"] = value.scale_keyframes;
    }
    if (value.node_targets != def.node_targets)
        js["node_targets"] = refs_to_json(value.node_targets, scene.nodes);
}
void from_json(const json& js, yocto_animation& value, yocto_scene& scene) {
    static const auto def    = yocto_animation();
    value.name               = js.value("name", def.name);
    value.filename           = js.value("filename", def.filename);
    value.animation_group    = js.value("animation_group", def.animation_group);
    value.interpolation_type = js.value(
        "interpolation_type", def.interpolation_type);
    value.keyframes_times = js.value("keyframes_times", def.keyframes_times);
    value.translation_keyframes = js.value(
        "translation_keyframes", def.translation_keyframes);
    value.rotation_keyframes = js.value(
        "rotation_keyframes", def.rotation_keyframes);
    value.scale_keyframes = js.value("scale_keyframes", def.scale_keyframes);
    value.node_targets    = refs_from_json(
        js.value("node_targets", vector<string>{}), scene.nodes);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Procedural commands for scenes
void from_json_procedural(
    const json& js, yocto_scene& value, yocto_scene& scene) {
    if (js.count("random_instances")) {
        auto& jjs          = js.at("random_instances");
        auto  num          = jjs.value("num", 100);
        auto  seed         = jjs.value("seed", 13);
        auto  shape_offset = (int)scene.shapes.size();
        auto  num_shapes   = 0;
        auto  base         = yocto_shape();
        from_json((json&)jjs.at("base"), base, scene);
        for (auto& j : jjs.at("shapes")) {
            value.shapes.push_back({});
            from_json((json&)j, value.shapes.back(), scene);
            num_shapes++;
        }

        auto pos      = vector<vec3f>{};
        auto norm     = vector<vec3f>{};
        auto texcoord = vector<vec2f>{};
        sample_triangles_points(pos, norm, texcoord, base.triangles,
            base.positions, base.normals, base.texturecoords, num, seed);

        auto rng = make_rng(seed, 17);
        for (auto i = 0; i < num; i++) {
            auto shape = get_random_int(rng, num_shapes - 1);
            value.instances.push_back({});
            value.instances.back().name  = "random_" + std::to_string(i);
            value.instances.back().frame = make_translation_frame(pos[i]);
            value.instances.back().shape = shape + shape_offset;
        }
    }
}

// serialize array of structs
template <typename T>
void to_json(json& js, const vector<T>& values, const yocto_scene& scene) {
    js = json::array_t{};
    for (auto& value : values) {
        js.push_back({});
        to_json(js.back(), value, scene);
    }
}
template <typename T>
void from_json(const json& js, vector<T>& values, yocto_scene& scene) {
    values.clear();
    for (auto& js_ : js) {
        values.push_back({});
        from_json(js_, values.back(), scene);
    }
}

// Serialize struct
void to_json(json& js, const yocto_scene& value, const yocto_scene& scene) {
    static const auto def = yocto_scene();
    js                    = json::object_t{};
    if (value.name != def.name) js["name"] = value.name;
    if (!value.cameras.empty()) to_json(js["cameras"], value.cameras, scene);
    if (!value.textures.empty()) to_json(js["textures"], value.textures, scene);
    if (!value.voltextures.empty())
        to_json(js["voltextures"], value.voltextures, scene);
    if (!value.materials.empty())
        to_json(js["materials"], value.materials, scene);
    if (!value.shapes.empty()) to_json(js["shapes"], value.shapes, scene);
    if (!value.surfaces.empty()) to_json(js["surfaces"], value.surfaces, scene);
    if (!value.instances.empty())
        to_json(js["instances"], value.instances, scene);
    if (!value.environments.empty())
        to_json(js["environments"], value.environments, scene);
    if (!value.nodes.empty()) to_json(js["nodes"], value.nodes, scene);
    if (!value.animations.empty())
        to_json(js["animations"], value.animations, scene);
}
void from_json(const json& js, yocto_scene& value, yocto_scene& scene) {
    static const auto def = yocto_scene();
    value.name            = js.value("name", def.name);
    if (js.count("cameras")) from_json(js.at("cameras"), value.cameras, scene);
    if (js.count("textures"))
        from_json(js.at("textures"), value.textures, scene);
    if (js.count("voltextures"))
        from_json(js.at("voltextures"), value.voltextures, scene);
    if (js.count("materials"))
        from_json(js.at("materials"), value.materials, scene);
    if (js.count("shapes")) from_json(js.at("shapes"), value.shapes, scene);
    if (js.count("surfaces"))
        from_json(js.at("surfaces"), value.surfaces, scene);
    if (js.count("instances"))
        from_json(js.at("instances"), value.instances, scene);
    if (js.count("environments"))
        from_json(js.at("environments"), value.environments, scene);
    if (js.count("nodes")) from_json(js.at("nodes"), value.nodes, scene);
    if (js.count("animations"))
        from_json(js.at("animations"), value.animations, scene);
    if (js.count("!!proc")) from_json_procedural(js.at("!!proc"), value, scene);
}

// Load json meshes
void load_json_meshes(yocto_scene& scene, const string& dirname,
    const load_scene_options& options) {
    if (options.skip_meshes) return;

    // load shapes
    parallel_foreach(
        scene.shapes,
        [&dirname](yocto_shape& shape) {
            if (shape.filename == "" || !empty(shape.positions)) return;
            auto filename = normalize_path(dirname + shape.filename);
            load_mesh(filename, shape.points, shape.lines, shape.triangles,
                shape.quads, shape.positions, shape.normals,
                shape.texturecoords, shape.colors, shape.radius, false);
        },
        options.cancel_flag, options.run_serially);

    // load surfaces
    parallel_foreach(
        scene.surfaces,
        [&dirname](yocto_surface& surface) {
            if (surface.filename == "" || !empty(surface.positions)) return;
            auto filename = normalize_path(dirname + surface.filename);
            load_facevarying_mesh(filename, surface.quads_positions,
                surface.quads_normals, surface.quads_texturecoords,
                surface.positions, surface.normals, surface.texturecoords,
                surface.quads_materials);
        },
        options.cancel_flag, options.run_serially);
}

// Load a scene in the builtin JSON format.
void load_json_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    // initialize
    scene = {};

    // wrap to rethrow better error
    try {
        // load jsonz
        auto js = json();
        load_json(filename, js);

        // deserialize json
        from_json(js, scene, scene);

        // load meshes and textures
        auto dirname = get_dirname(filename);
        load_json_meshes(scene, dirname, options);
        load_scene_textures(scene, dirname, options);
    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }

    // fix scene
    if (scene.name == "") scene.name = get_filename(filename);
    add_missing_cameras(scene);
    add_missing_materials(scene);
    add_missing_names(scene);
    update_transforms(scene);
}

// Save json meshes
void save_json_meshes(const yocto_scene& scene, const string& dirname,
    const save_scene_options& options) {
    if (options.skip_meshes) return;

    // save shapes
    parallel_foreach(
        scene.shapes,
        [&dirname](const yocto_shape& shape) {
            if (shape.filename == "") return;
            auto filename = normalize_path(dirname + shape.filename);
            save_mesh(filename, shape.points, shape.lines, shape.triangles,
                shape.quads, shape.positions, shape.normals,
                shape.texturecoords, shape.colors, shape.radius);
        },
        options.cancel_flag, options.run_serially);

    // save surfaces
    parallel_foreach(
        scene.surfaces,
        [&dirname](const yocto_surface& surface) {
            if (surface.filename == "") return;
            auto filename = normalize_path(dirname + surface.filename);
            save_facevarying_mesh(filename, surface.quads_positions,
                surface.quads_normals, surface.quads_texturecoords,
                surface.positions, surface.normals, surface.texturecoords,
                surface.quads_materials);
        },
        options.cancel_flag, options.run_serially);
}

// Save a scene in the builtin JSON format.
void save_json_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    try {
        // save json
        auto js               = json::object();
        js["asset"]           = json::object();
        js["asset"]["format"] = "Yocto/Scene";
        js["asset"]["generator"] =
            "Yocto/GL - https://github.com/xelatihy/yocto-gl";
        to_json(js, scene, scene);
        save_json(filename, js);

        // save meshes and textures
        auto dirname = get_dirname(filename);
        save_json_meshes(scene, dirname, options);
        save_scene_textures(scene, dirname, options);
    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// OBJ CONVERSION
// -----------------------------------------------------------------------------
namespace yocto {

inline bool operator==(const obj_vertex& a, const obj_vertex& b) {
    return a.position == b.position && a.texturecoord == b.texturecoord &&
           a.normal == b.normal;
}

struct obj_vertex_hash {
    size_t operator()(const obj_vertex& v) const {
        auto vh = std::hash<int>();
        auto h  = (size_t)0;
        for (auto i = 0; i < 3; i++)
            h ^= vh((&v.position)[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

inline void parse_value(char*& str, obj_vertex& value) {
    value = obj_vertex{0, 0, 0};
    parse_value(str, value.position);
    if (*str == '/') {
        str++;
        if (*str == '/') {
            str++;
            parse_value(str, value.normal);
        } else {
            parse_value(str, value.texturecoord);
            if (*str == '/') {
                str++;
                parse_value(str, value.normal);
            }
        }
    }
}

// Input for OBJ textures
inline void parse_value(char*& str, obj_texture_info& info) {
    // initialize
    info = obj_texture_info();

    // get tokens
    auto tokens = vector<string>();
    while (*str == ' ') str++;
    while (*str) {
        auto token = ""s;
        parse_value(str, token);
        tokens.push_back(token);
        while (*str == ' ') str++;
    }
    if (empty(tokens)) throw io_error("cannot parse value");

    // texture name
    info.path = normalize_path(tokens.back());

    // texture options
    auto last = string();
    for (auto i = 0; i < tokens.size() - 1; i++) {
        if (tokens[i] == "-bm") info.scale = atof(tokens[i + 1].c_str());
        if (tokens[i] == "-clamp") info.clamp = true;
    }
}

// Load obj materials
void load_mtl(const string& filename, const obj_callbacks& cb,
    const load_obj_options& options) {
    // open file
    auto fs = input_file(filename);

    // currently parsed material
    auto material = obj_material();
    auto first    = true;

    // read the file line by line
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        // line
        auto line = buffer;
        normalize_obj_line(line);
        if (!*line) continue;

        // get command
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "newmtl") {
            if (!first && cb.material) cb.material(material);
            first    = false;
            material = obj_material();
            parse_value(line, material.name);
        } else if (cmd == "illum") {
            parse_value(line, material.illum);
        } else if (cmd == "Ke") {
            parse_value(line, material.ke);
        } else if (cmd == "Kd") {
            parse_value(line, material.kd);
        } else if (cmd == "Ks") {
            parse_value(line, material.ks);
        } else if (cmd == "Kt") {
            parse_value(line, material.kt);
        } else if (cmd == "Tf") {
            material.kt = {-1, -1, -1};
            parse_value(line, material.kt);
            if (material.kt.y < 0)
                material.kt = {material.kt.x, material.kt.x, material.kt.x};
            if (options.flip_tr) material.kt = vec3f{1, 1, 1} - material.kt;
        } else if (cmd == "Tr") {
            parse_value(line, material.op);
            if (options.flip_tr) material.op = 1 - material.op;
        } else if (cmd == "Ns") {
            parse_value(line, material.ns);
            material.rs = pow(2 / (material.ns + 2), 1 / 4.0f);
            if (material.rs < 0.01f) material.rs = 0;
            if (material.rs > 0.99f) material.rs = 1;
        } else if (cmd == "d") {
            parse_value(line, material.op);
        } else if (cmd == "Pr" || cmd == "rs") {
            parse_value(line, material.rs);
        } else if (cmd == "map_Ke") {
            parse_value(line, material.ke_txt);
        } else if (cmd == "map_Kd") {
            parse_value(line, material.kd_txt);
        } else if (cmd == "map_Ks") {
            parse_value(line, material.ks_txt);
        } else if (cmd == "map_Tr") {
            parse_value(line, material.kt_txt);
        } else if (cmd == "map_d" || cmd == "map_Tr") {
            parse_value(line, material.op_txt);
        } else if (cmd == "map_Pr" || cmd == "map_rs") {
            parse_value(line, material.rs_txt);
        } else if (cmd == "map_occ" || cmd == "occ") {
            parse_value(line, material.occ_txt);
        } else if (cmd == "map_bump" || cmd == "bump") {
            parse_value(line, material.bump_txt);
        } else if (cmd == "map_disp" || cmd == "disp") {
            parse_value(line, material.disp_txt);
        } else if (cmd == "map_norm" || cmd == "norm") {
            parse_value(line, material.norm_txt);
        } else if (cmd == "Ve") {
            parse_value(line, material.ve);
        } else if (cmd == "Va") {
            parse_value(line, material.va);
        } else if (cmd == "Vd") {
            parse_value(line, material.vd);
        } else if (cmd == "Vg") {
            parse_value(line, material.vg);
        } else if (cmd == "map_Vd") {
            parse_value(line, material.vd_txt);
        }
    }

    // issue current material
    if (!first && cb.material) cb.material(material);
}

// Load obj extensions
void load_objx(const string& filename, const obj_callbacks& cb,
    const load_obj_options& options) {
    // open file
    auto fs = input_file(filename);

    // read the file line by line
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        // line
        auto line = buffer;
        normalize_obj_line(line);
        if (!*line) continue;

        // get command
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "c") {
            auto camera = obj_camera();
            parse_value(line, camera.name);
            parse_value(line, camera.ortho);
            parse_value(line, camera.width);
            parse_value(line, camera.height);
            parse_value(line, camera.focal);
            parse_value(line, camera.focus);
            parse_value(line, camera.aperture);
            parse_value(line, camera.frame);
            if (cb.camera) cb.camera(camera);
        } else if (cmd == "e") {
            auto environment = obj_environment();
            parse_value(line, environment.name);
            parse_value(line, environment.ke);
            parse_value(line, environment.ke_txt.path);
            parse_value(line, environment.frame);
            if (environment.ke_txt.path == "\"\"") environment.ke_txt.path = "";
            if (cb.environmnet) cb.environmnet(environment);
        } else if (cmd == "po") {
            auto procedural = obj_procedural();
            parse_value(line, procedural.name);
            parse_value(line, procedural.type);
            parse_value(line, procedural.material);
            parse_value(line, procedural.size);
            parse_value(line, procedural.level);
            parse_value(line, procedural.frame);
            if (cb.procedural) cb.procedural(procedural);
        } else {
            // unused
        }
    }
}

// Load obj scene
void load_obj(const string& filename, const obj_callbacks& cb,
    const load_obj_options& options) {
    // open file
    auto fs = input_file(filename);

    // track vertex size
    auto vert_size = obj_vertex();
    auto verts     = vector<obj_vertex>();  // buffer to avoid reallocation

    // read the file line by line
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        // line
        auto line = buffer;
        normalize_obj_line(line);
        if (!*line) continue;

        // get command
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "v") {
            auto vert = zero3f;
            parse_value(line, vert);
            if (cb.vert) cb.vert(vert);
            vert_size.position += 1;
        } else if (cmd == "vn") {
            auto vert = zero3f;
            parse_value(line, vert);
            if (cb.norm) cb.norm(vert);
            vert_size.normal += 1;
        } else if (cmd == "vt") {
            auto vert = zero2f;
            parse_value(line, vert);
            if (options.flip_texcoord) vert.y = 1 - vert.y;
            if (cb.texcoord) cb.texcoord(vert);
            vert_size.texturecoord += 1;
        } else if (cmd == "f" || cmd == "l" || cmd == "p") {
            verts.clear();
            while (*line == ' ') line++;
            while (*line) {
                auto vert = obj_vertex{};
                parse_value(line, vert);
                if (!vert.position) break;
                if (vert.position < 0)
                    vert.position = vert_size.position + vert.position + 1;
                if (vert.texturecoord < 0)
                    vert.texturecoord = vert_size.texturecoord +
                                        vert.texturecoord + 1;
                if (vert.normal < 0)
                    vert.normal = vert_size.normal + vert.normal + 1;
                verts.push_back(vert);
                while (*line == ' ') line++;
            }
            if (cmd == "f" && cb.face) cb.face(verts);
            if (cmd == "l" && cb.line) cb.line(verts);
            if (cmd == "p" && cb.point) cb.point(verts);
        } else if (cmd == "o") {
            auto name = ""s;
            parse_value(line, name, true);
            if (cb.object) cb.object(name);
        } else if (cmd == "usemtl") {
            auto name = ""s;
            parse_value(line, name, true);
            if (cb.usemtl) cb.usemtl(name);
        } else if (cmd == "g") {
            auto name = ""s;
            parse_value(line, name, true);
            if (cb.group) cb.group(name);
        } else if (cmd == "s") {
            auto name = ""s;
            parse_value(line, name, true);
            if (cb.smoothing) cb.smoothing(name);
        } else if (cmd == "mtllib") {
            if (options.geometry_only) continue;
            auto mtlname = ""s;
            parse_value(line, mtlname);
            if (cb.mtllib) cb.mtllib(mtlname);
            auto mtlpath = get_dirname(filename) + mtlname;
            load_mtl(mtlpath, cb, options);
        } else {
            // unused
        }
    }

    // parse extensions if presents
    if (!options.geometry_only) {
        auto extname    = replace_extension(filename, "objx");
        auto ext_exists = exists_file(extname);
        if (ext_exists) {
            load_objx(extname, cb, options);
        }
    }
}

// Parse obj texture
void parse_obj_texture(char*& str, int& texture_id, yocto_scene& scene) {
    auto info = obj_texture_info{};
    parse_value(str, info);
    for (texture_id = 0; texture_id < scene.textures.size(); texture_id++) {
        if (scene.textures[texture_id].filename == info.path) return;
    }
    scene.textures.push_back({});
    auto& texture         = scene.textures.back();
    texture.name          = info.path;
    texture.filename      = info.path;
    texture.clamp_to_edge = info.clamp;
}
void parse_obj_voltexture(char*& str, int& texture_id, yocto_scene& scene) {
    auto info = obj_texture_info{};
    parse_value(str, info);
    for (texture_id = 0; texture_id < scene.voltextures.size(); texture_id++) {
        if (scene.voltextures[texture_id].filename == info.path) return;
    }
    scene.voltextures.push_back({});
    auto& texture         = scene.voltextures.back();
    texture.name          = info.path;
    texture.filename      = info.path;
    texture.clamp_to_edge = info.clamp;
}

// Get material index
void set_obj_material(
    const string& name, int& material_id, const yocto_scene& scene) {
    for (material_id = 0; material_id < scene.materials.size(); material_id++) {
        if (scene.materials[material_id].name == name) return;
    }
    throw io_error("unknown material " + name);
}

// Load obj materials
void load_mtl(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    // open file
    auto fs = input_file(filename);

    // options
    auto flip_tr = true;

    // currently parsed material
    auto material = yocto_material();
    auto first    = true;

    // read the file line by line
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        // line
        auto line = buffer;
        normalize_obj_line(line);
        if (!*line) continue;

        // get command
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "newmtl") {
            if (!first) scene.materials.push_back(material);
            first    = false;
            material = yocto_material{};
            parse_value(line, material.name);
        } else if (cmd == "illum") {
        } else if (cmd == "Ke") {
            parse_value(line, material.emission);
        } else if (cmd == "Kd") {
            parse_value(line, material.diffuse);
        } else if (cmd == "Ks") {
            parse_value(line, material.specular);
        } else if (cmd == "Kt") {
            parse_value(line, material.transmission);
        } else if (cmd == "Tf") {
            parse_value(line, material.transmission);
            if (flip_tr) material.transmission = 1 - material.transmission;
        } else if (cmd == "Tr") {
            parse_value(line, material.opacity);
            if (flip_tr) material.opacity = 1 - material.opacity;
        } else if (cmd == "Ns") {
            parse_value(line, material.roughness);
            material.roughness = pow(2 / (material.roughness + 2), 1 / 4.0f);
            if (material.roughness < 0.01f) material.roughness = 0;
            if (material.roughness > 0.99f) material.roughness = 1;
        } else if (cmd == "d") {
            parse_value(line, material.opacity);
        } else if (cmd == "Pr" || cmd == "rs") {
            parse_value(line, material.roughness);
        } else if (cmd == "map_Ke") {
            parse_obj_texture(line, material.emission_texture, scene);
        } else if (cmd == "map_Kd") {
            parse_obj_texture(line, material.diffuse_texture, scene);
        } else if (cmd == "map_Ks") {
            parse_obj_texture(line, material.specular_texture, scene);
        } else if (cmd == "map_Tr") {
            parse_obj_texture(line, material.transmission_texture, scene);
        } else if (cmd == "map_d" || cmd == "map_Tr") {
            parse_obj_texture(line, material.opacity_texture, scene);
        } else if (cmd == "map_Pr" || cmd == "map_rs") {
            parse_obj_texture(line, material.roughness_texture, scene);
        } else if (cmd == "map_occ" || cmd == "occ") {
            parse_obj_texture(line, material.occlusion_texture, scene);
        } else if (cmd == "map_bump" || cmd == "bump") {
            parse_obj_texture(line, material.bump_texture, scene);
        } else if (cmd == "map_disp" || cmd == "disp") {
            parse_obj_texture(line, material.displacement_texture, scene);
        } else if (cmd == "map_norm" || cmd == "norm") {
            parse_obj_texture(line, material.normal_texture, scene);
        } else if (cmd == "Ve") {
            parse_value(line, material.volume_emission);
        } else if (cmd == "Va") {
            parse_value(line, material.volume_albedo);
        } else if (cmd == "Vd") {
            parse_value(line, material.volume_density);
        } else if (cmd == "Vg") {
            parse_value(line, material.volume_phaseg);
        } else if (cmd == "map_Vd") {
            parse_obj_texture(line, material.volume_density_texture, scene);
        }
    }

    // issue current material
    if (!first) scene.materials.push_back(material);
}

// Load obj extensions
void load_objx(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    // open file
    auto fs = input_file(filename);

    // read the file line by line
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        // line
        auto line = buffer;
        normalize_obj_line(line);
        if (!*line) continue;

        // get command
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "c") {
            auto camera = yocto_camera();
            parse_value(line, camera.name);
            parse_value(line, camera.orthographic);
            parse_value(line, camera.film_width);
            parse_value(line, camera.film_height);
            parse_value(line, camera.focal_length);
            parse_value(line, camera.focus_distance);
            parse_value(line, camera.lens_aperture);
            parse_value(line, camera.frame);
            scene.cameras.push_back(camera);
        } else if (cmd == "e") {
            auto environment = yocto_environment();
            parse_value(line, environment.name);
            parse_value(line, environment.emission);
            auto texture_path = ""s;
            parse_value(line, texture_path);
            parse_value(line, environment.frame);
            if (texture_path != "\"\"") {
                auto str = (char*)texture_path.c_str();
                parse_obj_texture(str, environment.emission_texture, scene);
            }
            scene.environments.push_back(environment);
        } else if (cmd == "po") {
            auto name = ""s, type = ""s, material = ""s;
            auto frame = identity_frame3f;
            auto size  = 1.0f;
            auto level = 0;
            parse_value(line, name);
            parse_value(line, type);
            parse_value(line, material);
            parse_value(line, size);
            parse_value(line, level);
            parse_value(line, frame);
            auto shape = yocto_shape{};
            shape.name = name;
            set_obj_material(material, shape.material, scene);
            if (type == "floor") {
                make_floor_shape(shape.quads, shape.positions, shape.normals,
                    shape.texturecoords,
                    {level < 0 ? 1 : pow2(level), level < 0 ? 20 : pow2(level)},
                    {size, size}, {size / 2, size / 2});
            } else {
                throw io_error("unknown obj procedural " + type);
            }
            scene.shapes.push_back(shape);
            auto instance  = yocto_instance{};
            instance.name  = name;
            instance.frame = frame;
            instance.shape = (int)scene.shapes.size() - 1;
            scene.instances.push_back(instance);
        } else {
            // unused
        }
    }
}

// Load obj scene
void load_obj(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    // open file
    auto fs = input_file(filename);

    // options
    auto flip_texcoord = true;

    // splitting policy
    auto split_material        = options.obj_split_shapes;
    auto split_group           = options.obj_split_shapes;
    auto split_smoothing       = false;
    auto preserve_face_varying = options.obj_preserve_face_varying;

    // current parsing values
    auto matname   = string();
    auto oname     = string();
    auto gname     = string();
    auto smoothing = true;

    // vertices
    auto opos      = vector<vec3f>();
    auto onorm     = vector<vec3f>();
    auto otexcoord = vector<vec2f>();

    // vertex maps
    auto vertex_map   = unordered_map<obj_vertex, int, obj_vertex_hash>();
    auto pos_map      = unordered_map<int, int>();
    auto norm_map     = unordered_map<int, int>();
    auto texcoord_map = unordered_map<int, int>();

    // track vertex size
    auto vert_size = obj_vertex();
    auto verts     = vector<obj_vertex>();  // buffer to avoid reallocation

    // helpers
    auto add_instance = [&]() {
        auto instance = yocto_instance();
        instance.name = !empty(oname) ? oname : gname;
        scene.instances.push_back(instance);
        vertex_map.clear();
        pos_map.clear();
        norm_map.clear();
        texcoord_map.clear();
    };

    // read the file line by line
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        // line
        auto line = buffer;
        normalize_obj_line(line);
        if (!*line) continue;

        // get command
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "v") {
            auto vert = zero3f;
            parse_value(line, vert);
            opos.push_back(vert);
            vert_size.position += 1;
        } else if (cmd == "vn") {
            auto vert = zero3f;
            parse_value(line, vert);
            onorm.push_back(vert);
            vert_size.normal += 1;
        } else if (cmd == "vt") {
            auto vert = zero2f;
            parse_value(line, vert);
            if (flip_texcoord) vert.y = 1 - vert.y;
            otexcoord.push_back(vert);
            vert_size.texturecoord += 1;
        } else if (cmd == "f" || cmd == "l" || cmd == "p") {
            if (empty(scene.instances) ||
                (cmd == "l" && scene.instances.back().surface >= 0) ||
                (cmd == "p" && scene.instances.back().surface >= 0)) {
                add_instance();
            }
            if (scene.instances.back().shape < 0 &&
                scene.instances.back().surface < 0) {
                if (cmd == "f" &&
                    (preserve_face_varying ||
                        scene.instances.back().name.find(
                            "[yocto::facevarying]") != string::npos)) {
                    scene.surfaces.push_back({});
                    scene.surfaces.back().name = scene.instances.back().name;
                    scene.surfaces.back().materials.push_back(-1);
                    set_obj_material(
                        matname, scene.surfaces.back().materials.back(), scene);
                    scene.instances.back().surface =
                        (int)scene.surfaces.size() - 1;
                } else {
                    scene.shapes.push_back({});
                    scene.shapes.back().name = scene.instances.back().name;
                    set_obj_material(
                        matname, scene.shapes.back().material, scene);
                    scene.instances.back().shape = (int)scene.shapes.size() - 1;
                }
            }
            verts.clear();
            while (*line == ' ') line++;
            while (*line) {
                auto vert = obj_vertex{};
                parse_value(line, vert);
                if (!vert.position) break;
                if (vert.position < 0)
                    vert.position = vert_size.position + vert.position + 1;
                if (vert.texturecoord < 0)
                    vert.texturecoord = vert_size.texturecoord +
                                        vert.texturecoord + 1;
                if (vert.normal < 0)
                    vert.normal = vert_size.normal + vert.normal + 1;
                verts.push_back(vert);
                while (*line == ' ') line++;
            }
            if (scene.instances.back().surface >= 0) {
                auto& surface = scene.surfaces.back();
                for (auto& vert : verts) {
                    if (!vert.position) continue;
                    auto pos_it = pos_map.find(vert.position);
                    if (pos_it != pos_map.end()) continue;
                    auto nverts = (int)surface.positions.size();
                    pos_map.insert(pos_it, {vert.position, nverts});
                    surface.positions.push_back(opos.at(vert.position - 1));
                }
                for (auto& vert : verts) {
                    if (!vert.texturecoord) continue;
                    auto texcoord_it = texcoord_map.find(vert.texturecoord);
                    if (texcoord_it != texcoord_map.end()) continue;
                    auto nverts = (int)surface.texturecoords.size();
                    texcoord_map.insert(
                        texcoord_it, {vert.texturecoord, nverts});
                    surface.texturecoords.push_back(
                        otexcoord.at(vert.texturecoord - 1));
                }
                for (auto& vert : verts) {
                    if (!vert.normal) continue;
                    auto norm_it = norm_map.find(vert.normal);
                    if (norm_it != norm_map.end()) continue;
                    auto nverts = (int)surface.normals.size();
                    norm_map.insert(norm_it, {vert.normal, nverts});
                    surface.normals.push_back(onorm.at(vert.normal - 1));
                }
            } else {
                auto& shape = scene.shapes.back();
                for (auto& vert : verts) {
                    auto it = vertex_map.find(vert);
                    if (it != vertex_map.end()) continue;
                    auto nverts = (int)shape.positions.size();
                    vertex_map.insert(it, {vert, nverts});
                    if (vert.position)
                        shape.positions.push_back(opos.at(vert.position - 1));
                    if (vert.texturecoord)
                        shape.texturecoords.push_back(
                            otexcoord.at(vert.texturecoord - 1));
                    if (vert.normal)
                        shape.normals.push_back(onorm.at(vert.normal - 1));
                }
            }
            if (cmd == "f" && scene.instances.back().surface >= 0) {
                auto& surface = scene.surfaces.back();
                if (verts.size() == 4) {
                    if (verts[0].position) {
                        surface.quads_positions.push_back(
                            {pos_map.at(verts[0].position),
                                pos_map.at(verts[1].position),
                                pos_map.at(verts[2].position),
                                pos_map.at(verts[3].position)});
                    }
                    if (verts[0].texturecoord) {
                        surface.quads_texturecoords.push_back(
                            {texcoord_map.at(verts[0].texturecoord),
                                texcoord_map.at(verts[1].texturecoord),
                                texcoord_map.at(verts[2].texturecoord),
                                texcoord_map.at(verts[3].texturecoord)});
                    }
                    if (verts[0].normal) {
                        surface.quads_normals.push_back(
                            {norm_map.at(verts[0].normal),
                                norm_map.at(verts[1].normal),
                                norm_map.at(verts[2].normal),
                                norm_map.at(verts[3].normal)});
                    }
                } else {
                    if (verts[0].position) {
                        for (auto i = 2; i < verts.size(); i++)
                            surface.quads_positions.push_back(
                                {pos_map.at(verts[0].position),
                                    pos_map.at(verts[i - 1].position),
                                    pos_map.at(verts[i].position),
                                    pos_map.at(verts[i].position)});
                    }
                    if (verts[0].texturecoord) {
                        for (auto i = 2; i < verts.size(); i++)
                            surface.quads_texturecoords.push_back(
                                {texcoord_map.at(verts[0].texturecoord),
                                    texcoord_map.at(verts[i - 1].texturecoord),
                                    texcoord_map.at(verts[i].texturecoord),
                                    texcoord_map.at(verts[i].texturecoord)});
                    }
                    if (verts[0].normal) {
                        for (auto i = 2; i < verts.size(); i++)
                            surface.quads_normals.push_back(
                                {norm_map.at(verts[0].normal),
                                    norm_map.at(verts[i - 1].normal),
                                    norm_map.at(verts[i].normal),
                                    norm_map.at(verts[i].normal)});
                    }
                }
            } else if (cmd == "f" && scene.instances.back().surface < 0) {
                auto& shape = scene.shapes.back();
                if (verts.size() == 4) {
                    shape.quads.push_back(
                        {vertex_map.at(verts[0]), vertex_map.at(verts[1]),
                            vertex_map.at(verts[2]), vertex_map.at(verts[3])});
                } else {
                    for (auto i = 2; i < verts.size(); i++)
                        shape.triangles.push_back({vertex_map.at(verts[0]),
                            vertex_map.at(verts[i - 1]),
                            vertex_map.at(verts[i])});
                }
            } else if (cmd == "l") {
                auto& shape = scene.shapes.back();
                for (auto i = 1; i < verts.size(); i++)
                    shape.lines.push_back(
                        {vertex_map.at(verts[i - 1]), vertex_map.at(verts[i])});
            } else if (cmd == "p") {
                auto& shape = scene.shapes.back();
                for (auto i = 0; i < verts.size(); i++)
                    shape.points.push_back(vertex_map.at(verts[i]));
            }
        } else if (cmd == "o") {
            oname     = "";
            gname     = "";
            matname   = "";
            smoothing = true;
            parse_value(line, oname, true);
            add_instance();
        } else if (cmd == "g") {
            parse_value(line, gname, true);
            if (split_group) add_instance();
        } else if (cmd == "usemtl") {
            parse_value(line, matname, true);
            if (split_material) add_instance();
        } else if (cmd == "s") {
            auto name = ""s;
            parse_value(line, name, true);
            smoothing = (name == "on");
            if (split_smoothing) add_instance();
        } else if (cmd == "mtllib") {
            auto mtlname = ""s;
            parse_value(line, mtlname);
            auto mtlpath = get_dirname(filename) + mtlname;
            load_mtl(mtlpath, scene, options);
        } else {
            // unused
        }
    }

    // parse extensions if presents
    auto extname    = replace_extension(filename, "objx");
    auto ext_exists = exists_file(extname);
    if (ext_exists) {
        load_objx(extname, scene, options);
    }

    // cleanup empty
    for (auto idx = 0; idx < scene.instances.size(); idx++) {
        auto& instance = scene.instances[idx];
        if (instance.shape >= 0 &&
            !empty(scene.shapes[instance.shape].positions))
            continue;
        if (instance.surface >= 0 &&
            !empty(scene.surfaces[instance.surface].positions))
            continue;
        scene.instances.erase(scene.instances.begin() + idx);
        idx--;
    }

    // merging quads and triangles
    for (auto& shape : scene.shapes) {
        if (empty(shape.triangles) || empty(shape.quads)) continue;
        merge_triangles_and_quads(shape.triangles, shape.quads, false);
    }
}

// Loads an OBJ
void load_obj_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    scene = {};

    // splitting policy
    auto split_material = options.obj_split_shapes;
    auto split_group    = options.obj_split_shapes;
    // auto split_smoothing = options.obj_split_shapes;
    auto split_smoothing = false;

    // current parsing values
    auto matname   = string();
    auto oname     = string();
    auto gname     = string();
    auto smoothing = true;

    // vertices
    auto opos      = deque<vec3f>();
    auto onorm     = deque<vec3f>();
    auto otexcoord = deque<vec2f>();

    // object maps
    auto tmap = unordered_map<string, int>();
    auto vmap = unordered_map<string, int>();
    auto mmap = unordered_map<string, int>();

    // vertex maps
    auto vertex_map   = unordered_map<obj_vertex, int, obj_vertex_hash>();
    auto pos_map      = unordered_map<int, int>();
    auto norm_map     = unordered_map<int, int>();
    auto texcoord_map = unordered_map<int, int>();

    // add object if needed
    auto is_instance_empty = [](const yocto_scene&     scene,
                                 const yocto_instance& instance) {
        if (instance.shape >= 0)
            return empty(scene.shapes[instance.shape].positions);
        if (instance.surface >= 0)
            return empty(scene.surfaces[instance.surface].positions);
        return true;
    };
    auto add_instance = [&](yocto_scene& scene, const string& objname,
                            const string& groupname) {
        auto instance = yocto_instance();
        instance.name = !empty(objname) ? objname : groupname;
        scene.instances.push_back(instance);
        vertex_map.clear();
        pos_map.clear();
        norm_map.clear();
        texcoord_map.clear();
    };
    // Get current material
    auto get_material_id = [&mmap](const string& matname) {
        if (matname != "") {
            auto it = mmap.find(matname);
            if (it == mmap.end()) {
                throw io_error("missing material " + matname);
                return -1;
            }
            return it->second;
        }
        return -1;
    };
    // Parse texture options and name
    auto add_texture = [&scene, &tmap](
                           const obj_texture_info& info, bool force_linear) {
        if (info.path == "") return -1;
        if (tmap.find(info.path) != tmap.end()) {
            return tmap.at(info.path);
        }

        // create texture
        auto texture          = yocto_texture{};
        texture.name          = info.path;
        texture.filename      = info.path;
        texture.clamp_to_edge = info.clamp;
        texture.height_scale  = info.scale;
        texture.ldr_as_linear = force_linear || is_hdr_filename(info.path);
        scene.textures.push_back(texture);
        auto index      = (int)scene.textures.size() - 1;
        tmap[info.path] = index;

        return index;
    };
    // Parse texture options and name
    auto add_voltexture = [&scene, &vmap](
                              const obj_texture_info& info, bool srgb) {
        if (info.path == "") return -1;
        if (vmap.find(info.path) != vmap.end()) {
            return vmap.at(info.path);
        }

        // create texture
        auto texture     = yocto_voltexture{};
        texture.name     = info.path;
        texture.filename = info.path;
        scene.voltextures.push_back(texture);
        auto index      = (int)scene.voltextures.size() - 1;
        vmap[info.path] = index;

        return index;
    };
    // Add  vertices to the current shape
    auto add_verts = [&](const vector<obj_vertex>& verts, yocto_shape& shape) {
        for (auto& vert : verts) {
            auto it = vertex_map.find(vert);
            if (it != vertex_map.end()) continue;
            auto& shape  = scene.shapes.back();
            auto  nverts = (int)shape.positions.size();
            vertex_map.insert(it, {vert, nverts});
            if (vert.position)
                shape.positions.push_back(opos.at(vert.position - 1));
            if (vert.texturecoord)
                shape.texturecoords.push_back(
                    otexcoord.at(vert.texturecoord - 1));
            if (vert.normal) shape.normals.push_back(onorm.at(vert.normal - 1));
        }
    };
    // add vertex
    auto add_fvverts = [&](const vector<obj_vertex>& verts,
                           yocto_surface&            surface) {
        for (auto& vert : verts) {
            if (!vert.position) continue;
            auto pos_it = pos_map.find(vert.position);
            if (pos_it != pos_map.end()) continue;
            auto nverts = (int)surface.positions.size();
            pos_map.insert(pos_it, {vert.position, nverts});
            surface.positions.push_back(opos.at(vert.position - 1));
        }
        for (auto& vert : verts) {
            if (!vert.texturecoord) continue;
            auto texcoord_it = texcoord_map.find(vert.texturecoord);
            if (texcoord_it != texcoord_map.end()) continue;
            auto nverts = (int)surface.texturecoords.size();
            texcoord_map.insert(texcoord_it, {vert.texturecoord, nverts});
            surface.texturecoords.push_back(
                otexcoord.at(vert.texturecoord - 1));
        }
        for (auto& vert : verts) {
            if (!vert.normal) continue;
            auto norm_it = norm_map.find(vert.normal);
            if (norm_it != norm_map.end()) continue;
            auto nverts = (int)surface.normals.size();
            norm_map.insert(norm_it, {vert.normal, nverts});
            surface.normals.push_back(onorm.at(vert.normal - 1));
        }
    };

    // callbacks
    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const vector<obj_vertex>& verts) {
        if (empty(scene.instances)) add_instance(scene, oname, gname);
        if (scene.instances.back().shape < 0 &&
            scene.instances.back().surface < 0) {
            if (options.obj_preserve_face_varying ||
                scene.instances.back().name.find("[yocto::facevarying]") !=
                    string::npos) {
                scene.surfaces.push_back({});
                scene.surfaces.back().name = scene.instances.back().name;
                scene.surfaces.back().materials.push_back(
                    get_material_id(matname));
                scene.instances.back().surface = (int)scene.surfaces.size() - 1;
            } else {
                scene.shapes.push_back({});
                scene.shapes.back().name     = scene.instances.back().name;
                scene.shapes.back().material = get_material_id(matname);
                scene.instances.back().shape = (int)scene.shapes.size() - 1;
            }
        }
        if (scene.instances.back().surface >= 0) {
            auto& surface = scene.surfaces.back();
            add_fvverts(verts, surface);
            if (verts.size() == 4) {
                if (verts[0].position) {
                    surface.quads_positions.push_back(
                        {pos_map.at(verts[0].position),
                            pos_map.at(verts[1].position),
                            pos_map.at(verts[2].position),
                            pos_map.at(verts[3].position)});
                }
                if (verts[0].texturecoord) {
                    surface.quads_texturecoords.push_back(
                        {texcoord_map.at(verts[0].texturecoord),
                            texcoord_map.at(verts[1].texturecoord),
                            texcoord_map.at(verts[2].texturecoord),
                            texcoord_map.at(verts[3].texturecoord)});
                }
                if (verts[0].normal) {
                    surface.quads_normals.push_back(
                        {norm_map.at(verts[0].normal),
                            norm_map.at(verts[1].normal),
                            norm_map.at(verts[2].normal),
                            norm_map.at(verts[3].normal)});
                }
            } else {
                if (verts[0].position) {
                    for (auto i = 2; i < verts.size(); i++)
                        surface.quads_positions.push_back(
                            {pos_map.at(verts[0].position),
                                pos_map.at(verts[i - 1].position),
                                pos_map.at(verts[i].position),
                                pos_map.at(verts[i].position)});
                }
                if (verts[0].texturecoord) {
                    for (auto i = 2; i < verts.size(); i++)
                        surface.quads_texturecoords.push_back(
                            {texcoord_map.at(verts[0].texturecoord),
                                texcoord_map.at(verts[i - 1].texturecoord),
                                texcoord_map.at(verts[i].texturecoord),
                                texcoord_map.at(verts[i].texturecoord)});
                }
                if (verts[0].normal) {
                    for (auto i = 2; i < verts.size(); i++)
                        surface.quads_normals.push_back(
                            {norm_map.at(verts[0].normal),
                                norm_map.at(verts[i - 1].normal),
                                norm_map.at(verts[i].normal),
                                norm_map.at(verts[i].normal)});
                }
            }
        } else {
            auto& shape = scene.shapes.back();
            add_verts(verts, shape);
            if (verts.size() == 4) {
                shape.quads.push_back(
                    {vertex_map.at(verts[0]), vertex_map.at(verts[1]),
                        vertex_map.at(verts[2]), vertex_map.at(verts[3])});
            } else {
                for (auto i = 2; i < verts.size(); i++)
                    shape.triangles.push_back({vertex_map.at(verts[0]),
                        vertex_map.at(verts[i - 1]), vertex_map.at(verts[i])});
            }
        }
    };
    cb.line = [&](const vector<obj_vertex>& verts) {
        if (empty(scene.instances)) add_instance(scene, oname, gname);
        if (scene.instances.back().surface >= 0)
            add_instance(scene, oname, gname);
        if (scene.instances.back().shape < 0) {
            scene.shapes.push_back({});
            scene.shapes.back().name     = scene.instances.back().name;
            scene.shapes.back().material = get_material_id(matname);
            scene.instances.back().shape = (int)scene.shapes.size() - 1;
        }
        auto& shape = scene.shapes.back();
        add_verts(verts, shape);
        for (auto i = 1; i < verts.size(); i++)
            shape.lines.push_back(
                {vertex_map.at(verts[i - 1]), vertex_map.at(verts[i])});
    };
    cb.point = [&](const vector<obj_vertex>& verts) {
        if (empty(scene.instances)) add_instance(scene, oname, gname);
        if (scene.instances.back().surface >= 0)
            add_instance(scene, oname, gname);
        if (scene.instances.back().shape < 0) {
            scene.shapes.push_back({});
            scene.shapes.back().name     = scene.instances.back().name;
            scene.shapes.back().material = get_material_id(matname);
            scene.instances.back().shape = (int)scene.shapes.size() - 1;
        }
        auto& shape = scene.shapes.back();
        add_verts(verts, shape);
        for (auto i = 0; i < verts.size(); i++)
            shape.points.push_back(vertex_map.at(verts[i]));
    };
    cb.object = [&](const string& name) {
        oname     = name;
        gname     = "";
        matname   = "";
        smoothing = true;
        add_instance(scene, oname, gname);
    };
    cb.group = [&](const string& name) {
        gname = name;
        if (split_group) {
            add_instance(scene, oname, gname);
        }
    };
    cb.smoothing = [&](const string& name) {
        smoothing = (name == "on");
        if (split_smoothing) {
            add_instance(scene, oname, gname);
        }
    };
    cb.usemtl = [&](const string& name) {
        matname = name;
        if (split_material) {
            add_instance(scene, oname, gname);
        }
    };
    cb.material = [&](const obj_material& omat) {
        auto material                   = yocto_material();
        material.name                   = omat.name;
        material.emission               = omat.ke;
        material.diffuse                = omat.kd;
        material.specular               = omat.ks;
        material.transmission           = omat.kt;
        material.roughness              = omat.rs;
        material.opacity                = omat.op;
        material.emission_texture       = add_texture(omat.ke_txt, false);
        material.diffuse_texture        = add_texture(omat.kd_txt, false);
        material.specular_texture       = add_texture(omat.ks_txt, false);
        material.transmission_texture   = add_texture(omat.kt_txt, false);
        material.opacity_texture        = add_texture(omat.op_txt, true);
        material.roughness_texture      = add_texture(omat.rs_txt, true);
        material.occlusion_texture      = add_texture(omat.occ_txt, true);
        material.bump_texture           = add_texture(omat.bump_txt, true);
        material.displacement_texture   = add_texture(omat.disp_txt, true);
        material.normal_texture         = add_texture(omat.norm_txt, true);
        material.volume_emission        = omat.ve;
        material.volume_albedo          = omat.va;
        material.volume_density         = omat.vd;
        material.volume_phaseg          = omat.vg;
        material.volume_density_texture = add_voltexture(omat.vd_txt, false);
        scene.materials.push_back(material);
        mmap[material.name] = (int)scene.materials.size() - 1;
    };
    cb.camera = [&](const obj_camera& ocam) {
        auto camera           = yocto_camera();
        camera.name           = ocam.name;
        camera.frame          = ocam.frame;
        camera.orthographic   = ocam.ortho;
        camera.film_width     = ocam.width;
        camera.film_height    = ocam.height;
        camera.focal_length   = ocam.focal;
        camera.focus_distance = ocam.focus;
        camera.lens_aperture  = ocam.aperture;
        scene.cameras.push_back(camera);
    };
    cb.environmnet = [&](const obj_environment& oenv) {
        auto environment             = yocto_environment();
        environment.name             = oenv.name;
        environment.frame            = oenv.frame;
        environment.emission         = oenv.ke;
        environment.emission_texture = add_texture(oenv.ke_txt, true);
        scene.environments.push_back(environment);
    };
    cb.procedural = [&](const obj_procedural& oproc) {
        auto shape = yocto_shape();
        shape.name = oproc.name;
        if (mmap.find(oproc.material) == mmap.end()) {
            throw io_error("missing material " + oproc.material);
        } else {
            shape.material = mmap.find(oproc.material)->second;
        }
        if (oproc.type == "floor") {
            make_floor_shape(shape.quads, shape.positions, shape.normals,
                shape.texturecoords,
                {oproc.level < 0 ? 1 : pow2(oproc.level),
                    oproc.level < 0 ? 20 : pow2(oproc.level)},
                {oproc.size, oproc.size}, {oproc.size / 2, oproc.size / 2});
        } else {
            throw io_error("unknown obj procedural");
        }
        scene.shapes.push_back(shape);
        auto instance  = yocto_instance();
        instance.name  = oproc.name;
        instance.frame = oproc.frame;
        instance.shape = (int)scene.shapes.size() - 1;
        scene.instances.push_back(instance);
    };

    try {
        // Parse obj
        auto obj_options          = load_obj_options();
        obj_options.geometry_only = false;
        load_obj(filename, cb, obj_options);

        // cleanup empty
        for (auto idx = 0; idx < scene.instances.size(); idx++) {
            if (!is_instance_empty(scene, scene.instances[idx])) continue;
            scene.instances.erase(scene.instances.begin() + idx);
            idx--;
        }

        // merging quads and triangles
        for (auto& shape : scene.shapes) {
            if (empty(shape.triangles) || empty(shape.quads)) continue;
            merge_triangles_and_quads(shape.triangles, shape.quads, false);
        }

        // load textures
        auto dirname = get_dirname(filename);
        load_scene_textures(scene, dirname, options);
    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }

    // fix scene
    scene.name = get_filename(filename);
    add_missing_cameras(scene);
    add_missing_materials(scene);
    add_missing_names(scene);
    update_transforms(scene);
}

void load_obj_scene_(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    scene = {};
    try {
        // Parse obj
        load_obj(filename, scene, options);

        // load textures
        auto dirname = get_dirname(filename);
        load_scene_textures(scene, dirname, options);
    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }

    // fix scene
    scene.name = get_filename(filename);
    add_missing_cameras(scene);
    add_missing_materials(scene);
    add_missing_names(scene);
    update_transforms(scene);
}

void save_mtl(
    const string& filename, const yocto_scene& scene, bool flip_tr = true) {
    // open file
    auto fs = output_file(filename);

    // for each material, dump all the values
    for (auto& material : scene.materials) {
        println_values(fs, "newmtl", material.name);
        println_values(fs, "  illum 2");
        println_values(fs, "  Ke", material.emission);
        println_values(fs, "  Kd", material.diffuse);
        println_values(fs, "  Ks", material.specular);
        println_values(fs, "  Kt", material.transmission);
        println_values(fs, "  Ns",
            (int)clamp(
                2 / pow(clamp(material.roughness, 0.0f, 0.99f) + 1e-10f, 4.0f) -
                    2,
                0.0f, 1.0e9f));
        println_values(fs, "  d", material.opacity);
        if (material.emission_texture >= 0)
            println_values(fs, "  map_Ke",
                scene.textures[material.emission_texture].filename);
        if (material.diffuse_texture >= 0)
            println_values(fs, "  map_Kd",
                scene.textures[material.diffuse_texture].filename);
        if (material.specular_texture >= 0)
            println_values(fs, "  map_Ks",
                scene.textures[material.specular_texture].filename);
        if (material.transmission_texture >= 0)
            println_values(fs, "  map_Kt",
                scene.textures[material.transmission_texture].filename);
        if (material.opacity_texture >= 0 &&
            material.opacity_texture != material.diffuse_texture)
            println_values(fs, "  map_d",
                scene.textures[material.opacity_texture].filename);
        if (material.roughness_texture >= 0)
            println_values(fs, "  map_Pr",
                scene.textures[material.roughness_texture].filename);
        if (material.occlusion_texture >= 0)
            println_values(fs, "  map_occ",
                scene.textures[material.occlusion_texture].filename);
        if (material.bump_texture >= 0)
            println_values(fs, "  map_bump",
                scene.textures[material.bump_texture].filename);
        if (material.displacement_texture >= 0)
            println_values(fs, "  map_disp",
                scene.textures[material.displacement_texture].filename);
        if (material.normal_texture >= 0)
            println_values(fs, "  map_norm",
                scene.textures[material.normal_texture].filename);
        if (material.volume_emission != zero3f)
            println_values(fs, "  Ve", material.volume_emission);
        if (material.volume_density != zero3f)
            println_values(fs, "  Vd", material.volume_density);
        if (material.volume_albedo != zero3f)
            println_values(fs, "  Va", material.volume_albedo);
        if (material.volume_phaseg != 0)
            println_values(fs, "  Vg", material.volume_phaseg);
        if (material.volume_density_texture >= 0)
            println_values(fs, "  map_Vd",
                scene.voltextures[material.volume_density_texture].filename);
        println_values(fs, "\n");
    }
}

void save_objx(const string& filename, const yocto_scene& scene) {
    // open file
    auto fs = output_file(filename);

    // cameras
    for (auto& camera : scene.cameras) {
        println_values(fs, "c", camera.name, (int)camera.orthographic,
            camera.film_width, camera.film_height, camera.focal_length,
            camera.focus_distance, camera.lens_aperture, camera.frame);
    }

    // environments
    for (auto& environment : scene.environments) {
        if (environment.emission_texture >= 0) {
            println_values(fs, "e", environment.name, environment.emission,
                scene.textures[environment.emission_texture].filename,
                environment.frame);
        } else {
            println_values(fs, "e", environment.name, environment.emission,
                "\"\"", environment.frame);
        }
    }
}

inline void print_value(const output_file& fs, const obj_vertex& value) {
    print_value(fs, value.position);
    if (value.texturecoord) {
        print_value(fs, '/');
        print_value(fs, value.texturecoord);
        if (value.normal) {
            print_value(fs, '/');
            print_value(fs, value.normal);
        }
    } else {
        if (value.normal) {
            print_value(fs, '/');
            print_value(fs, value.normal);
        }
    }
}

void save_obj(const string& filename, const yocto_scene& scene,
    bool flip_texcoord = true) {
    // open file
    auto fs = output_file(filename);

    println_values(
        fs, "# Saved by Yocto/GL - https://github.com/xelatihy/yocto-gl\n\n");

    // material library
    if (!empty(scene.materials)) {
        auto mtlname = replace_extension(get_filename(filename), "mtl");
        println_values(fs, "mtllib", mtlname);
    }

    // shapes
    auto offset = obj_vertex{0, 0, 0};
    for (auto& instance : scene.instances) {
        if (instance.surface >= 0) {
            auto& surface = scene.surfaces[instance.surface];
            println_values(fs, "o", instance.name);
            if (!empty(surface.materials) && empty(surface.quads_materials))
                println_values(fs, "usemtl",
                    scene.materials[surface.materials.front()].name);
            if (instance.frame == identity_frame3f) {
                for (auto& p : surface.positions) println_values(fs, "v", p);
                for (auto& n : surface.normals) println_values(fs, "vn", n);
                for (auto& t : surface.texturecoords)
                    println_values(
                        fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            } else {
                for (auto& pp : surface.positions) {
                    println_values(
                        fs, "v", transform_point(instance.frame, pp));
                }
                for (auto& nn : surface.normals) {
                    println_values(
                        fs, "vn", transform_direction(instance.frame, nn));
                }
                for (auto& t : surface.texturecoords)
                    println_values(
                        fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            }
            auto last_material_id = -1;
            for (auto i = 0; i < surface.quads_positions.size(); i++) {
                if (!empty(surface.quads_materials) &&
                    surface.quads_materials[i] != last_material_id) {
                    last_material_id = surface.quads_materials[i];
                    println_values(fs, "usemtl",
                        scene.materials[surface.materials[last_material_id]]
                            .name);
                }
                if (!empty(surface.texturecoords) && empty(surface.normals)) {
                    auto vert = [offset](int ip, int it) {
                        return obj_vertex{ip + offset.position + 1,
                            it + offset.texturecoord + 1, 0};
                    };
                    auto qp = surface.quads_positions[i];
                    auto qt = surface.quads_texturecoords[i];
                    if (qp.z == qp.w) {
                        println_values(fs, "f", vert(qp.x, qt.x),
                            vert(qp.y, qt.y), vert(qp.z, qt.z));
                    } else {
                        println_values(fs, "f", vert(qp.x, qt.x),
                            vert(qp.y, qt.y), vert(qp.z, qt.z),
                            vert(qp.w, qt.w));
                    }
                } else if (!empty(surface.texturecoords) &&
                           !empty(surface.normals)) {
                    auto vert = [offset](int ip, int it, int in) {
                        return obj_vertex{ip + offset.position + 1,
                            it + offset.texturecoord + 1,
                            in + offset.normal + 1};
                    };
                    auto qp = surface.quads_positions[i];
                    auto qt = surface.quads_texturecoords[i];
                    auto qn = surface.quads_normals[i];
                    if (qp.z == qp.w) {
                        println_values(fs, "f", vert(qp.x, qt.x, qn.x),
                            vert(qp.y, qt.y, qn.y), vert(qp.z, qt.z, qn.z));
                    } else {
                        println_values(fs, "f", vert(qp.x, qt.x, qn.x),
                            vert(qp.y, qt.y, qn.y), vert(qp.z, qt.z, qn.z),
                            vert(qp.w, qt.w, qn.w));
                    }
                } else if (!empty(surface.normals)) {
                    auto vert = [offset](int ip, int in) {
                        return obj_vertex{ip + offset.position + 1, 0,
                            in + offset.normal + 1};
                    };
                    auto qp = surface.quads_positions[i];
                    auto qn = surface.quads_normals[i];
                    if (qp.z == qp.w) {
                        println_values(fs, "f", vert(qp.x, qn.x),
                            vert(qp.y, qn.y), vert(qp.z, qn.z));
                    } else {
                        println_values(fs, "f", vert(qp.x, qn.x),
                            vert(qp.y, qn.y), vert(qp.z, qn.z),
                            vert(qp.w, qn.w));
                    }
                } else {
                    auto vert = [offset](int ip) {
                        return obj_vertex{ip + offset.position + 1, 0, 0};
                    };
                    auto q = surface.quads_positions[i];
                    if (q.z == q.w) {
                        println_values(
                            fs, "f", vert(q.x), vert(q.y), vert(q.z));
                    } else {
                        println_values(fs, "f", vert(q.x), vert(q.y), vert(q.z),
                            vert(q.w));
                    }
                }
            }
            offset.position += surface.positions.size();
            offset.texturecoord += surface.texturecoords.size();
            offset.normal += surface.normals.size();
        } else if (instance.shape >= 0) {
            auto& shape = scene.shapes[instance.shape];
            println_values(fs, "o", instance.name);
            if (shape.material >= 0)
                println_values(
                    fs, "usemtl", scene.materials[shape.material].name);
            if (instance.frame == identity_frame3f) {
                for (auto& p : shape.positions) println_values(fs, "v", p);
                for (auto& n : shape.normals) println_values(fs, "vn", n);
                for (auto& t : shape.texturecoords)
                    println_values(
                        fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            } else {
                for (auto& pp : shape.positions) {
                    println_values(
                        fs, "v", transform_point(instance.frame, pp));
                }
                for (auto& nn : shape.normals) {
                    println_values(
                        fs, "vn", transform_direction(instance.frame, nn));
                }
                for (auto& t : shape.texturecoords)
                    println_values(
                        fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            }
            auto mask = obj_vertex{1, empty(shape.texturecoords) ? 0 : 1,
                empty(shape.normals) ? 0 : 1};
            auto vert = [mask, offset](int i) {
                return obj_vertex{(i + offset.position + 1) * mask.position,
                    (i + offset.texturecoord + 1) * mask.texturecoord,
                    (i + offset.normal + 1) * mask.normal};
            };
            for (auto& p : shape.points) {
                println_values(fs, "p", vert(p));
            }
            for (auto& l : shape.lines) {
                println_values(fs, "l", vert(l.x), vert(l.y));
            }
            for (auto& t : shape.triangles) {
                println_values(fs, "f", vert(t.x), vert(t.y), vert(t.z));
            }
            for (auto& q : shape.quads) {
                if (q.z == q.w) {
                    println_values(fs, "f", vert(q.x), vert(q.y), vert(q.z));
                } else {
                    println_values(
                        fs, "f", vert(q.x), vert(q.y), vert(q.z), vert(q.w));
                }
            }
            offset.position += shape.positions.size();
            offset.texturecoord += shape.texturecoords.size();
            offset.normal += shape.normals.size();
        } else {
        }
    }
}

void save_obj_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    try {
        save_obj(filename, scene, true);
        if (!empty(scene.materials)) {
            save_mtl(replace_extension(filename, ".mtl"), scene, true);
        }
        if (!empty(scene.cameras) || !empty(scene.environments)) {
            save_objx(replace_extension(filename, ".objx"), scene);
        }

        // skip textures if needed
        auto dirname = get_dirname(filename);
        save_scene_textures(scene, dirname, options);
    } catch (const std::exception& e) {
        throw io_error("cannot save scene " + filename + "\n" + e.what());
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// PLY CONVERSION
// -----------------------------------------------------------------------------
namespace yocto {

void load_ply_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    scene = {};

    try {
        // load ply mesh
        scene.shapes.push_back({});
        auto& shape = scene.shapes.back();
        load_ply_mesh(filename, shape.points, shape.lines, shape.triangles,
            shape.quads, shape.positions, shape.normals, shape.texturecoords,
            shape.colors, shape.radius, false);

        // add instance
        auto instance  = yocto_instance{};
        instance.name  = shape.name;
        instance.shape = 0;
        scene.instances.push_back(instance);

    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }

    // fix scene
    scene.name = get_filename(filename);
    add_missing_cameras(scene);
    add_missing_materials(scene);
    add_missing_names(scene);
    update_transforms(scene);
}

void save_ply_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    if (scene.shapes.empty()) {
        throw io_error("cannot save empty scene " + filename);
    }
    try {
        auto& shape = scene.shapes.front();
        save_ply_mesh(filename, shape.points, shape.lines, shape.triangles,
            shape.quads, shape.positions, shape.normals, shape.texturecoords,
            shape.colors, shape.radius);
    } catch (const std::exception& e) {
        throw io_error("cannot save scene " + filename + "\n" + e.what());
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// GLTF CONVESION
// -----------------------------------------------------------------------------
namespace yocto {

static bool startswith(const string& str, const string& substr) {
    if (str.length() < substr.length()) return false;
    for (auto i = 0; i < substr.length(); i++)
        if (str[i] != substr[i]) return false;
    return true;
}

// convert gltf to scene
void gltf_to_scene(
    yocto_scene& scene, const json& gltf, const string& dirname) {
    // convert textures
    if (gltf.count("images")) {
        for (auto instance_id = 0; instance_id < gltf.at("images").size();
             instance_id++) {
            auto& gimg       = gltf.at("images").at(instance_id);
            auto  texture    = yocto_texture{};
            texture.name     = gimg.value("name", ""s);
            texture.filename = (startswith(gimg.value("uri", ""s), "data:")) ?
                                   string("[glTF-inline].png") :
                                   gimg.value("uri", ""s);
            scene.textures.push_back(texture);
        }
    }

    // load buffers
    auto bmap = vector<vector<byte>>();
    if (gltf.count("buffers")) {
        bmap.resize(gltf.at("buffers").size());
        for (auto bid = 0; bid < gltf.at("buffers").size(); bid++) {
            auto& gbuf = gltf.at("buffers").at(bid);
            auto& data = bmap.at(bid);
            auto  uri  = gbuf.value("uri", ""s);
            if (uri == "") continue;
            if (startswith(uri, "data:")) {
                // assume it is base64 and find ','
                auto pos = uri.find(',');
                if (pos == uri.npos) {
                    throw io_error("bad gltf buffer");
                }
                // decode
                auto data_char = base64_decode(uri.substr(pos + 1));
                data = vector<unsigned char>((unsigned char*)data_char.c_str(),
                    (unsigned char*)data_char.c_str() + data_char.length());
            } else {
                auto filename = normalize_path(dirname + uri);
                load_binary(filename, data);
            }
            if (gbuf.value("byteLength", -1) != data.size()) {
                throw io_error("bad gltf buffer");
            }
        }
    }

    // add a texture
    auto add_texture = [&scene, &gltf](const json& ginfo, bool force_linear) {
        if (!gltf.count("images") || !gltf.count("textures")) return -1;
        if (ginfo.is_null() || empty(ginfo)) return -1;
        if (ginfo.value("index", -1) < 0) return -1;
        auto& gtxt = gltf.at("textures").at(ginfo.value("index", -1));
        if (empty(gtxt) || gtxt.value("source", -1) < 0) return -1;
        auto texture_id = gtxt.value("source", -1);
        if (!gltf.count("samplers") || gtxt.value("sampler", -1) < 0)
            return texture_id;
        auto& gsmp = gltf.at("samplers").at(gtxt.value("sampler", -1));
        scene.textures[texture_id].clamp_to_edge = gsmp.value("wrapS", ""s) ==
                                                       "ClampToEdge" ||
                                                   gsmp.value("wrapT", ""s) ==
                                                       "ClampToEdge";
        scene.textures[texture_id].height_scale = gsmp.value("scale", 1.0f) *
                                                  gsmp.value("strength", 1.0f);
        scene.textures[texture_id].ldr_as_linear =
            force_linear ||
            is_hdr_filename(scene.textures[texture_id].filename);
        return texture_id;
    };

    // convert materials
    if (gltf.count("materials")) {
        for (auto mid = 0; mid < gltf.at("materials").size(); mid++) {
            auto& gmat        = gltf.at("materials").at(mid);
            auto  material    = yocto_material();
            material.name     = gmat.value("name", ""s);
            material.emission = gmat.value("emissiveFactor", zero3f);
            if (gmat.count("emissiveTexture"))
                material.emission_texture = add_texture(
                    gmat.at("emissiveTexture"), false);
            if (gmat.count("extensions") &&
                gmat.at("extensions")
                    .count("KHR_materials_pbrSpecularGlossiness")) {
                material.base_metallic = false;
                material.gltf_textures = true;
                auto& gsg              = gmat.at("extensions")
                                .at("KHR_materials_pbrSpecularGlossiness");
                auto kb = gsg.value("diffuseFactor", vec4f{1, 1, 1, 1});
                material.diffuse  = {kb.x, kb.y, kb.z};
                material.opacity  = kb.w;
                material.specular = gsg.value("specularFactor", vec3f{1, 1, 1});
                material.roughness = 1 - gsg.value("glossinessFactor", 1.0f);
                if (gsg.count("diffuseTexture"))
                    material.diffuse_texture = add_texture(
                        gsg.at("diffuseTexture"), false);
                if (gsg.count("specularGlossinessTexture"))
                    material.specular_texture = add_texture(
                        gsg.at("specularGlossinessTexture"), false);
                material.roughness_texture = material.specular_texture;
            } else if (gmat.count("pbrMetallicRoughness")) {
                material.base_metallic = true;
                material.gltf_textures = true;
                auto& gmr              = gmat.at("pbrMetallicRoughness");
                auto  kb = gmr.value("baseColorFactor", vec4f{1, 1, 1, 1});
                material.diffuse   = {kb.x, kb.y, kb.z};
                material.opacity   = kb.w;
                auto km            = gmr.value("metallicFactor", 1.0f);
                material.specular  = {km, km, km};
                material.roughness = gmr.value("roughnessFactor", 1.0f);
                if (gmr.count("baseColorTexture"))
                    material.diffuse_texture = add_texture(
                        gmr.at("baseColorTexture"), false);
                if (gmr.count("metallicRoughnessTexture"))
                    material.specular_texture = add_texture(
                        gmr.at("metallicRoughnessTexture"), true);
                material.roughness_texture = material.specular_texture;
            }
            if (gmat.count("occlusionTexture"))
                material.occlusion_texture = add_texture(
                    gmat.at("occlusionTexture"), true);
            if (gmat.count("normalTexture"))
                material.normal_texture = add_texture(
                    gmat.at("normalTexture"), true);
            scene.materials.push_back(material);
        }
    }

    // get values from accessors
    auto accessor_values =
        [&gltf, &bmap](const json& gacc,
            bool normalize = false) -> vector<std::array<double, 4>> {
        auto gview  = gltf.at("bufferViews").at(gacc.value("bufferView", -1));
        auto data   = bmap.at(gview.value("buffer", -1)).data();
        auto offset = gacc.value("byteOffset", 0) +
                      gview.value("byteOffset", 0);
        auto stride      = gview.value("byteStride", 0);
        auto compTypeNum = gacc.value("componentType", 5123);
        auto count       = gacc.value("count", -1);
        auto type        = gacc.value("type", ""s);
        auto ncomp       = 0;
        if (type == "SCALAR") ncomp = 1;
        if (type == "VEC2") ncomp = 2;
        if (type == "VEC3") ncomp = 3;
        if (type == "VEC4") ncomp = 4;
        auto compSize = 1;
        if (compTypeNum == 5122 || compTypeNum == 5123) {
            compSize = 2;
        }
        if (compTypeNum == 5124 || compTypeNum == 5125 || compTypeNum == 5126) {
            compSize = 4;
        }
        if (!stride) stride = compSize * ncomp;
        auto vals = vector<std::array<double, 4>>(
            count, {{0.0, 0.0, 0.0, 1.0}});
        for (auto i = 0; i < count; i++) {
            auto d = data + offset + i * stride;
            for (auto c = 0; c < ncomp; c++) {
                if (compTypeNum == 5120) {  // char
                    vals[i][c] = (double)(*(char*)d);
                    if (normalize) vals[i][c] /= SCHAR_MAX;
                } else if (compTypeNum == 5121) {  // byte
                    vals[i][c] = (double)(*(byte*)d);
                    if (normalize) vals[i][c] /= UCHAR_MAX;
                } else if (compTypeNum == 5122) {  // short
                    vals[i][c] = (double)(*(short*)d);
                    if (normalize) vals[i][c] /= SHRT_MAX;
                } else if (compTypeNum == 5123) {  // unsigned short
                    vals[i][c] = (double)(*(unsigned short*)d);
                    if (normalize) vals[i][c] /= USHRT_MAX;
                } else if (compTypeNum == 5124) {  // int
                    vals[i][c] = (double)(*(int*)d);
                    if (normalize) vals[i][c] /= INT_MAX;
                } else if (compTypeNum == 5125) {  // unsigned int
                    vals[i][c] = (double)(*(unsigned int*)d);
                    if (normalize) vals[i][c] /= UINT_MAX;
                } else if (compTypeNum == 5126) {  // float
                    vals[i][c] = (*(float*)d);
                }
                d += compSize;
            }
        }
        return vals;
    };

    // convert meshes
    auto meshes = vector<vector<int>>();
    if (gltf.count("meshes")) {
        for (auto mid = 0; mid < gltf.at("meshes").size(); mid++) {
            auto& gmesh = gltf.at("meshes").at(mid);
            meshes.push_back({});
            auto sid = 0;
            for (auto& gprim : gmesh.value("primitives", json::array())) {
                if (!gprim.count("attributes")) continue;
                auto shape = yocto_shape();
                shape.name = gmesh.value("name", ""s) +
                             ((sid) ? std::to_string(sid) : string());
                sid++;
                for (json::iterator gattr_it = gprim.at("attributes").begin();
                     gattr_it != gprim.at("attributes").end(); ++gattr_it) {
                    auto  semantic = gattr_it.key();
                    auto& gacc =
                        gltf.at("accessors").at(gattr_it.value().get<int>());
                    auto vals = accessor_values(gacc);
                    if (semantic == "POSITION") {
                        shape.positions.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shape.positions.push_back({(float)vals[i][0],
                                (float)vals[i][1], (float)vals[i][2]});
                    } else if (semantic == "NORMAL") {
                        shape.normals.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shape.normals.push_back({(float)vals[i][0],
                                (float)vals[i][1], (float)vals[i][2]});
                    } else if (semantic == "TEXCOORD" ||
                               semantic == "TEXCOORD_0") {
                        shape.texturecoords.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shape.texturecoords.push_back(
                                {(float)vals[i][0], (float)vals[i][1]});
                    } else if (semantic == "COLOR" || semantic == "COLOR_0") {
                        shape.colors.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shape.colors.push_back(
                                {(float)vals[i][0], (float)vals[i][1],
                                    (float)vals[i][2], (float)vals[i][3]});
                    } else if (semantic == "TANGENT") {
                        shape.tangentspaces.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shape.tangentspaces.push_back(
                                {(float)vals[i][0], (float)vals[i][1],
                                    (float)vals[i][2], (float)vals[i][3]});
                        for (auto& t : shape.tangentspaces) t.w = -t.w;
                    } else if (semantic == "RADIUS") {
                        shape.radius.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shape.radius.push_back((float)vals[i][0]);
                    } else {
                        // ignore
                    }
                }
                // indices
                auto mode = gprim.value("mode", 4);
                if (!gprim.count("indices")) {
                    if (mode == 4) {
                        // triangles
                        shape.triangles.reserve(shape.positions.size() / 3);
                        for (auto i = 0; i < shape.positions.size() / 3; i++)
                            shape.triangles.push_back(
                                {i * 3 + 0, i * 3 + 1, i * 3 + 2});
                    } else if (mode == 6) {
                        // triangle fan
                        shape.triangles.reserve(shape.positions.size() - 2);
                        for (auto i = 2; i < shape.positions.size(); i++)
                            shape.triangles.push_back({0, i - 1, i});
                    } else if (mode == 5) {
                        // triangle strip
                        shape.triangles.reserve(shape.positions.size() - 2);
                        for (auto i = 2; i < shape.positions.size(); i++)
                            shape.triangles.push_back({i - 2, i - 1, i});
                    } else if (mode == 1) {
                        // lines
                        shape.lines.reserve(shape.positions.size() / 2);
                        for (auto i = 0; i < shape.positions.size() / 2; i++)
                            shape.lines.push_back({i * 2 + 0, i * 2 + 1});
                    } else if (mode == 2) {
                        // line loop
                        shape.lines.reserve(shape.positions.size());
                        for (auto i = 1; i < shape.positions.size(); i++)
                            shape.lines.push_back({i - 1, i});
                        shape.lines.back() = {
                            (int)shape.positions.size() - 1, 0};
                    } else if (mode == 3) {
                        // line strip
                        shape.lines.reserve(shape.positions.size() - 1);
                        for (auto i = 1; i < shape.positions.size(); i++)
                            shape.lines.push_back({i - 1, i});
                    } else if (mode == -1 || mode == 0) {
                        // points
                        throw io_error("points not supported");
                    } else {
                        throw io_error("unknown primitive type");
                    }
                } else {
                    auto indices = accessor_values(
                        gltf.at("accessors").at(gprim.value("indices", -1)),
                        false);
                    if (mode == 4) {
                        // triangles
                        shape.triangles.reserve(indices.size());
                        for (auto i = 0; i < indices.size() / 3; i++)
                            shape.triangles.push_back(
                                {(int)indices[i * 3 + 0][0],
                                    (int)indices[i * 3 + 1][0],
                                    (int)indices[i * 3 + 2][0]});
                    } else if (mode == 6) {
                        // triangle fan
                        shape.triangles.reserve(indices.size() - 2);
                        for (auto i = 2; i < indices.size(); i++)
                            shape.triangles.push_back({(int)indices[0][0],
                                (int)indices[i - 1][0], (int)indices[i][0]});
                    } else if (mode == 5) {
                        // triangle strip
                        shape.triangles.reserve(indices.size() - 2);
                        for (auto i = 2; i < indices.size(); i++)
                            shape.triangles.push_back({(int)indices[i - 2][0],
                                (int)indices[i - 1][0], (int)indices[i][0]});
                    } else if (mode == 1) {
                        // lines
                        shape.lines.reserve(indices.size() / 2);
                        for (auto i = 0; i < indices.size() / 2; i++)
                            shape.lines.push_back({(int)indices[i * 2 + 0][0],
                                (int)indices[i * 2 + 1][0]});
                    } else if (mode == 2) {
                        // line loop
                        shape.lines.reserve(indices.size());
                        for (auto i = 1; i < indices.size(); i++)
                            shape.lines.push_back(
                                {(int)indices[i - 1][0], (int)indices[i][0]});
                        shape.lines.back() = {
                            (int)indices[indices.size() - 1][0],
                            (int)indices[0][0]};
                    } else if (mode == 3) {
                        // line strip
                        shape.lines.reserve(indices.size() - 1);
                        for (auto i = 1; i < indices.size(); i++)
                            shape.lines.push_back(
                                {(int)indices[i - 1][0], (int)indices[i][0]});
                    } else if (mode == -1 || mode == 0) {
                        // points
                        throw io_error("points not supported");
                    } else {
                        throw io_error("unknown primitive type");
                    }
                }
                shape.material = gprim.count("material") ?
                                     gprim.value("material", -1) :
                                     -1;
                scene.shapes.push_back(shape);
                meshes.back().push_back((int)scene.shapes.size() - 1);
            }
        }
    }

    // convert cameras
    if (gltf.count("cameras")) {
        for (auto cid = 0; cid < gltf.at("cameras").size(); cid++) {
            auto& gcam          = gltf.at("cameras").at(cid);
            auto  camera        = yocto_camera{};
            camera.name         = gcam.value("name", ""s);
            camera.orthographic = gcam.value("type", ""s) == "orthographic";
            if (camera.orthographic) {
                throw io_error("orthographic not supported well");
                auto ortho = gcam.value("orthographic", json::object());
                camera.lens_aperture = 0;
                set_camera_perspective(camera, ortho.value("ymag", 0.0f),
                    ortho.value("xmag", 0.0f) / ortho.value("ymag", 0.0f),
                    float_max);
            } else {
                auto persp = gcam.value("perspective", json::object());
                camera.lens_aperture = 0;
                set_camera_perspective(camera, persp.value("yfov", 1.0f),
                    persp.value("aspectRatio", 1.0f), float_max);
            }
            scene.cameras.push_back(camera);
        }
    }

    // convert cameras
    if (gltf.count("environments")) {
        for (auto eid = 0; eid < gltf.at("environments").size(); eid++) {
            auto& genv                   = gltf.at("environments").at(eid);
            auto  environment            = yocto_environment{};
            environment.name             = genv.value("name", ""s);
            environment.emission         = genv.value("emissiveFactor", zero3f);
            environment.emission_texture = add_texture(
                genv.at("emissiveTexture"), false);
            scene.environments.push_back(environment);
        }
    }

    // convert nodes
    if (gltf.count("nodes")) {
        for (auto nid = 0; nid < gltf.at("nodes").size(); nid++) {
            auto& gnde = gltf.at("nodes").at(nid);
            auto  node = yocto_scene_node{};
            node.name  = gnde.value("name", ""s);
            if (gnde.count("camera")) node.camera = gnde.value("camera", 0);
            if (gnde.count("environment"))
                node.environment = gnde.value("environment", 0);
            node.translation = gnde.value("translation", zero3f);
            node.rotation    = gnde.value("rotation", vec4f{0, 0, 0, 1});
            node.scale       = gnde.value("scale", vec3f{1, 1, 1});
            node.local = mat_to_frame(gnde.value("matrix", identity_mat4f));
            scene.nodes.push_back(node);
        }

        // set up parent pointers
        for (auto nid = 0; nid < gltf.at("nodes").size(); nid++) {
            auto& gnde = gltf.at("nodes").at(nid);
            if (!gnde.count("children")) continue;
            for (auto& cid : gnde.at("children"))
                scene.nodes[cid.get<int>()].parent = nid;
        }

        // set up instances
        for (auto nid = 0; nid < gltf.at("nodes").size(); nid++) {
            auto& gnde = gltf.at("nodes").at(nid);
            if (!gnde.count("mesh")) continue;
            auto& node = scene.nodes[nid];
            auto& shps = meshes.at(gnde.value("mesh", 0));
            if (empty(shps)) continue;
            if (shps.size() == 1) {
                auto instance  = yocto_instance();
                instance.name  = node.name;
                instance.shape = shps[0];
                scene.instances.push_back(instance);
                node.instance = (int)scene.instances.size() - 1;
            } else {
                for (auto shp : shps) {
                    auto& shape    = scene.shapes[shp];
                    auto  instance = yocto_instance();
                    instance.name  = node.name + "_" + shape.name;
                    instance.shape = shp;
                    scene.instances.push_back(instance);
                    auto child     = yocto_scene_node{};
                    child.name     = node.name + "_" + shape.name;
                    child.parent   = nid;
                    child.instance = (int)scene.instances.size() - 1;
                    scene.nodes.push_back(child);
                }
            }
        }
    }

    // convert animations
    if (gltf.count("animations")) {
        for (auto& ganm : gltf.at("animations")) {
            auto aid         = 0;
            auto sampler_map = unordered_map<vec2i, int>();
            for (auto& gchannel : ganm.at("channels")) {
                auto path_ = gchannel.at("target").at("path").get<string>();
                auto path  = -1;
                if (path_ == "translation") path = 0;
                if (path_ == "rotation") path = 1;
                if (path_ == "scale") path = 2;
                if (path_ == "weights") path = 3;
                if (sampler_map.find({gchannel.at("sampler").get<int>(),
                        path}) == sampler_map.end()) {
                    auto& gsampler = ganm.at("samplers")
                                         .at(gchannel.at("sampler").get<int>());
                    auto animation = yocto_animation{};
                    animation.name = (ganm.count("name") ?
                                             ganm.value("name", ""s) :
                                             "anim") +
                                     std::to_string(aid++);
                    animation.animation_group = ganm.value("name", ""s);
                    auto input_view           = accessor_values(
                        gltf.at("accessors").at(gsampler.value("input", -1)));
                    animation.keyframes_times.resize(input_view.size());
                    for (auto i = 0; i < input_view.size(); i++)
                        animation.keyframes_times[i] = input_view[i][0];
                    auto type = gsampler.value("interpolation", "LINEAR"s);
                    if (type == "LINEAR")
                        animation.interpolation_type =
                            yocto_interpolation_type::linear;
                    if (type == "STEP")
                        animation.interpolation_type =
                            yocto_interpolation_type::step;
                    if (type == "CUBICSPLINE")
                        animation.interpolation_type =
                            yocto_interpolation_type::bezier;
                    auto output_view = accessor_values(
                        gltf.at("accessors").at(gsampler.value("output", -1)));
                    switch (path) {
                        case 0: {  // translation
                            animation.translation_keyframes.reserve(
                                output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                animation.translation_keyframes.push_back(
                                    {(float)output_view[i][0],
                                        (float)output_view[i][1],
                                        (float)output_view[i][2]});
                        } break;
                        case 1: {  // rotation
                            animation.rotation_keyframes.reserve(
                                output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                animation.rotation_keyframes.push_back(
                                    {(float)output_view[i][0],
                                        (float)output_view[i][1],
                                        (float)output_view[i][2],
                                        (float)output_view[i][3]});
                        } break;
                        case 2: {  // scale
                            animation.scale_keyframes.reserve(
                                output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                animation.scale_keyframes.push_back(
                                    {(float)output_view[i][0],
                                        (float)output_view[i][1],
                                        (float)output_view[i][2]});
                        } break;
                        case 3: {  // weights
                            throw io_error("weights not supported for now");
#if 0
                        // get a node that it refers to
                        auto ncomp = 0;
                        auto gnode = gltf->get(gchannel->target->node);
                        auto gmesh = gltf->get(gnode->mesh);
                        if (gmesh) {
                            for (auto gshp : gmesh->primitives) {
                                ncomp = max((int)gshp->targets.size(), ncomp);
                            }
                        }
                        if (ncomp) {
                            auto values = vector<float>();
                            values.reserve(output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                values.push_back(output_view.get(i));
                            animation.weights.resize(values.size() / ncomp);
                            for (auto i = 0; i < animation.weights.size(); i++) {
                                animation.weights[i].resize(ncomp);
                                for (auto j = 0; j < ncomp; j++)
                                    animation.weights[i][j] = values[i * ncomp + j];
                            }
                        }
#endif
                        } break;
                        default: {
                            throw io_error("bad gltf animation");
                        }
                    }
                    sampler_map[{gchannel.at("sampler").get<int>(), path}] =
                        (int)scene.animations.size();
                    scene.animations.push_back(animation);
                }
                scene
                    .animations[sampler_map.at(
                        {gchannel.at("sampler").get<int>(), path})]
                    .node_targets.push_back(
                        (int)gchannel.at("target").at("node").get<int>());
            }
        }
    }
}

// Load a scene
void load_gltf_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    // initialization
    scene = {};

    try {
        // convert json
        auto js = json();
        load_json(filename, js);
        gltf_to_scene(scene, js, get_dirname(filename));

        // load textures
        auto dirname = get_dirname(filename);
        load_scene_textures(scene, dirname, options);

    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }

    // fix scene
    scene.name = get_filename(filename);
    add_missing_cameras(scene);
    add_missing_materials(scene);
    add_missing_names(scene);
    update_transforms(scene);

    // fix cameras
    auto bbox = compute_scene_bounds(scene);
    for (auto& camera : scene.cameras) {
        auto center   = (bbox.min + bbox.max) / 2;
        auto distance = dot(-camera.frame.z, center - camera.frame.o);
        if (distance > 0) camera.focus_distance = distance;
    }
}

// convert gltf scene to json
void scene_to_gltf(const yocto_scene& scene, json& js) {
    // init to emprt object
    js = json::object();

    // start creating json
    js["asset"]["version"] = "2.0";

    // prepare top level nodes
    if (!empty(scene.cameras)) js["cameras"] = json::array();
    if (!empty(scene.textures)) {
        js["textures"] = json::array();
        js["images"]   = json::array();
    }
    if (!empty(scene.materials)) js["materials"] = json::array();
    if (!empty(scene.shapes)) {
        js["meshes"]      = json::array();
        js["buffers"]     = json::array();
        js["bufferViews"] = json::array();
        js["accessors"]   = json::array();
    }
    if (!empty(scene.instances)) js["nodes"] = json::array();
    if (!empty(scene.nodes)) js["nodes"] = json::array();

    // convert cameras
    for (auto& camera : scene.cameras) {
        auto cjs    = json();
        cjs["name"] = camera.name;
        if (!camera.orthographic) {
            cjs["type"]         = "perspective";
            auto& pcjs          = cjs["perspective"];
            pcjs["yfov"]        = get_camera_fovy(camera);
            pcjs["aspectRatio"] = camera.film_width / camera.film_height;
            pcjs["znear"]       = 0.01f;
        } else {
            cjs["type"]   = "orthographic";
            auto& ocjs    = cjs["orthographic"];
            ocjs["xmag"]  = camera.film_width / 2;
            ocjs["ymag"]  = camera.film_height / 2;
            ocjs["znear"] = 0.01f;
        }
        js["cameras"].push_back(cjs);
    }

    // textures
    for (auto& texture : scene.textures) {
        auto tjs = json(), ijs = json();
        tjs["source"] = (int)js["images"].size();
        ijs["uri"]    = texture.filename;
        js["images"].push_back(ijs);
        js["textures"].push_back(tjs);
    }

    // material
    for (auto& material : scene.materials) {
        auto mjs    = json();
        mjs["name"] = material.name;
        if (material.emission != zero3f)
            mjs["emissiveFactor"] = material.emission;
        if (material.emission_texture >= 0)
            mjs["emissiveTexture"]["index"] = material.emission_texture;
        auto kd = vec4f{material.diffuse.x, material.diffuse.y,
            material.diffuse.z, material.opacity};
        if (material.base_metallic) {
            auto mmjs               = json();
            mmjs["baseColorFactor"] = kd;
            mmjs["metallicFactor"]  = material.specular.x;
            mmjs["roughnessFactor"] = material.roughness;
            if (material.diffuse_texture >= 0)
                mmjs["baseColorTexture"]["index"] = material.diffuse_texture;
            if (material.specular_texture >= 0)
                mmjs["metallicRoughnessTexture"]["index"] =
                    material.specular_texture;
            mjs["pbrMetallicRoughness"] = mmjs;
        } else {
            auto mmjs                = json();
            mmjs["diffuseFactor"]    = kd;
            mmjs["specularFactor"]   = material.specular;
            mmjs["glossinessFactor"] = 1 - material.roughness;
            if (material.diffuse_texture >= 0)
                mmjs["diffuseTexture"]["index"] = material.diffuse_texture;
            if (material.specular_texture >= 0)
                mmjs["specularGlossinessTexture"]["index"] =
                    material.specular_texture;
            mjs["extensions"]["KHR_materials_pbrSpecularGlossiness"] = mmjs;
        }
        if (material.normal_texture >= 0)
            mjs["normalTexture"]["index"] = material.normal_texture;
        if (material.occlusion_texture >= 0)
            mjs["occlusionTexture"]["index"] = material.occlusion_texture;
        js["materials"].push_back(mjs);
    }

    // shapes
    for (auto& shape : scene.shapes) {
        auto mjs = json(), bjs = json(), pjs = json();
        auto bid          = js["buffers"].size();
        mjs["name"]       = shape.name;
        mjs["primitives"] = json::array();
        bjs["name"]       = shape.name;
        bjs["byteLength"] = 0;
        bjs["uri"]        = replace_extension(shape.filename, ".bin");
        pjs["material"]   = shape.material;
        auto add_accessor = [&js, &bjs, bid](
                                int count, string type, bool indices = false) {
            auto bytes = count * 4;
            if (type == "VEC2") bytes *= 2;
            if (type == "VEC3") bytes *= 3;
            if (type == "VEC4") bytes *= 4;
            auto ajs = json(), vjs = json();
            vjs["buffer"]        = bid;
            vjs["byteLength"]    = bytes;
            vjs["byteOffset"]    = bjs["byteLength"].get<int>();
            vjs["target"]        = (!indices) ? 34962 : 34963;
            bjs["byteLength"]    = bjs["byteLength"].get<int>() + bytes;
            ajs["bufferView"]    = (int)js["bufferViews"].size();
            ajs["byteOffset"]    = 0;
            ajs["componentType"] = (!indices) ? 5126 : 5125;
            ajs["count"]         = count;
            ajs["type"]          = type;
            js["accessors"].push_back(ajs);
            js["bufferViews"].push_back(vjs);
            return (int)js["accessors"].size() - 1;
        };
        auto nverts = (int)shape.positions.size();
        if (!empty(shape.positions))
            pjs["attributes"]["POSITION"] = add_accessor(nverts, "VEC3");
        if (!empty(shape.normals))
            pjs["attributes"]["NORMAL"] = add_accessor(nverts, "VEC3");
        if (!empty(shape.texturecoords))
            pjs["attributes"]["TEXCOORD_0"] = add_accessor(nverts, "VEC2");
        if (!empty(shape.colors))
            pjs["attributes"]["COLOR_0"] = add_accessor(nverts, "VEC4");
        if (!empty(shape.radius))
            pjs["attributes"]["RADIUS"] = add_accessor(nverts, "SCALAR");
        if (!empty(shape.points)) {
            pjs["indices"] = add_accessor(
                (int)shape.points.size(), "SCALAR", true);
            pjs["mode"] = 1;
        }
        if (!empty(shape.lines)) {
            pjs["indices"] = add_accessor(
                (int)shape.lines.size() * 2, "SCALAR", true);
            pjs["mode"] = 1;
        }
        if (!empty(shape.triangles)) {
            pjs["indices"] = add_accessor(
                (int)shape.triangles.size() * 3, "SCALAR", true);
            pjs["mode"] = 4;
        }
        if (!empty(shape.quads)) {
            auto triangles = vector<vec3i>{};
            convert_quads_to_triangles(triangles, shape.quads);
            pjs["indices"] = add_accessor(
                (int)triangles.size() * 3, "SCALAR", true);
            pjs["mode"] = 4;
        }
        mjs["primitives"].push_back(pjs);
        js["meshes"].push_back(mjs);
        js["buffers"].push_back(bjs);
    }

    // nodes
    for (auto& node : scene.nodes) {
        auto njs           = json();
        njs["name"]        = node.name;
        njs["matrix"]      = frame_to_mat(node.local);
        njs["translation"] = node.translation;
        njs["rotation"]    = node.rotation;
        njs["scale"]       = node.scale;
        if (node.camera >= 0) njs["camera"] = node.camera;
        if (node.instance >= 0)
            njs["mesh"] = scene.instances[node.instance].shape;
        if (!empty(node.children)) {
            njs["children"] = json::array();
            for (auto& c : node.children) njs["children"].push_back(c);
        }
        js["nodes"].push_back(njs);
    }

    // animations not supported yet
    if (!empty(scene.animations)) throw io_error("animation not supported yet");

    // nodes from instances
    if (empty(scene.nodes)) {
        auto camera_id = 0;
        for (auto& camera : scene.cameras) {
            auto njs      = json();
            njs["name"]   = camera.name;
            njs["camera"] = camera_id++;
            njs["matrix"] = frame_to_mat(camera.frame);
            js["nodes"].push_back(njs);
        }
        for (auto& instance : scene.instances) {
            auto njs      = json();
            njs["name"]   = instance.name;
            njs["mesh"]   = instance.shape;
            njs["matrix"] = frame_to_mat(instance.frame);
            js["nodes"].push_back(njs);
        }
    }
}

// save gltf mesh
void save_gltf_mesh(const string& filename, const yocto_shape& shape) {
    // open file
    auto fs = output_file(filename, std::ios::binary);

    write_values(fs, shape.positions);
    write_values(fs, shape.normals);
    write_values(fs, shape.texturecoords);
    write_values(fs, shape.colors);
    write_values(fs, shape.radius);
    write_values(fs, shape.points);
    write_values(fs, shape.lines);
    write_values(fs, shape.triangles);
    auto qtriangles = vector<vec3i>{};
    convert_quads_to_triangles(qtriangles, shape.quads);
    write_values(fs, qtriangles);
}

// Save gltf json
void save_gltf_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    try {
        // save json
        auto js               = json::object();
        js["asset"]           = json::object();
        js["asset"]["format"] = "Yocto/Scene";
        js["asset"]["generator"] =
            "Yocto/GL - https://github.com/xelatihy/yocto-gl";
        scene_to_gltf(scene, js);
        save_json(filename, js);

        // meshes
        auto dirname = get_dirname(filename);
        for (auto& shape : scene.shapes) {
            if (shape.filename == "") continue;
            auto filename = normalize_path(dirname + shape.filename);
            filename      = replace_extension(filename, ".bin");
            save_gltf_mesh(filename, shape);
        }

        // save textures
        save_scene_textures(scene, dirname, options);
    } catch (const std::exception& e) {
        throw io_error("cannot save scene " + filename + "\n" + e.what());
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PBRT
// -----------------------------------------------------------------------------
namespace yocto {

// convert pbrt to json
void pbrt_to_json(const string& filename, json& js) {
    auto split = [](const string& str) {
        auto ret = vector<string>();
        if (empty(str)) return ret;
        auto lpos = (size_t)0;
        while (lpos != str.npos) {
            auto pos = str.find_first_of(" \t\n\r", lpos);
            if (pos != str.npos) {
                if (pos > lpos) ret.push_back(str.substr(lpos, pos - lpos));
                lpos = pos + 1;
            } else {
                if (lpos < str.size()) ret.push_back(str.substr(lpos));
                lpos = pos;
            }
        }
        return ret;
    };

    auto is_cmd = [](const vector<string>& tokens, int i) -> bool {
        auto& tok = tokens.at(i);
        return !(tok[0] == '[' || tok[0] == ']' || tok[0] == '\"' ||
                 tok[0] == '-' || tok[0] == '+' || tok[0] == '.' ||
                 std::isdigit(tok[0]));
    };
    auto is_number = [](const vector<string>& tokens, int i) -> bool {
        auto& tok = tokens.at(i);
        return tok[0] == '-' || tok[0] == '+' || tok[0] == '.' ||
               std::isdigit(tok[0]);
    };
    auto parse_string = [](const vector<string>& tokens, int& i) -> string {
        if (tokens[i][0] != '"') {
            throw io_error("string expected");
            return "";
        }
        auto tok = tokens[i++];
        tok      = tok.substr(1, tok.size() - 2);
        if (tok.find('|') != tok.npos) tok = tok.substr(tok.find('|') + 1);
        return tok;
    };
    auto parse_param = [&](const vector<string>& tokens, int& i,
                           json& js) -> void {
        auto list = false, first = true;
        while (i < tokens.size()) {
            if (is_cmd(tokens, i)) {
                break;
            } else if (tokens[i][0] == '[') {
                list = true;
                i++;
            } else if (tokens[i][0] == ']') {
                list = false;
                i++;
                break;
            } else if (tokens[i][0] == '"') {
                if (!first && !list) break;
                js.push_back(tokens[i].substr(1, tokens[i].size() - 2));
                i++;
                if (!list) break;
            } else {
                if (!first && !list) {
                    throw io_error("bad options");
                    break;
                }
                js.push_back(atof(tokens[i].c_str()));
                i++;
                if (!list) break;
            }
        }
    };
    auto parse_param_list = [&](const vector<string>& tokens, int& i,
                                json& js) -> void {
        while (i < tokens.size()) {
            if (is_cmd(tokens, i)) break;
            auto name = parse_string(tokens, i);
            js[name]  = json::array();
            parse_param(tokens, i, js.at(name));
            if (js.at(name).size() == 1) {
                js.at(name) = js.at(name).at(0);
            }
        }
    };
    auto parse_param_numbers = [&](const vector<string>& tokens, int& i,
                                   json& js) -> void {
        js["values"] = json::array();
        if (tokens[i][0] == '[') i++;
        while (is_number(tokens, i)) {
            js.at("values").push_back((float)atof(tokens[i++].c_str()));
        }
        if (tokens[i][0] == ']') i++;
    };

    auto fs = input_file(filename);

    auto pbrt = ""s;
    auto line = ""s;
    while (read_line(fs, line)) {
        if (line.find('#') == line.npos)
            pbrt += line + "\n";
        else
            pbrt += line.substr(0, line.find('#')) + "\n";
    }

    auto re = std::regex("\"(\\w+)\\s+(\\w+)\"");
    pbrt    = std::regex_replace(pbrt, re, "\"$1|$2\"");
    pbrt    = std::regex_replace(pbrt, std::regex("\\["), " [ ");
    pbrt    = std::regex_replace(pbrt, std::regex("\\]"), " ] ");
    js      = json::array();

    auto tokens = split(pbrt);
    auto i      = 0;
    while (i < tokens.size()) {
        if (!is_cmd(tokens, i)) {
            throw io_error("command expected");
            break;
        }
        auto& tok   = tokens[i++];
        auto  jcmd  = json::object();
        jcmd["cmd"] = tok;
        if (tok == "Transform" || tok == "LookAt" || tok == "Scale" ||
            tok == "Rotate" || tok == "Translate" || tok == "ConcatTransform") {
            parse_param_numbers(tokens, i, jcmd);
        } else if (tok == "Integrator" || tok == "Sampler" ||
                   tok == "PixelFilter" || tok == "Film" || tok == "Camera" ||
                   tok == "Shape" || tok == "AreaLightSource" ||
                   tok == "LightSource") {
            jcmd["type"] = parse_string(tokens, i);
            parse_param_list(tokens, i, jcmd);
        } else if (tok == "Texture") {
            jcmd["name"]       = parse_string(tokens, i);
            jcmd["value_type"] = parse_string(tokens, i);
            jcmd["type"]       = parse_string(tokens, i);
            parse_param_list(tokens, i, jcmd);
        } else if (tok == "MakeNamedMaterial") {
            jcmd["name"] = parse_string(tokens, i);
            parse_param_list(tokens, i, jcmd);
        } else if (tok == "Material") {
            jcmd["type"] = parse_string(tokens, i);
            parse_param_list(tokens, i, jcmd);
        } else if (tok == "NamedMaterial" || tok == "ObjectBegin" ||
                   tok == "ObjectInstance") {
            jcmd["name"] = parse_string(tokens, i);
        } else if (tok == "WorldBegin" || tok == "AttributeBegin" ||
                   tok == "TransformBegin" || tok == "WorldEnd" ||
                   tok == "AttributeEnd" || tok == "TransformEnd" ||
                   tok == "ObjectEnd" || tok == "ReverseOrientation") {
        } else {
            throw io_error("unsupported command " + tok);
        }
        js.push_back(jcmd);
    }
}

// Compute the fresnel term for dielectrics. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3f pbrt_fresnel_dielectric(float cosw, const vec3f& eta_) {
    auto eta = eta_;
    if (cosw < 0) {
        eta  = vec3f{1, 1, 1} / eta;
        cosw = -cosw;
    }

    auto sin2 = 1 - cosw * cosw;
    auto eta2 = eta * eta;

    auto cos2t = vec3f{1, 1, 1} - vec3f{sin2, sin2, sin2} / eta2;
    if (cos2t.x < 0 || cos2t.y < 0 || cos2t.z < 0)
        return vec3f{1, 1, 1};  // tir

    auto t0 = vec3f{sqrt(cos2t.x), sqrt(cos2t.y), sqrt(cos2t.z)};
    auto t1 = eta * t0;
    auto t2 = eta * cosw;

    auto rs = (vec3f{cosw, cosw, cosw} - t1) / (vec3f{cosw, cosw, cosw} + t1);
    auto rp = (t0 - t2) / (t0 + t2);

    return (rs * rs + rp * rp) / 2.0f;
}

// Compute the fresnel term for metals. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3f pbrt_fresnel_metal(float cosw, const vec3f& eta, const vec3f& etak) {
    if (etak == zero3f) return pbrt_fresnel_dielectric(cosw, eta);

    cosw       = clamp(cosw, (float)-1, (float)1);
    auto cos2  = cosw * cosw;
    auto sin2  = clamp(1 - cos2, (float)0, (float)1);
    auto eta2  = eta * eta;
    auto etak2 = etak * etak;

    auto t0         = eta2 - etak2 - vec3f{sin2, sin2, sin2};
    auto a2plusb2_2 = t0 * t0 + 4.0f * eta2 * etak2;
    auto a2plusb2   = vec3f{
        sqrt(a2plusb2_2.x), sqrt(a2plusb2_2.y), sqrt(a2plusb2_2.z)};
    auto t1  = a2plusb2 + vec3f{cos2, cos2, cos2};
    auto a_2 = (a2plusb2 + t0) / 2.0f;
    auto a   = vec3f{sqrt(a_2.x), sqrt(a_2.y), sqrt(a_2.z)};
    auto t2  = 2.0f * a * cosw;
    auto rs  = (t1 - t2) / (t1 + t2);

    auto t3 = vec3f{cos2, cos2, cos2} * a2plusb2 +
              vec3f{sin2, sin2, sin2} * vec3f{sin2, sin2, sin2};
    auto t4 = t2 * sin2;
    auto rp = rs * (t3 - t4) / (t3 + t4);

    return (rp + rs) / 2.0f;
}

// load pbrt scenes
void load_pbrt_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    scene = yocto_scene{};
    // convert to json
    auto js = json();
    try {
        pbrt_to_json(filename, js);
    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }

    auto dirname_ = get_dirname(filename);

    struct stack_item {
        frame3f        frame     = identity_frame3f;
        int            material  = -1;
        yocto_material light_mat = {};
        float          focus = 1, aspect = 1;
        bool           reverse = false;
    };

    // parse
    auto stack = vector<stack_item>();
    stack.push_back(stack_item());

    auto txt_map = unordered_map<string, int>();
    auto mat_map = unordered_map<string, int>();
    auto mid     = 0;

    auto get_vec3f = [](const json& js) -> vec3f {
        if (js.is_number())
            return {js.get<float>(), js.get<float>(), js.get<float>()};
        if (js.is_array() && js.size() == 1)
            return {js.at(0).get<float>(), js.at(0).get<float>(),
                js.at(0).get<float>()};
        if (js.is_array() && js.size() == 3)
            return {js.at(0).get<float>(), js.at(1).get<float>(),
                js.at(2).get<float>()};
        throw io_error("cannot handle vec3f");
        return zero3f;
    };

    auto get_emission_vec3f = [&get_vec3f](const json& js) -> vec3f {
        if (js.is_array() && js.size() == 2)
            return blackbody_to_rgb(js.at(0).get<float>()) *
                   js.at(1).get<float>();
        return get_vec3f(js);
    };

    auto get_vec4f = [](const json& js) -> vec4f {
        if (js.is_number())
            return {js.get<float>(), js.get<float>(), js.get<float>(),
                js.get<float>()};
        if (js.is_array() && js.size() == 4)
            return {js.at(0).get<float>(), js.at(1).get<float>(),
                js.at(2).get<float>(), js.at(3).get<float>()};
        throw io_error("cannot handle vec4f");
        return zero4f;
    };

    auto get_mat4f = [](const json& js) -> frame3f {
        if (!js.is_array() || js.size() != 16) {
            throw io_error("cannot handle vec4f");
            return identity_frame3f;
        }
        float m[16] = {0};
        for (auto i = 0; i < 16; i++) m[i] = js.at(i).get<float>();
        return {{m[0], m[1], m[2]}, {m[4], m[5], m[6]}, {m[8], m[9], m[10]},
            {m[12], m[13], m[14]}};
    };

    auto get_mat3f = [](const json& js) -> frame3f {
        if (!js.is_array() || js.size() != 9) {
            throw io_error("cannot handle mat3f");
            return identity_frame3f;
        }
        auto m = identity_frame3f;
        for (auto i = 0; i < 9; i++) (&m.x.x)[i] = js.at(i).get<float>();
        return m;
    };

    auto get_vector_vec3i = [](const json& js) -> vector<vec3i> {
        if (!js.is_array() || js.size() % 3) {
            throw io_error("cannot handle vector<vec3f");
            return {};
        }
        auto vals = vector<vec3i>(js.size() / 3);
        for (auto i = 0; i < vals.size(); i++) {
            vals[i].x = (int)std::round(js.at(i * 3 + 0).get<float>());
            vals[i].y = (int)std::round(js.at(i * 3 + 1).get<float>());
            vals[i].z = (int)std::round(js.at(i * 3 + 2).get<float>());
        }
        return vals;
    };

    auto get_vector_vec3f = [](const json& js) -> vector<vec3f> {
        if (!js.is_array() || js.size() % 3) {
            throw io_error("cannot handle vector<vec3f>");
            return {};
        }
        auto vals = vector<vec3f>(js.size() / 3);
        for (auto i = 0; i < vals.size(); i++) {
            vals[i].x = js.at(i * 3 + 0).get<float>();
            vals[i].y = js.at(i * 3 + 1).get<float>();
            vals[i].z = js.at(i * 3 + 2).get<float>();
        }
        return vals;
    };

    auto get_vector_vec2f = [](const json& js) -> vector<vec2f> {
        if (!js.is_array() || js.size() % 2) {
            throw io_error("cannot handle vector<vec3f>");
            return {};
        }
        auto vals = vector<vec2f>(js.size() / 2);
        for (auto i = 0; i < vals.size(); i++) {
            vals[i].x = js.at(i * 2 + 0).get<float>();
            vals[i].y = js.at(i * 2 + 1).get<float>();
        }
        return vals;
    };

    auto get_scaled_texture = [&txt_map, &get_vec3f](
                                  const json& js, vec3f& col, int& txt) {
        if (js.is_string()) {
            col = {1, 1, 1};
            txt = txt_map.at(js.get<string>());
        } else {
            col = get_vec3f(js);
            txt = -1;
        }
    };

    unordered_map<string, vector<yocto_instance>> objects;
    auto                                          lid = 0, sid = 0, cid = 0;
    auto                                          cur_object = ""s;
    for (auto& jcmd : js) {
        auto cmd = jcmd.at("cmd").get<string>();
        if (cmd == "Integrator" || cmd == "Sampler" || cmd == "PixelFilter") {
        } else if (cmd == "Transform") {
            stack.back().frame = get_mat4f(jcmd.at("values"));
        } else if (cmd == "ConcatTransform") {
            stack.back().frame = stack.back().frame *
                                 get_mat4f(jcmd.at("values"));
        } else if (cmd == "Scale") {
            auto v             = get_vec3f(jcmd.at("values"));
            stack.back().frame = stack.back().frame * make_scaling_frame(v);
        } else if (cmd == "Translate") {
            auto v             = get_vec3f(jcmd.at("values"));
            stack.back().frame = stack.back().frame * make_translation_frame(v);
        } else if (cmd == "Rotate") {
            auto v             = get_vec4f(jcmd.at("values"));
            stack.back().frame = stack.back().frame *
                                 make_rotation_frame(
                                     vec3f{v.y, v.z, v.w}, v.x * pif / 180);
        } else if (cmd == "LookAt") {
            auto m             = get_mat3f(jcmd.at("values"));
            stack.back().frame = stack.back().frame * inverse(make_lookat_frame(
                                                          m.x, m.y, m.z, true));
            stack.back().focus = length(m.x - m.y);
        } else if (cmd == "ReverseOrientation") {
            stack.back().reverse = !stack.back().reverse;
        } else if (cmd == "Film") {
            stack.back().aspect = jcmd.at("xresolution").get<float>() /
                                  jcmd.at("yresolution").get<float>();
        } else if (cmd == "Camera") {
            auto camera    = yocto_camera{};
            camera.name    = "camera" + std::to_string(cid++);
            camera.frame   = inverse(stack.back().frame);
            camera.frame.z = -camera.frame.z;
            auto focus     = stack.back().focus;
            auto aspect    = stack.back().aspect;
            auto fovy      = 1.0f;
            auto type      = jcmd.at("type").get<string>();
            if (type == "perspective") {
                fovy = jcmd.at("fov").get<float>() * pif / 180;
                if (aspect < 1) fovy = atan(tan(fovy) / aspect);
            } else {
                throw io_error("camera not supported " + type);
            }
            set_camera_perspective(camera, fovy, aspect, focus);
            scene.cameras.push_back(camera);
        } else if (cmd == "Texture") {
            auto found = false;
            auto name  = jcmd.at("name").get<string>();
            for (auto& texture : scene.textures) {
                if (texture.name == name) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                scene.textures.push_back({});
                auto& texture         = scene.textures.back();
                texture.name          = jcmd.at("name").get<string>();
                txt_map[texture.name] = (int)scene.textures.size() - 1;
                auto type             = jcmd.at("type").get<string>();
                if (type == "imagemap") {
                    texture.filename = jcmd.at("filename").get<string>();
                    if (get_extension(texture.filename) == "pfm")
                        texture.filename = replace_extension(texture.filename,
                            ".hd"
                            "r");
                } else {
                    throw io_error("texture not supported " + type);
                }
            }
        } else if (cmd == "MakeNamedMaterial" || cmd == "Material") {
            auto found = false;
            if (cmd == "MakeNamedMaterial") {
                auto name = jcmd.at("name").get<string>();
                for (auto& material : scene.materials) {
                    if (material.name == name) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                scene.materials.push_back({});
                auto& material = scene.materials.back();
                if (cmd == "Material") {
                    material.name = "unnamed_mat" + std::to_string(mid++);
                    stack.back().material = (int)scene.materials.size() - 1;
                } else {
                    material.name          = jcmd.at("name").get<string>();
                    mat_map[material.name] = (int)scene.materials.size() - 1;
                }
                auto type = "uber"s;
                if (jcmd.count("type")) type = jcmd.at("type").get<string>();
                if (type == "uber") {
                    if (jcmd.count("Kd"))
                        get_scaled_texture(jcmd.at("Kd"), material.diffuse,
                            material.diffuse_texture);
                    if (jcmd.count("Ks"))
                        get_scaled_texture(jcmd.at("Ks"), material.specular,
                            material.specular_texture);
                    if (jcmd.count("Kt"))
                        get_scaled_texture(jcmd.at("Kt"), material.transmission,
                            material.transmission_texture);
                    if (jcmd.count("opacity")) {
                        auto op     = vec3f{0, 0, 0};
                        auto op_txt = -1;
                        get_scaled_texture(jcmd.at("opacity"), op, op_txt);
                        material.opacity         = (op.x + op.y + op.z) / 3;
                        material.opacity_texture = op_txt;
                    }
                    material.roughness = 0;
                } else if (type == "plastic") {
                    if (jcmd.count("Kd"))
                        get_scaled_texture(jcmd.at("Kd"), material.diffuse,
                            material.diffuse_texture);
                    if (jcmd.count("Ks"))
                        get_scaled_texture(jcmd.at("Ks"), material.specular,
                            material.specular_texture);
                    material.roughness = 0;
                } else if (type == "translucent") {
                    if (jcmd.count("Kd"))
                        get_scaled_texture(jcmd.at("Kd"), material.diffuse,
                            material.diffuse_texture);
                    if (jcmd.count("Ks"))
                        get_scaled_texture(jcmd.at("Ks"), material.specular,
                            material.specular_texture);
                    material.roughness = 0;
                } else if (type == "mix") {
                    auto matname1 = (jcmd.count("namedmaterial1")) ?
                                        jcmd.at("namedmaterial1").get<string>() :
                                        ""s;
                    auto matname2 = (jcmd.count("namedmaterial2")) ?
                                        jcmd.at("namedmaterial2").get<string>() :
                                        ""s;
                    // auto amount = get_vec3f(jcmd.at("amount"));
                    auto matname = (!matname1.empty()) ? matname1 : matname2;
                    for (auto& mat : scene.materials) {
                        if (mat.name == matname) {
                            material = mat;
                            break;
                        }
                    }
                } else if (type == "matte") {
                    material.diffuse = {1, 1, 1};
                    if (jcmd.count("Kd"))
                        get_scaled_texture(jcmd.at("Kd"), material.diffuse,
                            material.diffuse_texture);
                    material.roughness = 1;
                } else if (type == "mirror") {
                    material.diffuse   = {0, 0, 0};
                    material.specular  = {1, 1, 1};
                    material.roughness = 0;
                } else if (type == "metal") {
                    auto eta           = get_vec3f(jcmd.at("eta"));
                    auto k             = get_vec3f(jcmd.at("k"));
                    material.specular  = pbrt_fresnel_metal(1, eta, k);
                    material.roughness = 0;
                } else if (type == "substrate") {
                    if (jcmd.count("Kd"))
                        get_scaled_texture(jcmd.at("Kd"), material.diffuse,
                            material.diffuse_texture);
                    material.specular = {0.04f, 0.04f, 0.04f};
                    if (jcmd.count("Ks"))
                        get_scaled_texture(jcmd.at("Ks"), material.specular,
                            material.specular_texture);
                    material.roughness = 0;
                } else if (type == "glass") {
                    material.specular     = {0.04f, 0.04f, 0.04f};
                    material.transmission = {1, 1, 1};
                    if (jcmd.count("Ks"))
                        get_scaled_texture(jcmd.at("Ks"), material.specular,
                            material.specular_texture);
                    if (jcmd.count("Kt"))
                        get_scaled_texture(jcmd.at("Kt"), material.transmission,
                            material.transmission_texture);
                    material.roughness = 0;
                } else if (type == "mix") {
                    printf("mix material not properly supported\n");
                    if (jcmd.count("namedmaterial1")) {
                        auto mat1 = jcmd.at("namedmaterial1").get<string>();
                        auto saved_name = material.name;
                        material        = scene.materials[mat_map.at(mat1)];
                        material.name   = saved_name;
                    } else {
                        throw io_error("mix material missing front material");
                    }
                } else {
                    material.diffuse = {1, 0, 0};
                    throw io_error("material not supported " + type);
                }
                if (jcmd.count("uroughness")) {
                    auto remap = js.count("remaproughness") &&
                                 js.at("remaproughness").get<bool>();
                    if (jcmd.count("uroughness"))
                        material.roughness = jcmd.at("uroughness").get<float>();
                    // if (!remap) material.rs = material.rs * material.rs;
                    if (remap) throw io_error("remap roughness not supported");
                }
                if (jcmd.count("roughness")) {
                    auto remap = js.count("remaproughness") &&
                                 js.at("remaproughness").get<bool>();
                    if (jcmd.count("roughness"))
                        material.roughness = jcmd.at("roughness").get<float>();
                    // if (!remap) material.rs = material.rs * material.rs;
                    if (remap) throw io_error("remap roughness not supported");
                }
                if (stack.back().light_mat.emission != zero3f) {
                    material.emission = stack.back().light_mat.emission;
                    material.emission_texture =
                        stack.back().light_mat.emission_texture;
                }
            }
        } else if (cmd == "NamedMaterial") {
            stack.back().material = mat_map.at(jcmd.at("name").get<string>());
            if (stack.back().light_mat.emission != zero3f) {
                auto material = yocto_material(
                    scene.materials[stack.back().material]);
                material.name += "_" + std::to_string(lid++);
                material.emission = stack.back().light_mat.emission;
                material.emission_texture =
                    stack.back().light_mat.emission_texture;
                scene.materials.push_back(material);
                stack.back().material = (int)scene.materials.size() - 1;
            }
        } else if (cmd == "Shape") {
            auto shape = yocto_shape();
            auto type  = jcmd.at("type").get<string>();
            if (type == "plymesh") {
                auto filename  = jcmd.at("filename").get<string>();
                shape.name     = get_filename(filename);
                shape.filename = filename;
                if (!options.skip_meshes) {
                    load_ply_mesh(dirname_ + filename, shape.points,
                        shape.lines, shape.triangles, shape.quads,
                        shape.positions, shape.normals, shape.texturecoords,
                        shape.colors, shape.radius, false);
                }
            } else if (type == "trianglemesh") {
                shape.name     = "mesh" + std::to_string(sid++);
                shape.filename = "models/" + shape.name + ".ply";
                if (jcmd.count("indices"))
                    shape.triangles = get_vector_vec3i(jcmd.at("indices"));
                if (jcmd.count("P"))
                    shape.positions = get_vector_vec3f(jcmd.at("P"));
                if (jcmd.count("N"))
                    shape.normals = get_vector_vec3f(jcmd.at("N"));
                if (jcmd.count("uv")) {
                    shape.texturecoords = get_vector_vec2f(jcmd.at("uv"));
                    for (auto& uv : shape.texturecoords) uv.y = 1 - uv.y;
                }
            } else if (type == "sphere") {
                shape.name     = "sphere" + std::to_string(sid++);
                shape.filename = "models/" + shape.name + ".ply";
                auto radius    = 1.0f;
                if (jcmd.count("radius"))
                    radius = jcmd.at("radius").get<float>();
                make_uvsphere_shape(shape.quads, shape.positions, shape.normals,
                    shape.texturecoords, {64, 32}, 2 * radius, {1, 1});
            } else if (type == "disk") {
                shape.name     = "disk" + std::to_string(sid++);
                shape.filename = "models/" + shape.name + ".ply";
                auto radius    = 1.0f;
                if (jcmd.count("radius"))
                    radius = jcmd.at("radius").get<float>();
                make_uvdisk_shape(shape.quads, shape.positions, shape.normals,
                    shape.texturecoords, {32, 16}, 2 * radius, {1, 1});
            } else {
                throw io_error("shape not supported " + type);
            }
            auto frame = stack.back().frame;
            auto scl = vec3f{length(frame.x), length(frame.y), length(frame.z)};
            for (auto& p : shape.positions) p *= scl;
            frame = {normalize(frame.x), normalize(frame.y), normalize(frame.z),
                frame.o};
            if (stack.back().reverse) {
                for (auto& t : shape.triangles) swap(t.y, t.z);
            }
            shape.material = stack.back().material;
            scene.shapes.push_back(shape);
            auto instance  = yocto_instance();
            instance.name  = shape.name;
            instance.frame = frame;
            instance.shape = (int)scene.shapes.size() - 1;
            if (cur_object != "") {
                objects[cur_object].push_back(instance);
            } else {
                scene.instances.push_back(instance);
            }
        } else if (cmd == "ObjectInstance") {
            static auto instances = unordered_map<string, int>();
            auto        name      = jcmd.at("name").get<string>();
            auto&       object    = objects.at(name);
            for (auto shape : object) {
                instances[shape.name] += 1;
                auto instance = yocto_instance();
                instance.name = shape.name + "_ist" +
                                std::to_string(instances[shape.name]);
                instance.frame = stack.back().frame * shape.frame;
                instance.shape = shape.shape;
                scene.instances.push_back(instance);
            }
        } else if (cmd == "AreaLightSource") {
            auto type = jcmd.at("type").get<string>();
            if (type == "diffuse") {
                auto ligth_mat         = yocto_material();
                ligth_mat.emission     = get_emission_vec3f(jcmd.at("L"));
                stack.back().light_mat = ligth_mat;
            } else {
                throw io_error("area light not supported " + type);
            }
        } else if (cmd == "LightSource") {
            auto type = jcmd.at("type").get<string>();
            if (type == "infinite") {
                auto environment = yocto_environment();
                environment.name = "environment" + std::to_string(lid++);
                // environment.frame =
                // frame3f{{1,0,0},{0,0,-1},{0,-1,0},{0,0,0}}
                // * stack.back().frame;
                environment.frame = stack.back().frame *
                                    frame3f{{1, 0, 0}, {0, 0, 1}, {0, 1, 0},
                                        {0, 0, 0}};
                environment.emission = {1, 1, 1};
                if (jcmd.count("scale"))
                    environment.emission *= get_vec3f(jcmd.at("scale"));
                if (jcmd.count("mapname")) {
                    auto texture     = yocto_texture{};
                    texture.filename = jcmd.at("mapname").get<string>();
                    texture.name     = environment.name;
                    scene.textures.push_back(texture);
                    environment.emission_texture = (int)scene.textures.size() -
                                                   1;
                }
                scene.environments.push_back(environment);
            } else if (type == "distant") {
                auto distant_dist = 100;
                scene.shapes.push_back({});
                auto& shape = scene.shapes.back();
                shape.name  = "distant" + std::to_string(lid++);
                auto from = vec3f{0, 0, 0}, to = vec3f{0, 0, 0};
                if (jcmd.count("from")) from = get_vec3f(jcmd.at("from"));
                if (jcmd.count("to")) to = get_vec3f(jcmd.at("to"));
                auto dir  = normalize(from - to);
                auto size = distant_dist * sin(5 * pif / 180);
                make_quad_shape(shape.quads, shape.positions, shape.normals,
                    shape.texturecoords, {1, 1}, {size, size}, {1, 1});
                scene.materials.push_back({});
                auto& material    = scene.materials.back();
                shape.material    = scene.materials.size() - 1;
                material.name     = shape.name;
                material.emission = {1, 1, 1};
                if (jcmd.count("L"))
                    material.emission *= get_emission_vec3f(jcmd.at("L"));
                if (jcmd.count("scale"))
                    material.emission *= get_vec3f(jcmd.at("scale"));
                material.emission *= (distant_dist * distant_dist) /
                                     (size * size);
                auto instance  = yocto_instance();
                instance.name  = shape.name;
                instance.shape = (int)scene.shapes.size() - 1;
                instance.frame = stack.back().frame *
                                 make_lookat_frame(dir * distant_dist, zero3f,
                                     {0, 1, 0}, true);
                scene.instances.push_back(instance);
            } else {
                throw io_error("light not supported " + type);
            }
        } else if (cmd == "WorldBegin") {
            stack.push_back(stack_item());
        } else if (cmd == "AttributeBegin") {
            stack.push_back(stack.back());
        } else if (cmd == "ObjectBegin") {
            auto name     = jcmd.at("name").get<string>();
            cur_object    = name;
            objects[name] = {};
        } else if (cmd == "ObjectEnd") {
            cur_object = "";
        } else if (cmd == "TransformBegin") {
            stack.push_back(stack.back());
        } else if (cmd == "WorldEnd" || cmd == "AttributeEnd" ||
                   cmd == "TransformEnd") {
            stack.pop_back();
        } else {
            throw io_error("command not supported " + cmd);
        }
    }

    // load textures
    auto dirname = get_dirname(filename);
    load_scene_textures(scene, dirname, options);

    // fix scene
    scene.name = get_filename(filename);
    add_missing_cameras(scene);
    add_missing_materials(scene);
    add_missing_names(scene);
    update_transforms(scene);
}

// Convert a scene to pbrt format
void save_pbrt(const string& filename, const yocto_scene& scene) {
    auto fs = output_file(filename);

    // convert camera and settings
    auto& camera         = scene.cameras.front();
    auto  from           = camera.frame.o;
    auto  to             = camera.frame.o - camera.frame.z;
    auto  up             = camera.frame.y;
    auto [width, height] = get_camera_image_size(camera, 0, 720);
    println_values(fs, "LookAt", from, to, up);
    println_values(fs, "Camera \"perspective\" \"float fov\"",
        get_camera_fovy(camera) * 180 / pif);

    // save renderer
    println_values(fs, "Sampler \"random\" \"integer pixelsamples\" [64]\n");
    // fprintf(f, "Sampler \"sobol\" \"interger pixelsamples\" [64]\n");
    println_values(fs, "Integrator \"path\"\n");
    println_values(fs,
        "Film \"image\" \"string filename\" [\"{}\"] "
        "\"integer xresolution\" [{}] \"integer yresolution\" [{}]\n",
        replace_extension(filename, "exr"), width, height);

    // start world
    println_values(fs, "WorldBegin\n");

    // convert textures
    for (auto& texture : scene.textures) {
        println_values(fs,
            "Texture \"{}\" \"spectrum\" \"imagemap\" "
            "\"string filename\" [\"{}\"]\n",
            texture.name, texture.filename);
    }

    // convert materials
    for (auto& material : scene.materials) {
        println_values(fs, "MakeNamedMaterial \"{}\" ", material.name);
        println_values(fs, "\"string type\" \"{}\" ", "uber");
        if (material.diffuse_texture >= 0)
            println_values(fs, "\"texture Kd\" [\"{}\"] ",
                scene.textures[material.diffuse_texture].name);
        else
            println_values(fs, "\"rgb Kd\" [{}] ", material.diffuse);
        if (material.specular_texture >= 0)
            println_values(fs, "\"texture Ks\" [\"{}\"] ",
                scene.textures[material.specular_texture].name);
        else
            println_values(fs, "\"rgb Ks\" [{}] ", material.specular);
        println_values(fs, "\"float roughness\" [{}] ", material.roughness);
        println_values(fs, "\n");
    }

    // convert instances
    for (auto& instance : scene.instances) {
        auto& shape    = scene.shapes[instance.shape];
        auto& material = scene.materials[shape.material];
        println_values(fs, "AttributeBegin\n");
        println_values(fs, "TransformBegin\n");
        println_values(
            fs, "ConcatTransform [{}]\n", frame_to_mat(instance.frame));
        if (material.emission != zero3f)
            println_values(fs, "AreaLightSource \"diffuse\" \"rgb L\" [ {} ]\n",
                material.emission);
        println_values(fs, "NamedMaterial \"{}\"\n", material.name);
        println_values(fs, "Shape \"plymesh\" \"string filename\" [\"{}\"]\n",
            shape.filename);
        println_values(fs, "TransformEnd\n");
        println_values(fs, "AttributeEnd\n");
    }

    // end world
    println_values(fs, "WorldEnd\n");
}

// Save a pbrt scene
void save_pbrt_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    try {
        // save json
        save_pbrt(filename, scene);

        // save meshes
        auto dirname = get_dirname(filename);
        for (auto& shape : scene.shapes) {
            if (shape.filename == "") continue;
            auto filename = normalize_path(dirname + shape.filename);
            save_mesh(filename, shape.points, shape.lines, shape.triangles,
                shape.quads, shape.positions, shape.normals,
                shape.texturecoords, shape.colors, shape.radius);
        }

        // skip textures
        save_scene_textures(scene, dirname, options);

    } catch (const std::exception& e) {
        throw io_error("cannot save scene " + filename + "\n" + e.what());
    }
}

// Attempt to fix pbrt z-up.
void pbrt_flipyz_scene(yocto_scene& scene) {
    // flip meshes
    for (auto& shape : scene.shapes) {
        for (auto& p : shape.positions) swap(p.y, p.z);
        for (auto& n : shape.normals) swap(n.y, n.z);
    }
    for (auto& instance : scene.instances) {
        instance.frame = instance.frame *
                         frame3f{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0, 0}};
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF BINARY SCENE FORMAT
// -----------------------------------------------------------------------------
namespace yocto {

// serialize_bin( ) can both save/load data to/from a binary file. The behaviour
// is set by the boolean 'save'. serialize_bin(name, file, true) : writes name
// as binary into file serialize_bin(name, file, false): read file as binary and
// set name

// Serialize vector
template <typename T>
void write_value(output_file& fs, const vector<T>& vec) {
    auto count = (size_t)vec.size();
    write_value(fs, count);
    write_values(fs, vec);
}
template <typename T>
void read_value(input_file& fs, vector<T>& vec) {
    auto count = (size_t)0;
    read_value(fs, count);
    vec = vector<T>(count);
    read_values(fs, vec);
}

// Serialize string
void write_value(output_file& fs, const string& str) {
    auto count = (size_t)str.size();
    write_value(fs, count);
    auto vec = vector<char>(str.begin(), str.end());
    write_values(fs, vec);
}
void read_value(input_file& fs, string& str) {
    auto count = (size_t)0;
    read_value(fs, count);
    auto vec = vector<char>(count);
    read_values(fs, vec);
    str = {vec.begin(), vec.end()};
}

// Serialize image
template <typename T>
void write_value(output_file& fs, const image<T>& img) {
    write_value(fs, img.width);
    write_value(fs, img.height);
    write_values(fs, img.pixels);
}
template <typename T>
void read_value(input_file& fs, image<T>& img) {
    read_value(fs, img.width);
    read_value(fs, img.height);
    img.pixels.resize(img.width * img.height);
    read_values(fs, img.pixels);
}

// Serialize image
template <typename T>
void write_value(output_file& fs, const volume<T>& vol) {
    write_value(fs, vol.width);
    write_value(fs, vol.height);
    write_value(fs, vol.depth);
    write_values(fs, vol.voxels);
}
template <typename T>
void read_value(input_file& fs, volume<T>& vol) {
    read_value(fs, vol.width);
    read_value(fs, vol.height);
    read_value(fs, vol.depth);
    vol.voxels.resize(vol.width * vol.height * vol.depth);
    read_values(fs, vol.voxels);
}

// Serialize vector of pointers
template <typename T>
void write_objects(output_file& fs, const vector<T>& vec) {
    auto count = (size_t)vec.size();
    write_value(fs, count);
    for (auto i = 0; i < vec.size(); ++i) {
        write_object(fs, vec[i]);
    }
}
template <typename T>
void read_objects(input_file& fs, vector<T>& vec) {
    auto count = (size_t)0;
    read_value(fs, count);
    vec = vector<T>(count);
    for (auto i = 0; i < vec.size(); ++i) {
        vec[i] = T{};
        read_object(fs, vec[i]);
    }
}

// Serialize yocto types. This is mostly boiler plate code.
void write_object(output_file& fs, const yocto_camera& camera) {
    write_value(fs, camera.name);
    write_value(fs, camera.frame);
    write_value(fs, camera.orthographic);
    write_value(fs, camera.film_width);
    write_value(fs, camera.film_height);
    write_value(fs, camera.focal_length);
    write_value(fs, camera.focus_distance);
    write_value(fs, camera.lens_aperture);
}
void read_object(input_file& fs, yocto_camera& camera) {
    read_value(fs, camera.name);
    read_value(fs, camera.frame);
    read_value(fs, camera.orthographic);
    read_value(fs, camera.film_width);
    read_value(fs, camera.film_height);
    read_value(fs, camera.focal_length);
    read_value(fs, camera.focus_distance);
    read_value(fs, camera.lens_aperture);
}

void write_object(output_file& fs, const bvh_shape& bvh) {
    write_value(fs, bvh.positions);
    write_value(fs, bvh.radius);
    write_value(fs, bvh.points);
    write_value(fs, bvh.lines);
    write_value(fs, bvh.triangles);
    write_value(fs, bvh.quads);
    write_value(fs, bvh.nodes);
    write_value(fs, bvh.nodes);
}
void read_object(input_file& fs, bvh_shape& bvh) {
    read_value(fs, bvh.positions);
    read_value(fs, bvh.radius);
    read_value(fs, bvh.points);
    read_value(fs, bvh.lines);
    read_value(fs, bvh.triangles);
    read_value(fs, bvh.quads);
    read_value(fs, bvh.nodes);
    read_value(fs, bvh.nodes);
}

void write_object(output_file& fs, const bvh_scene& bvh) {
    write_value(fs, bvh.nodes);
    write_value(fs, bvh.instances);
    write_objects(fs, bvh.shape_bvhs);
    write_value(fs, bvh.nodes);
}
void read_object(input_file& fs, bvh_scene& bvh) {
    read_value(fs, bvh.nodes);
    read_value(fs, bvh.instances);
    read_objects(fs, bvh.shape_bvhs);
    read_value(fs, bvh.nodes);
}

void write_object(output_file& fs, const yocto_shape& shape) {
    write_value(fs, shape.name);
    write_value(fs, shape.filename);
    write_value(fs, shape.material);
    write_value(fs, shape.subdivision_level);
    write_value(fs, shape.catmull_clark);
    write_value(fs, shape.compute_vertex_normals);
    write_value(fs, shape.points);
    write_value(fs, shape.lines);
    write_value(fs, shape.triangles);
    write_value(fs, shape.quads);
    write_value(fs, shape.positions);
    write_value(fs, shape.normals);
    write_value(fs, shape.texturecoords);
    write_value(fs, shape.colors);
    write_value(fs, shape.radius);
    write_value(fs, shape.tangentspaces);
}
void read_object(input_file& fs, yocto_shape& shape) {
    read_value(fs, shape.name);
    read_value(fs, shape.filename);
    read_value(fs, shape.material);
    read_value(fs, shape.subdivision_level);
    read_value(fs, shape.catmull_clark);
    read_value(fs, shape.compute_vertex_normals);
    read_value(fs, shape.points);
    read_value(fs, shape.lines);
    read_value(fs, shape.triangles);
    read_value(fs, shape.quads);
    read_value(fs, shape.positions);
    read_value(fs, shape.normals);
    read_value(fs, shape.texturecoords);
    read_value(fs, shape.colors);
    read_value(fs, shape.radius);
    read_value(fs, shape.tangentspaces);
}

void write_object(output_file& fs, const yocto_surface& surface) {
    write_value(fs, surface.name);
    write_value(fs, surface.filename);
    write_value(fs, surface.materials);
    write_value(fs, surface.subdivision_level);
    write_value(fs, surface.catmull_clark);
    write_value(fs, surface.compute_vertex_normals);
    write_value(fs, surface.quads_positions);
    write_value(fs, surface.quads_normals);
    write_value(fs, surface.quads_texturecoords);
    write_value(fs, surface.quads_materials);
    write_value(fs, surface.positions);
    write_value(fs, surface.normals);
    write_value(fs, surface.texturecoords);
}
void read_object(input_file& fs, yocto_surface& surface) {
    read_value(fs, surface.name);
    read_value(fs, surface.filename);
    read_value(fs, surface.materials);
    read_value(fs, surface.subdivision_level);
    read_value(fs, surface.catmull_clark);
    read_value(fs, surface.compute_vertex_normals);
    read_value(fs, surface.quads_positions);
    read_value(fs, surface.quads_normals);
    read_value(fs, surface.quads_texturecoords);
    read_value(fs, surface.quads_materials);
    read_value(fs, surface.positions);
    read_value(fs, surface.normals);
    read_value(fs, surface.texturecoords);
}

void write_object(output_file& fs, const yocto_texture& texture) {
    write_value(fs, texture.name);
    write_value(fs, texture.filename);
    write_value(fs, texture.hdr_image);
    write_value(fs, texture.ldr_image);
    write_value(fs, texture.clamp_to_edge);
    write_value(fs, texture.height_scale);
    write_value(fs, texture.no_interpolation);
    write_value(fs, texture.ldr_as_linear);
}
void read_object(input_file& fs, yocto_texture& texture) {
    read_value(fs, texture.name);
    read_value(fs, texture.filename);
    read_value(fs, texture.hdr_image);
    read_value(fs, texture.ldr_image);
    read_value(fs, texture.clamp_to_edge);
    read_value(fs, texture.height_scale);
    read_value(fs, texture.no_interpolation);
    read_value(fs, texture.ldr_as_linear);
}

void write_object(output_file& fs, const yocto_voltexture& texture) {
    write_value(fs, texture.name);
    write_value(fs, texture.filename);
    write_value(fs, texture.volume_data);
    write_value(fs, texture.clamp_to_edge);
}
void read_object(input_file& fs, yocto_voltexture& texture) {
    read_value(fs, texture.name);
    read_value(fs, texture.filename);
    read_value(fs, texture.volume_data);
    read_value(fs, texture.clamp_to_edge);
}

void write_object(output_file& fs, const yocto_environment& environment) {
    write_value(fs, environment.name);
    write_value(fs, environment.frame);
    write_value(fs, environment.emission);
    write_value(fs, environment.emission_texture);
}
void read_object(input_file& fs, yocto_environment& environment) {
    read_value(fs, environment.name);
    read_value(fs, environment.frame);
    read_value(fs, environment.emission);
    read_value(fs, environment.emission_texture);
}

void write_object(output_file& fs, const yocto_material& material) {
    write_value(fs, material.name);
    write_value(fs, material.base_metallic);
    write_value(fs, material.gltf_textures);
    write_value(fs, material.emission);
    write_value(fs, material.diffuse);
    write_value(fs, material.specular);
    write_value(fs, material.transmission);
    write_value(fs, material.roughness);
    write_value(fs, material.opacity);
    write_value(fs, material.fresnel);
    write_value(fs, material.refract);
    write_value(fs, material.emission_texture);
    write_value(fs, material.diffuse_texture);
    write_value(fs, material.specular_texture);
    write_value(fs, material.transmission_texture);
    write_value(fs, material.roughness_texture);
    write_value(fs, material.opacity_texture);
    write_value(fs, material.occlusion_texture);
    write_value(fs, material.bump_texture);
    write_value(fs, material.displacement_texture);
    write_value(fs, material.normal_texture);
    write_value(fs, material.volume_emission);
    write_value(fs, material.volume_albedo);
    write_value(fs, material.volume_density);
    write_value(fs, material.volume_phaseg);
    write_value(fs, material.volume_density_texture);
};
void read_object(input_file& fs, yocto_material& material) {
    read_value(fs, material.name);
    read_value(fs, material.base_metallic);
    read_value(fs, material.gltf_textures);
    read_value(fs, material.emission);
    read_value(fs, material.diffuse);
    read_value(fs, material.specular);
    read_value(fs, material.transmission);
    read_value(fs, material.roughness);
    read_value(fs, material.opacity);
    read_value(fs, material.fresnel);
    read_value(fs, material.refract);
    read_value(fs, material.emission_texture);
    read_value(fs, material.diffuse_texture);
    read_value(fs, material.specular_texture);
    read_value(fs, material.transmission_texture);
    read_value(fs, material.roughness_texture);
    read_value(fs, material.opacity_texture);
    read_value(fs, material.occlusion_texture);
    read_value(fs, material.bump_texture);
    read_value(fs, material.displacement_texture);
    read_value(fs, material.normal_texture);
    read_value(fs, material.volume_emission);
    read_value(fs, material.volume_albedo);
    read_value(fs, material.volume_density);
    read_value(fs, material.volume_phaseg);
    read_value(fs, material.volume_density_texture);
};

void write_object(output_file& fs, const yocto_instance& instance) {
    write_value(fs, instance.name);
    write_value(fs, instance.frame);
    write_value(fs, instance.shape);
    write_value(fs, instance.surface);
};
void read_object(input_file& fs, yocto_instance& instance) {
    read_value(fs, instance.name);
    read_value(fs, instance.frame);
    read_value(fs, instance.shape);
    read_value(fs, instance.surface);
};

void write_object(output_file& fs, const yocto_scene& scene) {
    write_value(fs, scene.name);
    write_objects(fs, scene.cameras);
    write_objects(fs, scene.shapes);
    write_objects(fs, scene.surfaces);
    write_objects(fs, scene.textures);
    write_objects(fs, scene.voltextures);
    write_objects(fs, scene.materials);
    write_objects(fs, scene.instances);
    write_objects(fs, scene.environments);
}
void read_object(input_file& fs, yocto_scene& scene) {
    read_value(fs, scene.name);
    read_objects(fs, scene.cameras);
    read_objects(fs, scene.shapes);
    read_objects(fs, scene.surfaces);
    read_objects(fs, scene.textures);
    read_objects(fs, scene.voltextures);
    read_objects(fs, scene.materials);
    read_objects(fs, scene.instances);
    read_objects(fs, scene.environments);
}

// Load/save a binary dump useful for very fast scene IO.
void load_ybin_scene(const string& filename, yocto_scene& scene,
    const load_scene_options& options) {
    scene = {};
    try {
        auto fs = input_file(filename, true);
        read_object(fs, scene);
    } catch (const std::exception& e) {
        throw io_error("cannot load scene " + filename + "\n" + e.what());
    }
}

// Load/save a binary dump useful for very fast scene IO.
void save_ybin_scene(const string& filename, const yocto_scene& scene,
    const save_scene_options& options) {
    try {
        auto fs = output_file(filename, true);
        write_object(fs, scene);
    } catch (const std::exception& e) {
        throw io_error("cannot save scene " + filename + "\n" + e.what());
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHAPE IO
// -----------------------------------------------------------------------------
namespace yocto {

// Reset mesh data
void reset_mesh_data(vector<int>& points, vector<vec2i>& lines,
    vector<vec3i>& triangles, vector<vec4i>& quads, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords, vector<vec4f>& colors,
    vector<float>& radius) {
    points        = {};
    lines         = {};
    triangles     = {};
    quads         = {};
    positions     = {};
    normals       = {};
    texturecoords = {};
    colors        = {};
    radius        = {};
}

// hack for CyHair data
void load_cyhair_mesh(const string& filename, vector<int>& points,
    vector<vec2i>& lines, vector<vec3i>& triangles, vector<vec4i>& quads,
    vector<vec3f>& positions, vector<vec3f>& normals,
    vector<vec2f>& texturecoords, vector<vec4f>& color, vector<float>& radius,
    bool force_triangles, bool flip_texcoord = true);

// Load ply mesh
void load_mesh(const string& filename, vector<int>& points,
    vector<vec2i>& lines, vector<vec3i>& triangles, vector<vec4i>& quads,
    vector<vec3f>& positions, vector<vec3f>& normals,
    vector<vec2f>& texturecoords, vector<vec4f>& colors, vector<float>& radius,
    bool force_triangles) {
    auto ext = get_extension(filename);
    if (ext == "ply" || ext == "PLY") {
        load_ply_mesh(filename, points, lines, triangles, quads, positions,
            normals, texturecoords, colors, radius, force_triangles);
    } else if (ext == "obj" || ext == "OBJ") {
        load_obj_mesh(filename, points, lines, triangles, quads, positions,
            normals, texturecoords, force_triangles);
    } else if (ext == "hair" || ext == "HAIR") {
        load_cyhair_mesh(filename, points, lines, triangles, quads, positions,
            normals, texturecoords, colors, radius, force_triangles);
    } else {
        reset_mesh_data(points, lines, triangles, quads, positions, normals,
            texturecoords, colors, radius);
        throw io_error("unsupported mesh type " + ext);
    }
}

// Save ply mesh
void save_mesh(const string& filename, const vector<int>& points,
    const vector<vec2i>& lines, const vector<vec3i>& triangles,
    const vector<vec4i>& quads, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    const vector<vec4f>& colors, const vector<float>& radius, bool ascii) {
    auto ext = get_extension(filename);
    if (ext == "ply" || ext == "PLY") {
        return save_ply_mesh(filename, points, lines, triangles, quads,
            positions, normals, texturecoords, colors, radius, ascii);
    } else if (ext == "obj" || ext == "OBJ") {
        return save_obj_mesh(filename, points, lines, triangles, quads,
            positions, normals, texturecoords);
    } else {
        throw io_error("unsupported mesh type " + ext);
    }
}

void load_ply_mesh(const string& filename, vector<int>& points,
    vector<vec2i>& lines, vector<vec3i>& triangles, vector<vec4i>& quads,
    vector<vec3f>& positions, vector<vec3f>& normals,
    vector<vec2f>& texturecoords, vector<vec4f>& color, vector<float>& radius,
    bool force_triangles, bool flip_texcoord) {
    // clear
    reset_mesh_data(points, lines, triangles, quads, positions, normals,
        texturecoords, color, radius);

    try {
        // load ply
        auto ply = ply_data{};
        load_ply(filename, ply);

        // copy vertex data
        for (auto& elem : ply.elements) {
            if (elem.name != "vertex") continue;
            auto count = elem.count;
            for (auto& prop : elem.properties) {
                auto vals        = data(prop.scalars);
                auto copy_floats = [vals, count](auto& vert, const auto& def,
                                       int stride, int offset) {
                    if (vert.size() != count) vert.resize(count, def);
                    auto dst = (float*)data(vert);
                    for (auto i = 0; i < count; i++)
                        dst[i * stride + offset] = vals[i];
                };
                if (prop.name == "x") copy_floats(positions, zero3f, 3, 0);
                if (prop.name == "y") copy_floats(positions, zero3f, 3, 1);
                if (prop.name == "z") copy_floats(positions, zero3f, 3, 2);
                if (prop.name == "nx") copy_floats(normals, zero3f, 3, 0);
                if (prop.name == "ny") copy_floats(normals, zero3f, 3, 1);
                if (prop.name == "nz") copy_floats(normals, zero3f, 3, 2);
                if (prop.name == "u" || prop.name == "s")
                    copy_floats(texturecoords, zero2f, 2, 0);
                if (prop.name == "v" || prop.name == "t")
                    copy_floats(texturecoords, zero2f, 2, 1);
                if (prop.name == "red")
                    copy_floats(color, vec4f{0, 0, 0, 1}, 4, 0);
                if (prop.name == "green")
                    copy_floats(color, vec4f{0, 0, 0, 1}, 4, 1);
                if (prop.name == "blue")
                    copy_floats(color, vec4f{0, 0, 0, 1}, 4, 2);
                if (prop.name == "alpha")
                    copy_floats(color, vec4f{0, 0, 0, 1}, 4, 3);
                if (prop.name == "radius") copy_floats(radius, 0.0f, 1, 0);
            }
        }

        // fix texture coordinated
        if (flip_texcoord && !empty(texturecoords)) {
            for (auto& uv : texturecoords) uv.y = 1 - uv.y;
        }

        // copy face data
        for (auto& elem : ply.elements) {
            if (elem.name != "face") continue;
            auto count = elem.count;
            for (auto& prop : elem.properties) {
                if (prop.name == "vertex_indices") {
                    for (auto fid = 0; fid < count; fid++) {
                        auto& list = prop.lists[fid];
                        auto  num  = (int)prop.scalars[fid];
                        if (num == 4) {
                            quads.push_back(
                                {list[0], list[1], list[2], list[3]});
                        } else {
                            for (auto i = 2; i < num; i++)
                                triangles.push_back(
                                    {list[0], list[i - 1], list[i]});
                        }
                    }
                }
            }
        }

        // copy face data
        for (auto& elem : ply.elements) {
            if (elem.name != "line") continue;
            auto count = elem.count;
            for (auto& prop : elem.properties) {
                if (prop.name == "vertex_indices") {
                    for (auto fid = 0; fid < count; fid++) {
                        auto& list = prop.lists[fid];
                        auto  num  = (int)prop.scalars[fid];
                        for (auto i = 1; i < num; i++)
                            lines.push_back({list[i], list[i - 1]});
                    }
                }
            }
        }

        merge_triangles_and_quads(triangles, quads, force_triangles);

    } catch (const std::exception& e) {
        throw io_error("cannot load mesh " + filename + "\n" + e.what());
    }
}

// Save ply mesh
void save_ply_mesh(const string& filename, const vector<int>& points,
    const vector<vec2i>& lines, const vector<vec3i>& triangles,
    const vector<vec4i>& quads, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    const vector<vec4f>& colors, const vector<float>& radius, bool ascii,
    bool flip_texcoord) {
    auto fs = output_file(filename, std::ios::binary);

    // header
    println_values(fs, "ply");
    if (ascii)
        println_values(fs, "format ascii 1.0");
    else
        println_values(fs, "format binary_little_endian 1.0");
    println_values(
        fs, "comment Saved by Yocto/GL - https://github.com/xelatihy/yocto-gl");
    println_values(fs, "element vertex", (int)positions.size());
    if (!empty(positions)) {
        println_values(fs, "property float x");
        println_values(fs, "property float y");
        println_values(fs, "property float z");
    }
    if (!empty(normals)) {
        println_values(fs, "property float nx");
        println_values(fs, "property float ny");
        println_values(fs, "property float nz");
    }
    if (!empty(texturecoords)) {
        println_values(fs, "property float u");
        println_values(fs, "property float v");
    }
    if (!empty(colors)) {
        println_values(fs, "property float red");
        println_values(fs, "property float green");
        println_values(fs, "property float blue");
        println_values(fs, "property float alpha");
    }
    if (!empty(radius)) {
        println_values(fs, "property float radius");
    }
    if (!empty(triangles) || !empty(quads)) {
        println_values(
            fs, "element face", (int)triangles.size() + (int)quads.size());
        println_values(fs, "property list uchar int vertex_indices");
    }
    if (!empty(lines)) {
        println_values(fs, "element line", (int)lines.size());
        println_values(fs, "property list uchar int vertex_indices");
    }
    println_values(fs, "end_header");

    // body
    if (ascii) {
        // write vertex data
        for (auto i = 0; i < positions.size(); i++) {
            if (!empty(positions)) print_value(fs, positions[i]);
            if (!empty(normals)) print_value(fs, normals[i]);
            if (!empty(texturecoords))
                print_value(fs, (!flip_texcoord) ? texturecoords[i] :
                                                   vec2f{texturecoords[i].x,
                                                       1 - texturecoords[i].y});
            if (!empty(colors)) print_value(fs, colors[i]);
            if (!empty(radius)) print_value(fs, radius[i]);
            print_value(fs, "\n");
        }

        // write face data
        for (auto& t : triangles) println_values(fs, "3", t);
        for (auto& q : quads) {
            if (q.z == q.w)
                println_values(fs, "3", vec3i{q.x, q.y, q.z});
            else
                println_values(fs, "4", q);
        }
        for (auto& l : lines) println_values(fs, "2", l);
    } else {
        // write vertex data
        for (auto i = 0; i < positions.size(); i++) {
            if (!empty(positions)) write_value(fs, positions[i]);
            if (!empty(normals)) write_value(fs, normals[i]);
            if (!empty(texturecoords))
                write_value(fs, (!flip_texcoord) ? texturecoords[i] :
                                                   vec2f{texturecoords[i].x,
                                                       1 - texturecoords[i].y});
            if (!empty(colors)) write_value(fs, colors[i]);
            if (!empty(radius)) write_value(fs, radius[i]);
        }

        // write face data
        for (auto& t : triangles) {
            auto n = (byte)3;
            write_value(fs, n);
            write_value(fs, t);
        }
        for (auto& q : quads) {
            if (q.z == q.w) {
                auto n = (byte)3;
                write_value(fs, n);
                write_value(fs, vec3i{q.x, q.y, q.z});
            } else {
                auto n = (byte)4;
                write_value(fs, n);
                write_value(fs, q);
            }
        }
        for (auto& l : lines) {
            auto n = (byte)2;
            write_value(fs, n);
            write_value(fs, l);
        }
    }
}

// Load ply mesh
void load_obj_mesh(const string& filename, vector<int>& points,
    vector<vec2i>& lines, vector<vec3i>& triangles, vector<vec4i>& quads,
    vector<vec3f>& positions, vector<vec3f>& normals,
    vector<vec2f>& texturecoords, bool force_triangles, bool flip_texcoord) {
    // clear
    auto colors = vector<vec4f>{};
    auto radius = vector<float>{};
    reset_mesh_data(points, lines, triangles, quads, positions, normals,
        texturecoords, colors, radius);

    // obj vertices
    auto opos      = std::deque<vec3f>();
    auto onorm     = std::deque<vec3f>();
    auto otexcoord = std::deque<vec2f>();

    // vertex maps
    auto vertex_map = unordered_map<obj_vertex, int, obj_vertex_hash>();

    // Add  vertices to the current shape
    auto add_verts = [&](const vector<obj_vertex>& verts) {
        for (auto& vert : verts) {
            auto it = vertex_map.find(vert);
            if (it != vertex_map.end()) continue;
            auto nverts = (int)positions.size();
            vertex_map.insert(it, {vert, nverts});
            if (vert.position) positions.push_back(opos.at(vert.position - 1));
            if (vert.texturecoord)
                texturecoords.push_back(otexcoord.at(vert.texturecoord - 1));
            if (vert.normal) normals.push_back(onorm.at(vert.normal - 1));
        }
    };

    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const vector<obj_vertex>& verts) {
        add_verts(verts);
        if (verts.size() == 4) {
            quads.push_back({vertex_map.at(verts[0]), vertex_map.at(verts[1]),
                vertex_map.at(verts[2]), vertex_map.at(verts[3])});
        } else {
            for (auto i = 2; i < verts.size(); i++)
                triangles.push_back({vertex_map.at(verts[0]),
                    vertex_map.at(verts[i - 1]), vertex_map.at(verts[i])});
        }
    };
    cb.line = [&](const vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 1; i < verts.size(); i++)
            lines.push_back(
                {vertex_map.at(verts[i - 1]), vertex_map.at(verts[i])});
    };
    cb.point = [&](const vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 0; i < verts.size(); i++)
            points.push_back(vertex_map.at(verts[i]));
    };

    try {
        // load obj
        auto obj_options          = load_obj_options();
        obj_options.exit_on_error = false;
        obj_options.geometry_only = true;
        obj_options.flip_texcoord = flip_texcoord;
        load_obj(filename, cb, obj_options);

        // merging quads and triangles
        merge_triangles_and_quads(triangles, quads, force_triangles);

    } catch (const std::exception& e) {
        throw io_error("cannot load mesh " + filename + "\n" + e.what());
    }
}

// Load ply mesh
void save_obj_mesh(const string& filename, const vector<int>& points,
    const vector<vec2i>& lines, const vector<vec3i>& triangles,
    const vector<vec4i>& quads, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    bool flip_texcoord) {
    auto fs = output_file(filename);

    println_values(
        fs, "# Saved by Yocto/GL - https://github.com/xelatihy/yocto-gl\n");

    for (auto& p : positions) println_values(fs, "v", p);
    for (auto& n : normals) println_values(fs, "vn", n);
    for (auto& t : texturecoords)
        println_values(fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});

    auto mask = obj_vertex{
        1, empty(texturecoords) ? 0 : 1, empty(normals) ? 0 : 1};
    auto vert = [mask](int i) {
        return obj_vertex{(i + 1) * mask.position, (i + 1) * mask.texturecoord,
            (i + 1) * mask.normal};
    };

    for (auto& p : points) {
        println_values(fs, "p", vert(p));
    }
    for (auto& l : lines) {
        println_values(fs, "l", vert(l.x), vert(l.y));
    }
    for (auto& t : triangles) {
        println_values(fs, "f", vert(t.x), vert(t.y), vert(t.z));
    }
    for (auto& q : quads) {
        if (q.z == q.w) {
            println_values(fs, "f", vert(q.x), vert(q.y), vert(q.z));
        } else {
            println_values(fs, "f", vert(q.x), vert(q.y), vert(q.z), vert(q.w));
        }
    }
}

// Reset mesh data
void reset_facevarying_mesh_data(vector<vec4i>& quads_positions,
    vector<vec4i>& quads_normals, vector<vec4i>& quads_textuercoords,
    vector<vec3f>& positions, vector<vec3f>& normals,
    vector<vec2f>& texturecoords, vector<int>& quads_materials) {
    quads_positions     = {};
    quads_normals       = {};
    quads_textuercoords = {};
    positions           = {};
    normals             = {};
    texturecoords       = {};
    quads_materials     = {};
}

// Load ply mesh
void load_facevarying_mesh(const string& filename,
    vector<vec4i>& quads_positions, vector<vec4i>& quads_normals,
    vector<vec4i>& quads_texturecoords, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    vector<int>& quads_materials) {
    auto ext = get_extension(filename);
    if (ext == "obj" || ext == "OBJ") {
        return load_obj_facevarying_mesh(filename, quads_positions,
            quads_normals, quads_texturecoords, positions, normals,
            texturecoords, quads_materials);
    } else {
        reset_facevarying_mesh_data(quads_positions, quads_normals,
            quads_texturecoords, positions, normals, texturecoords,
            quads_materials);
        throw io_error("unsupported mesh type " + ext);
    }
}

// Save ply mesh
void save_facevarying_mesh(const string& filename,
    const vector<vec4i>& quads_positions, const vector<vec4i>& quads_normals,
    const vector<vec4i>& quads_texturecoords, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    const vector<int>& quads_materials, bool ascii) {
    auto ext = get_extension(filename);
    if (ext == "obj" || ext == "OBJ") {
        return save_obj_facevarying_mesh(filename, quads_positions,
            quads_normals, quads_texturecoords, positions, normals,
            texturecoords, quads_materials);
    } else {
        throw io_error("unsupported mesh type " + ext);
    }
}

// Load ply mesh
void load_obj_facevarying_mesh(const string& filename,
    vector<vec4i>& quads_positions, vector<vec4i>& quads_normals,
    vector<vec4i>& quads_texturecoords, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    vector<int>& quads_materials, bool flip_texcoord) {
    // clear
    reset_facevarying_mesh_data(quads_positions, quads_normals,
        quads_texturecoords, positions, normals, texturecoords,
        quads_materials);

    // obj vertices
    auto opos      = std::deque<vec3f>();
    auto onorm     = std::deque<vec3f>();
    auto otexcoord = std::deque<vec2f>();

    // vertex maps
    auto pos_map      = unordered_map<int, int>();
    auto texcoord_map = unordered_map<int, int>();
    auto norm_map     = unordered_map<int, int>();

    // material group
    auto material_group      = vector<string>();
    auto current_material_id = -1;

    // add vertex
    auto add_fvverts = [&](const vector<obj_vertex>& verts) {
        for (auto& vert : verts) {
            if (!vert.position) continue;
            auto pos_it = pos_map.find(vert.position);
            if (pos_it != pos_map.end()) continue;
            auto nverts = (int)positions.size();
            pos_map.insert(pos_it, {vert.position, nverts});
            positions.push_back(opos.at(vert.position - 1));
        }
        for (auto& vert : verts) {
            if (!vert.texturecoord) continue;
            auto texcoord_it = texcoord_map.find(vert.texturecoord);
            if (texcoord_it != texcoord_map.end()) continue;
            auto nverts = (int)texturecoords.size();
            texcoord_map.insert(texcoord_it, {vert.texturecoord, nverts});
            texturecoords.push_back(otexcoord.at(vert.texturecoord - 1));
        }
        for (auto& vert : verts) {
            if (!vert.normal) continue;
            auto norm_it = norm_map.find(vert.normal);
            if (norm_it != norm_map.end()) continue;
            auto nverts = (int)normals.size();
            norm_map.insert(norm_it, {vert.normal, nverts});
            normals.push_back(onorm.at(vert.normal - 1));
        }
    };

    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const vector<obj_vertex>& verts) {
        add_fvverts(verts);
        if (verts.size() == 4) {
            if (verts[0].position) {
                quads_positions.push_back({pos_map.at(verts[0].position),
                    pos_map.at(verts[1].position),
                    pos_map.at(verts[2].position),
                    pos_map.at(verts[3].position)});
            }
            if (verts[0].texturecoord) {
                quads_texturecoords.push_back(
                    {texcoord_map.at(verts[0].texturecoord),
                        texcoord_map.at(verts[1].texturecoord),
                        texcoord_map.at(verts[2].texturecoord),
                        texcoord_map.at(verts[3].texturecoord)});
            }
            if (verts[0].normal) {
                quads_normals.push_back({norm_map.at(verts[0].normal),
                    norm_map.at(verts[1].normal), norm_map.at(verts[2].normal),
                    norm_map.at(verts[3].normal)});
            }
            quads_materials.push_back(current_material_id);
        } else {
            if (verts[0].position) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_positions.push_back({pos_map.at(verts[0].position),
                        pos_map.at(verts[1].position),
                        pos_map.at(verts[i].position),
                        pos_map.at(verts[i].position)});
            }
            if (verts[0].texturecoord) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_texturecoords.push_back(
                        {texcoord_map.at(verts[0].texturecoord),
                            texcoord_map.at(verts[1].texturecoord),
                            texcoord_map.at(verts[i].texturecoord),
                            texcoord_map.at(verts[i].texturecoord)});
            }
            if (verts[0].normal) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_normals.push_back({norm_map.at(verts[0].normal),
                        norm_map.at(verts[1].normal),
                        norm_map.at(verts[i].normal),
                        norm_map.at(verts[i].normal)});
            }
            for (auto i = 2; i < verts.size(); i++)
                quads_materials.push_back(current_material_id);
        }
    };
    cb.line = [&](const vector<obj_vertex>& verts) {
        throw io_error("lines not supported!");
    };
    cb.point = [&](const vector<obj_vertex>& verts) {
        throw io_error("points not supported!");
    };
    cb.usemtl = [&](const string& name) {
        auto pos = std::find(
            material_group.begin(), material_group.end(), name);
        if (pos == material_group.end()) {
            material_group.push_back(name);
            current_material_id = (int)material_group.size() - 1;
        } else {
            current_material_id = (int)(pos - material_group.begin());
        }
    };

    try {
        // load obj
        auto obj_options          = load_obj_options();
        obj_options.exit_on_error = false;
        obj_options.geometry_only = true;
        obj_options.flip_texcoord = flip_texcoord;
        load_obj(filename, cb, obj_options);

        // cleanup materials ids
        if (std::all_of(quads_materials.begin(), quads_materials.end(),
                [b = quads_materials.front()](auto a) { return a == b; })) {
            quads_materials.clear();
        }
    } catch (const std::exception& e) {
        throw io_error("cannot load mesh " + filename + "\n" + e.what());
    }
}

// Load ply mesh
void save_obj_facevarying_mesh(const string& filename,
    const vector<vec4i>& quads_positions, const vector<vec4i>& quads_normals,
    const vector<vec4i>& quads_texturecoords, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    const vector<int>& quads_materials, bool flip_texcoord) {
    auto fs = output_file(filename);

    println_values(
        fs, "# Saved by Yocto/GL - https://github.com/xelatihy/yocto-gl\n");

    for (auto& p : positions) println_values(fs, "v", p);
    for (auto& n : normals) println_values(fs, "vn", n);
    for (auto& t : texturecoords)
        println_values(fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});

    auto fvmask = obj_vertex{
        1, empty(texturecoords) ? 0 : 1, empty(normals) ? 0 : 1};
    auto fvvert = [fvmask](int pi, int ti, int ni) {
        return obj_vertex{(pi + 1) * fvmask.position,
            (ti + 1) * fvmask.texturecoord, (ni + 1) * fvmask.normal};
    };
    auto last_material_id = -1;
    for (auto i = 0; i < quads_positions.size(); i++) {
        if (!empty(quads_materials) && quads_materials[i] != last_material_id) {
            last_material_id = quads_materials[i];
            println_values(fs, "usemtl material_{}\n", last_material_id);
        }
        auto qp = quads_positions.at(i);
        auto qt = !empty(quads_texturecoords) ? quads_texturecoords.at(i) :
                                                vec4i{-1, -1, -1, -1};
        auto qn = !empty(quads_normals) ? quads_normals.at(i) :
                                          vec4i{-1, -1, -1, -1};
        if (qp.z != qp.w) {
            println_values(fs, "f", fvvert(qp.x, qt.x, qn.x),
                fvvert(qp.y, qt.y, qn.y), fvvert(qp.z, qt.z, qn.z),
                fvvert(qp.w, qt.w, qn.w));
        } else {
            println_values(fs, "f", fvvert(qp.x, qt.x, qn.x),
                fvvert(qp.y, qt.y, qn.y), fvvert(qp.z, qt.z, qn.z));
        }
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF OBJ IO
// -----------------------------------------------------------------------------
namespace yocto {}

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PLY IO
// -----------------------------------------------------------------------------
namespace yocto {

// normalize obj line for simpler parsing
inline void normalize_ply_line(char* str, char comment_char = '#') {
    auto has_content = false;
    auto start       = str;
    while (*str) {
        if (*str == ' ' || *str == '\t' || *str == '\r' || *str == '\n') {
            *str = ' ';
        } else {
            has_content = true;
        }
        str++;
    }
    if (!has_content) *start = 0;
}

// Load ply mesh
void load_ply(const string& filename, ply_data& ply) {
    // open file
    ply     = {};
    auto fs = input_file(filename, true);

    // parse header
    ply             = ply_data{};
    auto ascii      = false;
    auto big_endian = false;
    char buffer[4096];
    while (read_line(fs, buffer, sizeof(buffer))) {
        auto line = buffer;
        normalize_ply_line(line);
        if (!*line) continue;
        auto cmd = ""s;
        parse_value(line, cmd);
        if (cmd == "") continue;
        if (cmd == "ply") {
        } else if (cmd == "comment") {
        } else if (cmd == "format") {
            auto fmt = ""s;
            parse_value(line, fmt);
            if (fmt == "ascii") {
                ascii = true;
            } else if (fmt == "binary_little_endian") {
                ascii      = false;
                big_endian = false;
            } else if (fmt == "binary_big_endian") {
                ascii      = false;
                big_endian = true;
            } else {
                throw io_error("bad ply header");
            }
        } else if (cmd == "element") {
            auto elem = ply_element();
            parse_value(line, elem.name);
            parse_value(line, elem.count);
            ply.elements.push_back(elem);
        } else if (cmd == "property") {
            auto prop = ply_property();
            auto type = ""s;
            parse_value(line, type);
            if (type == "list") {
                auto count_type = ""s;
                parse_value(line, count_type);
                auto elem_type = ""s;
                parse_value(line, elem_type);
                if (count_type != "uchar" && count_type != "uint8")
                    throw io_error("unsupported ply list type");
                if (elem_type != "int" && elem_type != "uint")
                    throw io_error("unsupported ply list type");
                prop.type = ply_type::ply_int_list;
            } else if (type == "float") {
                prop.type = ply_type::ply_float;
            } else if (type == "uchar" || type == "uint8") {
                prop.type = ply_type::ply_uchar;
            } else if (type == "int") {
                prop.type = ply_type::ply_int;
            } else {
                throw io_error("unsupported ply format");
            }
            parse_value(line, prop.name);
            prop.scalars.resize(ply.elements.back().count);
            if (prop.type == ply_type::ply_int_list)
                prop.lists.resize(ply.elements.back().count);
            ply.elements.back().properties.push_back(prop);
        } else if (cmd == "end_header") {
            break;
        } else {
            throw io_error("unsupported ply format");
        }
    }

    // parse content
    if (ascii) {
        for (auto& elem : ply.elements) {
            for (auto vid = 0; vid < elem.count; vid++) {
                read_line(fs, buffer, sizeof(buffer));
                auto line = buffer;
                normalize_ply_line(line);
                for (auto pid = 0; pid < elem.properties.size(); pid++) {
                    auto& prop = elem.properties[pid];
                    if (prop.type == ply_type::ply_float) {
                        auto v = 0.0f;
                        parse_value(line, v);
                        prop.scalars[vid] = v;
                    } else if (prop.type == ply_type::ply_int) {
                        auto v = 0;
                        parse_value(line, v);
                        prop.scalars[vid] = v;
                    } else if (prop.type == ply_type::ply_uchar) {
                        auto vc = (unsigned char)0;
                        auto v  = 0;
                        parse_value(line, v);
                        vc                = (unsigned char)v;
                        prop.scalars[vid] = vc / 255.0f;
                    } else if (prop.type == ply_type::ply_int_list) {
                        auto vc = (unsigned char)0;
                        auto v  = 0;
                        parse_value(line, v);
                        vc                = (unsigned char)v;
                        prop.scalars[vid] = vc;
                        for (auto i = 0; i < (int)prop.scalars[vid]; i++) {
                            auto v = 0;
                            parse_value(line, v);
                            prop.lists[vid][i] = v;
                        }
                    } else {
                        throw io_error("unsupported ply format");
                    }
                }
            }
        }
    } else {
        auto byte_swap = [](auto val) {
            // https://stackoverflow.com/questions/2782725/converting-float-values-from-big-endian-to-little-endian
            auto retVal = val;
            auto pVal = (char*)&val, pRetVal = (char*)&retVal;
            auto size = (int)sizeof(val);
            for (auto i = 0; i < size; i++) pRetVal[size - 1 - i] = pVal[i];
            return retVal;
        };
        for (auto& elem : ply.elements) {
            for (auto vid = 0; vid < elem.count; vid++) {
                for (auto pid = 0; pid < elem.properties.size(); pid++) {
                    auto& prop = elem.properties[pid];
                    if (prop.type == ply_type::ply_float) {
                        auto v = 0.0f;
                        read_value(fs, v);
                        if (big_endian) v = byte_swap(v);
                        prop.scalars[vid] = v;
                    } else if (prop.type == ply_type::ply_int) {
                        auto v = 0;
                        read_value(fs, v);
                        if (big_endian) v = byte_swap(v);
                        prop.scalars[vid] = v;
                    } else if (prop.type == ply_type::ply_uchar) {
                        auto vc = (unsigned char)0;
                        read_value(fs, vc);
                        prop.scalars[vid] = vc / 255.0f;
                    } else if (prop.type == ply_type::ply_int_list) {
                        auto vc = (unsigned char)0;
                        read_value(fs, vc);
                        prop.scalars[vid] = vc;
                        for (auto i = 0; i < (int)prop.scalars[vid]; i++) {
                            auto v = 0;
                            read_value(fs, v);
                            if (big_endian) v = byte_swap(v);
                            prop.lists[vid][i] = v;
                        }
                    } else {
                        throw io_error("unsupported ply format");
                    }
                }
            }
        }
    }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF CYHAIR
// -----------------------------------------------------------------------------
namespace yocto {

struct cyhair_strand {
    vector<vec3f> positions;
    vector<float> radius;
    vector<float> transparency;
    vector<vec3f> color;
};

struct cyhair_data {
    vector<cyhair_strand> strands              = {};
    float                 default_thickness    = 0;
    float                 default_transparency = 0;
    vec3f                 default_color        = zero3f;
};

void load_cyhair(const string& filename, cyhair_data& hair) {
    // open file
    hair    = {};
    auto fs = input_file(filename, std::ios::binary);

    // Bytes 0-3    Must be "HAIR" in ascii code (48 41 49 52)
    // Bytes 4-7    Number of hair strands as unsigned int
    // Bytes 8-11    Total number of points of all strands as unsigned int
    // Bytes 12-15    Bit array of data in the file
    // Bit-0 is 1 if the file has segments array.
    // Bit-1 is 1 if the file has points array (this bit must be 1).
    // Bit-2 is 1 if the file has radius array.
    // Bit-3 is 1 if the file has transparency array.
    // Bit-4 is 1 if the file has color array.
    // Bit-5 to Bit-31 are reserved for future extension (must be 0).
    // Bytes 16-19    Default number of segments of hair strands as unsigned int
    // If the file does not have a segments array, this default value is used.
    // Bytes 20-23    Default radius hair strands as float
    // If the file does not have a radius array, this default value is used.
    // Bytes 24-27    Default transparency hair strands as float
    // If the file does not have a transparency array, this default value is
    // used. Bytes 28-39    Default color hair strands as float array of size 3
    // If the file does not have a radius array, this default value is used.
    // Bytes 40-127    File information as char array of size 88 in ascii

    // parse header
    hair = cyhair_data{};
    struct cyhair_header {
        char         magic[4]             = {0};
        unsigned int num_strands          = 0;
        unsigned int num_points           = 0;
        unsigned int flags                = 0;
        unsigned int default_segments     = 0;
        float        default_thickness    = 0;
        float        default_transparency = 0;
        vec3f        default_color        = zero3f;
        char         info[88]             = {0};
    };
    static_assert(sizeof(cyhair_header) == 128);
    auto header = cyhair_header{};
    read_value(fs, header);
    if (header.magic[0] != 'H' || header.magic[1] != 'A' ||
        header.magic[2] != 'I' || header.magic[3] != 'R')
        throw io_error("bad cyhair header");

    // set up data
    hair.default_thickness    = header.default_thickness;
    hair.default_transparency = header.default_transparency;
    hair.default_color        = header.default_color;
    hair.strands.resize(header.num_strands);

    // get segments length
    auto segments = vector<unsigned short>();
    if (header.flags & 1) {
        segments.resize(header.num_strands);
        read_values(fs, segments);
    } else {
        segments.assign(header.num_strands, header.default_segments);
    }

    // check segment length
    auto total_length = 0;
    for (auto segment : segments) total_length += segment + 1;
    if (total_length != header.num_points) {
        throw io_error("bad cyhair file");
    }

    // read positions data
    if (header.flags & 2) {
        for (auto strand_id = 0; strand_id < header.num_strands; strand_id++) {
            auto strand_size = (int)segments[strand_id] + 1;
            hair.strands[strand_id].positions.resize(strand_size);
            read_values(fs, hair.strands[strand_id].positions);
        }
    }
    // read radius data
    if (header.flags & 4) {
        for (auto strand_id = 0; strand_id < header.num_strands; strand_id++) {
            auto strand_size = (int)segments[strand_id] + 1;
            hair.strands[strand_id].radius.resize(strand_size);
            read_values(fs, hair.strands[strand_id].radius);
        }
    }
    // read transparency data
    if (header.flags & 8) {
        for (auto strand_id = 0; strand_id < header.num_strands; strand_id++) {
            auto strand_size = (int)segments[strand_id] + 1;
            hair.strands[strand_id].transparency.resize(strand_size);
            read_values(fs, hair.strands[strand_id].transparency);
        }
    }
    // read color data
    if (header.flags & 16) {
        for (auto strand_id = 0; strand_id < header.num_strands; strand_id++) {
            auto strand_size = (int)segments[strand_id] + 1;
            hair.strands[strand_id].color.resize(strand_size);
            read_values(fs, hair.strands[strand_id].color);
        }
    }
}

void load_cyhair_mesh(const string& filename, vector<int>& points,
    vector<vec2i>& lines, vector<vec3i>& triangles, vector<vec4i>& quads,
    vector<vec3f>& positions, vector<vec3f>& normals,
    vector<vec2f>& texturecoords, vector<vec4f>& color, vector<float>& radius,
    bool force_triangles, bool flip_texcoord) {
    // load hair file
    auto hair = cyhair_data();
    load_cyhair(filename, hair);

    // generate curve data
    for (auto& strand : hair.strands) {
        auto offset = (int)positions.size();
        for (auto segment = 0; segment < (int)strand.positions.size() - 1;
             segment++) {
            lines.push_back({offset + segment, offset + segment + 1});
        }
        positions.insert(
            positions.end(), strand.positions.begin(), strand.positions.end());
        if (strand.radius.empty()) {
            radius.insert(
                radius.end(), strand.positions.size(), hair.default_thickness);
        } else {
            radius.insert(
                radius.end(), strand.radius.begin(), strand.radius.end());
        }
        if (strand.color.empty()) {
            color.insert(color.end(), strand.positions.size(),
                {hair.default_color.x, hair.default_color.y,
                    hair.default_color.z, 1});
        } else {
            for (auto i = 0; i < strand.color.size(); i++) {
                auto scolor = strand.color[i];
                color.push_back({scolor.x, scolor.y, scolor.z, 1});
            }
        }
    }

    // flip yz
    for (auto& p : positions) std::swap(p.y, p.z);

    // compute tangents
    normals.resize(positions.size());
    compute_vertex_tangents(normals, lines, positions);

    // fix colors
    for (auto& c : color) c = srgb_to_linear(c);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHAPE IO
// -----------------------------------------------------------------------------
namespace yocto {

#if 0

// Load obj mesh
bool load_obj_fvmesh(const string& filename, vector<vec4i>& quads_positions,
    vector<vec3f>& pos, vector<vec4i>& quads_normals, vector<vec3f>& norm,
    vector<vec4i>& quads_texturecoords, vector<vec2f>& texcoord,
    bool flip_texcoord) {
    // clear
    vector<vec4i> quads_colors;
    vector<vec4f> color;
    reset_fvmesh_data(quads_positions, pos, quads_normals, norm,
        quads_texturecoords, texcoord, quads_colors, color);

    // obj vertex
    auto opos      = std::deque<vec3f>();
    auto onorm     = std::deque<vec3f>();
    auto otexcoord = std::deque<vec2f>();

    // vertex maps
    auto pos_map      = unordered_map<int, int>();
    auto texcoord_map = unordered_map<int, int>();
    auto norm_map     = unordered_map<int, int>();

    // add vertex
    auto add_verts = [&](const vector<obj_vertex>& verts) {
        for (auto& vert : verts) {
            if (!vert.position) continue;
            auto pos_it = pos_map.find(vert.position);
            if (pos_it != pos_map.end()) continue;
            auto nverts = (int)pos.size();
            pos_map.insert(pos_it, {vert.position, nverts});
            pos.push_back(opos.at(vert.position - 1));
        }
        for (auto& vert : verts) {
            if (!vert.texturecoord) continue;
            auto texcoord_it = texcoord_map.find(vert.texturecoord);
            if (texcoord_it != texcoord_map.end()) continue;
            auto nverts = (int)texcoord.size();
            texcoord_map.insert(texcoord_it, {vert.texturecoord, nverts});
            texcoord.push_back(otexcoord.at(vert.texturecoord - 1));
        }
        for (auto& vert : verts) {
            if (!vert.normal) continue;
            auto norm_it = norm_map.find(vert.normal);
            if (norm_it != norm_map.end()) continue;
            auto nverts = (int)norm.size();
            norm_map.insert(norm_it, {vert.normal, nverts});
            norm.push_back(onorm.at(vert.normal - 1));
        }
    };

    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const vector<obj_vertex>& verts) {
        add_verts(verts);
        if (verts.size() == 4) {
            if (verts[0].position) {
                quads_positions.push_back({pos_map.at(verts[0].position),
                    pos_map.at(verts[1].position), pos_map.at(verts[2].position),
                    pos_map.at(verts[3].position)});
            }
            if (verts[0].texturecoord) {
                quads_texturecoords.push_back(
                    {texcoord_map.at(verts[0].texturecoord),
                        texcoord_map.at(verts[1].texturecoord),
                        texcoord_map.at(verts[2].texturecoord),
                        texcoord_map.at(verts[3].texturecoord)});
            }
            if (verts[0].normal) {
                quads_normals.push_back({norm_map.at(verts[0].normal),
                    norm_map.at(verts[1].normal), norm_map.at(verts[2].normal),
                    norm_map.at(verts[3].normal)});
            }
        } else {
            if (verts[0].position) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_positions.push_back({pos_map.at(verts[0].position),
                        pos_map.at(verts[1].position),
                        pos_map.at(verts[i].position),
                        pos_map.at(verts[i].position)});
            }
            if (verts[0].texturecoord) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_texturecoords.push_back(
                        {texcoord_map.at(verts[0].texturecoord),
                            texcoord_map.at(verts[1].texturecoord),
                            texcoord_map.at(verts[i].texturecoord),
                            texcoord_map.at(verts[i].texturecoord)});
            }
            if (verts[0].normal) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_normals.push_back({norm_map.at(verts[0].normal),
                        norm_map.at(verts[1].normal), norm_map.at(verts[i].normal),
                        norm_map.at(verts[i].normal)});
            }
        }
    };

    // load obj
    return load_obj(filename, cb, true, true, flip_texcoord);
}

// Load ply mesh
bool save_obj_fvmesh(const string& filename, const vector<vec4i>& quads_positions,
    const vector<vec3f>& positions, const vector<vec4i>& quads_normals,
    const vector<vec3f>& normals, const vector<vec4i>& quads_texturecoords,
    const vector<vec2f>& texturecoords, bool flip_texcoord) {
    auto fs = open(filename, "wt");
    if (!fs) return false;

    for (auto& p : positions) println_values(fs, "v", p);
    for (auto& n : normals) println_values(fs, "vn", n);
    for (auto& t : texturecoords)
        println_values(fs, "vt", vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
    auto mask = obj_vertex{
        1, empty(texturecoords) ? 0 : 1, empty(normals) ? 0 : 1};
    auto vert = [mask](int pif, int ti, int ni) {
        return obj_vertex{(pif + 1) * mask.position,
            (ti + 1) * mask.texturecoord, (ni + 1) * mask.normal};
    };
    for (auto i = 0; i < quads_positions.size(); i++) {
        auto qp = quads_positions.at(i);
        auto qt = !empty(quads_texturecoords) ? quads_texturecoords.at(i) :
                                                 vec4i{-1, -1, -1, -1};
        auto qn = !empty(quads_normals) ? quads_normals.at(i) :
                                           vec4i{-1, -1, -1, -1};
        if (qp.z != qp.w)
            println_values(fs, "f",
                vert(qp.x, qt.x, qn.x).c_str(),
                vert(qp.y, qt.y, qn.y).c_str(),
                vert(qp.z, qt.z, qn.z).c_str(),
                vert(qp.w, qt.w, qn.w).c_str());
        else
            println_values(fs, "f", vert(qp.x, qt.x, qn.x).c_str(),
                vert(qp.y, qt.y, qn.y).c_str(),
                vert(qp.z, qt.z, qn.z).c_str());
    }

    return true;
}

#endif

}  // namespace yocto
