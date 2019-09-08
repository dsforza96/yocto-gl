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

#include "yocto_sceneio.h"
#include "yocto_modelio.h"
#include "yocto_random.h"
#include "yocto_shape.h"

#define CGLTF_IMPLEMENTATION
#include "ext/cgltf.h"

#include <limits.h>
#include <stdlib.h>
#include <array>
#include <atomic>
#include <cassert>
#include <deque>
#include <future>
#include <memory>
#include <regex>
#include <string_view>
#include <thread>

#include "ext/filesystem.hpp"
namespace fs = ghc::filesystem;

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

// Type aliases for readability
using string_view = std::string_view;
using namespace std::literals::string_view_literals;

}  // namespace yocto

// -----------------------------------------------------------------------------
// CONCURRENCY
// -----------------------------------------------------------------------------
namespace yocto {

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes a reference to a `T`.
template <typename T, typename Func>
static inline void parallel_foreach(
    vector<T>& values, const Func& func, std::atomic<bool>* cancel = nullptr) {
  auto                futures  = vector<std::future<void>>{};
  auto                nthreads = std::thread::hardware_concurrency();
  std::atomic<size_t> next_idx(0);
  for (auto thread_id = 0; thread_id < nthreads; thread_id++) {
    futures.emplace_back(
        std::async(std::launch::async, [&func, &next_idx, cancel, &values]() {
          while (true) {
            if (cancel && *cancel) break;
            auto idx = next_idx.fetch_add(1);
            if (idx >= values.size()) break;
            func(values[idx]);
          }
        }));
  }
  for (auto& f : futures) f.get();
}
template <typename T, typename Func>
static inline void parallel_foreach(const vector<T>& values, const Func& func,
    std::atomic<bool>* cancel = nullptr) {
  auto                futures  = vector<std::future<void>>{};
  auto                nthreads = std::thread::hardware_concurrency();
  std::atomic<size_t> next_idx(0);
  for (auto thread_id = 0; thread_id < nthreads; thread_id++) {
    futures.emplace_back(
        std::async(std::launch::async, [&func, &next_idx, cancel, &values]() {
          while (true) {
            if (cancel && *cancel) break;
            auto idx = next_idx.fetch_add(1);
            if (idx >= values.size()) break;
            func(values[idx]);
          }
        }));
  }
  for (auto& f : futures) f.get();
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// FILE IO
// -----------------------------------------------------------------------------
namespace yocto {

// Load a text file
inline void load_text(const string& filename, string& str) {
  // https://stackoverflow.com/questions/174531/how-to-read-the-content-of-a-file-to-a-string-in-c
  auto fs = fopen(filename.c_str(), "rt");
  if (!fs) throw std::runtime_error("cannot open file " + filename);
  fseek(fs, 0, SEEK_END);
  auto length = ftell(fs);
  fseek(fs, 0, SEEK_SET);
  str.resize(length);
  if (fread(str.data(), 1, length, fs) != length) {
    fclose(fs);
    throw std::runtime_error("cannot read file " + filename);
  }
  fclose(fs);
}

// Save a text file
inline void save_text(const string& filename, const string& str) {
  auto fs = fopen(filename.c_str(), "wt");
  if (!fs) throw std::runtime_error("cannot open file " + filename);
  if (fprintf(fs, "%s", str.c_str()) < 0) {
    fclose(fs);
    throw std::runtime_error("cannot write file " + filename);
  }
  fclose(fs);
}

// Load a binary file
inline void load_binary(const string& filename, vector<byte>& data) {
  // https://stackoverflow.com/questions/174531/how-to-read-the-content-of-a-file-to-a-string-in-c
  auto fs = fopen(filename.c_str(), "rb");
  if (!fs) throw std::runtime_error("cannot open file " + filename);
  fseek(fs, 0, SEEK_END);
  auto length = ftell(fs);
  fseek(fs, 0, SEEK_SET);
  data.resize(length);
  if (fread(data.data(), 1, length, fs) != length) {
    fclose(fs);
    throw std::runtime_error("cannot read file " + filename);
  }
  fclose(fs);
}

// Save a binary file
inline void save_binary(const string& filename, const vector<byte>& data) {
  auto fs = fopen(filename.c_str(), "wb");
  if (!fs) throw std::runtime_error("cannot open file " + filename);
  if (fwrite(data.data(), 1, data.size(), fs) != data.size()) {
    fclose(fs);
    throw std::runtime_error("cannot write file " + filename);
  }
  fclose(fs);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// GENERIC SCENE LOADING
// -----------------------------------------------------------------------------
namespace yocto {

// Load/save a scene in the builtin YAML format.
static void load_yaml_scene(
    const string& filename, yocto_scene& scene, const load_params& params);
static void save_yaml_scene(const string& filename, const yocto_scene& scene,
    const save_params& params);

// Load/save a scene from/to OBJ.
static void load_obj_scene(
    const string& filename, yocto_scene& scene, const load_params& params);
static void save_obj_scene(const string& filename, const yocto_scene& scene,
    const save_params& params);

// Load/save a scene from/to PLY. Loads/saves only one mesh with no other data.
static void load_ply_scene(
    const string& filename, yocto_scene& scene, const load_params& params);
static void save_ply_scene(const string& filename, const yocto_scene& scene,
    const save_params& params);

// Load/save a scene from/to glTF.
static void load_gltf_scene(
    const string& filename, yocto_scene& scene, const load_params& params);
static void save_gltf_scene(const string& filename, const yocto_scene& scene,
    const save_params& params);

// Load/save a scene from/to pbrt. This is not robust at all and only
// works on scene that have been previously adapted since the two renderers
// are too different to match.
static void load_pbrt_scene(
    const string& filename, yocto_scene& scene, const load_params& params);
static void save_pbrt_scene(const string& filename, const yocto_scene& scene,
    const save_params& params);

// Load a scene
void load_scene(
    const string& filename, yocto_scene& scene, const load_params& params) {
  auto ext = fs::path(filename).extension().string();
  if (ext == ".yaml" || ext == ".YAML") {
    load_yaml_scene(filename, scene, params);
  } else if (ext == ".obj" || ext == ".OBJ") {
    load_obj_scene(filename, scene, params);
  } else if (ext == ".gltf" || ext == ".GLTF") {
    load_gltf_scene(filename, scene, params);
  } else if (ext == ".pbrt" || ext == ".PBRT") {
    load_pbrt_scene(filename, scene, params);
  } else if (ext == ".ply" || ext == ".PLY") {
    load_ply_scene(filename, scene, params);
  } else {
    scene = {};
    throw std::runtime_error("unsupported scene format " + ext);
  }
}

// Save a scene
void save_scene(const string& filename, const yocto_scene& scene,
    const save_params& params) {
  auto ext = fs::path(filename).extension().string();
  if (ext == ".yaml" || ext == ".YAML") {
    save_yaml_scene(filename, scene, params);
  } else if (ext == ".obj" || ext == ".OBJ") {
    save_obj_scene(filename, scene, params);
  } else if (ext == ".gltf" || ext == ".GLTF") {
    save_gltf_scene(filename, scene, params);
  } else if (ext == ".pbrt" || ext == ".PBRT") {
    save_pbrt_scene(filename, scene, params);
  } else if (ext == ".ply" || ext == ".PLY") {
    save_ply_scene(filename, scene, params);
  } else {
    throw std::runtime_error("unsupported scene format " + ext);
  }
}

static string get_save_scene_message(
    const yocto_scene& scene, const string& indent) {
  auto str = ""s;
  str += indent + "\n";
  str += indent + "Written by Yocto/GL\n";
  str += indent + "https://github.com/xelatihy/yocto-gl\n";
  str += indent + "\n";
  str += format_stats(scene, indent);
  str += indent + "\n";
  return str;
}

// Return the preset type and the remaining filename
static inline bool is_preset_filename(const string& filename) {
  return filename.find("::yocto::") == 0;
}
// Return the preset type and the filename. Call only if this is a preset.
static inline pair<string, string> get_preset_type(const string& filename) {
  if (filename.find("::yocto::") == 0) {
    auto aux = filename.substr(string("::yocto::").size());
    auto pos = aux.find("::");
    if (pos == aux.npos) throw std::runtime_error("bad preset name" + filename);
    return {aux.substr(0, pos), aux.substr(pos + 2)};
  } else {
    return {"", filename};
  }
}

void load_texture(yocto_texture& texture, const string& dirname) {
  if (is_preset_filename(texture.uri)) {
    auto [type, nfilename] = get_preset_type(texture.uri);
    make_image_preset(texture.hdr, texture.ldr, type);
    texture.uri = nfilename;
  } else {
    if (is_hdr_filename(texture.uri)) {
      load_image(fs::path(dirname) / texture.uri, texture.hdr);
    } else {
      load_imageb(fs::path(dirname) / texture.uri, texture.ldr);
    }
  }
}

void load_voltexture(yocto_voltexture& texture, const string& dirname) {
  if (is_preset_filename(texture.uri)) {
    auto [type, nfilename] = get_preset_type(texture.uri);
    make_volpreset(texture.vol, type);
    texture.uri = nfilename;
  } else {
    load_volume(fs::path(dirname) / texture.uri, texture.vol);
  }
}

void load_textures(
    yocto_scene& scene, const string& dirname, const load_params& params) {
  if (params.notextures) return;

  // load images
  if (params.noparallel) {
    for (auto& texture : scene.textures) {
      if (params.cancel && *params.cancel) break;
      if (!texture.hdr.empty() || !texture.ldr.empty()) return;
      load_texture(texture, dirname);
    }
  } else {
    parallel_foreach(
        scene.textures,
        [&dirname](yocto_texture& texture) {
          if (!texture.hdr.empty() || !texture.ldr.empty()) return;
          load_texture(texture, dirname);
        },
        params.cancel);
  }

  // load volumes
  if (params.noparallel) {
    for (auto& texture : scene.voltextures) {
      if (params.cancel && *params.cancel) break;
      if (!texture.vol.empty()) return;
      load_voltexture(texture, dirname);
    }
  } else {
    parallel_foreach(
        scene.voltextures,
        [&dirname](yocto_voltexture& texture) {
          if (!texture.vol.empty()) return;
          load_voltexture(texture, dirname);
        },
        params.cancel);
  }
}

void save_texture(const yocto_texture& texture, const string& dirname) {
  if (!texture.hdr.empty()) {
    save_image(fs::path(dirname) / texture.uri, texture.hdr);
  } else {
    save_imageb(fs::path(dirname) / texture.uri, texture.ldr);
  }
}

void save_voltexture(const yocto_voltexture& texture, const string& dirname) {
  save_volume(fs::path(dirname) / texture.uri, texture.vol);
}

// helper to save textures
void save_textures(const yocto_scene& scene, const string& dirname,
    const save_params& params) {
  if (params.notextures) return;

  // save images
  if (params.noparallel) {
    for (auto& texture : scene.textures) {
      if (params.cancel && *params.cancel) break;
      save_texture(texture, dirname);
    }
  } else {
    parallel_foreach(
        scene.textures,
        [&dirname](
            const yocto_texture& texture) { save_texture(texture, dirname); },
        params.cancel);
  }

  // save volumes
  if (params.noparallel) {
    for (auto& texture : scene.voltextures) {
      if (params.cancel && *params.cancel) break;
      save_voltexture(texture, dirname);
    }
  } else {
    parallel_foreach(
        scene.voltextures,
        [&dirname](const yocto_voltexture& texture) {
          save_voltexture(texture, dirname);
        },
        params.cancel);
  }
}

void load_shape(yocto_shape& shape, const string& dirname) {
  if (is_preset_filename(shape.uri)) {
    auto [type, nfilename] = get_preset_type(shape.uri);
    make_shape_preset(shape.points, shape.lines, shape.triangles, shape.quads,
        shape.quadspos, shape.quadsnorm, shape.quadstexcoord, shape.positions,
        shape.normals, shape.texcoords, shape.colors, shape.radius, type);
    shape.uri = nfilename;
  } else {
    load_shape(fs::path(dirname) / shape.uri, shape.points, shape.lines,
        shape.triangles, shape.quads, shape.quadspos, shape.quadsnorm,
        shape.quadstexcoord, shape.positions, shape.normals, shape.texcoords,
        shape.colors, shape.radius, false);
  }
}

void save_shape(const yocto_shape& shape, const string& dirname) {
  save_shape(fs::path(dirname) / shape.uri, shape.points, shape.lines,
      shape.triangles, shape.quads, shape.quadspos, shape.quadsnorm,
      shape.quadstexcoord, shape.positions, shape.normals, shape.texcoords,
      shape.colors, shape.radius);
}

void load_subdiv(yocto_subdiv& subdiv, const string& dirname) {
  if (is_preset_filename(subdiv.uri)) {
    auto [type, nfilename] = get_preset_type(subdiv.uri);
    make_shape_preset(subdiv.points, subdiv.lines, subdiv.triangles,
        subdiv.quads, subdiv.quadspos, subdiv.quadsnorm, subdiv.quadstexcoord,
        subdiv.positions, subdiv.normals, subdiv.texcoords, subdiv.colors,
        subdiv.radius, type);
    subdiv.uri = nfilename;
  } else {
    load_shape(fs::path(dirname) / subdiv.uri, subdiv.points, subdiv.lines,
        subdiv.triangles, subdiv.quads, subdiv.quadspos, subdiv.quadsnorm,
        subdiv.quadstexcoord, subdiv.positions, subdiv.normals,
        subdiv.texcoords, subdiv.colors, subdiv.radius, subdiv.facevarying);
  }
}

void save_subdiv(const yocto_subdiv& subdiv, const string& dirname) {
  save_shape(fs::path(dirname) / subdiv.uri, subdiv.points, subdiv.lines,
      subdiv.triangles, subdiv.quads, subdiv.quadspos, subdiv.quadsnorm,
      subdiv.quadstexcoord, subdiv.positions, subdiv.normals, subdiv.texcoords,
      subdiv.colors, subdiv.radius);
}

// Load json meshes
void load_shapes(
    yocto_scene& scene, const string& dirname, const load_params& params) {
  // load shapes
  if (params.noparallel) {
    for (auto& shape : scene.shapes) {
      if (params.cancel && *params.cancel) break;
      load_shape(shape, dirname);
    }
  } else {
    parallel_foreach(
        scene.shapes,
        [&dirname](yocto_shape& shape) { load_shape(shape, dirname); },
        params.cancel);
  }

  // load subdivs
  if (params.noparallel) {
    for (auto& subdiv : scene.subdivs) {
      if (params.cancel && *params.cancel) break;
      load_subdiv(subdiv, dirname);
    }
  } else {
    parallel_foreach(
        scene.subdivs,
        [&dirname](yocto_subdiv& subdiv) { load_subdiv(subdiv, dirname); },
        params.cancel);
  }
}

// Save json meshes
void save_shapes(const yocto_scene& scene, const string& dirname,
    const save_params& params) {
  // save shapes
  if (params.noparallel) {
    for (auto& shape : scene.shapes) {
      if (params.cancel && *params.cancel) break;
      save_shape(shape, dirname);
    }
  } else {
    parallel_foreach(
        scene.shapes,
        [&dirname](const yocto_shape& shape) { save_shape(shape, dirname); },
        params.cancel);
  }
  // save subdivs
  if (params.noparallel) {
    for (auto& subdiv : scene.subdivs) {
      if (params.cancel && *params.cancel) break;
      save_subdiv(subdiv, dirname);
    }
  } else {
    parallel_foreach(
        scene.subdivs,
        [&dirname](
            const yocto_subdiv& subdiv) { save_subdiv(subdiv, dirname); },
        params.cancel);
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// YAML SUPPORT
// -----------------------------------------------------------------------------
namespace yocto {

void load_yaml(
    const string& filename, yocto_scene& scene, const load_params& params) {
  // open file
  auto fs = open_file(filename);

  // parse state
  enum struct parsing_type {
    // clang-format off
    none, camera, texture, voltexture, material, shape, subdiv, instance, environment
    // clang-format on
  };
  auto type = parsing_type::none;

  auto tmap = unordered_map<string, int>{{"", -1}};
  auto vmap = unordered_map<string, int>{{"", -1}};
  auto mmap = unordered_map<string, int>{{"", -1}};
  auto smap = unordered_map<string, int>{{"", -1}};

  // parse yaml reference
  auto get_yaml_ref = [](const yaml_value& yaml, int& value,
                          const unordered_map<string, int>& refs) {
    if (yaml.type != yaml_value_type::string)
      throw std::runtime_error("error parsing yaml value");
    if (yaml.string_ == "") return;
    try {
      value = refs.at(yaml.string_);
    } catch (...) {
      throw std::runtime_error("reference not found " + yaml.string_);
    }
  };

  // load yaml
  auto group  = ""s;
  auto key    = ""s;
  auto newobj = false;
  auto value  = yaml_value{};
  while (read_yaml_property(fs, group, key, newobj, value)) {
    if (group.empty()) {
      throw std::runtime_error("bad yaml");
    }
    if (key.empty()) {
      type = parsing_type::none;
      continue;
    }
    if (newobj) {
      if (group == "cameras") {
        type = parsing_type::camera;
        scene.cameras.push_back({});
      } else if (group == "textures") {
        type = parsing_type::texture;
        scene.textures.push_back({});
      } else if (group == "voltextures") {
        type = parsing_type::voltexture;
        scene.voltextures.push_back({});
      } else if (group == "materials") {
        type = parsing_type::material;
        scene.materials.push_back({});
      } else if (group == "shapes") {
        type = parsing_type::shape;
        scene.shapes.push_back({});
      } else if (group == "subdivs") {
        type = parsing_type::subdiv;
        scene.subdivs.push_back({});
      } else if (group == "instances") {
        type = parsing_type::instance;
        scene.instances.push_back({});
      } else if (group == "environments") {
        type = parsing_type::environment;
        scene.environments.push_back({});
      } else {
        type = parsing_type::none;
        throw std::runtime_error("unknown object type " + string(group));
      }
    }
    if (type == parsing_type::none) {
      throw std::runtime_error("bad yaml");
    } else if (type == parsing_type::camera) {
      auto& camera = scene.cameras.back();
      if (key == "uri") {
        get_yaml_value(value, camera.uri);
      } else if (key == "frame") {
        get_yaml_value(value, camera.frame);
      } else if (key == "orthographic") {
        get_yaml_value(value, camera.orthographic);
      } else if (key == "lens") {
        get_yaml_value(value, camera.lens);
      } else if (key == "film") {
        get_yaml_value(value, camera.film);
      } else if (key == "focus") {
        get_yaml_value(value, camera.focus);
      } else if (key == "aperture") {
        get_yaml_value(value, camera.aperture);
      } else if (key == "lookat") {
        auto lookat = identity3x3f;
        get_yaml_value(value, lookat);
        camera.frame = lookat_frame(lookat.x, lookat.y, lookat.z);
        camera.focus = length(lookat.x - lookat.y);
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::texture) {
      auto& texture = scene.textures.back();
      if (key == "uri") {
        get_yaml_value(value, texture.uri);
        auto refname = texture.uri;
        if (is_preset_filename(refname)) {
          refname = get_preset_type(refname).second;
        }
        tmap[refname] = (int)scene.textures.size() - 1;
      } else if (key == "filename") {
        get_yaml_value(value, texture.uri);
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::voltexture) {
      auto& texture = scene.voltextures.back();
      if (key == "uri") {
        get_yaml_value(value, texture.uri);
        auto refname = texture.uri;
        if (is_preset_filename(refname)) {
          refname = get_preset_type(refname).second;
        }
        vmap[refname] = (int)scene.voltextures.size() - 1;
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::material) {
      auto& material = scene.materials.back();
      if (key == "uri") {
        get_yaml_value(value, material.uri);
        mmap[material.uri] = (int)scene.materials.size() - 1;
      } else if (key == "emission") {
        get_yaml_value(value, material.emission);
      } else if (key == "diffuse") {
        get_yaml_value(value, material.diffuse);
      } else if (key == "metallic") {
        get_yaml_value(value, material.metallic);
      } else if (key == "specular") {
        get_yaml_value(value, material.specular);
      } else if (key == "roughness") {
        get_yaml_value(value, material.roughness);
      } else if (key == "coat") {
        get_yaml_value(value, material.coat);
      } else if (key == "transmission") {
        get_yaml_value(value, material.transmission);
      } else if (key == "refraction") {
        get_yaml_value(value, material.refraction);
      } else if (key == "voltransmission") {
        get_yaml_value(value, material.voltransmission);
      } else if (key == "volmeanfreepath") {
        get_yaml_value(value, material.volmeanfreepath);
      } else if (key == "volscatter") {
        get_yaml_value(value, material.volscatter);
      } else if (key == "volemission") {
        get_yaml_value(value, material.volemission);
      } else if (key == "volanisotropy") {
        get_yaml_value(value, material.volanisotropy);
      } else if (key == "volscale") {
        get_yaml_value(value, material.volscale);
      } else if (key == "opacity") {
        get_yaml_value(value, material.opacity);
      } else if (key == "coat") {
        get_yaml_value(value, material.coat);
      } else if (key == "emission_tex") {
        get_yaml_ref(value, material.emission_tex, tmap);
      } else if (key == "diffuse_tex") {
        get_yaml_ref(value, material.diffuse_tex, tmap);
      } else if (key == "metallic_tex") {
        get_yaml_ref(value, material.metallic_tex, tmap);
      } else if (key == "specular_tex") {
        get_yaml_ref(value, material.specular_tex, tmap);
      } else if (key == "transmission_tex") {
        get_yaml_ref(value, material.transmission_tex, tmap);
      } else if (key == "refraction_tex") {
        get_yaml_ref(value, material.refraction_tex, tmap);
      } else if (key == "roughness_tex") {
        get_yaml_ref(value, material.roughness_tex, tmap);
      } else if (key == "subsurface_tex") {
        get_yaml_ref(value, material.subsurface_tex, tmap);
      } else if (key == "opacity_tex") {
        get_yaml_ref(value, material.normal_tex, tmap);
      } else if (key == "normal_tex") {
        get_yaml_ref(value, material.normal_tex, tmap);
      } else if (key == "voldensity_tex") {
        get_yaml_ref(value, material.voldensity_tex, vmap);
      } else if (key == "gltf_textures") {
        get_yaml_value(value, material.gltf_textures);
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::shape) {
      auto& shape = scene.shapes.back();
      if (key == "uri") {
        get_yaml_value(value, shape.uri);
        auto refname = shape.uri;
        if (is_preset_filename(refname)) {
          refname = get_preset_type(refname).second;
        }
        smap[refname] = (int)scene.shapes.size() - 1;
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::subdiv) {
      auto& subdiv = scene.subdivs.back();
      if (key == "uri") {
        get_yaml_value(value, subdiv.uri);
      } else if (key == "shape") {
        get_yaml_ref(value, subdiv.shape, smap);
      } else if (key == "subdivisions") {
        get_yaml_value(value, subdiv.subdivisions);
      } else if (key == "catmullclark") {
        get_yaml_value(value, subdiv.catmullclark);
      } else if (key == "smooth") {
        get_yaml_value(value, subdiv.smooth);
      } else if (key == "facevarying") {
        get_yaml_value(value, subdiv.facevarying);
      } else if (key == "displacement_tex") {
        get_yaml_ref(value, subdiv.displacement_tex, tmap);
      } else if (key == "displacement") {
        get_yaml_value(value, subdiv.displacement);
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::instance) {
      auto& instance = scene.instances.back();
      if (key == "uri") {
        get_yaml_value(value, instance.uri);
      } else if (key == "frame") {
        get_yaml_value(value, instance.frame);
      } else if (key == "shape") {
        get_yaml_ref(value, instance.shape, smap);
      } else if (key == "material") {
        get_yaml_ref(value, instance.material, mmap);
      } else if (key == "lookat") {
        auto lookat = identity3x3f;
        get_yaml_value(value, lookat);
        instance.frame = lookat_frame(lookat.x, lookat.y, lookat.z, true);
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else if (type == parsing_type::environment) {
      auto& environment = scene.environments.back();
      if (key == "uri") {
        get_yaml_value(value, environment.uri);
      } else if (key == "frame") {
        get_yaml_value(value, environment.frame);
      } else if (key == "emission") {
        get_yaml_value(value, environment.emission);
      } else if (key == "emission_tex") {
        get_yaml_ref(value, environment.emission_tex, tmap);
      } else if (key == "lookat") {
        auto lookat = identity3x3f;
        get_yaml_value(value, lookat);
        environment.frame = lookat_frame(lookat.x, lookat.y, lookat.z, true);
      } else {
        throw std::runtime_error("unknown property " + string(key));
      }
    } else {
      assert(false);  // should not get here
    }
  }
}

// Save a scene in the builtin YAML format.
static void load_yaml_scene(
    const string& filename, yocto_scene& scene, const load_params& params) {
  scene = {};

  // Parse yaml
  load_yaml(filename, scene, params);

  // load shape and textures
  auto dirname = fs::path(filename).parent_path();
  load_shapes(scene, dirname, params);
  load_textures(scene, dirname, params);

  // fix scene
  scene.uri = fs::path(filename).filename();
  add_cameras(scene);
  add_materials(scene);
  add_radius(scene);
  normalize_uris(scene);
  trim_memory(scene);
  update_transforms(scene);
}

// Save yaml
static void save_yaml(const string& filename, const yocto_scene& scene,
    bool ply_instances = false, const string& instances_name = "") {
  // open file
  auto fs = open_file(filename, "w");

  write_yaml_comment(fs, get_save_scene_message(scene, ""));

  static const auto def_camera      = yocto_camera{};
  static const auto def_texture     = yocto_texture{};
  static const auto def_voltexture  = yocto_voltexture{};
  static const auto def_material    = yocto_material{};
  static const auto def_shape       = yocto_shape{};
  static const auto def_subdiv      = yocto_subdiv{};
  static const auto def_instance    = yocto_instance{};
  static const auto def_environment = yocto_environment{};

  auto yvalue = yaml_value{};

  if (!scene.cameras.empty()) write_yaml_object(fs, "cameras");
  for (auto& camera : scene.cameras) {
    write_yaml_property(
        fs, "cameras", "uri", true, make_yaml_value(camera.uri));
    if (camera.frame != identity3x4f)
      write_yaml_property(
          fs, "cameras", "frame", false, make_yaml_value(camera.frame));
    if (camera.orthographic)
      write_yaml_property(fs, "cameras", "orthographic", false,
          make_yaml_value(camera.orthographic));
    write_yaml_property(
        fs, "cameras", "lens", false, make_yaml_value(camera.lens));
    write_yaml_property(
        fs, "cameras", "film", false, make_yaml_value(camera.film));
    write_yaml_property(
        fs, "cameras", "focus", false, make_yaml_value(camera.focus));
    if (camera.aperture)
      write_yaml_property(
          fs, "cameras", "aperture", false, make_yaml_value(camera.aperture));
  }

  if (!scene.textures.empty()) write_yaml_object(fs, "textures");
  for (auto& texture : scene.textures) {
    write_yaml_property(
        fs, "textures", "uri", true, make_yaml_value(texture.uri));
  }

  if (!scene.voltextures.empty()) write_yaml_object(fs, "voltextures");
  for (auto& texture : scene.voltextures) {
    write_yaml_property(
        fs, "voltextures", "uri", true, make_yaml_value(texture.uri));
  }

  if (!scene.materials.empty()) write_yaml_object(fs, "materials");
  for (auto& material : scene.materials) {
    write_yaml_property(
        fs, "materials", "uri", true, make_yaml_value(material.uri));
    if (material.emission != zero3f)
      write_yaml_property(fs, "materials", "emission", false,
          make_yaml_value(material.emission));
    if (material.diffuse != zero3f)
      write_yaml_property(
          fs, "materials", "diffuse", false, make_yaml_value(material.diffuse));
    if (material.specular != zero3f)
      write_yaml_property(fs, "materials", "specular", false,
          make_yaml_value(material.specular));
    if (material.metallic)
      write_yaml_property(fs, "materials", "metallic", false,
          make_yaml_value(material.metallic));
    if (material.transmission != zero3f)
      write_yaml_property(fs, "materials", "transmission", false,
          make_yaml_value(material.transmission));
    if (material.refraction != zero3f)
      write_yaml_property(fs, "materials", "refraction", false,
          make_yaml_value(material.refraction));
    write_yaml_property(fs, "materials", "roughness", false,
        make_yaml_value(material.roughness));
    if (material.voltransmission != zero3f)
      write_yaml_property(fs, "materials", "voltransmission", false,
          make_yaml_value(material.voltransmission));
    if (material.volmeanfreepath != zero3f)
      write_yaml_property(fs, "materials", "volmeanfreepath", false,
          make_yaml_value(material.volmeanfreepath));
    if (material.volscatter != zero3f)
      write_yaml_property(fs, "materials", "volscatter", false,
          make_yaml_value(material.volscatter));
    if (material.volemission != zero3f)
      write_yaml_property(fs, "materials", "volemission", false,
          make_yaml_value(material.volemission));
    if (material.volanisotropy)
      write_yaml_property(fs, "materials", "volanisotropy", false,
          make_yaml_value(material.volanisotropy));
    if (material.voltransmission != zero3f ||
        material.volmeanfreepath != zero3f)
      write_yaml_property(fs, "materials", "volscale", false,
          make_yaml_value(material.volscale));
    if (material.coat != zero3f)
      write_yaml_property(
          fs, "materials", "coat", false, make_yaml_value(material.coat));
    if (material.opacity != 1)
      write_yaml_property(
          fs, "materials", "opacity", false, make_yaml_value(material.opacity));
    if (material.emission_tex >= 0)
      write_yaml_property(fs, "materials", "emission_tex", false,
          make_yaml_value(scene.textures[material.emission_tex].uri));
    if (material.diffuse_tex >= 0)
      write_yaml_property(fs, "materials", "diffuse_tex", false,
          make_yaml_value(scene.textures[material.diffuse_tex].uri));
    if (material.metallic_tex >= 0)
      write_yaml_property(fs, "materials", "metallic_tex", false,
          make_yaml_value(scene.textures[material.metallic_tex].uri));
    if (material.specular_tex >= 0)
      write_yaml_property(fs, "materials", "specular_tex", false,
          make_yaml_value(scene.textures[material.specular_tex].uri));
    if (material.roughness_tex >= 0)
      write_yaml_property(fs, "materials", "roughness_tex", false,
          make_yaml_value(scene.textures[material.roughness_tex].uri));
    if (material.transmission_tex >= 0)
      write_yaml_property(fs, "materials", "transmission_tex", false,
          make_yaml_value(scene.textures[material.transmission_tex].uri));
    if (material.refraction_tex >= 0)
      write_yaml_property(fs, "materials", "refraction_tex", false,
          make_yaml_value(scene.textures[material.refraction_tex].uri));
    if (material.subsurface_tex >= 0)
      write_yaml_property(fs, "materials", "subsurface_tex", false,
          make_yaml_value(scene.textures[material.subsurface_tex].uri));
    if (material.coat_tex >= 0)
      write_yaml_property(fs, "materials", "coat_tex", false,
          make_yaml_value(scene.textures[material.coat_tex].uri));
    if (material.opacity_tex >= 0)
      write_yaml_property(fs, "materials", "opacity_tex", false,
          make_yaml_value(scene.textures[material.opacity_tex].uri));
    if (material.normal_tex >= 0)
      write_yaml_property(fs, "materials", "normal_tex", false,
          make_yaml_value(scene.textures[material.normal_tex].uri));
    if (material.gltf_textures)
      write_yaml_property(fs, "materials", "gltf_textures", false,
          make_yaml_value(material.gltf_textures));
    if (material.voldensity_tex >= 0)
      write_yaml_property(fs, "materials", "voldensity_tex", false,
          make_yaml_value(scene.voltextures[material.voldensity_tex].uri));
  }

  if (!scene.shapes.empty()) write_yaml_object(fs, "shapes");
  for (auto& shape : scene.shapes) {
    write_yaml_property(fs, "shapes", "uri", true, make_yaml_value(shape.uri));
  }

  if (!scene.subdivs.empty()) write_yaml_object(fs, "subdivs");
  for (auto& subdiv : scene.subdivs) {
    write_yaml_property(
        fs, "subdivs", "uri", true, make_yaml_value(subdiv.uri));
    if (subdiv.shape >= 0)
      write_yaml_property(fs, "subdivs", "shape", false,
          make_yaml_value(scene.shapes[subdiv.shape].uri));
    write_yaml_property(fs, "subdivs", "subdivisions", false,
        make_yaml_value(subdiv.subdivisions));
    write_yaml_property(fs, "subdivs", "catmullclark", false,
        make_yaml_value(subdiv.catmullclark));
    write_yaml_property(
        fs, "subdivs", "smooth", false, make_yaml_value(subdiv.smooth));
    if (subdiv.facevarying)
      write_yaml_property(fs, "subdivs", "facevarying", false,
          make_yaml_value(subdiv.facevarying));
    if (subdiv.displacement_tex >= 0)
      write_yaml_property(fs, "subdivs", "displacement_tex", false,
          make_yaml_value(scene.textures[subdiv.displacement_tex].uri));
    if (subdiv.displacement_tex >= 0)
      write_yaml_property(fs, "subdivs", "displacement", false,
          make_yaml_value(subdiv.displacement));
  }

  if (!ply_instances) {
    if (!scene.instances.empty()) write_yaml_object(fs, "instances");
    for (auto& instance : scene.instances) {
      write_yaml_property(
          fs, "instances", "uri", true, make_yaml_value(instance.uri));
      if (instance.frame != identity3x4f)
        write_yaml_property(
            fs, "instances", "frame", false, make_yaml_value(instance.frame));
      if (instance.shape >= 0)
        write_yaml_property(fs, "instances", "shape", false,
            make_yaml_value(scene.shapes[instance.shape].uri));
      if (instance.material >= 0)
        write_yaml_property(fs, "instances", "material", false,
            make_yaml_value(scene.materials[instance.material].uri));
    }
  } else {
    if (!scene.instances.empty()) write_yaml_object(fs, "ply_instances");
    write_yaml_property(
        fs, "ply_instances", "uri", true, make_yaml_value(instances_name));
  }

  if (!scene.environments.empty()) write_yaml_object(fs, "environments");
  for (auto& environment : scene.environments) {
    write_yaml_property(
        fs, "environments", "uri", true, make_yaml_value(environment.uri));
    if (environment.frame != identity3x4f)
      write_yaml_property(fs, "environments", "frame", false,
          make_yaml_value(environment.frame));
    write_yaml_property(fs, "environments", "emission", false,
        make_yaml_value(environment.emission));
    if (environment.emission_tex >= 0)
      write_yaml_property(fs, "environments", "emission_tex", false,
          make_yaml_value(scene.textures[environment.emission_tex].uri));
  }
}

// Save a scene in the builtin YAML format.
static void save_yaml_scene(const string& filename, const yocto_scene& scene,
    const save_params& params) {
  try {
    // save yaml file
    save_yaml(filename, scene);

    // save meshes and textures
    auto dirname = fs::path(filename).parent_path();
    save_shapes(scene, dirname, params);
    save_textures(scene, dirname, params);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot save scene " + filename + "\n" + e.what());
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// OBJ CONVERSION
// -----------------------------------------------------------------------------
namespace yocto {

// Loads an MTL
static void load_mtl(const string& filename, yocto_scene& scene,
    unordered_map<string, int>& mmap, unordered_map<string, int>& tmap,
    const load_params& params) {
  // open file
  auto fs = open_file(filename);

  // parsing type
  enum struct parsing_type { none, material };
  auto ptype = parsing_type::none;

  // Parse texture params and name
  auto add_texture = [&scene, &tmap](const obj_texture_info& info,
                         bool force_linear) -> int {
    if (info.path == "") return -1;
    if (tmap.find(info.path) != tmap.end()) {
      return tmap.at(info.path);
    }

    // create texture
    auto texture = yocto_texture{};
    texture.uri  = info.path;
    for (auto& c : texture.uri)
      if (c == '\\') c = '/';
    scene.textures.push_back(texture);
    auto index      = (int)scene.textures.size() - 1;
    tmap[info.path] = index;

    return index;
  };

  // read mtl elements
  auto command = mtl_command{};
  auto value   = obj_value{};
  auto texture = obj_texture_info{};
  while (read_mtl_command(fs, command, value, texture)) {
    if (command == mtl_command::material) {
      auto& material = scene.materials.emplace_back();
      get_obj_value(value, material.uri);
      mmap[material.uri] = (int)scene.materials.size() - 1;
      ptype              = parsing_type::material;
      continue;
    }
    if (ptype == parsing_type::none) {
      throw std::runtime_error("bad mtl");
    } else {
      auto& material = scene.materials.back();
      switch (command) {
        case mtl_command::emission:
          get_obj_value(value, material.emission);
          break;
        case mtl_command::diffuse:
          get_obj_value(value, material.diffuse);
          break;
        case mtl_command::specular:
          get_obj_value(value, material.specular);
          break;
        case mtl_command::transmission:
          get_obj_value(value, material.transmission);
          break;
        case mtl_command::exponent:
          get_obj_value(value, material.roughness);
          material.roughness = pow(2 / (material.roughness + 2), 1 / 4.0f);
          if (material.roughness < 0.01f) material.roughness = 0;
          if (material.roughness > 0.99f) material.roughness = 1;
          break;
        case mtl_command::opacity:
          get_obj_value(value, material.opacity);
          break;
        case mtl_command::emission_map:
          material.emission_tex = add_texture(texture, false);
          break;
        case mtl_command::diffuse_map:
          material.diffuse_tex = add_texture(texture, false);
          break;
        case mtl_command::specular_map:
          material.specular_tex = add_texture(texture, false);
          break;
        case mtl_command::transmission_map:
          material.transmission_tex = add_texture(texture, false);
          break;
        case mtl_command::opacity_map:
          material.opacity_tex = add_texture(texture, true);
          break;
        case mtl_command::normal_map:
          material.normal_tex = add_texture(texture, true);
          break;
        case mtl_command::pbr_roughness:
          get_obj_value(value, material.roughness);
          break;
        case mtl_command::pbr_metallic:
          get_obj_value(value, material.metallic);
          break;
        case mtl_command::pbr_roughness_map:
          material.roughness_tex = add_texture(texture, true);
          break;
        case mtl_command::pbr_metallic_map:
          material.metallic_tex = add_texture(texture, true);
          break;
        case mtl_command::vol_transmission:
          get_obj_value(value, material.voltransmission);
          break;
        case mtl_command::vol_meanfreepath:
          get_obj_value(value, material.volmeanfreepath);
          break;
        case mtl_command::vol_scattering:
          get_obj_value(value, material.volscatter);
          break;
        case mtl_command::vol_emission:
          get_obj_value(value, material.volemission);
          break;
        case mtl_command::vol_anisotropy:
          get_obj_value(value, material.volanisotropy);
          break;
        case mtl_command::vol_scale:
          get_obj_value(value, material.volscale);
          break;
        case mtl_command::vol_scattering_map:
          material.subsurface_tex = add_texture(texture, false);
          break;
        default: break;  // ignore other values
      }
    }
  }
}

// Loads an OBJX
static void load_objx(const string& filename, yocto_scene& scene,
    const unordered_map<string, int>& mmap, unordered_map<string, int>& tmap,
    const unordered_map<string, vector<int>>& object_shapes,
    const load_params&                        params) {
  // open file
  auto fs = open_file(filename);

  // parsing types
  enum struct parsing_type { none, camera, environment, instance, procedural };
  auto ptype = parsing_type::none;

  // Parse texture params and name
  auto add_texture = [&scene, &tmap](const obj_texture_info& info,
                         bool force_linear) -> int {
    if (info.path == "") return -1;
    if (tmap.find(info.path) != tmap.end()) {
      return tmap.at(info.path);
    }

    // create texture
    auto texture = yocto_texture{};
    texture.uri  = info.path;
    for (auto& c : texture.uri)
      if (c == '\\') c = '/';
    scene.textures.push_back(texture);
    auto index      = (int)scene.textures.size() - 1;
    tmap[info.path] = index;

    return index;
  };

  // parsed geometry and materials
  bool first_instance = true;
  auto instances_idx  = vector<int>{};

  // read mtl elements
  auto command = objx_command{};
  auto value   = obj_value{};
  auto texture = obj_texture_info{};
  while (read_objx_command(fs, command, value, texture)) {
    if (command == objx_command::camera) {
      auto& camera = scene.cameras.emplace_back();
      get_obj_value(value, camera.uri);
      ptype = parsing_type::camera;
      continue;
    }
    if (command == objx_command::environment) {
      auto& environment = scene.environments.emplace_back();
      get_obj_value(value, environment.uri);
      ptype = parsing_type::environment;
      continue;
    }
    if (command == objx_command::instance) {
      if (first_instance) {
        scene.instances.clear();
        first_instance = false;
      }
      auto& instance = scene.instances.emplace_back();
      get_obj_value(value, instance.uri);
      ptype         = parsing_type::instance;
      instances_idx = {(int)scene.instances.size() - 1};
      continue;
    }
    if (command == objx_command::procedural) {
      auto& shape = scene.shapes.emplace_back();
      get_obj_value(value, shape.uri);
      auto& instance = scene.instances.emplace_back();
      instance.uri   = shape.uri;
      instance.shape = (int)scene.shapes.size() - 1;
      ptype          = parsing_type::procedural;
      continue;
    }

    if (ptype == parsing_type::none) {
      throw std::runtime_error("bad objx");
    } else if (ptype == parsing_type::camera) {
      auto& camera = scene.cameras.back();
      switch (command) {
        case objx_command::frame: get_obj_value(value, camera.frame); break;
        case objx_command::ortho:
          get_obj_value(value, camera.orthographic);
          break;
        case objx_command::width: get_obj_value(value, camera.film.x); break;
        case objx_command::height: get_obj_value(value, camera.film.y); break;
        case objx_command::lens: get_obj_value(value, camera.lens); break;
        case objx_command::aperture:
          get_obj_value(value, camera.aperture);
          break;
        case objx_command::focus: get_obj_value(value, camera.focus); break;
        default: throw std::runtime_error("bad objx"); break;
      }
    } else if (ptype == parsing_type::environment) {
      auto& environment = scene.environments.back();
      switch (command) {
        case objx_command::frame:
          get_obj_value(value, environment.frame);
          break;
        case objx_command::emission:
          get_obj_value(value, environment.emission);
          break;
        case objx_command::emission_map:
          environment.emission_tex = add_texture(texture, false);
          break;
        default: throw std::runtime_error("bad objx"); break;
      }
    } else if (ptype == parsing_type::instance) {
      switch (command) {
        case objx_command::frame: {
          for (auto& ist : instances_idx) {
            get_obj_value(value, scene.instances[ist].frame);
          }
        } break;
        case objx_command::material: {
          auto name = ""s;
          get_obj_value(value, name);
          auto ist_material = mmap.at(name);
          for (auto& ist : instances_idx) {
            scene.instances[ist].material = ist_material;
          }
        } break;
        case objx_command::object: {
          auto name = ""s;
          get_obj_value(value, name);
          auto& shapes = object_shapes.at(name);
          if (instances_idx.size() != shapes.size()) {
            auto to_add = shapes.size() - instances_idx.size();
            auto name   = scene.instances.back().uri;
            for (auto i = 0; i < to_add; i++) {
              auto& instance = scene.instances.emplace_back();
              instance.uri   = name;
              instances_idx.push_back((int)scene.instances.size() - 1);
            }
          }
          for (auto i = 0; i < shapes.size(); i++) {
            scene.instances[instances_idx[i]].shape = shapes[i];
          }
        } break;
        default: throw std::runtime_error("bad objx"); break;
      }
    } else if (ptype == parsing_type::procedural) {
      auto& shape    = scene.shapes.back();
      auto& instance = scene.instances.back();
      switch (command) {
        case objx_command::frame: get_obj_value(value, instance.frame); break;
        case objx_command::material: {
          auto name = ""s;
          get_obj_value(value, name);
          if (mmap.find(name) == mmap.end()) {
            throw std::runtime_error("missing material " + name);
          } else {
            instance.material = mmap.find(name)->second;
          }
        } break;
        case objx_command::object: {
          auto name = ""s;
          get_obj_value(value, name);
          if (name == "floor") {
            auto params         = proc_shape_params{};
            params.type         = proc_shape_params::type_t::floor;
            params.subdivisions = 0;
            params.scale        = 40 / 2;
            params.uvscale      = 40;
            make_proc_shape(shape.triangles, shape.quads, shape.positions,
                shape.normals, shape.texcoords, params);
          } else {
            throw std::runtime_error("unknown obj procedural");
          }
        } break;
        default: throw std::runtime_error("bad objx"); break;
      }
    } else {
      // skip other
      throw std::runtime_error("bad objx");
    }
  }
}

// Loads an OBJ
static void load_obj(
    const string& filename, yocto_scene& scene, const load_params& params) {
  // current parsing values
  string mname = ""s;
  string oname = ""s;
  string gname = ""s;

  // vertices
  auto opos      = std::deque<vec3f>{};
  auto onorm     = std::deque<vec3f>{};
  auto otexcoord = std::deque<vec2f>{};

  // object maps
  auto tmap = unordered_map<string, int>{{"", -1}};
  auto vmap = unordered_map<string, int>{{"", -1}};
  auto mmap = unordered_map<string, int>{{"", -1}};

  // vertex maps
  auto vertex_map   = unordered_map<obj_vertex, int>();
  auto pos_map      = unordered_map<int, int>();
  auto norm_map     = unordered_map<int, int>();
  auto texcoord_map = unordered_map<int, int>();

  // parsed geometry and materials
  auto object_shapes = unordered_map<string, vector<int>>{};

  // material libraries read already
  auto mlibs = vector<string>{};

  // current parse state
  bool facevarying_now = false;

  // Add  vertices to the current shape
  auto add_verts = [&](const vector<obj_vertex>& verts, yocto_shape& shape) {
    for (auto& vert : verts) {
      auto it = vertex_map.find(vert);
      if (it != vertex_map.end()) continue;
      auto& shape  = scene.shapes.back();
      auto  nverts = (int)shape.positions.size();
      vertex_map.insert(it, {vert, nverts});
      if (vert.position) shape.positions.push_back(opos.at(vert.position - 1));
      if (vert.texcoord)
        shape.texcoords.push_back(otexcoord.at(vert.texcoord - 1));
      if (vert.normal) shape.normals.push_back(onorm.at(vert.normal - 1));
      if (shape.normals.size() != 0 &&
          shape.normals.size() != shape.positions.size()) {
        while (shape.normals.size() != shape.positions.size())
          shape.normals.push_back({0, 0, 1});
      }
      if (shape.texcoords.size() != 0 &&
          shape.texcoords.size() != shape.positions.size()) {
        while (shape.texcoords.size() != shape.positions.size())
          shape.texcoords.push_back({0, 0});
      }
    }
  };

  // add vertex
  auto add_fvverts = [&](const vector<obj_vertex>& verts, yocto_shape& shape) {
    for (auto& vert : verts) {
      if (!vert.position) continue;
      auto pos_it = pos_map.find(vert.position);
      if (pos_it != pos_map.end()) continue;
      auto nverts = (int)shape.positions.size();
      pos_map.insert(pos_it, {vert.position, nverts});
      shape.positions.push_back(opos.at(vert.position - 1));
    }
    for (auto& vert : verts) {
      if (!vert.texcoord) continue;
      auto texcoord_it = texcoord_map.find(vert.texcoord);
      if (texcoord_it != texcoord_map.end()) continue;
      auto nverts = (int)shape.texcoords.size();
      texcoord_map.insert(texcoord_it, {vert.texcoord, nverts});
      shape.texcoords.push_back(otexcoord.at(vert.texcoord - 1));
    }
    for (auto& vert : verts) {
      if (!vert.normal) continue;
      auto norm_it = norm_map.find(vert.normal);
      if (norm_it != norm_map.end()) continue;
      auto nverts = (int)shape.normals.size();
      norm_map.insert(norm_it, {vert.normal, nverts});
      shape.normals.push_back(onorm.at(vert.normal - 1));
    }
  };

  // add object if needed
  auto add_shape = [&]() {
    auto shape      = yocto_shape{};
    shape.uri       = oname + gname;
    facevarying_now = params.facevarying ||
                      shape.uri.find("[yocto::facevarying]") != string::npos;
    scene.shapes.push_back(shape);
    auto instance     = yocto_instance{};
    instance.uri      = shape.uri;
    instance.shape    = (int)scene.shapes.size() - 1;
    instance.material = mmap.at(mname);
    scene.instances.push_back(instance);
    object_shapes[oname].push_back((int)scene.shapes.size() - 1);
    vertex_map.clear();
    pos_map.clear();
    norm_map.clear();
    texcoord_map.clear();
  };

  // open file
  auto fs = open_file(filename);

  // load obj elements
  auto element   = obj_command{};
  auto value     = obj_value{};
  auto vertices  = vector<obj_vertex>{};
  auto vert_size = obj_vertex{};
  while (read_obj_command(fs, element, value, vertices, vert_size)) {
    if (element == obj_command::vertex) {
      get_obj_value(value, opos.emplace_back());
    } else if (element == obj_command::normal) {
      get_obj_value(value, onorm.emplace_back());
    } else if (element == obj_command::texcoord) {
      get_obj_value(value, otexcoord.emplace_back());
      otexcoord.back().y = 1 - otexcoord.back().y;
    } else if (element == obj_command::face) {
      if (scene.shapes.empty()) add_shape();
      if (!scene.shapes.back().positions.empty() &&
          (!scene.shapes.back().lines.empty() ||
              !scene.shapes.back().points.empty())) {
        add_shape();
      }
      auto& shape = scene.shapes.back();
      if (!facevarying_now) {
        add_verts(vertices, shape);
        if (vertices.size() == 4) {
          shape.quads.push_back(
              {vertex_map.at(vertices[0]), vertex_map.at(vertices[1]),
                  vertex_map.at(vertices[2]), vertex_map.at(vertices[3])});
        } else {
          for (auto i = 2; i < vertices.size(); i++)
            shape.triangles.push_back({vertex_map.at(vertices[0]),
                vertex_map.at(vertices[i - 1]), vertex_map.at(vertices[i])});
        }
      } else {
        add_fvverts(vertices, shape);
        if (vertices.size() == 4) {
          if (vertices[0].position) {
            shape.quadspos.push_back({pos_map.at(vertices[0].position),
                pos_map.at(vertices[1].position),
                pos_map.at(vertices[2].position),
                pos_map.at(vertices[3].position)});
          }
          if (vertices[0].texcoord) {
            shape.quadstexcoord.push_back(
                {texcoord_map.at(vertices[0].texcoord),
                    texcoord_map.at(vertices[1].texcoord),
                    texcoord_map.at(vertices[2].texcoord),
                    texcoord_map.at(vertices[3].texcoord)});
          }
          if (vertices[0].normal) {
            shape.quadsnorm.push_back({norm_map.at(vertices[0].normal),
                norm_map.at(vertices[1].normal),
                norm_map.at(vertices[2].normal),
                norm_map.at(vertices[3].normal)});
          }
        } else {
          if (vertices[0].position) {
            for (auto i = 2; i < vertices.size(); i++)
              shape.quadspos.push_back({pos_map.at(vertices[0].position),
                  pos_map.at(vertices[i - 1].position),
                  pos_map.at(vertices[i].position),
                  pos_map.at(vertices[i].position)});
          }
          if (vertices[0].texcoord) {
            for (auto i = 2; i < vertices.size(); i++)
              shape.quadstexcoord.push_back(
                  {texcoord_map.at(vertices[0].texcoord),
                      texcoord_map.at(vertices[i - 1].texcoord),
                      texcoord_map.at(vertices[i].texcoord),
                      texcoord_map.at(vertices[i].texcoord)});
          }
          if (vertices[0].normal) {
            for (auto i = 2; i < vertices.size(); i++)
              shape.quadsnorm.push_back({norm_map.at(vertices[0].normal),
                  norm_map.at(vertices[i - 1].normal),
                  norm_map.at(vertices[i].normal),
                  norm_map.at(vertices[i].normal)});
          }
        }
      }
    } else if (element == obj_command::line) {
      if (scene.shapes.empty()) add_shape();
      if (!scene.shapes.back().positions.empty() &&
          scene.shapes.back().lines.empty()) {
        add_shape();
      }
      auto& shape = scene.shapes.back();
      add_verts(vertices, shape);
      for (auto i = 1; i < vertices.size(); i++)
        shape.lines.push_back(
            {vertex_map.at(vertices[i - 1]), vertex_map.at(vertices[i])});
    } else if (element == obj_command::point) {
      if (scene.shapes.empty()) add_shape();
      if (!scene.shapes.back().positions.empty() &&
          scene.shapes.back().points.empty()) {
        add_shape();
      }
      auto& shape = scene.shapes.back();
      add_verts(vertices, shape);
      for (auto i = 0; i < vertices.size(); i++)
        shape.points.push_back(vertex_map.at(vertices[i]));
    } else if (element == obj_command::object) {
      get_obj_value(value, oname);
      gname = "";
      mname = "";
      add_shape();
    } else if (element == obj_command::group) {
      get_obj_value(value, gname);
      add_shape();
    } else if (element == obj_command::usemtl) {
      get_obj_value(value, mname);
      add_shape();
    } else if (element == obj_command::mtllib) {
      auto name = ""s;
      get_obj_value(value, name);
      if (std::find(mlibs.begin(), mlibs.end(), name) != mlibs.end()) continue;
      mlibs.push_back(name);
      auto mtlpath = fs::path(filename).parent_path() / name;
      load_mtl(mtlpath, scene, mmap, tmap, params);
    } else {
      // skip all other commands
    }
  }

  // check for extension
  auto extname = fs::path(filename).replace_extension(".objx");
  if (fs::exists(extname))
    load_objx(extname, scene, mmap, tmap, object_shapes, params);

  // cleanup empty
  auto shape_count = 0;
  auto shape_pos   = vector<int>(scene.shapes.size(), -1);
  for (auto shape = 0; shape < scene.shapes.size(); shape++) {
    if (!scene.shapes[shape].positions.empty())
      shape_pos[shape] = shape_count++;
  }
  scene.shapes.erase(std::remove_if(scene.shapes.begin(), scene.shapes.end(),
                         [](auto& shape) { return shape.positions.empty(); }),
      scene.shapes.end());
  for (auto& instance : scene.instances)
    instance.shape = shape_pos[instance.shape];
  scene.instances.erase(
      std::remove_if(scene.instances.begin(), scene.instances.end(),
          [](auto& instance) { return instance.shape < 0; }),
      scene.instances.end());

  // check if any empty shape is left
  for (auto& shape : scene.shapes) {
    if (shape.positions.empty())
      throw std::runtime_error("empty shapes not supported");
  }

  // merging quads and triangles
  for (auto& shape : scene.shapes) {
    if (shape.triangles.empty() || shape.quads.empty()) continue;
    merge_triangles_and_quads(shape.triangles, shape.quads, false);
  }
}

// Loads an OBJ
static void load_obj_scene(
    const string& filename, yocto_scene& scene, const load_params& params) {
  scene = {};

  // Parse obj
  load_obj(filename, scene, params);

  // load textures
  auto dirname = fs::path(filename).parent_path();
  load_textures(scene, dirname, params);

  // fix scene
  scene.uri = fs::path(filename).filename();
  add_cameras(scene);
  add_materials(scene);
  add_radius(scene);
  normalize_uris(scene);
  trim_memory(scene);
  update_transforms(scene);
}

static void save_obj(const string& filename, const yocto_scene& scene,
    bool preserve_instances, bool flip_texcoord = true) {
  // open writer
  auto fs = open_file(filename, "w");

  // stats
  write_obj_comment(fs, get_save_scene_message(scene, ""));

  // material library
  if (!scene.materials.empty())
    write_obj_command(fs, obj_command::mtllib,
        make_obj_value(
            fs::path(filename).replace_extension(".mtl").filename().string()));

  // shapes
  auto offset    = obj_vertex{0, 0, 0};
  auto instances = vector<yocto_instance>{};
  if (preserve_instances) {
    instances.reserve(scene.shapes.size());
    for (auto shape = 0; shape < scene.shapes.size(); shape++) {
      instances.push_back({scene.shapes[shape].uri, identity3x4f, shape, -1});
    }
  }
  for (auto& instance : preserve_instances ? instances : scene.instances) {
    auto& shape = scene.shapes[instance.shape];
    write_obj_command(fs, obj_command::object,
        make_obj_value(fs::path(instance.uri).stem().string()));
    if (instance.material >= 0)
      write_obj_command(fs, obj_command::usemtl,
          make_obj_value(fs::path(scene.materials[instance.material].uri)
                             .stem()
                             .string()));
    if (instance.frame == identity3x4f) {
      for (auto& p : shape.positions)
        write_obj_command(fs, obj_command::vertex, make_obj_value(p));
      for (auto& n : shape.normals)
        write_obj_command(fs, obj_command::normal, make_obj_value(n));
      for (auto& t : shape.texcoords)
        write_obj_command(fs, obj_command::texcoord,
            make_obj_value(vec2f{t.x, flip_texcoord ? 1 - t.y : t.y}));
    } else {
      for (auto& p : shape.positions) {
        write_obj_command(fs, obj_command::vertex,
            make_obj_value(transform_point(instance.frame, p)));
      }
      for (auto& n : shape.normals) {
        write_obj_command(fs, obj_command::normal,
            make_obj_value(transform_direction(instance.frame, n)));
      }
      for (auto& t : shape.texcoords)
        write_obj_command(fs, obj_command::texcoord,
            make_obj_value(vec2f{t.x, flip_texcoord ? 1 - t.y : t.y}));
    }
    auto mask = obj_vertex{
        1, shape.texcoords.empty() ? 0 : 1, shape.normals.empty() ? 0 : 1};
    auto vert = [mask, offset](int i) {
      return obj_vertex{(i + offset.position + 1) * mask.position,
          (i + offset.texcoord + 1) * mask.texcoord,
          (i + offset.normal + 1) * mask.normal};
    };
    auto fvvert = [mask, offset](int pi, int ti, int ni) {
      return obj_vertex{(pi + offset.position + 1) * mask.position,
          (ti + offset.texcoord + 1) * mask.texcoord,
          (ni + offset.normal + 1) * mask.normal};
    };
    auto elems = vector<obj_vertex>{};
    elems.resize(1);
    for (auto& p : shape.points) {
      elems[0] = vert(p);
      write_obj_command(fs, obj_command::point, {}, elems);
    }
    elems.resize(2);
    for (auto& l : shape.lines) {
      elems[0] = vert(l.x);
      elems[1] = vert(l.y);
      write_obj_command(fs, obj_command::line, {}, elems);
    }
    elems.resize(3);
    for (auto& t : shape.triangles) {
      elems[0] = vert(t.x);
      elems[1] = vert(t.y);
      elems[2] = vert(t.z);
      write_obj_command(fs, obj_command::face, {}, elems);
    }
    elems.resize(4);
    for (auto& q : shape.quads) {
      elems[0] = vert(q.x);
      elems[1] = vert(q.y);
      elems[2] = vert(q.z);
      if (q.z == q.w) {
        elems.resize(3);
      } else {
        elems.resize(4);
        elems[3] = vert(q.w);
      }
      write_obj_command(fs, obj_command::face, {}, elems);
    }
    elems.resize(4);
    for (auto i = 0; i < shape.quadspos.size(); i++) {
      auto qp = shape.quadspos.at(i);
      auto qt = !shape.quadstexcoord.empty() ? shape.quadstexcoord.at(i)
                                             : vec4i{-1, -1, -1, -1};
      auto qn = !shape.quadsnorm.empty() ? shape.quadsnorm.at(i)
                                         : vec4i{-1, -1, -1, -1};
      elems[0] = fvvert(qp.x, qt.x, qn.x);
      elems[1] = fvvert(qp.y, qt.y, qn.y);
      elems[2] = fvvert(qp.z, qt.z, qn.z);
      if (qp.z == qp.w) {
        elems.resize(3);
      } else {
        elems.resize(4);
        elems[3] = fvvert(qp.w, qt.w, qn.w);
      }
      write_obj_command(fs, obj_command::face, {}, elems);
    }
    offset.position += shape.positions.size();
    offset.texcoord += shape.texcoords.size();
    offset.normal += shape.normals.size();
  }
}

static void save_mtl(const string& filename, const yocto_scene& scene) {
  // open writer
  auto fs = open_file(filename, "w");

  // stats
  write_obj_comment(fs, get_save_scene_message(scene, ""));

  // materials
  for (auto& material : scene.materials) {
    write_mtl_command(fs, mtl_command::material,
        make_obj_value(fs::path(material.uri).stem()));
    write_mtl_command(fs, mtl_command::illum, make_obj_value(2));
    if (material.emission != zero3f)
      write_mtl_command(
          fs, mtl_command::emission, make_obj_value(material.emission));
    auto kd = material.diffuse * (1 - material.metallic);
    auto ks = material.specular * (1 - material.metallic) +
              material.metallic * material.diffuse;
    write_mtl_command(fs, mtl_command::diffuse, make_obj_value(kd));
    write_mtl_command(fs, mtl_command::specular, make_obj_value(ks));
    if (material.transmission != zero3f)
      write_mtl_command(
          fs, mtl_command::transmission, make_obj_value(material.transmission));
    auto ns = (int)clamp(
        2 / pow(clamp(material.roughness, 0.0f, 0.99f) + 1e-10f, 4.0f) - 2,
        0.0f, 1.0e9f);
    write_mtl_command(fs, mtl_command::exponent, make_obj_value(ns));
    if (material.opacity != 1)
      write_mtl_command(
          fs, mtl_command::opacity, make_obj_value(material.opacity));
    if (material.emission_tex >= 0)
      write_mtl_command(fs, mtl_command::emission_map, {},
          scene.textures[material.emission_tex].uri);
    if (material.diffuse_tex >= 0)
      write_mtl_command(fs, mtl_command::diffuse_map, {},
          scene.textures[material.diffuse_tex].uri);
    if (material.specular_tex >= 0)
      write_mtl_command(fs, mtl_command::specular_map, {},
          scene.textures[material.specular_tex].uri);
    if (material.transmission_tex >= 0)
      write_mtl_command(fs, mtl_command::transmission_map, {},
          scene.textures[material.transmission_tex].uri);
    if (material.normal_tex >= 0)
      write_mtl_command(fs, mtl_command::normal_map, {},
          scene.textures[material.normal_tex].uri);
    if (material.voltransmission != zero3f ||
        material.volmeanfreepath != zero3f) {
      write_mtl_command(fs, mtl_command::vol_transmission,
          make_obj_value(material.voltransmission));
      write_mtl_command(fs, mtl_command::vol_meanfreepath,
          make_obj_value(material.volmeanfreepath));
      write_mtl_command(
          fs, mtl_command::vol_emission, make_obj_value(material.volemission));
      write_mtl_command(
          fs, mtl_command::vol_scattering, make_obj_value(material.volscatter));
      write_mtl_command(fs, mtl_command::vol_anisotropy,
          make_obj_value(material.volanisotropy));
      write_mtl_command(
          fs, mtl_command::vol_scale, make_obj_value(material.volscale));
    }
  }
}

static void save_objx(
    const string& filename, const yocto_scene& scene, bool preserve_instances) {
  // open writer
  auto fs = open_file(filename, "w");

  // stats
  write_obj_comment(fs, get_save_scene_message(scene, ""));

  // cameras
  for (auto& camera : scene.cameras) {
    write_objx_command(fs, objx_command::camera, make_obj_value(camera.uri));
    if (camera.orthographic)
      write_objx_command(
          fs, objx_command::ortho, make_obj_value((float)camera.orthographic));
    write_objx_command(fs, objx_command::width, make_obj_value(camera.film.x));
    write_objx_command(fs, objx_command::height, make_obj_value(camera.film.y));
    write_objx_command(fs, objx_command::lens, make_obj_value(camera.lens));
    write_objx_command(fs, objx_command::focus, make_obj_value(camera.focus));
    write_objx_command(
        fs, objx_command::aperture, make_obj_value(camera.aperture));
    write_objx_command(fs, objx_command::frame, make_obj_value(camera.frame));
  }

  // environments
  for (auto& environment : scene.environments) {
    write_objx_command(
        fs, objx_command::environment, make_obj_value(environment.uri));
    write_objx_command(
        fs, objx_command::emission, make_obj_value(environment.emission));
    if (environment.emission_tex >= 0)
      write_objx_command(fs, objx_command::emission_map, {},
          scene.textures[environment.emission_tex].uri);
    write_objx_command(
        fs, objx_command::frame, make_obj_value(environment.frame));
  }

  // instances
  if (preserve_instances) {
    for (auto& instance : scene.instances) {
      write_objx_command(
          fs, objx_command::instance, make_obj_value(instance.uri));
      write_objx_command(fs, objx_command::object,
          make_obj_value(scene.shapes[instance.shape].uri));
      write_objx_command(fs, objx_command::material,
          make_obj_value(scene.materials[instance.material].uri));
      write_objx_command(
          fs, objx_command::frame, make_obj_value(instance.frame));
    }
  }
}

static void save_obj_scene(const string& filename, const yocto_scene& scene,
    const save_params& params) {
  try {
    save_obj(filename, scene, params.objinstances, true);
    if (!scene.materials.empty())
      save_mtl(fs::path(filename).replace_extension(".mtl"), scene);
    if (!scene.cameras.empty() || !scene.cameras.empty() ||
        (!scene.instances.empty() && params.objinstances))
      save_objx(fs::path(filename).replace_extension(".objx"), scene,
          params.objinstances);
    auto dirname = fs::path(filename).parent_path();
    save_textures(scene, dirname, params);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot save scene " + filename + "\n" + e.what());
  }
}

void print_obj_camera(const yocto_camera& camera) {
  printf("c %s %d %g %g %g %g %g %g %g %g %g %g%g %g %g %g %g %g %g\n",
      fs::path(camera.uri).stem().c_str(), (int)camera.orthographic,
      camera.film.x, camera.film.y, camera.lens, camera.focus, camera.aperture,
      camera.frame.x.x, camera.frame.x.y, camera.frame.x.z, camera.frame.y.x,
      camera.frame.y.y, camera.frame.y.z, camera.frame.z.x, camera.frame.z.y,
      camera.frame.z.z, camera.frame.o.x, camera.frame.o.y, camera.frame.o.z);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// PLY CONVERSION
// -----------------------------------------------------------------------------
namespace yocto {

static void load_ply_scene(
    const string& filename, yocto_scene& scene, const load_params& params) {
  scene = {};

  try {
    // load ply mesh
    scene.shapes.push_back({});
    auto& shape = scene.shapes.back();
    load_shape(filename, shape.points, shape.lines, shape.triangles,
        shape.quads, shape.quadspos, shape.quadsnorm, shape.quadstexcoord,
        shape.positions, shape.normals, shape.texcoords, shape.colors,
        shape.radius, false);

    // add instance
    auto instance  = yocto_instance{};
    instance.uri   = shape.uri;
    instance.shape = 0;
    scene.instances.push_back(instance);

  } catch (const std::exception& e) {
    throw std::runtime_error("cannot load scene " + filename + "\n" + e.what());
  }

  // fix scene
  scene.uri = fs::path(filename).filename();
  add_cameras(scene);
  add_materials(scene);
  add_radius(scene);
  normalize_uris(scene);
  trim_memory(scene);
  update_transforms(scene);
}

static void save_ply_scene(const string& filename, const yocto_scene& scene,
    const save_params& params) {
  if (scene.shapes.empty()) {
    throw std::runtime_error("cannot save empty scene " + filename);
  }
  try {
    auto& shape = scene.shapes.front();
    save_shape(filename, shape.points, shape.lines, shape.triangles,
        shape.quads, shape.quadspos, shape.quadsnorm, shape.quadstexcoord,
        shape.positions, shape.normals, shape.texcoords, shape.colors,
        shape.radius);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot save scene " + filename + "\n" + e.what());
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// GLTF CONVESION
// -----------------------------------------------------------------------------
namespace yocto {

// convert gltf to scene
static void gltf_to_scene(const string& filename, yocto_scene& scene) {
  // load gltf
  auto params = cgltf_options{};
  memset(&params, 0, sizeof(params));
  auto data   = (cgltf_data*)nullptr;
  auto result = cgltf_parse_file(&params, filename.c_str(), &data);
  if (result != cgltf_result_success) {
    throw std::runtime_error("could not load gltf " + filename);
  }
  auto gltf = std::unique_ptr<cgltf_data, void (*)(cgltf_data*)>{
      data, cgltf_free};
  auto dirname = fs::path(filename).parent_path().string();
  if (dirname != "") dirname += "/";
  if (cgltf_load_buffers(&params, data, dirname.c_str()) !=
      cgltf_result_success) {
    throw std::runtime_error("could not load gltf buffers " + filename);
  }

  // convert textures
  auto _startswith = [](string_view str, string_view substr) {
    if (str.size() < substr.size()) return false;
    return str.substr(0, substr.size()) == substr;
  };
  auto imap = unordered_map<cgltf_image*, int>{};
  for (auto tid = 0; tid < gltf->images_count; tid++) {
    auto gimg    = &gltf->images[tid];
    auto texture = yocto_texture{};
    texture.uri  = (_startswith(gimg->uri, "data:"))
                      ? string("[glTF-static inline].png")
                      : gimg->uri;
    scene.textures.push_back(texture);
    imap[gimg] = tid;
  }

  // add a texture
  auto add_texture = [&imap](
                         const cgltf_texture_view& ginfo, bool force_linear) {
    if (!ginfo.texture || !ginfo.texture->image) return -1;
    auto gtxt = ginfo.texture;
    return imap.at(gtxt->image);
  };

  // convert materials
  auto mmap = unordered_map<cgltf_material*, int>{{nullptr, -1}};
  for (auto mid = 0; mid < gltf->materials_count; mid++) {
    auto gmat             = &gltf->materials[mid];
    auto material         = yocto_material();
    material.uri          = gmat->name ? gmat->name : "";
    material.emission     = {gmat->emissive_factor[0], gmat->emissive_factor[1],
        gmat->emissive_factor[2]};
    material.emission_tex = add_texture(gmat->emissive_texture, false);
    if (gmat->has_pbr_specular_glossiness) {
      material.gltf_textures = true;
      auto gsg               = &gmat->pbr_specular_glossiness;
      auto kb            = vec4f{gsg->diffuse_factor[0], gsg->diffuse_factor[1],
          gsg->diffuse_factor[2], gsg->diffuse_factor[3]};
      material.diffuse   = {kb.x, kb.y, kb.z};
      material.opacity   = kb.w;
      material.specular  = {gsg->specular_factor[0], gsg->specular_factor[1],
          gsg->specular_factor[2]};
      material.roughness = 1 - gsg->glossiness_factor;
      material.diffuse_tex  = add_texture(gsg->diffuse_texture, false);
      material.specular_tex = add_texture(
          gsg->specular_glossiness_texture, false);
      material.roughness_tex = material.specular_tex;
    } else if (gmat->has_pbr_metallic_roughness) {
      material.gltf_textures = true;
      auto gmr               = &gmat->pbr_metallic_roughness;
      auto kb = vec4f{gmr->base_color_factor[0], gmr->base_color_factor[1],
          gmr->base_color_factor[2], gmr->base_color_factor[3]};
      material.diffuse      = {kb.x, kb.y, kb.z};
      material.opacity      = kb.w;
      material.specular     = {0.04, 0.04, 0.04};
      material.metallic     = gmr->metallic_factor;
      material.roughness    = gmr->roughness_factor;
      material.diffuse_tex  = add_texture(gmr->base_color_texture, false);
      material.metallic_tex = add_texture(
          gmr->metallic_roughness_texture, true);
      material.roughness_tex = material.specular_tex;
    }
    material.normal_tex = add_texture(gmat->normal_texture, true);
    scene.materials.push_back(material);
    mmap[gmat] = (int)scene.materials.size() - 1;
  }

  // get values from accessors
  auto accessor_values =
      [](const cgltf_accessor* gacc,
          bool normalize = false) -> vector<std::array<double, 4>> {
    auto gview       = gacc->buffer_view;
    auto data        = (byte*)gview->buffer->data;
    auto offset      = gacc->offset + gview->offset;
    auto stride      = gview->stride;
    auto compTypeNum = gacc->component_type;
    auto count       = gacc->count;
    auto type        = gacc->type;
    auto ncomp       = 0;
    if (type == cgltf_type_scalar) ncomp = 1;
    if (type == cgltf_type_vec2) ncomp = 2;
    if (type == cgltf_type_vec3) ncomp = 3;
    if (type == cgltf_type_vec4) ncomp = 4;
    auto compSize = 1;
    if (compTypeNum == cgltf_component_type_r_16 ||
        compTypeNum == cgltf_component_type_r_16u) {
      compSize = 2;
    }
    if (compTypeNum == cgltf_component_type_r_32u ||
        compTypeNum == cgltf_component_type_r_32f) {
      compSize = 4;
    }
    if (!stride) stride = compSize * ncomp;
    auto vals = vector<std::array<double, 4>>(count, {{0.0, 0.0, 0.0, 1.0}});
    for (auto i = 0; i < count; i++) {
      auto d = data + offset + i * stride;
      for (auto c = 0; c < ncomp; c++) {
        if (compTypeNum == cgltf_component_type_r_8) {  // char
          vals[i][c] = (double)(*(char*)d);
          if (normalize) vals[i][c] /= SCHAR_MAX;
        } else if (compTypeNum == cgltf_component_type_r_8u) {  // byte
          vals[i][c] = (double)(*(byte*)d);
          if (normalize) vals[i][c] /= UCHAR_MAX;
        } else if (compTypeNum == cgltf_component_type_r_16) {  // short
          vals[i][c] = (double)(*(short*)d);
          if (normalize) vals[i][c] /= SHRT_MAX;
        } else if (compTypeNum ==
                   cgltf_component_type_r_16u) {  // unsigned short
          vals[i][c] = (double)(*(unsigned short*)d);
          if (normalize) vals[i][c] /= USHRT_MAX;
        } else if (compTypeNum == cgltf_component_type_r_32u) {  // unsigned int
          vals[i][c] = (double)(*(unsigned int*)d);
          if (normalize) vals[i][c] /= UINT_MAX;
        } else if (compTypeNum == cgltf_component_type_r_32f) {  // float
          vals[i][c] = (*(float*)d);
        }
        d += compSize;
      }
    }
    return vals;
  };

  // convert meshes
  auto meshes = unordered_map<cgltf_mesh*, vector<vec2i>>{{nullptr, {}}};
  for (auto mid = 0; mid < gltf->meshes_count; mid++) {
    auto gmesh    = &gltf->meshes[mid];
    meshes[gmesh] = {};
    for (auto sid = 0; sid < gmesh->primitives_count; sid++) {
      auto gprim = &gmesh->primitives[sid];
      if (!gprim->attributes_count) continue;
      auto shape = yocto_shape();
      shape.uri  = (gmesh->name ? gmesh->name : "") +
                  ((sid) ? std::to_string(sid) : string());
      for (auto aid = 0; aid < gprim->attributes_count; aid++) {
        auto gattr    = &gprim->attributes[aid];
        auto semantic = string(gattr->name ? gattr->name : "");
        auto gacc     = gattr->data;
        auto vals     = accessor_values(gacc);
        if (semantic == "POSITION") {
          shape.positions.reserve(vals.size());
          for (auto i = 0; i < vals.size(); i++)
            shape.positions.push_back(
                {(float)vals[i][0], (float)vals[i][1], (float)vals[i][2]});
        } else if (semantic == "NORMAL") {
          shape.normals.reserve(vals.size());
          for (auto i = 0; i < vals.size(); i++)
            shape.normals.push_back(
                {(float)vals[i][0], (float)vals[i][1], (float)vals[i][2]});
        } else if (semantic == "TEXCOORD" || semantic == "TEXCOORD_0") {
          shape.texcoords.reserve(vals.size());
          for (auto i = 0; i < vals.size(); i++)
            shape.texcoords.push_back({(float)vals[i][0], (float)vals[i][1]});
        } else if (semantic == "COLOR" || semantic == "COLOR_0") {
          shape.colors.reserve(vals.size());
          for (auto i = 0; i < vals.size(); i++)
            shape.colors.push_back({(float)vals[i][0], (float)vals[i][1],
                (float)vals[i][2], (float)vals[i][3]});
        } else if (semantic == "TANGENT") {
          shape.tangents.reserve(vals.size());
          for (auto i = 0; i < vals.size(); i++)
            shape.tangents.push_back({(float)vals[i][0], (float)vals[i][1],
                (float)vals[i][2], (float)vals[i][3]});
          for (auto& t : shape.tangents) t.w = -t.w;
        } else if (semantic == "RADIUS") {
          shape.radius.reserve(vals.size());
          for (auto i = 0; i < vals.size(); i++)
            shape.radius.push_back((float)vals[i][0]);
        } else {
          // ignore
        }
      }
      // indices
      if (!gprim->indices) {
        if (gprim->type == cgltf_primitive_type_triangles) {
          shape.triangles.reserve(shape.positions.size() / 3);
          for (auto i = 0; i < shape.positions.size() / 3; i++)
            shape.triangles.push_back({i * 3 + 0, i * 3 + 1, i * 3 + 2});
        } else if (gprim->type == cgltf_primitive_type_triangle_fan) {
          shape.triangles.reserve(shape.positions.size() - 2);
          for (auto i = 2; i < shape.positions.size(); i++)
            shape.triangles.push_back({0, i - 1, i});
        } else if (gprim->type == cgltf_primitive_type_triangle_strip) {
          shape.triangles.reserve(shape.positions.size() - 2);
          for (auto i = 2; i < shape.positions.size(); i++)
            shape.triangles.push_back({i - 2, i - 1, i});
        } else if (gprim->type == cgltf_primitive_type_lines) {
          shape.lines.reserve(shape.positions.size() / 2);
          for (auto i = 0; i < shape.positions.size() / 2; i++)
            shape.lines.push_back({i * 2 + 0, i * 2 + 1});
        } else if (gprim->type == cgltf_primitive_type_line_loop) {
          shape.lines.reserve(shape.positions.size());
          for (auto i = 1; i < shape.positions.size(); i++)
            shape.lines.push_back({i - 1, i});
          shape.lines.back() = {(int)shape.positions.size() - 1, 0};
        } else if (gprim->type == cgltf_primitive_type_line_strip) {
          shape.lines.reserve(shape.positions.size() - 1);
          for (auto i = 1; i < shape.positions.size(); i++)
            shape.lines.push_back({i - 1, i});
        } else if (gprim->type == cgltf_primitive_type_points) {
          // points
          throw std::runtime_error("points not supported");
        } else {
          throw std::runtime_error("unknown primitive type");
        }
      } else {
        auto indices = accessor_values(gprim->indices);
        if (gprim->type == cgltf_primitive_type_triangles) {
          shape.triangles.reserve(indices.size() / 3);
          for (auto i = 0; i < indices.size() / 3; i++)
            shape.triangles.push_back({(int)indices[i * 3 + 0][0],
                (int)indices[i * 3 + 1][0], (int)indices[i * 3 + 2][0]});
        } else if (gprim->type == cgltf_primitive_type_triangle_fan) {
          shape.triangles.reserve(indices.size() - 2);
          for (auto i = 2; i < indices.size(); i++)
            shape.triangles.push_back({(int)indices[0][0],
                (int)indices[i - 1][0], (int)indices[i][0]});
        } else if (gprim->type == cgltf_primitive_type_triangle_strip) {
          shape.triangles.reserve(indices.size() - 2);
          for (auto i = 2; i < indices.size(); i++)
            shape.triangles.push_back({(int)indices[i - 2][0],
                (int)indices[i - 1][0], (int)indices[i][0]});
        } else if (gprim->type == cgltf_primitive_type_lines) {
          shape.lines.reserve(indices.size() / 2);
          for (auto i = 0; i < indices.size() / 2; i++)
            shape.lines.push_back(
                {(int)indices[i * 2 + 0][0], (int)indices[i * 2 + 1][0]});
        } else if (gprim->type == cgltf_primitive_type_line_loop) {
          shape.lines.reserve(indices.size());
          for (auto i = 1; i < indices.size(); i++)
            shape.lines.push_back({(int)indices[i - 1][0], (int)indices[i][0]});
          shape.lines.back() = {
              (int)indices[indices.size() - 1][0], (int)indices[0][0]};
        } else if (gprim->type == cgltf_primitive_type_line_strip) {
          shape.lines.reserve(indices.size() - 1);
          for (auto i = 1; i < indices.size(); i++)
            shape.lines.push_back({(int)indices[i - 1][0], (int)indices[i][0]});
        } else if (gprim->type == cgltf_primitive_type_points) {
          throw std::runtime_error("points not supported");
        } else {
          throw std::runtime_error("unknown primitive type");
        }
      }
      scene.shapes.push_back(shape);
      meshes[gmesh].push_back(
          {(int)scene.shapes.size() - 1, mmap.at(gprim->material)});
    }
  }

  // convert cameras
  auto cmap = unordered_map<cgltf_camera*, int>{{nullptr, -1}};
  for (auto cid = 0; cid < gltf->cameras_count; cid++) {
    auto gcam           = &gltf->cameras[cid];
    auto camera         = yocto_camera{};
    camera.uri          = gcam->name ? gcam->name : "";
    camera.orthographic = gcam->type == cgltf_camera_type_orthographic;
    if (camera.orthographic) {
      // throw std::runtime_error("orthographic not supported well");
      auto ortho          = &gcam->orthographic;
      camera.aperture     = 0;
      camera.orthographic = true;
      camera.film         = {ortho->xmag, ortho->ymag};
    } else {
      auto persp      = &gcam->perspective;
      camera.aperture = 0;
      set_yperspective(camera, persp->yfov, persp->aspect_ratio, flt_max);
    }
    scene.cameras.push_back(camera);
    cmap[gcam] = (int)scene.cameras.size() - 1;
  }

  // convert nodes
  auto nmap = unordered_map<cgltf_node*, int>{{nullptr, -1}};
  for (auto nid = 0; nid < gltf->nodes_count; nid++) {
    auto gnde = &gltf->nodes[nid];
    auto node = yocto_scene_node{};
    node.uri  = gnde->name ? gnde->name : "";
    if (gnde->camera) node.camera = cmap.at(gnde->camera);
    if (gnde->has_translation) {
      node.translation = {
          gnde->translation[0], gnde->translation[1], gnde->translation[2]};
    }
    if (gnde->has_rotation) {
      node.rotation = {gnde->rotation[0], gnde->rotation[1], gnde->rotation[2],
          gnde->rotation[3]};
    }
    if (gnde->has_scale) {
      node.scale = {gnde->scale[0], gnde->scale[1], gnde->scale[2]};
    }
    if (gnde->has_matrix) {
      auto m     = gnde->matrix;
      node.local = frame3f(
          mat4f{{m[0], m[1], m[2], m[3]}, {m[4], m[5], m[6], m[7]},
              {m[8], m[9], m[10], m[11]}, {m[12], m[13], m[14], m[15]}});
    }
    scene.nodes.push_back(node);
    nmap[gnde] = (int)scene.nodes.size();
  }

  // set up parent pointers
  for (auto nid = 0; nid < gltf->nodes_count; nid++) {
    auto gnde = &gltf->nodes[nid];
    if (!gnde->children_count) continue;
    for (auto cid = 0; cid < gnde->children_count; cid++) {
      scene.nodes[nmap.at(gnde->children[cid])].parent = nid;
    }
  }

  // set up instances
  for (auto nid = 0; nid < gltf->nodes_count; nid++) {
    auto gnde = &gltf->nodes[nid];
    if (!gnde->mesh) continue;
    auto& node = scene.nodes[nid];
    auto& shps = meshes.at(gnde->mesh);
    if (shps.empty()) continue;
    if (shps.size() == 1) {
      auto instance     = yocto_instance();
      instance.uri      = node.uri;
      instance.shape    = shps[0].x;
      instance.material = shps[0].y;
      scene.instances.push_back(instance);
      node.instance = (int)scene.instances.size() - 1;
    } else {
      for (auto shp : shps) {
        auto& shape       = scene.shapes[shp.x];
        auto  instance    = yocto_instance();
        instance.uri      = node.uri + "_" + shape.uri;
        instance.shape    = shp.x;
        instance.material = shp.y;
        scene.instances.push_back(instance);
        auto child     = yocto_scene_node{};
        child.uri      = node.uri + "_" + shape.uri;
        child.parent   = nid;
        child.instance = (int)scene.instances.size() - 1;
        scene.nodes.push_back(child);
      }
    }
  }

  // hasher for later
  struct sampler_map_hash {
    size_t operator()(
        const pair<cgltf_animation_sampler*, cgltf_animation_path_type>& value)
        const {
      auto hasher1 = std::hash<cgltf_animation_sampler*>();
      auto hasher2 = std::hash<int>();
      auto h       = (size_t)0;
      h ^= hasher1(value.first) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= hasher2(value.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  // convert animations
  for (auto gid = 0; gid < gltf->animations_count; gid++) {
    auto ganm = &gltf->animations[gid];
    auto aid  = 0;
    auto sampler_map =
        unordered_map<pair<cgltf_animation_sampler*, cgltf_animation_path_type>,
            int, sampler_map_hash>();
    for (auto cid = 0; cid < ganm->channels_count; cid++) {
      auto gchannel = &ganm->channels[cid];
      auto path     = gchannel->target_path;
      if (sampler_map.find({gchannel->sampler, path}) == sampler_map.end()) {
        auto gsampler  = gchannel->sampler;
        auto animation = yocto_animation{};
        animation.uri  = (ganm->name ? ganm->name : "anim") +
                        std::to_string(aid++);
        animation.group = ganm->name ? ganm->name : "";
        auto input_view = accessor_values(gsampler->input);
        animation.times.resize(input_view.size());
        for (auto i = 0; i < input_view.size(); i++)
          animation.times[i] = input_view[i][0];
        switch (gsampler->interpolation) {
          case cgltf_interpolation_type_linear:
            animation.interpolation =
                yocto_animation::interpolation_type::linear;
            break;
          case cgltf_interpolation_type_step:
            animation.interpolation = yocto_animation::interpolation_type::step;
            break;
          case cgltf_interpolation_type_cubic_spline:
            animation.interpolation =
                yocto_animation::interpolation_type::bezier;
            break;
        }
        auto output_view = accessor_values(gsampler->output);
        switch (path) {
          case cgltf_animation_path_type_translation: {
            animation.translations.reserve(output_view.size());
            for (auto i = 0; i < output_view.size(); i++)
              animation.translations.push_back({(float)output_view[i][0],
                  (float)output_view[i][1], (float)output_view[i][2]});
          } break;
          case cgltf_animation_path_type_rotation: {
            animation.rotations.reserve(output_view.size());
            for (auto i = 0; i < output_view.size(); i++)
              animation.rotations.push_back(
                  {(float)output_view[i][0], (float)output_view[i][1],
                      (float)output_view[i][2], (float)output_view[i][3]});
          } break;
          case cgltf_animation_path_type_scale: {
            animation.scales.reserve(output_view.size());
            for (auto i = 0; i < output_view.size(); i++)
              animation.scales.push_back({(float)output_view[i][0],
                  (float)output_view[i][1], (float)output_view[i][2]});
          } break;
          case cgltf_animation_path_type_weights: {
            throw std::runtime_error("weights not supported for now");
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
            throw std::runtime_error("bad gltf animation");
          }
        }
        sampler_map[{gchannel->sampler, path}] = (int)scene.animations.size();
        scene.animations.push_back(animation);
      }
      scene.animations[sampler_map.at({gchannel->sampler, path})]
          .targets.push_back(nmap.at(gchannel->target_node));
    }
  }
}

// Load a scene
static void load_gltf_scene(
    const string& filename, yocto_scene& scene, const load_params& params) {
  // initialization
  scene = {};

  try {
    // load gltf
    gltf_to_scene(filename, scene);

    // load textures
    auto dirname = fs::path(filename).parent_path();
    load_textures(scene, dirname, params);

  } catch (const std::exception& e) {
    throw std::runtime_error("cannot load scene " + filename + "\n" + e.what());
  }

  // fix scene
  scene.uri = fs::path(filename).filename();
  add_cameras(scene);
  add_materials(scene);
  add_radius(scene);
  normalize_uris(scene);
  trim_memory(scene);
  update_transforms(scene);

  // fix cameras
  auto bbox = compute_bounds(scene);
  for (auto& camera : scene.cameras) {
    auto center   = (bbox.min + bbox.max) / 2;
    auto distance = dot(-camera.frame.z, center - camera.frame.o);
    if (distance > 0) camera.focus = distance;
  }
}

// begin/end objects and arrays
struct write_json_state {
  FILE*                    fs = nullptr;
  vector<pair<bool, bool>> stack;
};
static inline void write_json_text(write_json_state& state, const char* text) {
  if (fprintf(state.fs, "%s", text) < 0)
    throw std::runtime_error("cannot write json");
}
// static inline void write_json_text(
//     write_json_state& state, const string& text) {
//   if (fprintf(state.fs, "%s", text.c_str()) < 0)
//     throw std::runtime_error("cannot write json");
// }
static inline void _write_json_next(
    write_json_state& state, bool dedent = false) {
  static const char* indents[7] = {
      "", "  ", "    ", "      ", "        ", "          ", "            "};
  if (state.stack.empty()) return;
  write_json_text(state, state.stack.back().second ? ",\n" : "\n");
  write_json_text(
      state, indents[clamp((int)state.stack.size() + (dedent ? -1 : 0), 0, 6)]);
  state.stack.back().second = true;
}
static inline void _write_json_value(write_json_state& state, int value) {
  if (fprintf(state.fs, "%d", value) < 0)
    throw std::runtime_error("cannot write json");
}
static inline void _write_json_value(write_json_state& state, size_t value) {
  if (fprintf(state.fs, "%llu", (unsigned long long)value) < 0)
    throw std::runtime_error("cannot write json");
}
static inline void _write_json_value(write_json_state& state, float value) {
  if (fprintf(state.fs, "%g", value) < 0)
    throw std::runtime_error("cannot write json");
}
// static inline void _write_json_value(write_json_state& state, bool value) {
//   if (fprintf(state.fs, "%s", value ? "true" : "false") < 0)
//     throw std::runtime_error("cannot write json");
// }
static inline void _write_json_value(
    write_json_state& state, const string& value) {
  if (fprintf(state.fs, "\"%s\"", value.c_str()) < 0)
    throw std::runtime_error("cannot write json");
}
static inline void _write_json_value(
    write_json_state& state, const char* value) {
  if (fprintf(state.fs, "\"%s\"", value) < 0)
    throw std::runtime_error("cannot write json");
}
// static inline void _write_json_value(
//     write_json_state& state, const vec2f& value) {
//   if (fprintf(state.fs, "[%g, %g]", value.x, value.y) < 0)
//     throw std::runtime_error("cannot write json");
// }
static inline void _write_json_value(
    write_json_state& state, const vec3f& value) {
  if (fprintf(state.fs, "[%g, %g, %g]", value.x, value.y, value.z) < 0)
    throw std::runtime_error("cannot write json");
}
static inline void _write_json_value(
    write_json_state& state, const vec4f& value) {
  if (fprintf(
          state.fs, "[%g, %g, %g, %g]", value.x, value.y, value.z, value.w) < 0)
    throw std::runtime_error("cannot write json");
}
static inline void _write_json_value(
    write_json_state& state, const mat4f& value) {
  if (fprintf(state.fs, "[ ") < 0)
    throw std::runtime_error("cannot write json");
  for (auto i = 0; i < 16; i++) {
    if (fprintf(state.fs, i ? ", %g" : "%g", (&value.x.x)[i]) < 0)
      throw std::runtime_error("cannot write json");
  }
  if (fprintf(state.fs, " ]") < 0)
    throw std::runtime_error("cannot write json");
}
static inline void _write_json_value(
    write_json_state& state, const vector<int>& value) {
  write_json_text(state, "[ ");
  for (auto i = 0; i < value.size(); i++) {
    if (i) write_json_text(state, ", ");
    _write_json_value(state, value[i]);
  }
  write_json_text(state, " ]");
}
static inline void write_json_object(write_json_state& state) {
  _write_json_next(state);
  write_json_text(state, "{ ");
  state.stack.push_back({true, false});
}
static inline void write_json_object(write_json_state& state, const char* key) {
  _write_json_next(state);
  _write_json_value(state, key);
  write_json_text(state, ": {");
  state.stack.push_back({true, false});
}
// static inline void write_json_array(write_json_state& state) {
//   _write_json_next(state);
//   write_json_text(state, "[ ");
//   state.stack.push_back({false, false});
// }
static inline void write_json_array(write_json_state& state, const char* key) {
  _write_json_next(state);
  _write_json_value(state, key);
  write_json_text(state, ": [");
  state.stack.push_back({false, false});
}
static inline void write_json_pop(write_json_state& state) {
  _write_json_next(state, true);
  write_json_text(state, state.stack.back().first ? "}" : "]");
  state.stack.pop_back();
}
template <typename T>
static inline void write_json_value(write_json_state& state, const T& value) {
  _write_json_next(state);
  _write_json_value(state, value);
}
template <typename T>
static inline void write_json_value(
    write_json_state& state, const char* key, const T& value) {
  _write_json_next(state);
  _write_json_value(state, key);
  write_json_text(state, ": ");
  _write_json_value(state, value);
}
static inline void write_json_begin(write_json_state& state) {
  state.stack.clear();
  write_json_object(state);
}
static inline void write_json_end(write_json_state& state) {
  write_json_pop(state);
  if (state.stack.empty()) throw std::runtime_error("bad json stack");
}

// convert gltf scene to json
static void save_gltf(const string& filename, const yocto_scene& scene) {
  // shapes
  struct gltf_shape {
    string        uri       = "";
    int           material  = -1;
    int           mode      = 0;
    vector<int>   indices   = {};
    vector<vec3f> positions = {};
    vector<vec3f> normals   = {};
    vector<vec2f> texcoords = {};
    vector<vec4f> colors    = {};
    vector<float> radius    = {};
    vector<vec4f> tangents  = {};
  };

  // json writer
  auto fs_   = open_file(filename, "w");
  auto fs    = fs_.fs;
  auto state = write_json_state{fs};

  // begin writing
  write_json_begin(state);

  // start creating json
  write_json_object(state, "asset");
  write_json_value(state, "version", "2.0");
  write_json_value(
      state, "generator", "Yocto/GL - https://github.com/xelatihy/yocto-gl");
  write_json_pop(state);

  // convert cameras
  write_json_array(state, "cameras");
  for (auto& camera : scene.cameras) {
    write_json_object(state);
    write_json_value(state, "name", camera.uri);
    if (!camera.orthographic) {
      write_json_value(state, "type", "perspective");
      write_json_object(state, "perspective");
      write_json_value(state, "yfov", camera_yfov(camera));
      write_json_value(state, "aspectRatio", camera.film.x / camera.film.y);
      write_json_value(state, "znear", 0.01f);
      write_json_pop(state);
    } else {
      write_json_value(state, "type", "orthographic");
      write_json_object(state, "orthographic");
      write_json_value(state, "xmag", camera.film.x / 2);
      write_json_value(state, "ymag", camera.film.y / 2);
      write_json_value(state, "znear", 0.01f);
      write_json_pop(state);
    }
    write_json_pop(state);
  }
  write_json_pop(state);

  // textures
  write_json_array(state, "images");
  for (auto& texture : scene.textures) {
    write_json_object(state);
    write_json_value(state, "uri", texture.uri);
    write_json_pop(state);
  }
  write_json_pop(state);
  auto tid = 0;
  write_json_array(state, "textures");
  for (auto& texture : scene.textures) {
    write_json_object(state);
    write_json_value(state, "source", tid++);
    write_json_pop(state);
  }
  write_json_pop(state);

  // material
  auto write_json_texture = [](write_json_state& state, const char* key,
                                int tid) {
    write_json_object(state, key);
    write_json_value(state, "index", tid);
    write_json_pop(state);
  };
  write_json_array(state, "materials");
  for (auto& material : scene.materials) {
    write_json_object(state);
    write_json_value(state, "name", material.uri);
    if (material.emission != zero3f)
      write_json_value(state, "emissiveFactor", material.emission);
    if (material.emission_tex >= 0)
      write_json_texture(state, "emissiveTexture", material.emission_tex);
    auto kd = vec4f{material.diffuse.x, material.diffuse.y, material.diffuse.z,
        material.opacity};
    if (material.metallic || material.metallic_tex >= 0) {
      write_json_object(state, "pbrMetallicRoughness");
      write_json_value(state, "baseColorFactor", kd);
      write_json_value(state, "metallicFactor", material.metallic);
      write_json_value(state, "roughnessFactor", material.roughness);
      if (material.diffuse_tex >= 0)
        write_json_texture(state, "baseColorTexture", material.diffuse_tex);
      if (material.metallic_tex >= 0)
        write_json_texture(
            state, "metallicRoughnessTexture", material.metallic_tex);
      write_json_pop(state);
    } else {
      write_json_object(state, "extensions");
      write_json_object(state, "KHR_materials_pbrSpecularGlossiness");
      write_json_value(state, "diffuseFactor", kd);
      write_json_value(state, "specularFactor", material.specular);
      write_json_value(state, "glossinessFactor", 1 - material.roughness);
      if (material.diffuse_tex >= 0)
        write_json_texture(state, "diffuseTexture", material.diffuse_tex);
      if (material.specular_tex >= 0)
        write_json_texture(
            state, "specularGlossinessTexture", material.specular_tex);
      write_json_pop(state);
      write_json_pop(state);
    }
    if (material.normal_tex >= 0)
      write_json_texture(state, "normalTexture", material.normal_tex);
    write_json_pop(state);
  }
  write_json_pop(state);

  auto shapes = vector<gltf_shape>(scene.shapes.size());
  auto sid    = 0;
  for (auto& shape : scene.shapes) {
    auto& split = shapes[sid++];
    split.uri   = fs::path(shape.uri).replace_extension(".bin");
    split.mode  = 4;
    if (!shape.points.empty()) split.mode = 1;
    if (!shape.lines.empty()) split.mode = 1;
    if (shape.quadspos.empty()) {
      split.positions = shape.positions;
      split.normals   = shape.normals;
      split.texcoords = shape.texcoords;
      split.colors    = shape.colors;
      split.radius    = shape.radius;
      split.indices.insert(split.indices.end(), shape.points.data(),
          shape.points.data() + shape.points.size());
      split.indices.insert(split.indices.end(), (int*)shape.lines.data(),
          (int*)shape.lines.data() + shape.lines.size() * 2);
      split.indices.insert(split.indices.end(), (int*)shape.triangles.data(),
          (int*)shape.triangles.data() + shape.triangles.size() * 3);
      if (!shape.quads.empty()) {
        auto triangles = quads_to_triangles(shape.quads);
        split.indices.insert(split.indices.end(), (int*)triangles.data(),
            (int*)triangles.data() + triangles.size() * 3);
      }
    } else {
      auto quads = vector<vec4i>{};
      split_facevarying(quads, split.positions, split.normals, split.texcoords,
          shape.quadspos, shape.quadsnorm, shape.quadstexcoord, shape.positions,
          shape.normals, shape.texcoords);
      auto triangles = quads_to_triangles(quads);
      split.indices.insert(split.indices.end(), (int*)triangles.data(),
          (int*)triangles.data() + triangles.size() * 3);
    }
  }
  for (auto& instance : scene.instances) {
    shapes[instance.shape].material = instance.material;
  }

  // buffers
  write_json_array(state, "buffers");
  for (auto& shape : shapes) {
    auto buffer_size = sizeof(int) * shape.indices.size() +
                       sizeof(vec3f) * shape.positions.size() +
                       sizeof(vec3f) * shape.normals.size() +
                       sizeof(vec2f) * shape.texcoords.size() +
                       sizeof(vec4f) * shape.colors.size() +
                       sizeof(float) * shape.radius.size();
    write_json_object(state);
    write_json_value(state, "name", shape.uri);
    write_json_value(state, "uri", shape.uri);
    write_json_value(state, "byteLength", buffer_size);
    write_json_pop(state);
  }
  write_json_pop(state);

  // buffer views
  auto write_json_bufferview = [](write_json_state& state, auto& values,
                                   size_t& offset, int bid, bool indices) {
    if (values.empty()) return;
    auto bytes = values.size() * sizeof(values[0]);
    write_json_object(state);
    write_json_value(state, "buffer", bid);
    write_json_value(state, "byteLength", bytes);
    write_json_value(state, "byteOffset", offset);
    write_json_value(state, "target", (!indices) ? 34962 : 34963);
    write_json_pop(state);
    offset += bytes;
  };
  write_json_array(state, "bufferViews");
  auto bid = 0;
  for (auto& shape : shapes) {
    auto offset = (size_t)0;
    write_json_bufferview(state, shape.indices, offset, bid, true);
    write_json_bufferview(state, shape.positions, offset, bid, false);
    write_json_bufferview(state, shape.normals, offset, bid, false);
    write_json_bufferview(state, shape.texcoords, offset, bid, false);
    write_json_bufferview(state, shape.colors, offset, bid, false);
    write_json_bufferview(state, shape.radius, offset, bid, false);
    bid++;
  }
  write_json_pop(state);

  // accessors
  auto write_json_accessor = [](write_json_state& state, auto& values, int& vid,
                                 bool indices) {
    if (values.empty()) return;
    auto count = values.size();
    auto type  = "SCALAR";
    if (!indices) {
      if (sizeof(values[0]) / sizeof(float) == 2) type = "VEC2";
      if (sizeof(values[0]) / sizeof(float) == 3) type = "VEC3";
      if (sizeof(values[0]) / sizeof(float) == 4) type = "VEC4";
    }
    write_json_object(state);
    write_json_value(state, "bufferView", vid++);
    write_json_value(state, "byteOffset", 0);
    write_json_value(state, "componentType", (!indices) ? 5126 : 5125);
    write_json_value(state, "count", count);
    write_json_value(state, "type", type);
    write_json_pop(state);
  };
  auto vid = 0;
  write_json_array(state, "accessors");
  for (auto& shape : shapes) {
    write_json_accessor(state, shape.indices, vid, true);
    write_json_accessor(state, shape.positions, vid, false);
    write_json_accessor(state, shape.normals, vid, false);
    write_json_accessor(state, shape.texcoords, vid, false);
    write_json_accessor(state, shape.colors, vid, false);
    write_json_accessor(state, shape.radius, vid, false);
  }
  write_json_pop(state);

  // meshes
  auto aid = 0;
  write_json_array(state, "meshes");
  for (auto& shape : shapes) {
    write_json_object(state);
    write_json_value(state, "name", shape.uri);
    write_json_array(state, "primitives");
    write_json_value(state, "material", shape.material);
    if (!shape.indices.empty()) write_json_value(state, "indices", aid++);
    write_json_object(state, "attributes");
    if (!shape.positions.empty()) write_json_value(state, "POSITION", aid++);
    if (!shape.normals.empty()) write_json_value(state, "NORMAL", aid++);
    if (!shape.texcoords.empty()) write_json_value(state, "indices", aid++);
    if (!shape.colors.empty()) write_json_value(state, "TEXCOORD_0", aid++);
    if (!shape.radius.empty()) write_json_value(state, "RADIUS", aid++);
    write_json_accessor(state, shape.positions, vid, false);
    write_json_accessor(state, shape.normals, vid, false);
    write_json_accessor(state, shape.texcoords, vid, false);
    write_json_accessor(state, shape.colors, vid, false);
    write_json_accessor(state, shape.radius, vid, false);
    write_json_pop(state);
    write_json_pop(state);
  }
  write_json_pop(state);

  // nodes
  write_json_array(state, "nodes");
  if (scene.nodes.empty()) {
    auto camera_id = 0;
    for (auto& camera : scene.cameras) {
      write_json_object(state);
      write_json_value(state, "name", camera.uri);
      write_json_value(state, "camera", camera_id++);
      write_json_value(state, "matrix", mat4f(camera.frame));
      write_json_pop(state);
    }
    for (auto& instance : scene.instances) {
      write_json_object(state);
      write_json_value(state, "name", instance.uri);
      write_json_value(state, "mesh", instance.shape);
      write_json_value(state, "matrix", mat4f(instance.frame));
      write_json_pop(state);
    }
  } else {
    for (auto& node : scene.nodes) {
      write_json_object(state);
      write_json_value(state, "name", node.uri);
      write_json_value(state, "matrix", mat4f(node.local));
      write_json_value(state, "translation", node.translation);
      write_json_value(state, "rotation", node.rotation);
      write_json_value(state, "scale", node.scale);
      if (node.camera >= 0) write_json_value(state, "camera", node.camera);
      if (node.instance >= 0) {
        auto& instance = scene.instances[node.instance];
        write_json_value(state, "mesh", instance.shape);
      }
      if (!node.children.empty()) {
        write_json_value(state, "children", node.children);
      }
      write_json_pop(state);
    }
  }
  write_json_pop(state);

  // animations not supported yet
  if (!scene.animations.empty())
    throw std::runtime_error("animation not supported yet");

  // end writing
  write_json_end(state);

  // meshes
  auto write_values = [](FILE* fs, const auto& values) {
    if (values.empty()) return;
    if (fwrite(values.data(), sizeof(values.front()), values.size(), fs) !=
        values.size())
      throw std::runtime_error("cannot write to file");
  };
  auto dirname = fs::path(filename).parent_path();
  for (auto& shape : shapes) {
    auto fs_ = open_file(dirname / shape.uri, "w");
    auto fs  = fs_.fs;
    write_values(fs, shape.indices);
    write_values(fs, shape.positions);
    write_values(fs, shape.normals);
    write_values(fs, shape.texcoords);
    write_values(fs, shape.colors);
    write_values(fs, shape.radius);
  }
}

// Save gltf json
static void save_gltf_scene(const string& filename, const yocto_scene& scene,
    const save_params& params) {
  try {
    // save json
    save_gltf(filename, scene);

    // save textures
    auto dirname = fs::path(filename).parent_path();
    save_textures(scene, dirname, params);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot save scene " + filename + "\n" + e.what());
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PBRT
// -----------------------------------------------------------------------------
namespace yocto {

// Compute the fresnel term for dielectrics. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
static vec3f pbrt_fresnel_dielectric(float cosw, const vec3f& eta_) {
  auto eta = eta_;
  if (cosw < 0) {
    eta  = vec3f{1, 1, 1} / eta;
    cosw = -cosw;
  }

  auto sin2 = 1 - cosw * cosw;
  auto eta2 = eta * eta;

  auto cos2t = vec3f{1, 1, 1} - vec3f{sin2, sin2, sin2} / eta2;
  if (cos2t.x < 0 || cos2t.y < 0 || cos2t.z < 0) return vec3f{1, 1, 1};  // tir

  auto t0 = vec3f{sqrt(cos2t.x), sqrt(cos2t.y), sqrt(cos2t.z)};
  auto t1 = eta * t0;
  auto t2 = eta * cosw;

  auto rs = (vec3f{cosw, cosw, cosw} - t1) / (vec3f{cosw, cosw, cosw} + t1);
  auto rp = (t0 - t2) / (t0 + t2);

  return (rs * rs + rp * rp) / 2.0f;
}

// Compute the fresnel term for metals. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
static vec3f pbrt_fresnel_metal(
    float cosw, const vec3f& eta, const vec3f& etak) {
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

// convert pbrt elements
void add_pbrt_camera(yocto_scene& scene, const pbrt_camera& pcamera,
    const pbrt_context& ctx, float last_film_aspect, bool verbose = false) {
  auto camera    = yocto_camera{};
  camera.frame   = inverse((frame3f)ctx.transform_start);
  camera.frame.z = -camera.frame.z;
  switch (pcamera.type) {
    case pbrt_camera::type_t::perspective: {
      auto& perspective = pcamera.perspective;
      auto  aspect      = perspective.frameaspectratio;
      if (aspect < 0) aspect = last_film_aspect;
      if (aspect < 0) aspect = 1;
      if (aspect >= 1) {
        set_yperspective(camera, radians(perspective.fov), aspect,
            clamp(perspective.focaldistance, 1.0e-2f, 1.0e4f));
      } else {
        auto yfov = 2 * atan(tan(radians(perspective.fov) / 2) / aspect);
        set_yperspective(camera, yfov, aspect,
            clamp(perspective.focaldistance, 1.0e-2f, 1.0e4f));
      }
    } break;
    case pbrt_camera::type_t::orthographic: {
      throw std::runtime_error("unsupported Camera type");
    } break;
    case pbrt_camera::type_t::environment: {
      throw std::runtime_error("unsupported Camera type");
    } break;
    case pbrt_camera::type_t::realistic: {
      auto& realistic = pcamera.realistic;
      camera.lens     = max(realistic.approx_focallength, 35.0f) * 0.001f;
      auto aspect     = 1.0f;
      if (aspect < 0) aspect = last_film_aspect;
      if (aspect < 0) aspect = 1;
      if (aspect >= 1) {
        camera.film.y = camera.film.x / aspect;
      } else {
        camera.film.x = camera.film.y * aspect;
      }
      camera.focus    = realistic.focusdistance;
      camera.aperture = realistic.aperturediameter / 2;
    } break;
  }
  scene.cameras.push_back(camera);
}
static void add_pbrt_film(yocto_scene& scene, const pbrt_film& pfilm,
    const pbrt_context& ctx, float& last_film_aspect) {
  switch (pfilm.type) {
    case pbrt_film::type_t::image: {
      auto& image      = pfilm.image;
      last_film_aspect = (float)image.xresolution / (float)image.yresolution;
      for (auto& camera : scene.cameras) {
        camera.film.x = camera.film.y * last_film_aspect;
      }
    } break;
  }
}
static void add_pbrt_shape(yocto_scene& scene, const pbrt_shape& pshape,
    const pbrt_context& ctx, const string& name, const string& filename,
    const string&                                  cur_object,
    unordered_map<string, vector<yocto_instance>>& omap,
    const unordered_map<string, yocto_material>&   mmap,
    const unordered_map<string, vec3f>&            amap,
    unordered_map<string, int>&                    ammap) {
  auto get_material = [&](const pbrt_context& ctx) -> int {
    static auto light_id    = 0;
    auto        lookup_name = ctx.material + "_______" + ctx.arealight;
    if (ammap.find(lookup_name) != ammap.end()) return ammap.at(lookup_name);
    auto material = mmap.at(ctx.material);
    if (amap.at(ctx.arealight) != zero3f) {
      material.emission = amap.at(ctx.arealight);
      material.uri += "_arealight_" + std::to_string(light_id++);
    }
    scene.materials.push_back(material);
    ammap[lookup_name] = (int)scene.materials.size() - 1;
    return (int)scene.materials.size() - 1;
  };
  auto shape = yocto_shape{};
  shape.uri  = name;
  switch (pshape.type) {
    case pbrt_shape::type_t::trianglemesh: {
      auto& mesh      = pshape.trianglemesh;
      shape.positions = mesh.P;
      shape.normals   = mesh.N;
      shape.texcoords = mesh.uv;
      for (auto& uv : shape.texcoords) uv.y = (1 - uv.y);
      shape.triangles = mesh.indices;
    } break;
    case pbrt_shape::type_t::loopsubdiv: {
      auto& mesh      = pshape.loopsubdiv;
      shape.positions = mesh.P;
      shape.triangles = mesh.indices;
      shape.normals.resize(shape.positions.size());
      compute_normals(shape.normals, shape.triangles, shape.positions);
    } break;
    case pbrt_shape::type_t::plymesh: {
      auto& mesh = pshape.plymesh;
      shape.uri  = mesh.filename;
      load_shape(fs::path(filename).parent_path() / mesh.filename, shape.points,
          shape.lines, shape.triangles, shape.quads, shape.quadspos,
          shape.quadsnorm, shape.quadstexcoord, shape.positions, shape.normals,
          shape.texcoords, shape.colors, shape.radius, false);
    } break;
    case pbrt_shape::type_t::sphere: {
      auto& sphere        = pshape.sphere;
      auto  params        = proc_shape_params{};
      params.type         = proc_shape_params::type_t::uvsphere;
      params.subdivisions = 5;
      params.scale        = sphere.radius;
      make_proc_shape(shape.triangles, shape.quads, shape.positions,
          shape.normals, shape.texcoords, params);
    } break;
    case pbrt_shape::type_t::disk: {
      auto& disk          = pshape.disk;
      auto  params        = proc_shape_params{};
      params.type         = proc_shape_params::type_t::uvdisk;
      params.subdivisions = 4;
      params.scale        = disk.radius;
      make_proc_shape(shape.triangles, shape.quads, shape.positions,
          shape.normals, shape.texcoords, params);
    } break;
    default: {
      throw std::runtime_error(
          "unsupported shape type " + std::to_string((int)pshape.type));
    }
  }
  scene.shapes.push_back(shape);
  auto instance     = yocto_instance{};
  instance.frame    = (frame3f)ctx.transform_start;
  instance.shape    = (int)scene.shapes.size() - 1;
  instance.material = get_material(ctx);
  if (cur_object == "") {
    scene.instances.push_back(instance);
  } else {
    omap[cur_object].push_back(instance);
  }
}
static void add_pbrt_texture(yocto_scene& scene, const pbrt_texture& ptexture,
    const pbrt_context& ctx, const string& name,
    unordered_map<string, int>& tmap, unordered_map<string, vec3f>& ctmap,
    unordered_map<string, bool>& timap, bool remove_contant_textures = true,
    bool verbose = false) {
  if (remove_contant_textures &&
      ptexture.type == pbrt_texture::type_t::constant) {
    auto& constant = ptexture.constant;
    ctmap[name]    = (vec3f)constant.value.value;
    timap[name]    = false;
    return;
  }
  auto texture = yocto_texture{};
  texture.uri  = "textures/" + name + ".png";
  switch (ptexture.type) {
    case pbrt_texture::type_t::imagemap: {
      auto& imagemap = ptexture.imagemap;
      texture.uri    = imagemap.filename;
    } break;
    case pbrt_texture::type_t::constant: {
      auto& constant = ptexture.constant;
      texture.ldr.resize({1, 1});
      texture.ldr[{0, 0}] = float_to_byte(
          vec4f{(vec3f)constant.value.value, 1});
    } break;
    case pbrt_texture::type_t::bilerp: {
      // auto& bilerp   = get<pbrt_texture::bilerp_t>(ptexture);
      texture.ldr.resize({1, 1});
      texture.ldr[{0, 0}] = {255, 0, 0, 255};
      if (verbose) printf("texture bilerp not supported well");
    } break;
    case pbrt_texture::type_t::checkerboard: {
      auto& checkerboard = ptexture.checkerboard;
      auto  rgb1         = checkerboard.tex1.texture == ""
                      ? checkerboard.tex1.value
                      : pbrt_spectrum3f{0.4f, 0.4f, 0.4f};
      auto rgb2 = checkerboard.tex1.texture == ""
                      ? checkerboard.tex2.value
                      : pbrt_spectrum3f{0.6f, 0.6f, 0.6f};
      auto params   = proc_image_params{};
      params.type   = proc_image_params::type_t::checker;
      params.color0 = {rgb1.x, rgb1.y, rgb1.z, 1};
      params.color1 = {rgb2.x, rgb2.y, rgb2.z, 1};
      params.scale  = 2;
      make_proc_image(texture.hdr, params);
      float_to_byte(texture.ldr, texture.hdr);
      texture.hdr = {};
      if (verbose) printf("texture checkerboard not supported well");
    } break;
    case pbrt_texture::type_t::dots: {
      // auto& dots   = get<pbrt_texture::dots_t>(ptexture);
      texture.ldr.resize({1, 1});
      texture.ldr[{0, 0}] = {255, 0, 0, 255};
      if (verbose) printf("texture dots not supported well");
    } break;
    case pbrt_texture::type_t::fbm: {
      // auto& fbm = ptexture.fbm;
      auto params = proc_image_params{};
      params.type = proc_image_params::type_t::fbm;
      make_proc_image(texture.hdr, params);
      float_to_byte(texture.ldr, texture.hdr);
      texture.hdr = {};
      if (verbose) printf("texture fbm not supported well");
    } break;
    case pbrt_texture::type_t::marble: {
      // auto& marble = ptexture.marble;
      auto params = proc_image_params{};
      params.type = proc_image_params::type_t::fbm;
      make_proc_image(texture.hdr, params);
      float_to_byte(texture.ldr, texture.hdr);
      texture.hdr = {};
      if (verbose) printf("texture marble not supported well");
    } break;
    case pbrt_texture::type_t::mix: {
      auto& mix = ptexture.mix;
      if (timap.at(mix.tex1.texture)) {
        texture.uri = scene.textures.at(tmap.at(mix.tex1.texture)).uri;
      } else if (timap.at(mix.tex2.texture)) {
        texture.uri = scene.textures.at(tmap.at(mix.tex2.texture)).uri;
      } else {
        texture.ldr.resize({1, 1});
        texture.ldr[{0, 0}] = {255, 0, 0, 255};
      }
      if (verbose) printf("texture mix not supported well");
    } break;
    case pbrt_texture::type_t::scale: {
      auto& scale = ptexture.scale;
      if (timap.at(scale.tex1.texture)) {
        texture.uri = scene.textures.at(tmap.at(scale.tex1.texture)).uri;
      } else if (timap.at(scale.tex2.texture)) {
        texture.uri = scene.textures.at(tmap.at(scale.tex2.texture)).uri;
      } else {
        texture.ldr.resize({1, 1});
        texture.ldr[{0, 0}] = {255, 0, 0, 255};
      }
      if (verbose) printf("texture scale not supported well");
    } break;
    case pbrt_texture::type_t::uv: {
      // auto& uv   = get<pbrt_texture::uv_t>(ptexture);
      texture.ldr.resize({1, 1});
      texture.ldr[{0, 0}] = {255, 0, 0, 255};
      if (verbose) printf("texture uv not supported well");
    } break;
    case pbrt_texture::type_t::windy: {
      // auto& uv   = get<pbrt_texture::uv_t>(ptexture);
      texture.ldr.resize({1, 1});
      texture.ldr[{0, 0}] = {255, 0, 0, 255};
      if (verbose) printf("texture windy not supported well");
    } break;
    case pbrt_texture::type_t::wrinkled: {
      // auto& uv   = get<pbrt_texture::wrinkled_t>(ptexture);
      texture.ldr.resize({1, 1});
      texture.ldr[{0, 0}] = {255, 0, 0, 255};
      if (verbose) printf("texture wrinkled not supported well");
    } break;
  }
  scene.textures.push_back(texture);
  tmap[name]  = (int)scene.textures.size() - 1;
  timap[name] = ptexture.type == pbrt_texture::type_t::imagemap;
}
static void add_pbrt_material(yocto_scene& scnee,
    const pbrt_material& pmaterial, const pbrt_context& ctx, const string& name,
    unordered_map<string, yocto_material>& mmap,
    const unordered_map<string, int>&      tmap,
    const unordered_map<string, vec3f>& ctmap, bool verbose = false) {
  auto is_constant_texture = [&](const string& name) -> bool {
    return ctmap.find(name) != ctmap.end();
  };
  auto get_constant_texture_color = [&](const string& name) -> vec3f {
    return ctmap.at(name);
  };

  auto get_scaled_texture = [&](const pbrt_textured3f& textured, vec3f& color,
                                int& texture) {
    if (textured.texture == "") {
      color   = {textured.value.x, textured.value.y, textured.value.z};
      texture = -1;
    } else if (is_constant_texture(textured.texture)) {
      color   = get_constant_texture_color(textured.texture);
      texture = -1;
    } else {
      color   = {1, 1, 1};
      texture = tmap.at(textured.texture);
    }
  };

  auto get_scaled_texturef = [&](const pbrt_textured3f& textured, float& factor,
                                 vec3f& color, int& texture) {
    if (textured.texture == "") {
      color  = {textured.value.x, textured.value.y, textured.value.z};
      factor = color == zero3f ? 0 : 1;
      if (!factor) color = {1, 1, 1};
      texture = -1;
    } else if (is_constant_texture(textured.texture)) {
      color  = get_constant_texture_color(textured.texture);
      factor = color == zero3f ? 0 : 1;
      if (!factor) color = {1, 1, 1};
      texture = -1;
    } else {
      color   = {1, 1, 1};
      factor  = 1;
      texture = tmap.at(textured.texture);
    }
  };

  auto get_pbrt_roughness = [&](float uroughness, float vroughness,
                                bool remap) -> float {
    if (uroughness == 0 && vroughness == 0) return 0;
    auto roughness = (uroughness + vroughness) / 2;
    // from pbrt code
    if (remap) {
      roughness = max(roughness, 1e-3f);
      auto x    = log(roughness);
      roughness = 1.62142f + 0.819955f * x + 0.1734f * x * x +
                  0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
    }
    return sqrt(roughness);
  };

  auto material = yocto_material{};
  material.uri  = name;
  switch (pmaterial.type) {
    case pbrt_material::type_t::uber: {
      auto& uber = pmaterial.uber;
      get_scaled_texture(uber.Kd, material.diffuse, material.diffuse_tex);
      get_scaled_texture(uber.Ks, material.specular, material.specular_tex);
      get_scaled_texture(
          uber.Kt, material.transmission, material.transmission_tex);
      float op_f = 1;
      auto  op   = vec3f{0, 0, 0};
      get_scaled_texturef(uber.opacity, op_f, op, material.opacity_tex);
      material.opacity   = (op.x + op.y + op.z) / 3;
      material.roughness = get_pbrt_roughness(
          uber.uroughness.value, uber.vroughness.value, uber.remaproughness);
    } break;
    case pbrt_material::type_t::plastic: {
      auto& plastic = pmaterial.plastic;
      get_scaled_texture(plastic.Kd, material.diffuse, material.diffuse_tex);
      get_scaled_texture(plastic.Ks, material.specular, material.specular_tex);
      material.specular *= 0.04f;
      material.roughness = get_pbrt_roughness(plastic.uroughness.value,
          plastic.vroughness.value, plastic.remaproughness);
    } break;
    case pbrt_material::type_t::translucent: {
      auto& translucent = pmaterial.translucent;
      get_scaled_texture(
          translucent.Kd, material.diffuse, material.diffuse_tex);
      get_scaled_texture(
          translucent.Ks, material.specular, material.specular_tex);
      material.specular *= 0.04f;
      material.roughness = get_pbrt_roughness(translucent.uroughness.value,
          translucent.vroughness.value, translucent.remaproughness);
    } break;
    case pbrt_material::type_t::matte: {
      auto& matte = pmaterial.matte;
      get_scaled_texture(matte.Kd, material.diffuse, material.diffuse_tex);
      material.roughness = 1;
    } break;
    case pbrt_material::type_t::mirror: {
      auto& mirror = pmaterial.mirror;
      get_scaled_texturef(
          mirror.Kr, material.metallic, material.diffuse, material.diffuse_tex);
      material.roughness = 0;
    } break;
    case pbrt_material::type_t::metal: {
      auto& metal = pmaterial.metal;
      float eta_f = 0, etak_f = 0;
      auto  eta = zero3f, k = zero3f;
      auto  eta_texture = -1, k_texture = -1;
      get_scaled_texturef(metal.eta, eta_f, eta, eta_texture);
      get_scaled_texturef(metal.k, etak_f, k, k_texture);
      material.specular  = pbrt_fresnel_metal(1, eta, k);
      material.roughness = get_pbrt_roughness(
          metal.uroughness.value, metal.vroughness.value, metal.remaproughness);
    } break;
    case pbrt_material::type_t::substrate: {
      auto& substrate = pmaterial.substrate;
      get_scaled_texture(substrate.Kd, material.diffuse, material.diffuse_tex);
      get_scaled_texture(
          substrate.Ks, material.specular, material.specular_tex);
      material.roughness = get_pbrt_roughness(substrate.uroughness.value,
          substrate.vroughness.value, substrate.remaproughness);
    } break;
    case pbrt_material::type_t::glass: {
      auto& glass = pmaterial.glass;
      get_scaled_texture(glass.Kr, material.specular, material.specular_tex);
      material.specular *= 0.04f;
      get_scaled_texture(
          glass.Kt, material.transmission, material.transmission_tex);
      material.roughness = get_pbrt_roughness(
          glass.uroughness.value, glass.vroughness.value, glass.remaproughness);
    } break;
    case pbrt_material::type_t::hair: {
      auto& hair = pmaterial.hair;
      get_scaled_texture(hair.color, material.diffuse, material.diffuse_tex);
      material.roughness = 1;
      if (verbose) printf("hair material not properly supported\n");
    } break;
    case pbrt_material::type_t::disney: {
      auto& disney = pmaterial.disney;
      get_scaled_texture(disney.color, material.diffuse, material.diffuse_tex);
      material.roughness = 1;
      if (verbose) printf("disney material not properly supported\n");
    } break;
    case pbrt_material::type_t::kdsubsurface: {
      auto& kdsubsurface = pmaterial.kdsubsurface;
      get_scaled_texture(
          kdsubsurface.Kd, material.diffuse, material.diffuse_tex);
      get_scaled_texture(
          kdsubsurface.Kr, material.specular, material.specular_tex);
      material.specular *= 0.04f;
      material.roughness = get_pbrt_roughness(kdsubsurface.uroughness.value,
          kdsubsurface.vroughness.value, kdsubsurface.remaproughness);
      if (verbose) printf("kdsubsurface material not properly supported\n");
    } break;
    case pbrt_material::type_t::subsurface: {
      auto& subsurface = pmaterial.subsurface;
      get_scaled_texture(
          subsurface.Kr, material.specular, material.specular_tex);
      material.specular *= 0.04f;
      get_scaled_texture(
          subsurface.Kt, material.transmission, material.transmission_tex);
      material.roughness = get_pbrt_roughness(subsurface.uroughness.value,
          subsurface.vroughness.value, subsurface.remaproughness);
      material.volscale  = 1 / subsurface.scale;
      auto sigma_a = zero3f, sigma_s = zero3f;
      auto sigma_a_tex = -1, sigma_s_tex = -1;
      get_scaled_texture(subsurface.sigma_a, sigma_a, sigma_a_tex);
      get_scaled_texture(subsurface.sigma_prime_s, sigma_s, sigma_s_tex);
      material.volmeanfreepath = 1 / (sigma_a + sigma_s);
      material.volscatter      = sigma_s / (sigma_a + sigma_s);
      if (verbose) printf("subsurface material not properly supported\n");
    } break;
    case pbrt_material::type_t::mix: {
      auto& mix     = pmaterial.mix;
      auto  matname = (!mix.namedmaterial1.empty()) ? mix.namedmaterial1
                                                   : mix.namedmaterial2;
      material = mmap.at(matname);
      if (verbose) printf("mix material not properly supported\n");
    } break;
    case pbrt_material::type_t::fourier: {
      auto& fourier = pmaterial.fourier;
      if (fourier.approx_type ==
          pbrt_material::fourier_t::approx_type_t::plastic) {
        auto& plastic = fourier.approx_plastic;
        get_scaled_texture(plastic.Kd, material.diffuse, material.diffuse_tex);
        get_scaled_texture(
            plastic.Ks, material.specular, material.specular_tex);
        material.specular *= 0.04f;
        material.roughness = get_pbrt_roughness(plastic.uroughness.value,
            plastic.vroughness.value, plastic.remaproughness);
      } else if (fourier.approx_type ==
                 pbrt_material::fourier_t::approx_type_t::metal) {
        auto& metal = fourier.approx_metal;
        float eta_f = 0, etak_f = 0;
        auto  eta = zero3f, k = zero3f;
        auto  eta_texture = -1, k_texture = -1;
        get_scaled_texturef(metal.eta, eta_f, eta, eta_texture);
        get_scaled_texturef(metal.k, etak_f, k, k_texture);
        material.specular  = pbrt_fresnel_metal(1, eta, k);
        material.roughness = get_pbrt_roughness(metal.uroughness.value,
            metal.vroughness.value, metal.remaproughness);
      } else if (fourier.approx_type ==
                 pbrt_material::fourier_t::approx_type_t::glass) {
        auto& glass = fourier.approx_glass;
        get_scaled_texture(glass.Kr, material.specular, material.specular_tex);
        material.specular *= 0.04f;
        get_scaled_texture(
            glass.Kt, material.transmission, material.transmission_tex);
      }
    } break;
  }
  mmap[name] = material;
}
static void add_pbrt_arealight(yocto_scene& scene, const pbrt_arealight& plight,
    const pbrt_context& ctx, const string& name,
    unordered_map<string, vec3f>& amap) {
  auto emission = zero3f;
  switch (plight.type) {
    case pbrt_arealight::type_t::diffuse: {
      auto& diffuse = plight.diffuse;
      emission      = (vec3f)diffuse.L * (vec3f)diffuse.scale;
    } break;
    case pbrt_arealight::type_t::none: {
      throw std::runtime_error("should not have gotten here");
    } break;
  }
  amap[name] = emission;
}
static void add_pbrt_light(
    yocto_scene& scene, const pbrt_light& plight, const pbrt_context& ctx) {
  static auto light_id = 0;
  auto        name     = "light_" + std::to_string(light_id++);
  switch (plight.type) {
    case pbrt_light::type_t::infinite: {
      auto& infinite    = plight.infinite;
      auto  environment = yocto_environment();
      environment.uri   = name;
      // environment.frame =
      // frame3f{{1,0,0},{0,0,-1},{0,-1,0},{0,0,0}}
      // * stack.back().frame;
      environment.frame = (frame3f)ctx.transform_start *
                          frame3f{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0, 0}};
      environment.emission = (vec3f)infinite.scale * (vec3f)infinite.L;
      if (infinite.mapname != "") {
        auto texture = yocto_texture{};
        texture.uri  = infinite.mapname;
        scene.textures.push_back(texture);
        environment.emission_tex = (int)scene.textures.size() - 1;
      }
      scene.environments.push_back(environment);
    } break;
    case pbrt_light::type_t::distant: {
      auto& distant      = plight.distant;
      auto  distant_dist = 100;
      scene.shapes.push_back({});
      auto& shape  = scene.shapes.back();
      shape.uri    = name;
      auto dir     = normalize(distant.from - distant.to);
      auto size    = distant_dist * sin(5 * pif / 180);
      auto params  = proc_shape_params{};
      params.type  = proc_shape_params::type_t::quad;
      params.scale = size / 2;
      make_proc_shape(shape.triangles, shape.quads, shape.positions,
          shape.normals, shape.texcoords, params);
      scene.materials.push_back({});
      auto& material    = scene.materials.back();
      material.uri      = shape.uri;
      material.emission = (vec3f)distant.L * (vec3f)distant.scale;
      material.emission *= (distant_dist * distant_dist) / (size * size);
      auto instance     = yocto_instance();
      instance.uri      = shape.uri;
      instance.shape    = (int)scene.shapes.size() - 1;
      instance.material = (int)scene.materials.size() - 1;
      instance.frame    = (frame3f)ctx.transform_start *
                       lookat_frame(
                           dir * distant_dist, zero3f, {0, 1, 0}, true);
      scene.instances.push_back(instance);
    } break;
    case pbrt_light::type_t::point: {
      auto& point = plight.point;
      scene.shapes.push_back({});
      auto& shape         = scene.shapes.back();
      shape.uri           = name;
      auto size           = 0.005f;
      auto params         = proc_shape_params{};
      params.type         = proc_shape_params::type_t::sphere;
      params.scale        = size;
      params.subdivisions = 2;
      make_proc_shape(shape.triangles, shape.quads, shape.positions,
          shape.normals, shape.texcoords, params);
      scene.materials.push_back({});
      auto& material    = scene.materials.back();
      material.uri      = shape.uri;
      material.emission = (vec3f)point.I * (vec3f)point.scale;
      // TODO: fix emission
      auto instance     = yocto_instance();
      instance.uri      = shape.uri;
      instance.shape    = (int)scene.shapes.size() - 1;
      instance.material = (int)scene.materials.size() - 1;
      instance.frame    = (frame3f)ctx.transform_start *
                       translation_frame(point.from);
      scene.instances.push_back(instance);
    } break;
    case pbrt_light::type_t::goniometric: {
      auto& goniometric = plight.goniometric;
      scene.shapes.push_back({});
      auto& shape         = scene.shapes.back();
      shape.uri           = name;
      auto size           = 0.005f;
      auto params         = proc_shape_params{};
      params.type         = proc_shape_params::type_t::sphere;
      params.scale        = size;
      params.subdivisions = 2;
      make_proc_shape(shape.triangles, shape.quads, shape.positions,
          shape.normals, shape.texcoords, params);
      scene.materials.push_back({});
      auto& material    = scene.materials.back();
      material.uri      = shape.uri;
      material.emission = (vec3f)goniometric.I * (vec3f)goniometric.scale;
      // TODO: fix emission
      auto instance     = yocto_instance();
      instance.uri      = shape.uri;
      instance.shape    = (int)scene.shapes.size() - 1;
      instance.material = (int)scene.materials.size() - 1;
      instance.frame    = (frame3f)ctx.transform_start;
      scene.instances.push_back(instance);
    } break;
    case pbrt_light::type_t::spot: {
      auto& spot = plight.spot;
      scene.shapes.push_back({});
      auto& shape         = scene.shapes.back();
      shape.uri           = name;
      auto size           = 0.005f;
      auto params         = proc_shape_params{};
      params.type         = proc_shape_params::type_t::sphere;
      params.scale        = size;
      params.subdivisions = 2;
      make_proc_shape(shape.triangles, shape.quads, shape.positions,
          shape.normals, shape.texcoords, params);
      scene.materials.push_back({});
      auto& material    = scene.materials.back();
      material.uri      = shape.uri;
      material.emission = (vec3f)spot.I * (vec3f)spot.scale;
      // TODO: fix emission
      auto instance     = yocto_instance();
      instance.uri      = shape.uri;
      instance.shape    = (int)scene.shapes.size() - 1;
      instance.material = (int)scene.materials.size() - 1;
      instance.frame    = (frame3f)ctx.transform_start;
      scene.instances.push_back(instance);
    } break;
    default: {
      throw std::runtime_error(
          "light type not supported " + std::to_string((int)plight.type));
    }
  }
}

// load pbrt
static void load_pbrt(
    const string& filename, yocto_scene& scene, const load_params& params) {
  auto files = vector<file_wrapper>{};
  open_file(files.emplace_back(), filename);

  // parse state
  auto        mmap       = unordered_map<string, yocto_material>{{"", {}}};
  auto        amap       = unordered_map<string, vec3f>{{"", zero3f}};
  auto        ammap      = unordered_map<string, int>{};
  auto        tmap       = unordered_map<string, int>{{"", -1}};
  auto        ctmap      = unordered_map<string, vec3f>{{"", zero3f}};
  auto        timap      = unordered_map<string, bool>{{"", false}};
  auto        omap       = unordered_map<string, vector<yocto_instance>>{};
  string      cur_object = ""s;
  float       last_film_aspect = -1.0f;
  static auto shape_id         = 0;

  // parse command by command
  auto element = pbrt_element{};
  auto name    = ""s;
  auto data    = pbrt_element_data{};
  auto stack   = vector<pbrt_context>{};
  auto state   = pbrt_parser_state{};
  while (!files.empty()) {
    if (!read_pbrt_element(files.back(), element, name, data, stack, state)) {
      files.pop_back();
      continue;
    }
    if (element == pbrt_element::film) {
      add_pbrt_film(scene, data.film, stack.back(), last_film_aspect);
    } else if (element == pbrt_element::camera) {
      add_pbrt_camera(scene, data.camera, stack.back(), last_film_aspect);
    } else if (element == pbrt_element::shape) {
      add_pbrt_shape(scene, data.shape, stack.back(),
          "shapes/shape__" + std::to_string(shape_id++) + ".ply", filename,
          cur_object, omap, mmap, amap, ammap);
    } else if (element == pbrt_element::light) {
      add_pbrt_light(scene, data.light, stack.back());
    } else if (element == pbrt_element::texture) {
      add_pbrt_texture(
          scene, data.texture, stack.back(), name, tmap, ctmap, timap);
    } else if (element == pbrt_element::material) {
      add_pbrt_material(
          scene, data.material, stack.back(), name, mmap, tmap, ctmap);
    } else if (element == pbrt_element::arealight) {
      add_pbrt_arealight(scene, data.arealight, stack.back(), name, amap);
    } else if (element == pbrt_element::object_instance) {
      auto& pinstances = omap.at(name);
      for (auto& pinstance : pinstances) {
        auto instance  = yocto_instance();
        instance.frame = (frame3f)stack.back().transform_start *
                         pinstance.frame;
        instance.shape    = pinstance.shape;
        instance.material = pinstance.material;
        scene.instances.push_back(instance);
      }
    } else if (element == pbrt_element::begin_object) {
      cur_object       = name;
      omap[cur_object] = {};
    } else if (element == pbrt_element::end_object) {
      cur_object = "";
    } else if (element == pbrt_element::include) {
      open_file(files.emplace_back(), fs::path(filename).parent_path() / name);
    } else {
      // skip other
    }
  }
}

// load pbrt scenes
static void load_pbrt_scene(
    const string& filename, yocto_scene& scene, const load_params& params) {
  scene = yocto_scene{};

  try {
    // Parse pbrt
    load_pbrt(filename, scene, params);

    // load textures
    auto dirname = fs::path(filename).parent_path();
    load_textures(scene, dirname, params);
  } catch (const std::exception& e) {
    throw std::runtime_error("cannot load scene " + filename + "\n" + e.what());
  }

  // fix scene
  scene.uri = fs::path(filename).filename();
  add_cameras(scene);
  add_materials(scene);
  add_radius(scene);
  normalize_uris(scene);
  trim_memory(scene);
  update_transforms(scene);
}

// Write text to file
static inline void write_pbrt_text(FILE* fs, const char* value) {
  if (fprintf(fs, "%s", value) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_text(FILE* fs, const string& value) {
  if (fprintf(fs, "%s", value.c_str()) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_value(FILE* fs, int value, bool parens = true) {
  if (fprintf(fs, parens ? "[ %d ]" : "%d", value) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_value(FILE* fs, float value, bool parens = true) {
  if (fprintf(fs, parens ? "[ %g ]" : "%g", value) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_value(
    FILE* fs, const char* value, bool parens = true) {
  if (fprintf(fs, parens ? "\"%s\"" : "\"%s\"", value) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_value(
    FILE* fs, const string& value, bool parens = true) {
  if (fprintf(fs, parens ? "[ \"%s\" ]" : "\"%s\"", value.c_str()) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_value(
    FILE* fs, const vec3f& value, bool parens = true) {
  if (fprintf(fs, parens ? "[ %g %g %g ]" : "%g %g %g", value.x, value.y,
          value.z) < 0)
    throw std::runtime_error("cannot print value");
}
static inline void write_pbrt_value(
    FILE* fs, const mat3f& value, bool parens = true) {
  if (parens) {
    if (fprintf(fs, "[ ") < 0) throw std::runtime_error("cannot print value");
  }
  for (auto i = 0; i < 9; i++)
    if (fprintf(fs, i ? " %g" : "%g", (&value.x.x)[i]) < 0)
      throw std::runtime_error("cannot print value");
  if (parens) {
    if (fprintf(fs, " ]") < 0) throw std::runtime_error("cannot print value");
  }
}
// static inline void write_pbrt_value(
//     FILE* fs, const frame3f& value, bool parens = true) {
//   if (parens) {
//     if (fprintf(fs, "[ ") < 0) throw std::runtime_error("cannot print
//     value");
//   }
//   for (auto i = 0; i < 12; i++)
//     if (fprintf(fs, i ? " %g" : "%g", (&value.x.x)[i]) < 0)
//       throw std::runtime_error("cannot print value");
//   if (parens) {
//     if (fprintf(fs, " ]") < 0) throw std::runtime_error("cannot print
//     value");
//   }
// }
static inline void write_pbrt_value(
    FILE* fs, const mat4f& value, bool parens = true) {
  if (parens) {
    if (fprintf(fs, "[ ") < 0) throw std::runtime_error("cannot print value");
  }
  for (auto i = 0; i < 16; i++)
    if (fprintf(fs, i ? " %g" : "%g", (&value.x.x)[i]) < 0)
      throw std::runtime_error("cannot print value");
  if (parens) {
    if (fprintf(fs, " ]") < 0) throw std::runtime_error("cannot print value");
  }
}

template <typename T>
struct pbrt_noparens {
  T value;
  pbrt_noparens(const T& value) : value{value} {}
};
template <typename T>
static inline void write_pbrt_value(FILE* fs, const pbrt_noparens<T>& value) {
  write_pbrt_value(fs, value.value, false);
}

template <typename T, typename... Ts>
static inline void write_pbrt_line(
    FILE* fs, const T& value, const Ts... values) {
  write_pbrt_value(fs, value);
  if constexpr (sizeof...(values) == 0) {
    write_pbrt_text(fs, "\n");
  } else {
    write_pbrt_text(fs, " ");
    write_pbrt_line(fs, values...);
  }
}

template <typename... Ts>
static inline void write_pbrt_command(
    FILE* fs, const char* cmd, const Ts... values) {
  write_pbrt_text(fs, cmd);
  if constexpr (sizeof...(values) == 0) {
    write_pbrt_text(fs, "\n");
  } else {
    write_pbrt_text(fs, " ");
    write_pbrt_line(fs, values...);
  }
}

// Convert a scene to pbrt format
static void save_pbrt(const string& filename, const yocto_scene& scene) {
  // open file
  auto fs_ = open_file(filename, "w");
  auto fs  = fs_.fs;

  // embed data
  write_pbrt_text(fs, get_save_scene_message(scene, "# ") + "\n\n");

  // convert camera and settings
  auto& camera     = scene.cameras.front();
  auto  from       = camera.frame.o;
  auto  to         = camera.frame.o - camera.frame.z;
  auto  up         = camera.frame.y;
  auto  image_size = camera_resolution(camera, 1280);
  write_pbrt_command(fs, "LookAt", pbrt_noparens<mat3f>{{from, to, up}});
  write_pbrt_command(fs, "Camera", "perspective", "float fov",
      camera_fov(camera).x * 180 / pif);

  // save renderer
  write_pbrt_command(fs, "Sampler", "random", "integer pixelsamples", 64);
  write_pbrt_command(fs, "Integrator", "path");
  write_pbrt_command(fs, "Film", "image", "string filename",
      fs::path(filename).stem().string() + ".exr", "integer xresolution",
      image_size.x, "integer yresolution", image_size.y);

  // start world
  write_pbrt_command(fs, "WorldBegin");

  // convert textures
  for (auto& texture : scene.textures) {
    write_pbrt_command(fs, "Texture",
        pbrt_noparens<string>{fs::path(texture.uri).stem().string()},
        "spectrum", "imagemap", "string filename", texture.uri);
  }

  // convert materials
  for (auto& material : scene.materials) {
    write_pbrt_command(fs, "MakeNamedMaterial",
        fs::path(material.uri).stem().string().c_str(), "string type", "uber");
    if (material.diffuse_tex >= 0) {
      write_pbrt_command(fs, "    ", "texture Kd",
          fs::path(scene.textures[material.diffuse_tex].uri).stem().string());
    } else {
      write_pbrt_command(fs, "    ", "rgb Kd", material.diffuse);
    }
    if (material.specular_tex >= 0) {
      write_pbrt_command(fs, "    ", "texture Ks",
          fs::path(scene.textures[material.specular_tex].uri).stem().string());
    } else {
      auto specular = vec3f{1};
      write_pbrt_command(fs, "    ", "rgb Ks", specular);
    }
    if (material.transmission != zero3f) {
      if (material.transmission_tex >= 0) {
        write_pbrt_command(fs, "    ", "texture Kt",
            fs::path(scene.textures[material.transmission_tex].uri)
                .stem()
                .string());
      } else {
        auto transmission = vec3f{1};
        write_pbrt_command(fs, "    ", "rgb Kt", transmission);
      }
    }
    write_pbrt_command(
        fs, "    ", "float roughness", material.roughness * material.roughness);
  }

  // convert instances
  for (auto& instance : scene.instances) {
    auto& shape    = scene.shapes[instance.shape];
    auto& material = scene.materials[instance.material];
    write_pbrt_command(fs, "AttributeBegin");
    write_pbrt_command(fs, "  TransformBegin");
    write_pbrt_command(fs, "    Transform", mat4f{instance.frame});
    write_pbrt_command(fs, "    NamedMaterial",
        pbrt_noparens{fs::path(material.uri).stem().string()});
    if (material.emission != zero3f) {
      write_pbrt_command(
          fs, "    AreaLightSource", "diffuse", "rgb L", material.emission);
    }
    write_pbrt_command(fs, "    Shape", "plymesh", "string filename",
        fs::path(shape.uri).replace_extension(".ply").string());
    write_pbrt_command(fs, "  TransformEnd");
    write_pbrt_command(fs, "AttributeEnd");
  }

  // end world
  write_pbrt_command(fs, "WorldEnd");
}

// Save a pbrt scene
void save_pbrt_scene(const string& filename, const yocto_scene& scene,
    const save_params& params) {
  try {
    // save json
    save_pbrt(filename, scene);

    // save meshes
    auto dirname = fs::path(filename).parent_path();
    for (auto& shape : scene.shapes) {
      save_shape((dirname / shape.uri).replace_extension(".ply"), shape.points,
          shape.lines, shape.triangles, shape.quads, shape.quadspos,
          shape.quadsnorm, shape.quadstexcoord, shape.positions, shape.normals,
          shape.texcoords, shape.colors, shape.radius);
    }

    // save textures
    save_textures(scene, dirname, params);

  } catch (const std::exception& e) {
    throw std::runtime_error("cannot save scene " + filename + "\n" + e.what());
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// EXAMPLE SCENES
// -----------------------------------------------------------------------------
namespace yocto {
void make_cornellbox_scene(yocto_scene& scene) {
  scene.uri               = "cornellbox";
  auto& camera            = scene.cameras.emplace_back();
  camera.uri              = "cam";
  camera.frame            = frame3f{{0, 1, 3.9}};
  camera.lens             = 0.035;
  camera.aperture         = 0.0;
  camera.film             = {0.024, 0.024};
  auto& floor_mat         = scene.materials.emplace_back();
  floor_mat.uri           = "floor";
  floor_mat.diffuse       = {0.725, 0.71, 0.68};
  auto& ceiling_mat       = scene.materials.emplace_back();
  ceiling_mat.uri         = "ceiling";
  ceiling_mat.diffuse     = {0.725, 0.71, 0.68};
  auto& backwall_mat      = scene.materials.emplace_back();
  backwall_mat.uri        = "backwall";
  backwall_mat.diffuse    = {0.725, 0.71, 0.68};
  auto& rightwall_mat     = scene.materials.emplace_back();
  rightwall_mat.uri       = "rightwall";
  rightwall_mat.diffuse   = {0.14, 0.45, 0.091};
  auto& leftwall_mat      = scene.materials.emplace_back();
  leftwall_mat.uri        = "leftwall";
  leftwall_mat.diffuse    = {0.63, 0.065, 0.05};
  auto& shortbox_mat      = scene.materials.emplace_back();
  shortbox_mat.uri        = "shortbox";
  shortbox_mat.diffuse    = {0.725, 0.71, 0.68};
  auto& tallbox_mat       = scene.materials.emplace_back();
  tallbox_mat.uri         = "tallbox";
  tallbox_mat.diffuse     = {0.725, 0.71, 0.68};
  auto& light_mat         = scene.materials.emplace_back();
  light_mat.uri           = "light";
  light_mat.emission      = {17, 12, 4};
  auto& floor_shp         = scene.shapes.emplace_back();
  floor_shp.uri           = "floor";
  floor_shp.positions     = {{-1, 0, 1}, {1, 0, 1}, {1, 0, -1}, {-1, 0, -1}};
  floor_shp.triangles     = {{0, 1, 2}, {2, 3, 0}};
  auto& ceiling_shp       = scene.shapes.emplace_back();
  ceiling_shp.uri         = "ceiling";
  ceiling_shp.positions   = {{-1, 2, 1}, {-1, 2, -1}, {1, 2, -1}, {1, 2, 1}};
  ceiling_shp.triangles   = {{0, 1, 2}, {2, 3, 0}};
  auto& backwall_shp      = scene.shapes.emplace_back();
  backwall_shp.uri        = "backwall";
  backwall_shp.positions  = {{-1, 0, -1}, {1, 0, -1}, {1, 2, -1}, {-1, 2, -1}};
  backwall_shp.triangles  = {{0, 1, 2}, {2, 3, 0}};
  auto& rightwall_shp     = scene.shapes.emplace_back();
  rightwall_shp.uri       = "rightwall";
  rightwall_shp.positions = {{1, 0, -1}, {1, 0, 1}, {1, 2, 1}, {1, 2, -1}};
  rightwall_shp.triangles = {{0, 1, 2}, {2, 3, 0}};
  auto& leftwall_shp      = scene.shapes.emplace_back();
  leftwall_shp.uri        = "leftwall";
  leftwall_shp.positions  = {{-1, 0, 1}, {-1, 0, -1}, {-1, 2, -1}, {-1, 2, 1}};
  leftwall_shp.triangles  = {{0, 1, 2}, {2, 3, 0}};
  auto& shortbox_shp      = scene.shapes.emplace_back();
  shortbox_shp.uri        = "shortbox";
  shortbox_shp.positions  = {{0.53, 0.6, 0.75}, {0.7, 0.6, 0.17},
      {0.13, 0.6, 0.0}, {-0.05, 0.6, 0.57}, {-0.05, 0.0, 0.57},
      {-0.05, 0.6, 0.57}, {0.13, 0.6, 0.0}, {0.13, 0.0, 0.0}, {0.53, 0.0, 0.75},
      {0.53, 0.6, 0.75}, {-0.05, 0.6, 0.57}, {-0.05, 0.0, 0.57},
      {0.7, 0.0, 0.17}, {0.7, 0.6, 0.17}, {0.53, 0.6, 0.75}, {0.53, 0.0, 0.75},
      {0.13, 0.0, 0.0}, {0.13, 0.6, 0.0}, {0.7, 0.6, 0.17}, {0.7, 0.0, 0.17},
      {0.53, 0.0, 0.75}, {0.7, 0.0, 0.17}, {0.13, 0.0, 0.0},
      {-0.05, 0.0, 0.57}};
  shortbox_shp.triangles  = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {6, 7, 4},
      {8, 9, 10}, {10, 11, 8}, {12, 13, 14}, {14, 15, 12}, {16, 17, 18},
      {18, 19, 16}, {20, 21, 22}, {22, 23, 20}};
  auto& tallbox_shp       = scene.shapes.emplace_back();
  tallbox_shp.uri         = "tallbox";
  tallbox_shp.positions   = {{-0.53, 1.2, 0.09}, {0.04, 1.2, -0.09},
      {-0.14, 1.2, -0.67}, {-0.71, 1.2, -0.49}, {-0.53, 0.0, 0.09},
      {-0.53, 1.2, 0.09}, {-0.71, 1.2, -0.49}, {-0.71, 0.0, -0.49},
      {-0.71, 0.0, -0.49}, {-0.71, 1.2, -0.49}, {-0.14, 1.2, -0.67},
      {-0.14, 0.0, -0.67}, {-0.14, 0.0, -0.67}, {-0.14, 1.2, -0.67},
      {0.04, 1.2, -0.09}, {0.04, 0.0, -0.09}, {0.04, 0.0, -0.09},
      {0.04, 1.2, -0.09}, {-0.53, 1.2, 0.09}, {-0.53, 0.0, 0.09},
      {-0.53, 0.0, 0.09}, {0.04, 0.0, -0.09}, {-0.14, 0.0, -0.67},
      {-0.71, 0.0, -0.49}};
  tallbox_shp.triangles   = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {6, 7, 4},
      {8, 9, 10}, {10, 11, 8}, {12, 13, 14}, {14, 15, 12}, {16, 17, 18},
      {18, 19, 16}, {20, 21, 22}, {22, 23, 20}};
  auto& light_shp         = scene.shapes.emplace_back();
  light_shp.uri           = "light";
  light_shp.positions     = {{-0.25, 1.99, 0.25}, {-0.25, 1.99, -0.25},
      {0.25, 1.99, -0.25}, {0.25, 1.99, 0.25}};
  light_shp.triangles     = {{0, 1, 2}, {2, 3, 0}};
  scene.instances.push_back({"floor", identity3x4f, 0, 0});
  scene.instances.push_back({"ceiling", identity3x4f, 1, 1});
  scene.instances.push_back({"backwall", identity3x4f, 2, 2});
  scene.instances.push_back({"rightwall", identity3x4f, 3, 3});
  scene.instances.push_back({"leftwall", identity3x4f, 4, 4});
  scene.instances.push_back({"shortbox", identity3x4f, 5, 5});
  scene.instances.push_back({"tallbox", identity3x4f, 6, 6});
  scene.instances.push_back({"light", identity3x4f, 7, 7});
}

}  // namespace yocto
