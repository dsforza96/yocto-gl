//
// # Yocto/SceneIO: Tiny library for Yocto/Scene input and output
//
// Yocto/SceneIO provides loading and saving functionality for scenes
// in Yocto/GL. We support a simple to use JSON format, PLY, OBJ and glTF.
// The JSON serialization is a straight copy of the in-memory scene data.
// To speed up testing, we also support a binary format that is a dump of
// the current scene. This format should not be use for archival though.
//
//
// ## Scene Loading and Saving
//
// 1. load a scene with `load_scene()` and save it with `save_scene()`
//
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

#ifndef _YOCTO_SCENEIO_H_
#define _YOCTO_SCENEIO_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include <functional>
#include <memory>

#include "yocto_image.h"
#include "yocto_math.h"

// -----------------------------------------------------------------------------
// SCENE DATA
// -----------------------------------------------------------------------------
namespace yocto {

// Using directives.
using std::function;
using std::make_unique;
using std::unique_ptr;

// Camera based on a simple lens model. The camera is placed using a frame.
// Camera projection is described in photographic terms. In particular,
// we specify film size (35mm by default), film aspect ration,
// the lens' focal length, the focus distance and the lens aperture.
// All values are in meters. Here are some common aspect ratios used in video
// and still photography.
// 3:2    on 35 mm:  0.036 x 0.024
// 16:9   on 35 mm:  0.036 x 0.02025 or 0.04267 x 0.024
// 2.35:1 on 35 mm:  0.036 x 0.01532 or 0.05640 x 0.024
// 2.39:1 on 35 mm:  0.036 x 0.01506 or 0.05736 x 0.024
// 2.4:1  on 35 mm:  0.036 x 0.015   or 0.05760 x 0.024 (approx. 2.39 : 1)
// To compute good apertures, one can use the F-stop number from photography
// and set the aperture to focal length over f-stop.
struct sceneio_camera {
  string  name         = "";
  frame3f frame        = identity3x4f;
  bool    orthographic = false;
  float   lens         = 0.050;
  float   film         = 0.036;
  float   aspect       = 1.500;
  float   focus        = flt_max;
  float   aperture     = 0;
};

// Texture containing either an LDR or HDR image. HdR images are encoded
// in linear color space, while LDRs are encoded as sRGB.
struct sceneio_texture {
  string       name = "";
  image<vec4f> hdr  = {};
  image<vec4b> ldr  = {};
};

// Material for surfaces, lines and triangles.
// For surfaces, uses a microfacet model with thin sheet transmission.
// The model is based on OBJ, but contains glTF compatibility.
// For the documentation on the values, please see the OBJ format.
struct sceneio_material {
  // material data
  string name = "";

  // material
  vec3f emission     = {0, 0, 0};
  vec3f color        = {0, 0, 0};
  float specular     = 0;
  float roughness    = 0;
  float metallic     = 0;
  float ior          = 1.5;
  vec3f spectint     = {1, 1, 1};
  float coat         = 0;
  float transmission = 0;
  vec3f scattering   = {0, 0, 0};
  float scanisotropy = 0;
  float trdepth      = 0.01;
  float opacity      = 1;
  float displacement = 0;
  bool  thin         = true;

  // textures
  sceneio_texture* emission_tex     = nullptr;
  sceneio_texture* color_tex        = nullptr;
  sceneio_texture* specular_tex     = nullptr;
  sceneio_texture* metallic_tex     = nullptr;
  sceneio_texture* roughness_tex    = nullptr;
  sceneio_texture* transmission_tex = nullptr;
  sceneio_texture* spectint_tex     = nullptr;
  sceneio_texture* scattering_tex   = nullptr;
  sceneio_texture* coat_tex         = nullptr;
  sceneio_texture* opacity_tex      = nullptr;
  sceneio_texture* normal_tex       = nullptr;
  sceneio_texture* displacement_tex = nullptr;
  bool             gltf_textures    = false;  // glTF packed textures

  // [experimental] properties to drive subdiv and displacement
  int  subdivisions = 2;
  bool smooth       = true;
};

// Shape data represented as indexed meshes of elements.
// May contain either points, lines, triangles and quads.
// Additionally, we support face-varying primitives where
// each vertex data has its own topology.
struct sceneio_shape {
  // shape data
  string name = "";

  // primitives
  vector<int>   points    = {};
  vector<vec2i> lines     = {};
  vector<vec3i> triangles = {};
  vector<vec4i> quads     = {};

  // vertex data
  vector<vec3f> positions = {};
  vector<vec3f> normals   = {};
  vector<vec2f> texcoords = {};
  vector<vec4f> colors    = {};
  vector<float> radius    = {};
  vector<vec4f> tangents  = {};
};

// Subdiv data represented as indexed meshes of elements.
// May contain points, lines, triangles, quads or
// face-varying quads.
struct sceneio_subdiv {
  // shape data
  string name = "";

  // face-varying primitives
  vector<vec4i> quadspos      = {};
  vector<vec4i> quadsnorm     = {};
  vector<vec4i> quadstexcoord = {};

  // vertex data
  vector<vec3f> positions = {};
  vector<vec3f> normals   = {};
  vector<vec2f> texcoords = {};
};

// Instance data.
struct sceneio_instance {
  // instance data
  string          name   = "";
  vector<frame3f> frames = {};
};

// Object.
struct sceneio_object {
  // object data
  string            name     = "";
  frame3f           frame    = identity3x4f;
  sceneio_shape*    shape    = nullptr;
  sceneio_material* material = nullptr;
  sceneio_instance* instance = nullptr;
  sceneio_subdiv*   subdiv   = nullptr;
};

// Environment map.
struct sceneio_environment {
  string           name         = "";
  frame3f          frame        = identity3x4f;
  vec3f            emission     = {0, 0, 0};
  sceneio_texture* emission_tex = nullptr;
};

// Scene comprised an array of objects whose memory is owened by the scene.
// All members are optional,Scene objects (camera, instances, environments)
// have transforms defined internally. A scene can optionally contain a
// node hierarchy where each node might point to a camera, instance or
// environment. In that case, the element transforms are computed from
// the hierarchy. Animation is also optional, with keyframe data that
// updates node transformations only if defined.
struct sceneio_model {
  string                       name         = "";
  vector<sceneio_camera*>      cameras      = {};
  vector<sceneio_object*>      objects      = {};
  vector<sceneio_environment*> environments = {};
  vector<sceneio_shape*>       shapes       = {};
  vector<sceneio_subdiv*>      subdivs      = {};
  vector<sceneio_texture*>     textures     = {};
  vector<sceneio_material*>    materials    = {};
  vector<sceneio_instance*>    instances    = {};
  ~sceneio_model();
};

// create a scene
unique_ptr<sceneio_model> make_sceneio_model();

// add element to a scene
sceneio_camera*      add_camera(sceneio_model* scene);
sceneio_environment* add_environment(sceneio_model* scene);
sceneio_object*      add_object(sceneio_model* scene);
sceneio_instance*    add_instance(sceneio_model* scene);
sceneio_material*    add_material(sceneio_model* scene);
sceneio_shape*       add_shape(sceneio_model* scene);
sceneio_subdiv*      add_subdiv(sceneio_model* scene);
sceneio_texture*     add_texture(sceneio_model* scene);
sceneio_object*      add_complete_object(
         sceneio_model* scene, const string& basename = "");

}  // namespace yocto

// -----------------------------------------------------------------------------
// SCENE IO FUNCTIONS
// -----------------------------------------------------------------------------

namespace yocto {

// Progress callback called when loading.
using sceneio_progress =
    function<void(const string& message, int current, int total)>;

// Load/save a scene in the supported formats. Throws on error.
// Calls the progress callback, if defined, as we process more data.
unique_ptr<sceneio_model> load_scene(const string& filename, string& error,
    sceneio_progress progress = {}, bool noparallel = false);
[[nodiscard]] bool load_scene(const string& filename, sceneio_model* scene,
    string& error, sceneio_progress progress_cb = {}, bool noparallel = false);
[[nodiscard]] bool save_scene(const string& filename,
    const sceneio_model* scene, string& error,
    sceneio_progress progress_cb = {}, bool noparallel = false);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SCENE STATS AND VALIDATION
// -----------------------------------------------------------------------------
namespace yocto {

// Return scene statistics as list of strings.
vector<string> scene_stats(const sceneio_model* scene, bool verbose = false);
// Return validation errors as list of strings.
vector<string> scene_validation(
    const sceneio_model* scene, bool notextures = false);

// Return an approximate scene bounding box.
bbox3f compute_bounds(const sceneio_model* scene);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SCENE UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Apply subdivision and displacement rules.
void tesselate_subdivs(sceneio_model* scene, sceneio_progress progress_cb = {});
void tesselate_subdiv(sceneio_model* scene, sceneio_subdiv* subdiv);

// Update node transforms. Eventually this will be deprecated as we do not
// support animation in this manner long term.
void update_transforms(
    sceneio_model* scene, float time = 0, const string& anim_group = "");

// TODO: remove
inline vec3f eta_to_reflectivity(float eta) {
  return vec3f{((eta - 1) * (eta - 1)) / ((eta + 1) * (eta + 1))};
}

}  // namespace yocto

#endif
