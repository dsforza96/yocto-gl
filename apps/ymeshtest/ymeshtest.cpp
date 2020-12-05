//
// LICENSE:
//
// Copyright (c) 2016 -- 2020 Fabio Pellacini
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

#include <yocto/yocto_commonio.h>
#include <yocto/yocto_math.h>
#include <yocto/yocto_mesh.h>
#include <yocto/yocto_modelio.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_sceneio.h>
#include <yocto/yocto_shape.h>
using namespace yocto;

#include "ext/json.hpp"

using json = nlohmann::json;

// -----------------------------------------------------------------------------
// JSON SUPPORT
// -----------------------------------------------------------------------------
namespace yocto {

using json = nlohmann::json;
using std::array;

// support for json conversions
inline void to_json(json& j, const vec2f& value) {
  nlohmann::to_json(j, (const array<float, 2>&)value);
}
inline void to_json(json& j, const vec3f& value) {
  nlohmann::to_json(j, (const array<float, 3>&)value);
}
inline void to_json(json& j, const vec4f& value) {
  nlohmann::to_json(j, (const array<float, 4>&)value);
}
inline void to_json(json& j, const frame3f& value) {
  nlohmann::to_json(j, (const array<float, 12>&)value);
}
inline void to_json(json& j, const mat4f& value) {
  nlohmann::to_json(j, (const array<float, 16>&)value);
}

inline void from_json(const json& j, vec3f& value) {
  nlohmann::from_json(j, (array<float, 3>&)value);
}
inline void from_json(const json& j, mat3f& value) {
  nlohmann::from_json(j, (array<float, 9>&)value);
}
inline void from_json(const json& j, frame3f& value) {
  nlohmann::from_json(j, (array<float, 12>&)value);
}

inline void to_json(json& j, const mesh_point& value) {
  nlohmann::to_json(j, pair{value.face, value.uv});
}

// load/save json
bool load_json(const string& filename, json& js, string& error) {
  // error helpers
  auto parse_error = [filename, &error]() {
    error = filename + ": parse error in json";
    return false;
  };
  auto text = ""s;
  if (!load_text(filename, text, error)) return false;
  try {
    js = json::parse(text);
    return true;
  } catch (std::exception&) {
    return parse_error();
  }
}

bool save_json(const string& filename, const json& js, string& error) {
  return save_text(filename, js.dump(2), error);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// SCENE CREATION SUPPORT
// -----------------------------------------------------------------------------
namespace yocto {

sceneio_camera* add_camera(sceneio_scene* scene, const string& name,
    const frame3f& frame, float lens, float aspect, float aperture = 0,
    float focus = 10, bool orthographic = false, float film = 0.036) {
  auto camera          = add_camera(scene, name);
  camera->frame        = frame;
  camera->lens         = lens;
  camera->aspect       = aspect;
  camera->film         = film;
  camera->orthographic = orthographic;
  camera->aperture     = aperture;
  camera->focus        = focus;
  return camera;
}

sceneio_instance* add_instance(sceneio_scene* scene, const string& name,
    const frame3f& frame, sceneio_shape* shape, sceneio_material* material) {
  auto instance      = add_instance(scene, name);
  instance->frame    = frame;
  instance->shape    = shape;
  instance->material = material;
  return instance;
}

sceneio_shape* add_shape(sceneio_scene* scene, const string& name,
    const quads_shape& shape_data, int subdivisions = 0, float displacement = 0,
    sceneio_texture* displacement_tex = nullptr) {
  auto shape              = add_shape(scene, name);
  shape->quads            = shape_data.quads;
  shape->positions        = shape_data.positions;
  shape->normals          = shape_data.normals;
  shape->texcoords        = shape_data.texcoords;
  shape->subdivisions     = subdivisions;
  shape->smooth           = subdivisions > 0 || displacement_tex;
  shape->displacement     = displacement;
  shape->displacement_tex = displacement_tex;
  return shape;
}

sceneio_shape* add_shape(sceneio_scene* scene, const string& name,
    const quads_fvshape& shape_data, int subdivisions = 0,
    float displacement = 0, sceneio_texture* displacement_tex = nullptr) {
  auto shape              = add_shape(scene, name);
  shape->quadspos         = shape_data.quadspos;
  shape->quadsnorm        = shape_data.quadsnorm;
  shape->quadstexcoord    = shape_data.quadstexcoord;
  shape->positions        = shape_data.positions;
  shape->normals          = shape_data.normals;
  shape->texcoords        = shape_data.texcoords;
  shape->subdivisions     = subdivisions;
  shape->smooth           = subdivisions > 0 || displacement_tex;
  shape->displacement     = displacement;
  shape->displacement_tex = displacement_tex;
  return shape;
}

sceneio_shape* add_shape(sceneio_scene* scene, const string& name,
    const triangles_shape& shape_data, int subdivisions = 0,
    float displacement = 0, sceneio_texture* displacement_tex = nullptr) {
  auto shape              = add_shape(scene, name);
  shape->triangles        = shape_data.triangles;
  shape->positions        = shape_data.positions;
  shape->normals          = shape_data.normals;
  shape->texcoords        = shape_data.texcoords;
  shape->subdivisions     = subdivisions;
  shape->smooth           = subdivisions > 0 || displacement_tex;
  shape->displacement     = displacement;
  shape->displacement_tex = displacement_tex;
  return shape;
}

sceneio_shape* add_shape(sceneio_scene* scene, const string& name,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texcoords,
    int subdivisions = 0, float displacement = 0,
    sceneio_texture* displacement_tex = nullptr) {
  auto shape              = add_shape(scene, name);
  shape->triangles        = triangles;
  shape->positions        = positions;
  shape->normals          = normals;
  shape->texcoords        = texcoords;
  shape->subdivisions     = subdivisions;
  shape->smooth           = subdivisions > 0 || displacement_tex;
  shape->displacement     = displacement;
  shape->displacement_tex = displacement_tex;
  return shape;
}

sceneio_shape* add_shape(sceneio_scene* scene, const string& name,
    const lines_shape& shape_data, int subdivisions = 0, float displacement = 0,
    sceneio_texture* displacement_tex = nullptr) {
  auto shape              = add_shape(scene, name);
  shape->lines            = shape_data.lines;
  shape->positions        = shape_data.positions;
  shape->normals          = shape_data.normals;
  shape->texcoords        = shape_data.texcoords;
  shape->radius           = shape_data.radius;
  shape->subdivisions     = subdivisions;
  shape->smooth           = subdivisions > 0 || displacement_tex;
  shape->displacement     = displacement;
  shape->displacement_tex = displacement_tex;
  return shape;
}

sceneio_shape* add_shape(sceneio_scene* scene, const string& name,
    const points_shape& shape_data, int subdivisions = 0,
    float displacement = 0, sceneio_texture* displacement_tex = nullptr) {
  auto shape              = add_shape(scene, name);
  shape->points           = shape_data.points;
  shape->positions        = shape_data.positions;
  shape->normals          = shape_data.normals;
  shape->texcoords        = shape_data.texcoords;
  shape->radius           = shape_data.radius;
  shape->subdivisions     = subdivisions;
  shape->smooth           = subdivisions > 0 || displacement_tex;
  shape->displacement     = displacement;
  shape->displacement_tex = displacement_tex;
  return shape;
}

sceneio_material* add_emission_material(sceneio_scene* scene,
    const string& name, const vec3f& emission, sceneio_texture* emission_tex) {
  auto material          = add_material(scene, name);
  material->emission     = emission;
  material->emission_tex = emission_tex;
  return material;
}

sceneio_material* add_matte_material(sceneio_scene* scene, const string& name,
    const vec3f& color, sceneio_texture* color_tex,
    sceneio_texture* normal_tex = nullptr) {
  auto material        = add_material(scene, name);
  material->color      = color;
  material->color_tex  = color_tex;
  material->roughness  = 1;
  material->normal_tex = normal_tex;
  return material;
}

sceneio_material* add_specular_material(sceneio_scene* scene,
    const string& name, const vec3f& color, sceneio_texture* color_tex,
    float roughness, sceneio_texture* roughness_tex = nullptr,
    sceneio_texture* normal_tex = nullptr, float ior = 1.5, float specular = 1,
    sceneio_texture* specular_tex = nullptr, const vec3f& spectint = {1, 1, 1},
    sceneio_texture* spectint_tex = nullptr) {
  auto material           = add_material(scene, name);
  material->color         = color;
  material->color_tex     = color_tex;
  material->specular      = specular;
  material->specular_tex  = specular_tex;
  material->spectint      = spectint;
  material->spectint_tex  = spectint_tex;
  material->roughness     = roughness;
  material->roughness_tex = roughness_tex;
  material->ior           = ior;
  material->normal_tex    = normal_tex;
  return material;
}

sceneio_environment* add_environment(sceneio_scene* scene, const string& name,
    const frame3f& frame, const vec3f& emission,
    sceneio_texture* emission_tex = nullptr) {
  auto environment          = add_environment(scene, name);
  environment->frame        = frame;
  environment->emission     = emission;
  environment->emission_tex = emission_tex;
  return environment;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// TEST CREATION  SUPPORT
// -----------------------------------------------------------------------------
vector<mesh_point> sample_points(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const shape_bvh& bvh,
    const vec3f& camera_from, const vec3f& camera_to, float camera_lens,
    float camera_aspect, int ray_trials = 10000, int num_points = 4) {
  // init data
  auto points  = vector<mesh_point>{};
  auto rng_ray = make_rng(9867198237913);
  // try to pick in the camera
  auto ray_trial = 0;
  while (points.size() < num_points) {
    if (ray_trial++ >= ray_trials) break;
    auto ray  = camera_ray(lookat_frame(camera_from, camera_to, {0, 1, 0}),
        camera_lens, camera_aspect, 0.036f, rand2f(rng_ray));
    auto isec = intersect_triangles_bvh(bvh, triangles, positions, ray);
    if (isec.hit) points.push_back({isec.element, isec.uv});
  }
  // pick based on area
  auto rng_area = make_rng(9867198237913);
  auto cdf      = sample_triangles_cdf(triangles, positions);
  while (points.size() < num_points) {
    auto [triangle, uv] = sample_triangles(
        cdf, rand1f(rng_area), rand2f(rng_area));
    points.push_back({mesh_point{triangle, uv}});
  }
  return points;
}

#if 0
  ​ ​ void update_path_shape(shade_shape * shape, const bool_mesh& mesh,
      const geodesic_path& path, float radius, bool thin_lines = false) {
    auto positions = path_positions(
        path, mesh.triangles, mesh.positions, mesh.adjacencies);
    ​ if (thin_lines) {
      set_positions(shape, positions);
      shape->shape->elements = ogl_element_type::line_strip;
      set_instances(shape, {}, {});
      return;
    }
    ​ auto froms = vector<vec3f>();
    auto     tos   = vector<vec3f>();
    froms.reserve(positions.size() - 1);
    tos.reserve(positions.size() - 1);
    for (int i = 0; i < positions.size() - 1; i++) {
      auto from = positions[i];
      auto to   = positions[i + 1];
      if (from == to) continue;
      froms.push_back(from);
      tos.push_back(to);
    }
    ​ auto cylinder = make_uvcylinder({4, 1, 1}, {radius, 1});
    for (auto& p : cylinder.positions) {
      p.z = p.z * 0.5 + 0.5;
    }
    ​ set_quads(shape, cylinder.quads);
    set_positions(shape, cylinder.positions);
    set_normals(shape, cylinder.normals);
    set_texcoords(shape, cylinder.texcoords);
    set_instances(shape, froms, tos);
  }
#endif

vector<mesh_point> trace_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& points) {
  auto trace_line =
      [](const dual_geodesic_solver& graph, const vector<vec3i>& triangles,
          const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
          const mesh_point& start, const mesh_point& end) {
        auto path = geodesic_path{};
        if (start.face == end.face) {
          path.start = start;
          path.end   = end;
          path.strip = {start.face};
        } else {
          auto strip = strip_on_dual_graph(
              graph, triangles, positions, end.face, start.face);
          path = shortest_path(
              triangles, positions, adjacencies, start, end, strip);
        }
        // get mesh points
        return convert_mesh_path(triangles, adjacencies, path.strip, path.lerps,
            path.start, path.end)
            .points;
      };

  // geodesic path
  auto path = vector<mesh_point>{};
  for (auto idx = 0; idx < (int)points.size() - 1; idx++) {
    auto segment = trace_line(
        graph, triangles, positions, adjacencies, points[idx], points[idx + 1]);
    path.insert(path.end(), segment.begin(), segment.end());
  }
  return path;
}

vector<vec3f> path_positions(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<mesh_point>& path) {
  auto ppositions = vector<vec3f>{};
  ppositions.reserve(path.size());
  for (auto& point : path) {
    auto& triangle = triangles[point.face];
    ppositions.push_back(interpolate_triangle(positions[triangle.x],
        positions[triangle.y], positions[triangle.z], point.uv));
  }
  return ppositions;
}

lines_shape path_to_lines(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<mesh_point>& path,
    float radius) {
  auto shape      = lines_shape{};
  shape.positions = path_positions(triangles, positions, path);
  // shape.normals   = ...;  // TODO(fabio): tangents
  shape.radius = vector<float>(shape.positions.size(), radius);
  shape.lines  = vector<vec2i>(shape.positions.size() - 1);
  for (auto k = 0; k < shape.lines.size(); k++) shape.lines[k] = {k, k + 1};
  return shape;
}

points_shape path_to_points(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<mesh_point>& path,
    float radius) {
  auto shape      = points_shape{};
  shape.positions = path_positions(triangles, positions, path);
  // shape.normals   = ...;  // TODO(fabio): tangents
  shape.radius = vector<float>(shape.positions.size(), radius);
  shape.points = vector<int>(shape.positions.size());
  for (auto k = 0; k < shape.points.size(); k++) shape.points[k] = k;
  return shape;
}

quads_shape path_to_quads(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<mesh_point>& path,
    float point_thickness, float line_thickness) {
  auto ppositions = path_positions(triangles, positions, path);
  auto shape      = quads_shape{};
  for (auto idx = 0; idx < ppositions.size(); idx++) {
    if (point_thickness > 0) {
      auto sphere = make_sphere(4, point_thickness);
      for (auto& p : sphere.positions) p += ppositions[idx];
      merge_quads(shape.quads, shape.positions, shape.normals, shape.texcoords,
          sphere.quads, sphere.positions, sphere.normals, sphere.texcoords);
    }
    // if (line_thickness > 0 && idx < (int)ppositions.size() - 1 &&
    //     length(ppositions[idx] - ppositions[idx + 1]) > point_thickness) {
    if (line_thickness > 0 && idx < (int)ppositions.size() - 1) {
      auto cylinder = make_uvcylinder({32, 1, 1},
          {line_thickness, length(ppositions[idx] - ppositions[idx + 1]) / 2});
      auto frame    = frame_fromz((ppositions[idx] + ppositions[idx + 1]) / 2,
          normalize(ppositions[idx + 1] - ppositions[idx]));
      for (auto& p : cylinder.positions) p = transform_point(frame, p);
      merge_quads(shape.quads, shape.positions, shape.normals, shape.texcoords,
          cylinder.quads, cylinder.positions, cylinder.normals,
          cylinder.texcoords);
    }
  }
  return shape;
}

void make_scene_floor(sceneio_scene* scene, const vec3f& camera_from,
    const vec3f& camera_to, float camera_lens, float camera_aspect,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<mesh_point>& points, const vector<mesh_point>& path,
    bool points_as_meshes = true, float point_thickness = 0.02f,
    float line_thickness = 0.01f) {
  scene->name = "name";
  // camera
  add_camera(scene, "camera", lookat_frame(camera_from, camera_to, {0, 1, 0}),
      camera_lens, camera_aspect, 0, length(camera_from - camera_to));

  // mesh
  // TODO(fabio): normals?
  add_instance(scene, "mesh", identity3x4f,
      add_shape(scene, "mesh", triangles, positions, {}, {}, {}),
      add_specular_material(scene, "mesh", {0.6, 0.6, 0.6}, nullptr, 0));

  // curve
  if (points_as_meshes) {
    add_instance(scene, "path", identity3x4f,
        add_shape(scene, "path",
            path_to_quads(
                triangles, positions, path, line_thickness, line_thickness)),
        add_matte_material(scene, "path", {0.8, 0.1, 0.1}, nullptr));
  } else {
    add_instance(scene, "path", identity3x4f,
        add_shape(scene, "path",
            path_to_lines(triangles, positions, path, line_thickness)),
        add_matte_material(scene, "path", {0.8, 0.1, 0.1}, nullptr));
  }

  // points
  if (points_as_meshes) {
    add_instance(scene, "points", identity3x4f,
        add_shape(scene, "points",
            path_to_quads(triangles, positions, points, point_thickness, 0)),
        add_matte_material(scene, "points", {0.1, 0.8, 0.1}, nullptr));
  } else {
    add_instance(scene, "points", identity3x4f,
        add_shape(scene, "points",
            path_to_points(triangles, positions, points, point_thickness)),
        add_matte_material(scene, "points", {0.1, 0.8, 0.1}, nullptr));
  }

  // environment
  // TODO(fabio): environment
  add_environment(scene, "environment", identity3x4f, {0, 0, 0}, nullptr);
  // environment->emission_tex = add_texture(scene, "env");
  //   load_image("data/env.png", environment_tex->hdr, error);

  // lights
  add_instance(scene, "arealight1",
      lookat_frame({-2, 2, 2}, {0, 0.5, 0}, {0, 1, 0}, true),
      add_shape(scene, "arealight1", make_rect({1, 1}, {0.2, 0.2})),
      add_emission_material(scene, "arealight1", {40, 40, 40}, nullptr));
  add_instance(scene, "arealight2",
      lookat_frame({2, 2, 1}, {0, 0.5, 0}, {0, 1, 0}, true),
      add_shape(scene, "arealight2", make_rect({1, 1}, {0.2, 0.2})),
      add_emission_material(scene, "arealight2", {40, 40, 40}, nullptr));
  add_instance(scene, "arealight3",
      lookat_frame({0, 2, -2}, {0, 0.5, 0}, {0, 1, 0}, true),
      add_shape(scene, "arealight3", make_rect({1, 1}, {0.2, 0.2})),
      add_emission_material(scene, "arealight3", {40, 40, 40}, nullptr));

  // add floor
  // TODO(fabio): floor material
  // floor_frame.o.y = -0.5;
  add_instance(scene, "floor", identity3x4f,
      add_shape(scene, "floor", make_floor({1, 1}, {10, 10})),
      add_matte_material(scene, "floor", {1, 1, 1}, nullptr));
}

void make_scene_floating(sceneio_scene* scene, const vec3f& camera_from,
    const vec3f& camera_to, float camera_lens, float camera_aspect,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<mesh_point>& points, const vector<mesh_point>& path,
    bool points_as_meshes = true, float point_thickness = 0.02f,
    float line_thickness = 0.01f) {
  scene->name = "name";
  // camera
  add_camera(scene, "camera", lookat_frame(camera_from, camera_to, {0, 1, 0}),
      camera_lens, camera_aspect, 0, length(camera_from - camera_to));

  // mesh
  // TODO(fabio): normals?
  add_instance(scene, "mesh", identity3x4f,
      add_shape(scene, "mesh", triangles, positions, {}, {}, {}),
      add_specular_material(scene, "mesh", {0.6, 0.6, 0.6}, nullptr, 0));

  // curve
  if (points_as_meshes) {
    add_instance(scene, "path", identity3x4f,
        add_shape(scene, "path",
            path_to_quads(
                triangles, positions, path, line_thickness, line_thickness)),
        add_matte_material(scene, "path", {0.8, 0.1, 0.1}, nullptr));
  } else {
    add_instance(scene, "path", identity3x4f,
        add_shape(scene, "path",
            path_to_lines(triangles, positions, path, line_thickness)),
        add_matte_material(scene, "path", {0.8, 0.1, 0.1}, nullptr));
  }

  // points
  if (points_as_meshes) {
    add_instance(scene, "points", identity3x4f,
        add_shape(scene, "points",
            path_to_quads(triangles, positions, points, point_thickness, 0)),
        add_matte_material(scene, "points", {0.1, 0.8, 0.1}, nullptr));
  } else {
    add_instance(scene, "points", identity3x4f,
        add_shape(scene, "points",
            path_to_points(triangles, positions, points, point_thickness)),
        add_matte_material(scene, "points", {0.1, 0.8, 0.1}, nullptr));
  }

  // environment
  // TODO(fabio): environment
  add_environment(scene, "environment", identity3x4f, {1, 1, 1}, nullptr);
}

// Save a path
bool save_mesh_points(
    const string& filename, const vector<mesh_point>& path, string& error) {
  auto format_error = [filename, &error]() {
    error = filename + ": unknown format";
    return false;
  };

  auto ext = path_extension(filename);
  if (ext == ".json" || ext == ".JSON") {
    auto js    = json{};
    js["path"] = path;
    return save_json(filename, js, error);
  } else if (ext == ".ply" || ext == ".PLY") {
    auto ply_guard = std::make_unique<ply_model>();
    auto ply       = ply_guard.get();
    auto ids       = vector<int>(path.size());
    auto uvs       = vector<vec2f>(path.size());
    for (auto idx = 0; idx < path.size(); idx++) {
      ids[idx] = path[idx].face;
      uvs[idx] = path[idx].uv;
    }
    add_value(ply, "mesh_points", "face", ids);
    add_values(ply, "mesh_points", {"u", "v"}, uvs);
    return save_ply(filename, ply, error);
  } else {
    return format_error();
  }
}

// -----------------------------------------------------------------------------
// MAIN FUNCTION
// -----------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
  // command line parameters
  auto smooth    = false;
  auto faceted   = false;
  auto validate  = false;
  auto pathname  = "path.ply"s;
  auto statsname = "stats.json"s;
  auto scenename = "scene.json"s;
  auto meshname  = "mesh.ply"s;

  // parse command line
  auto cli = make_cli("ymshproc", "Applies operations on a triangle mesh");
  add_option(cli, "--smooth", smooth, "Compute smooth normals");
  add_option(cli, "--faceted", faceted, "Remove normals");
  add_option(cli, "--validate,-v", validate, "validate mesh");
  add_option(cli, "--path,-p", pathname, "output path");
  add_option(cli, "--stats,-s", statsname, "output stats");
  add_option(cli, "--scene,-S", scenename, "output scene");
  add_option(cli, "mesh", meshname, "input mesh", true);
  parse_cli(cli, argc, argv);

  // mesh data
  auto positions = vector<vec3f>{};
  auto normals   = vector<vec3f>{};
  auto texcoords = vector<vec2f>{};
  auto colors    = vector<vec3f>{};
  auto triangles = vector<vec3i>{};

  // stats
  auto stats = json{};

  // load mesh
  auto ioerror    = ""s;
  auto load_timer = print_timed("load mesh");
  if (!load_mesh(
          meshname, triangles, positions, normals, texcoords, colors, ioerror))
    print_fatal(ioerror);
  stats["mesh"]["load_time"] = print_elapsed(load_timer);
  stats["mesh"]["filename"]  = meshname;
  stats["mesh"]["valid"]     = false;
  stats["mesh"]["triangles"] = triangles.size();
  stats["mesh"]["vertices"]  = positions.size();

  // check if valid
  if (validate) {
    // TODO(fabio): validation code here
  } else {
    stats["mesh"]["valid"] = true;
  }

  // transform
  auto rescale_timer = print_timed("rescale bbox");
  auto bbox          = invalidb3f;
  for (auto& position : positions) bbox = merge(bbox, position);
  for (auto& position : positions)
    position = (position - center(bbox)) / max(size(bbox));
  stats["mesh"]["rescale_time"] = print_elapsed(rescale_timer);

  // default camera
  auto camera_from   = vec3f{0, 0, 3};
  auto camera_to     = vec3f{0, 0, 0};
  auto camera_lens   = 0.100f;
  auto camera_aspect = size(bbox).x / size(bbox).y;

  // build bvh
  auto bvh_timer       = print_timed("build bvh");
  auto bvh             = make_triangles_bvh(triangles, positions, {});
  stats["bvh"]["time"] = print_elapsed(bvh_timer);

  // pick points
  auto points_timer = print_timed("sample points");
  auto points = sample_points(triangles, positions, bvh, camera_from, camera_to,
      camera_lens, camera_aspect);
  stats["points"]["time"]      = print_elapsed(points_timer);
  stats["points"]["vertices"]  = points.size();
  stats["points"]["positions"] = points;

  // build graph
  auto graph_timer = print_timed("graph bvh");
  auto adjacencies = face_adjacencies(triangles);
  auto graph = make_dual_geodesic_solver(triangles, positions, adjacencies);
  stats["solver"]["time"] = print_elapsed(graph_timer);

  // trace path
  auto path_timer = print_timed("trace path");
  auto path = trace_path(graph, triangles, positions, adjacencies, points);
  stats["path"]["time"]     = print_elapsed(path_timer);
  stats["path"]["filename"] = pathname;
  stats["path"]["vertices"] = path.size();

  // create output directories
  if (!make_directory(path_dirname(statsname), ioerror)) print_fatal(ioerror);
  if (!make_directory(path_dirname(pathname), ioerror)) print_fatal(ioerror);
  if (!make_directory(path_dirname(scenename), ioerror)) print_fatal(ioerror);
  if (!make_directory(path_join(path_dirname(scenename), "shapes"), ioerror))
    print_fatal(ioerror);
  if (!make_directory(path_join(path_dirname(scenename), "textures"), ioerror))
    print_fatal(ioerror);

  // save path
  if (!save_mesh_points(pathname, path, ioerror)) print_fatal(ioerror);

  // save scene
  auto scene_timer = print_timed("save scene");
  auto scene_guard = std::make_unique<sceneio_scene>();
  auto scene       = scene_guard.get();
  make_scene_floating(scene, camera_from, camera_to, camera_lens, camera_aspect,
      triangles, positions, points, path);
  if (!save_scene(scenename, scene, ioerror)) print_fatal(ioerror);
  stats["scene"]["time"]     = print_elapsed(scene_timer);
  stats["scene"]["filename"] = scenename;

  // save stats
  auto stats_timer = print_timed("save stats");
  if (!save_json(statsname, stats, ioerror)) print_fatal(ioerror);
  stats["stats"]["time"] = print_elapsed(stats_timer);

  // done
  return 0;
}
