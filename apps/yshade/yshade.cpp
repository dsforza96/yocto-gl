//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
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

#include <yocto/yocto_cli.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_image.h>
#include <yocto/yocto_mesh.h>
#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sceneio.h>
#include <yocto/yocto_shape.h>
#include <yocto_gui/yocto_imgui.h>
#include <yocto_gui/yocto_shade.h>

#include <queue>
#include <unordered_set>

#ifdef _WIN32
#undef near
#undef far
#endif

using namespace yocto;

static void init_glscene(shade_scene &glscene, const sceneio_scene &ioscene,
    progress_callback progress_cb) {
  // handle progress
  auto progress = vec2i{
      0, (int)ioscene.cameras.size() + (int)ioscene.materials.size() +
             (int)ioscene.textures.size() + (int)ioscene.shapes.size() +
             (int)ioscene.instances.size()};

  // init scene
  init_scene(glscene);

  // camera
  for (auto &iocamera : ioscene.cameras) {
    if (progress_cb) progress_cb("convert camera", progress.x++, progress.y);
    auto &camera = glscene.cameras.at(add_camera(glscene));
    set_frame(camera, iocamera.frame);
    set_lens(camera, iocamera.lens, iocamera.aspect, iocamera.film);
    set_nearfar(camera, 0.001, 10000);
  }

  // textures
  for (auto &iotexture : ioscene.textures) {
    if (progress_cb) progress_cb("convert texture", progress.x++, progress.y);
    auto  handle    = add_texture(glscene);
    auto &gltexture = glscene.textures[handle];
    if (!iotexture.pixelsf.empty()) {
      set_texture(
          gltexture, iotexture.width, iotexture.height, iotexture.pixelsf);
    } else if (!iotexture.pixelsb.empty()) {
      set_texture(
          gltexture, iotexture.width, iotexture.height, iotexture.pixelsb);
    }
  }

  // material
  for (auto &iomaterial : ioscene.materials) {
    if (progress_cb) progress_cb("convert material", progress.x++, progress.y);
    auto  handle     = add_material(glscene);
    auto &glmaterial = glscene.materials[handle];
    set_emission(glmaterial, iomaterial.emission, iomaterial.emission_tex);
    set_opacity(glmaterial, iomaterial.opacity, invalid_handle);
    set_normalmap(glmaterial, iomaterial.normal_tex);
    switch (iomaterial.type) {
      case material_type::matte: {
        set_color(glmaterial, iomaterial.color, iomaterial.color_tex);
        set_specular(glmaterial, 0, invalid_handle);
        set_metallic(glmaterial, 0, invalid_handle);
        set_roughness(glmaterial, 0, invalid_handle);
      } break;
      case material_type::plastic: {
        set_color(glmaterial, iomaterial.color, iomaterial.color_tex);
        set_specular(glmaterial, 1, invalid_handle);
        set_metallic(glmaterial, 0, invalid_handle);
        set_roughness(glmaterial, iomaterial.roughness, invalid_handle);
      } break;
      case material_type::metal: {
        set_color(glmaterial, iomaterial.color, iomaterial.color_tex);
        set_specular(glmaterial, 0, invalid_handle);
        set_metallic(glmaterial, 1, invalid_handle);
        set_roughness(
            glmaterial, iomaterial.roughness, iomaterial.roughness_tex);
      } break;
      case material_type::metallic: {
        set_color(glmaterial, iomaterial.color, iomaterial.color_tex);
        set_specular(glmaterial, 1, invalid_handle);
        set_metallic(glmaterial, iomaterial.metallic, invalid_handle);
        set_roughness(glmaterial, iomaterial.roughness, invalid_handle);
      } break;
      default: {
        set_color(glmaterial, iomaterial.color, iomaterial.color_tex);
        set_specular(glmaterial, 0, invalid_handle);
        set_metallic(glmaterial, 0, invalid_handle);
        set_roughness(
            glmaterial, iomaterial.roughness, iomaterial.roughness_tex);
      } break;
    }
  }

  // shapes
  for (auto &ioshape : ioscene.shapes) {
    if (progress_cb) progress_cb("convert shape", progress.x++, progress.y);
    add_shape(glscene, ioshape.points, ioshape.lines, ioshape.triangles,
        ioshape.quads, ioshape.positions, ioshape.normals, ioshape.texcoords,
        ioshape.colors);
  }

  // shapes
  for (auto &ioinstance : ioscene.instances) {
    if (progress_cb) progress_cb("convert instance", progress.x++, progress.y);
    auto  handle     = add_instance(glscene);
    auto &glinstance = glscene.instances[handle];
    set_frame(glinstance, ioinstance.frame);
    set_shape(glinstance, ioinstance.shape);
    set_material(glinstance, ioinstance.material);
  }

  // environments
  for (auto &ioenvironment : ioscene.environments) {
    auto  handle        = add_environment(glscene);
    auto &glenvironment = glscene.environments[handle];
    set_frame(glenvironment, ioenvironment.frame);
    set_emission(
        glenvironment, ioenvironment.emission, ioenvironment.emission_tex);
  }

  // init environments
  init_environments(glscene);

  // done
  if (progress_cb) progress_cb("convert done", progress.x++, progress.y);
}

using run_shade_scene_callback = std::function<void(gui_window *win,
    const gui_input &input, scene_scene &scene, shade_scene &glscene)>;

void run_shade_scene(scene_scene &scene, const string &name,
    const string &camname, const progress_callback &progress_cb,
    const run_shade_scene_callback &widgets_callback  = {},
    const run_shade_scene_callback &uiupdate_callback = {}) {
  // glscene
  auto glscene = shade_scene{};

  // draw params
  auto params = shade_params{};

  // callbacks
  auto callbacks    = gui_callbacks{};
  callbacks.init_cb = [&glscene, &scene, &progress_cb](
                          gui_window *win, const gui_input &input) {
    init_glscene(glscene, scene, progress_cb);
  };
  callbacks.clear_cb = [&glscene](gui_window *win, const gui_input &input) {
    clear_scene(glscene);
  };
  callbacks.draw_cb = [&glscene, &params](
                          gui_window *win, const gui_input &input) {
    draw_scene(
        glscene, glscene.cameras.at(0), input.framebuffer_viewport, params);
  };
  callbacks.widgets_cb = [&glscene, &scene, &params, &widgets_callback](
                             gui_window *win, const gui_input &input) {
    draw_checkbox(win, "wireframe", params.wireframe);
    continue_line(win);
    draw_checkbox(win, "faceted", params.faceted);
    continue_line(win);
    draw_checkbox(win, "double sided", params.double_sided);
    draw_combobox(
        win, "lighting", (int &)params.lighting, shade_lighting_names);
    draw_slider(win, "exposure", params.exposure, -10, 10);
    draw_slider(win, "gamma", params.gamma, 0.1f, 4);
    draw_slider(win, "near", params.near, 0.01f, 1.0f);
    draw_slider(win, "far", params.far, 1000.0f, 10000.0f);
    if (widgets_callback) widgets_callback(win, input, scene, glscene);
  };
  callbacks.update_cb = [](gui_window *win, const gui_input &input) {
    // update(win, apps);
  };
  callbacks.uiupdate_cb = [&glscene, &scene, &uiupdate_callback](
                              gui_window *win, const gui_input &input) {
    // handle mouse and keyboard for navigation
    if (uiupdate_callback) uiupdate_callback(win, input, scene, glscene);
    if ((input.mouse_left || input.mouse_right) && !input.modifier_alt &&
        !input.modifier_ctrl && !input.widgets_active) {
      auto dolly  = 0.0f;
      auto pan    = zero2f;
      auto rotate = zero2f;
      if (input.mouse_left && !input.modifier_shift)
        rotate = (input.mouse_pos - input.mouse_last) / 100.0f;
      if (input.mouse_right)
        dolly = (input.mouse_pos.x - input.mouse_last.x) / 100.0f;
      if (input.mouse_left && input.modifier_shift)
        pan = (input.mouse_pos - input.mouse_last) / 100.0f;
      auto &camera                         = scene.cameras.at(0);
      std::tie(camera.frame, camera.focus) = camera_turntable(
          camera.frame, camera.focus, rotate, dolly, pan);
      set_frame(glscene.cameras.at(0), camera.frame);
    }
  };

  // run ui
  run_ui({1280 + 320, 720}, "yshade", callbacks);
}

struct shade_scene_params {
  string scene = "scene.json"s;
};

// Cli
void add_command(cli_command &cli, const string &name,
    shade_scene_params &value, const string &usage) {
  auto &cmd = add_command(cli, name, usage);
  add_argument(cmd, "scene", value.scene, "Input scene.");
}

int run_shade_scene(const shade_scene_params &params) {
  // loading scene
  auto ioerror = ""s;
  auto scene   = scene_scene{};
  if (!load_scene(params.scene, scene, ioerror, print_progress))
    print_fatal(ioerror);

  // tesselation
  tesselate_shapes(scene, print_progress);

  // run viewer
  run_shade_scene(
      scene, params.scene, "", print_progress,
      [](gui_window *win, const gui_input &input, scene_scene &scene,
          shade_scene &glscene) {},
      [](gui_window *win, const gui_input &input, scene_scene &scene,
          shade_scene &glscene) {});

  // done
  return 0;
}

// Create a shape with small spheres for each point
static shape_data make_spheres(
    const vector<vec3f> &positions, float radius, int steps) {
  auto shape = shape_data{};
  for (auto position : positions) {
    auto sphere = make_sphere(steps, radius);
    for (auto &p : sphere.positions) p += position;
    merge_quads(shape.quads, shape.positions, shape.normals, shape.texcoords,
        sphere.quads, sphere.positions, sphere.normals, sphere.texcoords);
  }
  return shape;
}
static shape_data make_cylinders(const vector<vec2i> &lines,
    const vector<vec3f> &positions, float radius, const vec3i &steps) {
  auto shape = shape_data{};
  for (auto line : lines) {
    auto len      = length(positions[line.x] - positions[line.y]);
    auto dir      = normalize(positions[line.x] - positions[line.y]);
    auto center   = (positions[line.x] + positions[line.y]) / 2;
    auto cylinder = make_uvcylinder({4, 1, 1}, {radius, len / 2});
    auto frame    = frame_fromz(center, dir);
    for (auto &p : cylinder.positions) p = transform_point(frame, p);
    for (auto &n : cylinder.normals) n = transform_direction(frame, n);
    merge_quads(shape.quads, shape.positions, shape.normals, shape.texcoords,
        cylinder.quads, cylinder.positions, cylinder.normals,
        cylinder.texcoords);
  }
  return shape;
}

static scene_scene make_shapescene(
    const scene_shape &ioshape_, progress_callback progress_cb) {
  // Frame camera
  auto camera_frame = [](float lens, float aspect,
                          float film = 0.036) -> frame3f {
    auto camera_dir  = normalize(vec3f{0, 0.5, 1});
    auto bbox_radius = 2.0f;
    auto camera_dist = bbox_radius * lens / (film / aspect);
    return lookat_frame(camera_dir * camera_dist, {0, 0, 0}, {0, 1, 0});
  };

  // handle progress
  auto progress = vec2i{0, 5};
  if (progress_cb) progress_cb("create scene", progress.x++, progress.y);

  // init scene
  auto scene = scene_scene{};

  // rescale shape to unit
  auto ioshape = ioshape_;
  auto bbox    = invalidb3f;
  for (auto &pos : ioshape.positions) bbox = merge(bbox, pos);
  for (auto &pos : ioshape.positions) pos -= center(bbox);
  for (auto &pos : ioshape.positions) pos /= max(size(bbox));
  // TODO(fabio): this should be a math function

  // camera
  if (progress_cb) progress_cb("create camera", progress.x++, progress.y);
  auto &camera  = scene.cameras.emplace_back();
  camera.frame  = camera_frame(0.050, 16.0f / 9.0f, 0.036);
  camera.lens   = 0.050;
  camera.aspect = 16.0f / 9.0f;
  camera.film   = 0.036;
  camera.focus  = length(camera.frame.o - center(bbox));

  // material
  if (progress_cb) progress_cb("create material", progress.x++, progress.y);
  auto &shape_material     = scene.materials.emplace_back();
  shape_material.type      = material_type::plastic;
  shape_material.color     = {0.5, 1, 0.5};
  shape_material.roughness = 0.2;
  auto &edges_material     = scene.materials.emplace_back();
  edges_material.type      = material_type::matte;
  edges_material.color     = {0, 0, 0};
  auto &vertices_material  = scene.materials.emplace_back();
  vertices_material.type   = material_type::matte;
  vertices_material.color  = {0, 0, 0};

  // shapes
  if (progress_cb) progress_cb("create shape", progress.x++, progress.y);
  scene.shapes.emplace_back(ioshape);
  auto &edges_shape        = scene.shapes.emplace_back();
  edges_shape.positions    = ioshape.positions;
  edges_shape.lines        = get_edges(ioshape.triangles, ioshape.quads);
  auto &vertices_shape     = scene.shapes.emplace_back(ioshape);
  vertices_shape.positions = ioshape.positions;
  vertices_shape.points    = vector<int>(ioshape.positions.size());
  for (auto idx = 0; idx < (int)vertices_shape.points.size(); idx++)
    vertices_shape.points[idx] = idx;

#if 0
  auto froms = vector<vec3f>();
  auto tos   = vector<vec3f>();
  froms.reserve(edges.size());
  tos.reserve(edges.size());
  float avg_edge_length = 0;
  for (auto& edge : edges) {
    auto from = ioshape.positions[edge.x];
    auto to   = ioshape.positions[edge.y];
    froms.push_back(from);
    tos.push_back(to);
    avg_edge_length += length(from - to);
  }
  avg_edge_length /= edges.size();
  auto cylinder_radius = 0.05f * avg_edge_length;
  auto cylinder        = make_uvcylinder({4, 1, 1}, {cylinder_radius, 1});
  for (auto& p : cylinder.positions) {
    p.z = p.z * 0.5 + 0.5;
  }
  auto edges_shape = add_shape(glscene, {}, {}, {}, cylinder.quads,
      cylinder.positions, cylinder.normals, cylinder.texcoords, {});
  set_instances(glscene.shapes[edges_shape], froms, tos);

  auto vertices_radius = 3.0f * cylinder_radius;
  auto vertices        = make_spheres(ioshape.positions, vertices_radius, 2);
  auto vertices_shape  = add_shape(glscene, {}, {}, {}, vertices.quads,
      vertices.positions, vertices.normals, vertices.texcoords, {});
  set_instances(glscene.shapes[vertices_shape], {}, {});
#endif

  // instances
  if (progress_cb) progress_cb("create instance", progress.x++, progress.y);
  auto &shape_instance       = scene.instances.emplace_back();
  shape_instance.shape       = 0;
  shape_instance.material    = 0;
  auto &edges_instance       = scene.instances.emplace_back();
  edges_instance.shape       = 1;
  edges_instance.material    = 1;
  auto &vertices_instance    = scene.instances.emplace_back();
  vertices_instance.shape    = 2;
  vertices_instance.material = 2;

  // done
  if (progress_cb) progress_cb("create scene", progress.x++, progress.y);
  return scene;
}

struct shade_shape_params {
  string shape = "shape.ply";
};

// Cli
void add_command(cli_command &cli, const string &name,
    shade_shape_params &value, const string &usage) {
  auto &cmd = add_command(cli, name, usage);
  add_argument(cmd, "shape", value.shape, "Input shape.");
}

int run_shade_shape(const shade_shape_params &params) {
  // loading shape
  auto ioerror = ""s;
  auto shape   = scene_shape{};
  print_progress("load shape", 0, 1);
  if (!load_shape(params.shape, shape, ioerror)) print_fatal(ioerror);
  print_progress("load shape", 1, 1);

  // create scene
  auto scene = make_shapescene(shape, print_progress);

  // run viewer
  run_shade_scene(
      scene, params.shape, "", print_progress,
      [](gui_window *win, const gui_input &input, scene_scene &scene,
          shade_scene &glscene) {
        auto &ioshape = scene.shapes.at(0);
        draw_label(win, "points", std::to_string(ioshape.points.size()));
        draw_label(win, "lines", std::to_string(ioshape.lines.size()));
        draw_label(win, "triangles", std::to_string(ioshape.triangles.size()));
        draw_label(win, "quads", std::to_string(ioshape.quads.size()));
        draw_label(win, "positions", std::to_string(ioshape.positions.size()));
        draw_label(win, "normals", std::to_string(ioshape.normals.size()));
        draw_label(win, "texcoords", std::to_string(ioshape.texcoords.size()));
        draw_label(win, "colors", std::to_string(ioshape.colors.size()));
        draw_label(win, "radius", std::to_string(ioshape.radius.size()));
      },
      [](gui_window *win, const gui_input &input, scene_scene &scene,
          shade_scene &glscene) {
        glscene.instances[1].hidden = true;
        glscene.instances[2].hidden = true;
      });

  // done
  return 0;
}

enum struct sculpt_brush_type { gaussian, texture, smooth };
auto const sculpt_brush_names = vector<std::string>{
    "gaussian brush", "texture brush", "smooth brush"};

struct sculpt_params {
  sculpt_brush_type type     = sculpt_brush_type::gaussian;
  float             radius   = 0.35f;
  float             strength = 1.0f;
  bool              negative = false;
};

struct sculpt_state {
  // data structures
  shape_bvh           bvh         = {};
  hash_grid           grid        = {};
  vector<vector<int>> adjacencies = {};
  geodesic_solver     solver      = {};
  // brush
  image_data tex_image = {};
  // stroke
  vector<shape_point> stroke   = {};
  vec2f               last_uv  = {};
  bool                instroke = false;
  // shape at the beginning of the stroke
  scene_shape base_shape = {};
};

// Initialize all sculpting parameters.
sculpt_state make_sculpt_state(
    const shape_data &shape, const scene_texture &texture) {
  auto state = sculpt_state{};
  state.bvh  = make_triangles_bvh(
      shape.triangles, shape.positions, shape.radius);
  state.grid       = make_hash_grid(shape.positions, 0.05f);
  auto adjacencies = face_adjacencies(shape.triangles);
  state.solver     = make_geodesic_solver(
      shape.triangles, adjacencies, shape.positions);
  state.adjacencies = vertex_adjacencies(shape.triangles, adjacencies);
  state.tex_image   = texture;
  state.base_shape  = shape;
  state.base_shape.texcoords.assign(shape.positions.size(), {0, 0});
  return state;
}

shape_data make_circle(
    const vec3f &center, const mat3f &basis, float radius, int steps) {
  // 4 initial vertices
  auto  lines    = make_lines({1, 4});
  vec3f next_dir = basis.x;
  for (auto line : lines.lines) {
    auto v1             = line.x;
    auto v2             = line.y;
    lines.positions[v1] = center + next_dir * radius;
    lines.normals[v1]   = basis.z;
    next_dir            = cross(basis.z, next_dir);
    lines.positions[v2] = center + next_dir * radius;
    lines.normals[v2]   = basis.z;
  }

  // create polylines and fix lenght to radius
  auto positions = vector<vec3f>{};
  auto normals   = vector<vec3f>{};
  auto lins      = vector<vec2i>{};
  steps          = int(steps / 4);
  for (auto line : lines.lines) {
    auto v1  = line.x;
    auto v2  = line.y;
    auto pos = lines.positions[v1];
    positions.push_back(pos);
    normals.push_back(lines.normals[v1]);
    auto dir    = lines.positions[v2] - lines.positions[v1];
    auto lenght = length(dir) / steps;
    dir         = normalize(dir);
    for (int i = 0; i < steps; i++) {
      auto new_pos = pos + dir * lenght * (i + 1);
      auto new_dir = normalize(new_pos - center);
      new_pos      = center + new_dir * radius;
      positions.push_back(new_pos);
      normals.push_back(basis.z);
      lins.push_back({int(positions.size() - 2), int(positions.size() - 1)});
    }
  }

  // apply
  lines.positions = positions;
  lines.normals   = normals;
  lines.lines     = lins;
  return lines;
}

// To visualize mouse intersection on mesh
shape_data make_cursor(const vec3f &position, const vec3f &normal, float radius,
    float height = 0.05f) {
  auto basis  = basis_fromz(normal);
  auto cursor = make_circle(position, basis, radius, 32);
  cursor.normals.clear();
  cursor.texcoords.clear();
  cursor.positions.push_back(position);
  cursor.positions.push_back(position + normal * 0.05f);
  cursor.lines.push_back(
      {int(cursor.positions.size() - 2), int(cursor.positions.size() - 1)});
  return cursor;
}

// To take shape positions indices associate with planar coordinates
vector<int> stroke_parameterization(vector<vec2f> &coords,
    const geodesic_solver &solver, const vector<int> &sampling,
    const vector<vec3f> &positions, const vector<vec3f> &normals,
    float radius) {
  if (normals.empty()) return vector<int>{};

  // Planar coordinates by local computation between neighbors
  auto compute_coordinates = [](vector<vec2f> &       coords,
                                 const vector<mat3f> &frames,
                                 const vector<vec3f> &positions, int node,
                                 int neighbor, float weight) -> void {
    // Project a vector on a plane, maintaining vector length
    auto project_onto_plane = [](const mat3f &basis, const vec3f &p) -> vec2f {
      auto v  = p - dot(p, basis.z) * basis.z;
      auto v1 = vec2f{dot(v, basis.x), dot(v, basis.y)};
      return v1 * (length(p) / length(v1));
    };

    auto current_coord = coords[node];
    auto edge          = positions[node] - positions[neighbor];
    auto projection    = project_onto_plane(frames[neighbor], edge);
    auto new_coord     = coords[neighbor] + projection;
    auto avg_lenght    = (length(current_coord) + length(new_coord)) / 2;
    auto new_dir       = normalize(current_coord + new_coord);
    coords[node] = current_coord == zero2f ? new_coord : new_dir * avg_lenght;

    // following doesn't work
    // coords[node] = current_coord + (weight * (coords[neighbor] +
    // projection));
  };

  // Frame by local computation between neighbors
  auto compute_frame = [](vector<mat3f> &frames, const vector<vec3f> &normals,
                           int node, int neighbor, float weight) -> void {
    auto current_dir = frames[node].x;
    auto rotation    = basis_fromz(normals[neighbor]) *
                    transpose(basis_fromz(normals[node]));
    auto neighbor_dir = frames[neighbor].x;
    current_dir       = current_dir + (rotation * weight * neighbor_dir);
    frames[node].z    = normals[node];
    frames[node].y    = cross(frames[node].z, normalize(current_dir));
    frames[node].x    = cross(frames[node].y, frames[node].z);
  };

  // Classic Dijkstra
  auto dijkstra = [](const geodesic_solver &solver, const vector<int> &sources,
                      vector<float> &distances, float max_distance,
                      auto &&update) -> void {
    auto compare = [&](int i, int j) { return distances[i] > distances[j]; };
    std::priority_queue<int, vector<int>, decltype(compare)> queue(compare);

    // setup queue
    for (auto source : sources) queue.push(source);

    while (!queue.empty()) {
      int node = queue.top();
      queue.pop();

      auto distance = distances[node];
      if (distance > max_distance) continue;  // early exit

      for (auto arc : solver.graph[node]) {
        auto new_distance = distance + arc.length;

        update(node, arc.node, new_distance);

        if (new_distance < distances[arc.node]) {
          distances[arc.node] = new_distance;
          queue.push(arc.node);
        }
      }
    }
  };

  // Compute initial frames of stroke sampling vertices
  auto compute_stroke_frames = [](vector<mat3f> &       frames,
                                   const vector<vec3f> &positions,
                                   const vector<vec3f> &normals,
                                   const vector<int> &stroke_sampling) -> void {
    // frames follow stroke direction
    for (int i = 0; i < (int)stroke_sampling.size() - 1; i++) {
      int  curr    = stroke_sampling[i];
      int  next    = stroke_sampling[i + 1];
      auto dir     = positions[next] - positions[curr];
      auto z       = normals[curr];
      auto y       = cross(z, normalize(dir));
      auto x       = cross(y, z);
      frames[curr] = {x, y, z};
    }

    if (stroke_sampling.size() == 1) {
      frames[stroke_sampling[0]] = basis_fromz(normals[stroke_sampling[0]]);
    } else {
      int  final    = stroke_sampling[stroke_sampling.size() - 1];
      int  prev     = stroke_sampling[stroke_sampling.size() - 2];
      auto dir      = positions[final] - positions[prev];
      auto z        = normals[final];
      auto y        = cross(z, normalize(dir));
      auto x        = cross(y, z);
      frames[final] = {x, y, z};
    }

    // average frames direction of middle stroke vertices
    for (int i = 1; i < stroke_sampling.size() - 1; i++) {
      int  curr    = stroke_sampling[i];
      int  next    = stroke_sampling[i + 1];
      int  prev    = stroke_sampling[i - 1];
      auto dir     = frames[prev].x + frames[next].x;
      auto z       = normals[curr];
      auto y       = cross(z, normalize(dir));
      auto x       = cross(y, z);
      frames[curr] = {x, y, z};
    }
  };

  // init params
  auto vertices = std::unordered_set<int>{};  // to avoid duplicates

  auto visited = vector<bool>(positions.size(), false);
  for (auto sample : sampling) visited[sample] = true;

  coords              = vector<vec2f>(solver.graph.size(), zero2f);
  coords[sampling[0]] = {radius, radius};
  vertices.insert(sampling[0]);
  for (int i = 1; i < sampling.size(); i++) {
    auto edge           = positions[sampling[i]] - positions[sampling[i - 1]];
    coords[sampling[i]] = {coords[sampling[i - 1]].x + length(edge), radius};
    vertices.insert(sampling[i]);
  }

  auto distances = vector<float>(solver.graph.size(), flt_max);
  for (auto sample : sampling) distances[sample] = 0.0f;

  auto frames = vector<mat3f>(positions.size(), identity3x3f);
  compute_stroke_frames(frames, positions, normals, sampling);

  auto update = [&](int node, int neighbor, float new_distance) {
    vertices.insert(node);
    if (!visited[neighbor]) return;
    float weight = length(positions[neighbor] - positions[node]) + flt_eps;
    weight       = 1.0f / weight;
    compute_coordinates(coords, frames, positions, node, neighbor, weight);
    compute_frame(frames, normals, node, neighbor, weight);
    visited[node] = true;
  };
  dijkstra(solver, sampling, distances, radius, update);

  auto vec_vertices = vector<int>(vertices.begin(), vertices.end());

  // conversion in [0, 1]
  for (int i = 0; i < vertices.size(); i++)
    coords[vec_vertices[i]] /= radius * 2.0f;

  return vec_vertices;
}

// Compute gaussian function
float gaussian_distribution(const vec3f &origin, const vec3f &position,
    float standard_dev, float scale_factor, float strength, float radius) {
  auto scaled_strength = strength / ((((radius - 0.1f) * 0.5f) / 0.7f) + 0.2f);
  auto N               = 1.0f / (((standard_dev * scaled_strength) *
                       (standard_dev * scaled_strength) *
                       (standard_dev * scaled_strength)) *
                      sqrt((2.0f * pi) * (2.0f * pi) * (2.0f * pi)));
  auto d               = (origin - position) * scale_factor;
  auto E               = dot(d, d) / (2.0f * standard_dev * standard_dev);
  return N * yocto::exp(-E);
}

// To apply brush on intersected points' neighbors
bool gaussian_brush(vector<vec3f> &positions, const hash_grid &grid,
    const vector<vec3i> &triangles, const vector<vec3f> &base_positions,
    const vector<vec3f> &base_normals, const vector<shape_point> &stroke,
    const sculpt_params &params) {
  if (stroke.empty()) return false;

  // helpers
  auto eval_position = [](const vector<vec3i> & triangles,
                           const vector<vec3f> &positions, int element,
                           const vec2f &uv) {
    auto &triangle = triangles[element];
    return interpolate_triangle(positions[triangle.x], positions[triangle.y],
        positions[triangle.z], uv);
  };
  auto eval_normal = [](const vector<vec3i> & triangles,
                         const vector<vec3f> &normals, int element,
                         const vec2f &uv) {
    auto &triangle = triangles[element];
    return normalize(interpolate_triangle(
        normals[triangle.x], normals[triangle.y], normals[triangle.z], uv));
  };

  // for a correct gaussian distribution
  float scale_factor = 3.5f / params.radius;

  auto neighbors = vector<int>{};
  for (auto [element, uv] : stroke) {
    auto position = eval_position(triangles, base_positions, element, uv);
    auto normal   = eval_normal(triangles, base_normals, element, uv);
    find_neighbors(grid, neighbors, position, params.radius);
    if (params.negative) normal = -normal;
    for (auto neighbor : neighbors) {
      auto gauss_height = gaussian_distribution(position, positions[neighbor],
          0.7f, scale_factor, params.strength, params.radius);
      positions[neighbor] += normal * gauss_height;
    }
    neighbors.clear();
  }

  return true;
}

// Compute texture values through the parameterization
bool texture_brush(vector<vec3f> &positions, vector<vec2f> &texcoords,
    const geodesic_solver &solver, const image_data &texture,
    const vector<vec3i> &triangles, const vector<vec3f> &base_positions,
    const vector<vec3f> &base_normals, const vector<shape_point> &stroke,
    const sculpt_params &params) {
  if (texture.pixelsf.empty() && texture.pixelsb.empty()) return false;

  // Taking closest vertex of an intersection
  auto closest_vertex = [](const vector<vec3i> &triangles, int element,
                            const vec2f &uv) -> int {
    auto &triangle = triangles[element];
    if (uv.x < 0.5f && uv.y < 0.5f) return triangle.x;
    if (uv.x > uv.y) return triangle.y;
    return triangle.z;
  };

  auto sampling = vector<int>{};
  for (auto [element, uv] : stroke) {
    sampling.push_back(closest_vertex(triangles, element, uv));
  }

  auto vertices = stroke_parameterization(
      texcoords, solver, sampling, base_positions, base_normals, params.radius);
  if (vertices.empty()) return false;

  positions = base_positions;

  auto scale_factor = 3.5f / params.radius;
  auto max_height   = gaussian_distribution(
      zero3f, zero3f, 0.7f, scale_factor, params.strength, params.radius);

  for (auto idx : vertices) {
    auto uv     = texcoords[idx];
    auto height = max(xyz(eval_image(texture, uv)));
    auto normal = base_normals[idx];
    if (params.negative) normal = -normal;
    height *= max_height;
    positions[idx] += normal * height;
  }

  return true;
}

// Cotangent operator

// Compute edge cotangents weights
float laplacian_weight(const vector<vec3f> &positions,
    const vector<vector<int>> &adjacencies, int node, int neighbor) {
  auto cotan = [](vec3f &a, vec3f &b) {
    return dot(a, b) / length(cross(a, b));
  };

  auto num_neighbors = int(adjacencies[node].size());

  int ind = -1;
  for (int i = 0; i < adjacencies[node].size(); i++) {
    if (adjacencies[node][i] == neighbor) {
      ind = i;
      break;
    }
  }

  auto prev = adjacencies[node][(ind - 1) >= 0 ? ind - 1 : num_neighbors - 1];
  auto next = adjacencies[node][(ind + 1) % num_neighbors];

  auto v1 = positions[node] - positions[prev];
  auto v2 = positions[neighbor] - positions[prev];
  auto v3 = positions[node] - positions[next];
  auto v4 = positions[neighbor] - positions[next];

  float cot_alpha = cotan(v1, v2);
  float cot_beta  = cotan(v3, v4);
  float weight    = (cot_alpha + cot_beta) / 2;

  float cotan_max = yocto::cos(flt_min) / yocto::sin(flt_min);
  weight          = clamp(weight, cotan_max, -cotan_max);
  return weight;
}

// Smooth brush with Laplace Operator Discretization and Cotangents Weights
bool smooth_brush(vector<vec3f> &positions, const geodesic_solver &solver,
    const vector<vec3i> &triangles, const vector<vector<int>> &adjacencies,
    vector<shape_point> &stroke, const sculpt_params &params) {
  if (stroke.empty()) return false;

  // Taking closest vertex of an intersection
  auto closest_vertex = [](const vector<vec3i> &triangles, int element,
                            const vec2f &uv) -> int {
    auto &triangle = triangles[element];
    if (uv.x < 0.5f && uv.y < 0.5f) return triangle.x;
    if (uv.x > uv.y) return triangle.y;
    return triangle.z;
  };

  auto stroke_vertices = vector<int>{};
  for (auto [element, uv] : stroke)
    stroke_vertices.push_back(closest_vertex(triangles, element, uv));

  // Classic Dijkstra
  auto dijkstra = [](const geodesic_solver &solver, const vector<int> &sources,
                      vector<float> &distances, float max_distance,
                      auto &&update) -> void {
    auto compare = [&](int i, int j) { return distances[i] > distances[j]; };
    std::priority_queue<int, vector<int>, decltype(compare)> queue(compare);

    // setup queue
    for (auto source : sources) queue.push(source);

    while (!queue.empty()) {
      int node = queue.top();
      queue.pop();

      auto distance = distances[node];
      if (distance > max_distance) continue;  // early exit

      for (auto arc : solver.graph[node]) {
        auto new_distance = distance + arc.length;

        update(node, arc.node, new_distance);

        if (new_distance < distances[arc.node]) {
          distances[arc.node] = new_distance;
          queue.push(arc.node);
        }
      }
    }
  };

  auto distances = vector<float>(solver.graph.size(), flt_max);
  for (auto sample : stroke_vertices) distances[sample] = 0.0f;

  auto current_node = -1;
  auto neighbors    = vector<int>{};
  auto weights      = vector<float>{};
  auto update       = [&](int node, int neighbor, float new_distance) {
    if (current_node == -1) current_node = node;
    if (node != current_node) {
      vec3f sum1 = zero3f;
      float sum2 = 0.0f;
      for (int i = 0; i < neighbors.size(); i++) {
        sum1 += positions[neighbors[i]] * weights[i];
        sum2 += weights[i];
      }
      positions[current_node] += 0.5f *
                                 ((sum1 / sum2) - positions[current_node]);
      current_node = node;
      neighbors.clear();
      weights.clear();
    }
    neighbors.push_back(neighbor);
    weights.push_back(laplacian_weight(positions, adjacencies, node, neighbor));
  };
  dijkstra(solver, stroke_vertices, distances, params.radius, update);

  return true;
}

static scene_scene make_sculptscene(
    const scene_shape &ioshape_, progress_callback progress_cb) {
  // Frame camera
  auto camera_frame = [](float lens, float aspect,
                          float film = 0.036) -> frame3f {
    auto camera_dir  = normalize(vec3f{0, 0.5, 1});
    auto bbox_radius = 2.0f;
    auto camera_dist = bbox_radius * lens / (film / aspect);
    return lookat_frame(camera_dir * camera_dist, {0, 0, 0}, {0, 1, 0});
  };

  // handle progress
  auto progress = vec2i{0, 5};
  if (progress_cb) progress_cb("create scene", progress.x++, progress.y);

  // init scene
  auto scene = scene_scene{};

  // rescale shape to unit
  auto ioshape = ioshape_;
  auto bbox    = invalidb3f;
  for (auto &pos : ioshape.positions) bbox = merge(bbox, pos);
  for (auto &pos : ioshape.positions) pos -= center(bbox);
  for (auto &pos : ioshape.positions) pos /= max(size(bbox));

  // camera
  if (progress_cb) progress_cb("create camera", progress.x++, progress.y);
  auto &camera  = scene.cameras.emplace_back();
  camera.frame  = camera_frame(0.050, 16.0f / 9.0f, 0.036);
  camera.lens   = 0.050;
  camera.aspect = 16.0f / 9.0f;
  camera.film   = 0.036;
  camera.focus  = length(camera.frame.o - center(bbox));

  // material
  if (progress_cb) progress_cb("create material", progress.x++, progress.y);
  auto &shape_material  = scene.materials.emplace_back();
  shape_material.type   = material_type::matte;
  shape_material.color  = {0.78f, 0.31f, 0.23f};
  auto &cursor_material = scene.materials.emplace_back();
  cursor_material.type  = material_type::matte;

  // shapes
  if (progress_cb) progress_cb("create shape", progress.x++, progress.y);
  scene.shapes.emplace_back(ioshape);
  scene.shapes.emplace_back(make_cursor({0, 0, 0}, {0, 0, 1}, 1));

  // instances
  if (progress_cb) progress_cb("create instance", progress.x++, progress.y);
  auto &shape_instance     = scene.instances.emplace_back();
  shape_instance.shape     = 0;
  shape_instance.material  = 0;
  auto &cursor_instance    = scene.instances.emplace_back();
  cursor_instance.shape    = 1;
  cursor_instance.material = 1;

  // done
  if (progress_cb) progress_cb("create scene", progress.x++, progress.y);
  return scene;
}

// To make the stroke sampling (position, normal) following the mouse
static pair<vector<shape_point>, vec2f> sample_stroke(const shape_bvh &bvh,
    const scene_shape &shape, const vec2f &last_uv, const vec2f &mouse_uv,
    const scene_camera &camera, const sculpt_params &params) {
  // helper
  auto intersect_shape = [&](const vec2f &uv) {
    auto ray = camera_ray(
        camera.frame, camera.lens, camera.aspect, camera.film, uv);
    return intersect_triangles_bvh(
        bvh, shape.triangles, shape.positions, ray, false);
  };

  // eval current intersection
  auto last  = intersect_shape(last_uv);
  auto mouse = intersect_shape(mouse_uv);
  if (!mouse.hit || !last.hit) return {{}, last_uv};

  // sample
  auto delta_pos   = distance(eval_position(shape, last.element, last.uv),
      eval_position(shape, mouse.element, mouse.uv));
  auto stroke_dist = params.radius * 0.2f;
  auto steps       = int(delta_pos / stroke_dist);
  if (steps == 0) return {};
  auto update_uv = (mouse_uv - last_uv) * stroke_dist / delta_pos;
  auto cur_uv    = last_uv;
  auto samples   = vector<shape_point>{};
  for (auto step = 0; step < steps; step++) {
    cur_uv += update_uv;
    auto isec = intersect_shape(cur_uv);
    if (!isec.hit) continue;
    samples.push_back({isec.element, isec.uv});
  }

  return {samples, cur_uv};
}

static pair<bool, bool> sculpt_update(sculpt_state &state, scene_shape &shape,
    scene_shape &cursor, const scene_camera &camera, const vec2f &mouse_uv,
    bool mouse_pressed, const sculpt_params &params) {
  auto updated_shape = false, updated_cursor = false;

  auto ray = camera_ray(
      camera.frame, camera.lens, camera.aspect, camera.film, mouse_uv);
  auto isec = intersect_triangles_bvh(
      state.bvh, shape.triangles, shape.positions, ray, false);
  if (isec.hit) {
    cursor         = make_cursor(eval_position(shape, isec.element, isec.uv),
        eval_normal(shape, isec.element, isec.uv),
        params.radius *
            (params.type == sculpt_brush_type::gaussian ? 0.5f : 1.0f));
    updated_cursor = true;
  }

  // sculpting
  if (isec.hit && mouse_pressed) {
    if (!state.instroke) {
      state.instroke = true;
      state.last_uv  = mouse_uv;
    } else {
      auto last_uv           = state.last_uv;
      auto [samples, cur_uv] = sample_stroke(
          state.bvh, shape, last_uv, mouse_uv, camera, params);
      if (!samples.empty()) {
        state.last_uv = cur_uv;
        if (params.type == sculpt_brush_type::gaussian) {
          state.stroke  = samples;
          updated_shape = gaussian_brush(shape.positions, state.grid,
              shape.triangles, state.base_shape.positions,
              state.base_shape.normals, state.stroke, params);
        } else if (params.type == sculpt_brush_type::smooth) {
          state.stroke  = samples;
          updated_shape = smooth_brush(shape.positions, state.solver,
              shape.triangles, state.adjacencies, state.stroke, params);
        } else if (params.type == sculpt_brush_type::texture) {
          state.stroke.insert(
              state.stroke.end(), samples.begin(), samples.end());
          updated_shape = texture_brush(shape.positions,
              state.base_shape.texcoords, state.solver, state.tex_image,
              state.base_shape.triangles, state.base_shape.positions,
              state.base_shape.normals, state.stroke, params);
        }
      }
    }
    if (updated_shape) {
      triangles_normals(shape.normals, shape.triangles, shape.positions);
      update_triangles_bvh(state.bvh, shape.triangles, shape.positions);
      state.grid = make_hash_grid(shape.positions, state.grid.cell_size);
    }
  } else {
    if (state.instroke) {
      state.instroke = false;
      state.stroke.clear();
      state.base_shape.positions = shape.positions;
      state.base_shape.normals   = shape.normals;
      state.base_shape.texcoords.assign(
          state.base_shape.texcoords.size(), {0, 0});
    }
  }

  return {updated_shape, updated_cursor};
}

struct shade_sculpt_params {
  string shape   = "shape.ply"s;
  string texture = "";
};

// Cli
inline void add_command(cli_command &cli, const string &name,
    shade_sculpt_params &value, const string &usage) {
  auto &cmd = add_command(cli, name, usage);
  add_argument(cmd, "shape", value.shape, "Input shape.");
  add_option(cmd, "texture", value.texture, "Brush texture.");
}

int run_shade_sculpt(const shade_sculpt_params &params_) {
  // loading shape
  auto ioerror = ""s;
  auto ioshape = scene_shape{};
  print_progress("load shape", 0, 1);
  if (!load_shape(params_.shape, ioshape, ioerror)) print_fatal(ioerror);
  if (!ioshape.quads.empty()) {
    ioshape.triangles = quads_to_triangles(ioshape.quads);
    ioshape.quads.clear();
  }
  print_progress("load shape", 1, 1);

  // loading texture
  auto texture = scene_texture{};
  if (!params_.texture.empty()) {
    print_progress("load texture", 0, 1);
    if (!load_texture(params_.texture, texture, ioerror)) print_fatal(ioerror);
    print_progress("load texture", 1, 1);
  }

  // setup app
  auto scene = make_sculptscene(ioshape, print_progress);

  // sculpt params
  auto params = sculpt_params{};
  auto state  = make_sculpt_state(scene.shapes.front(), texture);

  // callbacks
  run_shade_scene(
      scene, params_.shape, "", print_progress,
      [&params](gui_window *win, const gui_input &input, scene_scene &scene,
          shade_scene &glscene) {
        draw_combobox(
            win, "brush type", (int &)params.type, sculpt_brush_names);
        if (params.type == sculpt_brush_type::gaussian) {
          if (params.strength < 0.8f || params.strength > 1.5f)
            params.strength = 1.0f;
          draw_slider(win, "radius", params.radius, 0.1f, 0.8f);
          draw_slider(win, "strength", params.strength, 1.5f, 0.9f);
          draw_checkbox(win, "negative", params.negative);
        } else if (params.type == sculpt_brush_type::texture) {
          if (params.strength < 0.8f || params.strength > 1.5f)
            params.strength = 1.0f;
          draw_slider(win, "radius", params.radius, 0.1f, 0.8f);
          draw_slider(win, "strength", params.strength, 1.5f, 0.9f);
          draw_checkbox(win, "negative", params.negative);
        } else if (params.type == sculpt_brush_type::smooth) {
          draw_slider(win, "radius", params.radius, 0.1f, 0.8f);
          draw_slider(win, "strength", params.strength, 0.1f, 1.0f);
        }
      },
      [&state, &params](gui_window *win, const gui_input &input,
          scene_scene &scene, shade_scene &glscene) {
        auto  mouse_uv = vec2f{input.mouse_pos.x / float(input.window_size.x),
            input.mouse_pos.y / float(input.window_size.y)};
        auto &shape    = scene.shapes.at(0);
        auto &cursor   = scene.shapes.at(1);
        auto &camera   = scene.cameras.at(0);
        auto &glshape  = glscene.shapes.at(0);
        auto [updated_shape, updated_cursor] = sculpt_update(state, shape,
            cursor, camera, mouse_uv, input.mouse_left && input.modifier_ctrl,
            params);
        if (updated_cursor) {
          set_positions(glscene.shapes.at(1), cursor.positions);
          set_lines(glscene.shapes.at(1), cursor.lines);
          glscene.instances.at(1).hidden = false;
        } else {
          glscene.instances.at(1).hidden = true;
        }
        if (updated_shape) {
          set_positions(glshape, shape.positions);
          set_normals(glshape, shape.normals);
        }
      });

  // done
  return 0;
}

struct app_params {
  string              command = "scene";
  shade_scene_params  scene   = {};
  shade_shape_params  shape   = {};
  shade_sculpt_params sculpt  = {};
};

// Cli
void add_commands(cli_command &cli, const string &name, app_params &value,
    const string &usage) {
  cli = make_cli(name, usage);
  add_command_name(cli, "command", value.command, "Command.");
  add_command(cli, "scene", value.scene, "View scenes.");
  add_command(cli, "shape", value.shape, "View shapes.");
  add_command(cli, "sculpt", value.sculpt, "Sculpt shapes.");
}

// Parse cli
void parse_cli(app_params &params, int argc, const char **argv) {
  auto cli = cli_command{};
  add_commands(cli, "yshade", params, "View and edit interactively");
  parse_cli(cli, argc, argv);
}

int main(int argc, const char *argv[]) {
  // command line parameters
  auto params = app_params{};
  parse_cli(params, argc, argv);

  // dispatch commands
  if (params.command == "scene") {
    return run_shade_scene(params.scene);
  } else if (params.command == "shape") {
    return run_shade_shape(params.shape);
  } else if (params.command == "sculpt") {
    return run_shade_sculpt(params.sculpt);
  } else {
    return print_fatal("unknown command " + params.command);
  }
}
