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

#include "../yocto/yocto_commonio.h"
#include "../yocto/yocto_math.h"
#include "../yocto/yocto_scene.h"
#include "../yocto/yocto_sceneio.h"
#include "../yocto/yocto_shape.h"
using namespace yocto;

int main(int argc, const char** argv) {
  // command line parameters
  auto geodesic_source      = -1;
  int  p0                   = -1;
  int  p1                   = -1;
  int  p2                   = -1;
  auto num_geodesic_samples = 0;
  auto geodesic_scale       = 30.0f;
  auto slice                = false;
  auto facevarying          = false;
  auto positiononly         = false;
  auto trianglesonly        = false;
  auto normals              = false;
  auto rotate               = zero3f;
  auto scale                = vec3f{1};
  auto uscale               = 1.0f;
  auto translate            = zero3f;
  auto output               = "out.ply"s;
  auto filename             = "mesh.ply"s;

  // parse command line
  auto cli = make_cli("ymshproc", "Applies operations on a triangle mesh");
  add_cli_option(
      cli, "--geodesic-source,-g", geodesic_source, "Geodesic source");
  add_cli_option(cli, "--path-vertex0,-p0", p0, "Path vertex 0");
  add_cli_option(cli, "--path-vertex1,-p1", p1, "Path vertex 1");
  add_cli_option(cli, "--path-vertex2,-p2", p2, "Path vertex 2");
  add_cli_option(cli, "--num-geodesic-samples", num_geodesic_samples,
      "Number of sampled geodesic sources");
  add_cli_option(cli, "--geodesic-scale", geodesic_scale, "Geodesic scale");
  add_cli_option(cli, "--slice", slice, "Slice mesh along field isolines");
  add_cli_option(cli, "--facevarying", facevarying, "Preserve facevarying");
  add_cli_option(
      cli, "--positiononly", positiononly, "Remove all but positions");
  add_cli_option(
      cli, "--trianglesonly", trianglesonly, "Remove all but triangles");
  add_cli_option(cli, "--normals", normals, "Compute smooth normals");
  add_cli_option(cli, "--rotatey", rotate.y, "Rotate around y axis");
  add_cli_option(cli, "--rotatex", rotate.x, "Rotate around x axis");
  add_cli_option(cli, "--rotatez", rotate.z, "Rotate around z axis");
  add_cli_option(cli, "--translatey", translate.y, "Translate along y axis");
  add_cli_option(cli, "--translatex", translate.x, "Translate along x axis");
  add_cli_option(cli, "--translatez", translate.z, "Translate along z axis");
  add_cli_option(cli, "--scale", uscale, "Scale along xyz axes");
  add_cli_option(cli, "--scaley", scale.y, "Scale along y axis");
  add_cli_option(cli, "--scalex", scale.x, "Scale along x axis");
  add_cli_option(cli, "--scalez", scale.z, "Scale along z axis");
  add_cli_option(cli, "--output,-o", output, "output mesh", true);
  add_cli_option(cli, "mesh", filename, "input mesh", true);
  if (!parse_cli(cli, argc, argv)) exit(1);

  // load mesh
  auto shape = yocto_shape{};
  try {
    auto timer = print_timed("loading shape");
    if (!facevarying) {
      load_shape(filename, shape.points, shape.lines, shape.triangles,
          shape.quads, shape.positions, shape.normals, shape.texcoords,
          shape.colors, shape.radius);
    } else {
      load_fvshape(filename, shape.quadspos, shape.quadsnorm,
          shape.quadstexcoord, shape.positions, shape.normals, shape.texcoords);
    }
  } catch (const std::exception& e) {
    print_fatal(e.what());
  }

  // remove data
  if (positiononly) {
    shape.normals       = {};
    shape.texcoords     = {};
    shape.colors        = {};
    shape.radius        = {};
    shape.quadsnorm     = {};
    shape.quadstexcoord = {};
    if (!shape.quadspos.empty()) swap(shape.quads, shape.quadspos);
  }

  // convert data
  if (trianglesonly) {
    if (!shape.quadspos.empty()) {
      print_fatal("cannot convert facevarying data to triangles");
    }
    if (!shape.quads.empty()) {
      shape.triangles = quads_to_triangles(shape.quads);
      shape.quads     = {};
    }
  }

  // transform
  if (uscale != 1) scale *= uscale;
  if (translate != zero3f || rotate != zero3f || scale != vec3f{1}) {
    auto timer = print_timed("transforming shape");
    auto xform = translation_frame(translate) * scaling_frame(scale) *
                 rotation_frame({1, 0, 0}, radians(rotate.x)) *
                 rotation_frame({0, 0, 1}, radians(rotate.z)) *
                 rotation_frame({0, 1, 0}, radians(rotate.y));
    for (auto& p : shape.positions) p = transform_point(xform, p);
    for (auto& n : shape.normals)
      n = transform_normal(xform, n, max(scale) != min(scale));
  }

  // compute normals
  if (normals) {
    auto timer = print_timed("computing normals");
    update_normals(shape);
    if (!shape.quadspos.empty()) shape.quadsnorm = shape.quadspos;
  }

  // compute geodesics and store them as colors
  if (geodesic_source >= 0 || num_geodesic_samples > 0) {
    auto timer       = print_timed("computing geodesics");
    auto adjacencies = face_adjacencies(shape.triangles);
    auto solver      = make_geodesic_solver(
        shape.triangles, adjacencies, shape.positions);
    auto sources = vector<int>();
    if (geodesic_source >= 0) {
      sources = {geodesic_source};
    } else {
      sources = sample_vertices_poisson(solver, num_geodesic_samples);
    }
    auto field = compute_geodesic_distances(solver, sources);

    if (slice) {
      auto tags = vector<int>(shape.triangles.size(), 0);
      meandering_triangles(field, geodesic_scale, 0, 1, 2, shape.triangles,
          tags, shape.positions, shape.normals);
      for (int i = 0; i < shape.triangles.size(); i++) {
        if (tags[i] == 1) shape.triangles[i] = {-1, -1, -1};
      }
    } else {
      shape.colors = vector<vec4f>(shape.positions.size());
      for (int i = 0; i < shape.colors.size(); ++i) {
        shape.colors[i] = vec4f(sinf(geodesic_scale * field[i]));
      }
      // distance_to_color(shape.colors, field, geodesic_scale);
    }
  }

  if (p0 != -1) {
    auto tags        = vector<int>(shape.triangles.size(), 0);
    auto adjacencies = face_adjacencies(shape.triangles);
    auto solver      = make_geodesic_solver(
        shape.triangles, adjacencies, shape.positions);

    auto          paths = vector<surface_path>();
    vector<float> fields[3];
    fields[0] = compute_geodesic_distances(solver, {p0});
    fields[1] = compute_geodesic_distances(solver, {p1});
    fields[2] = compute_geodesic_distances(solver, {p2});
    for (int i = 0; i < 3; ++i) {
      for (auto& f : fields[i]) f = -f;
    }

    paths.push_back(follow_gradient_field(shape.triangles, shape.positions,
        adjacencies, tags, 0, fields[1], p0, p1));

    paths.push_back(follow_gradient_field(shape.triangles, shape.positions,
        adjacencies, tags, 0, fields[2], p1, p2));

    paths.push_back(follow_gradient_field(shape.triangles, shape.positions,
        adjacencies, tags, 0, fields[0], p2, p0));

    vector<vec2i> lines;
    vector<vec3f> positions;
    for (int i = 0; i < 3; i++) {
      auto pos = make_positions_from_path(paths[i], shape.positions);
      auto line = vector<vec2i>(pos.size() - 1);
      for (int k = 0; k < line.size(); k++) {
          line[k] = {k, k+1};
        line[k] += (int)lines.size();
      }
      lines += line;
      positions += pos;
    }
    shape           = {};
    shape.lines     = lines;
    shape.positions = positions;
  }

  // save mesh
  try {
    auto timer = print_timed("saving shape");
    if (!shape.quadspos.empty()) {
      save_fvshape(output, shape.quadspos, shape.quadsnorm, shape.quadstexcoord,
          shape.positions, shape.normals, shape.texcoords);
    } else {
      save_shape(output, shape.points, shape.lines, shape.triangles,
          shape.quads, shape.positions, shape.normals, shape.texcoords,
          shape.colors, shape.radius);
    }
  } catch (const std::exception& e) {
    print_fatal(e.what());
  }

  // done
  return 0;
}
