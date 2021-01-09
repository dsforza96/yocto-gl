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

#include <yocto/yocto_commonio.h>
#include <yocto/yocto_math.h>
#include <yocto/yocto_sceneio.h>
#include <yocto/yocto_shape.h>
#if YOCTO_OPENGL == 1
#include <yocto_gui/yocto_glview.h>
#endif
using namespace yocto;

#include "yscene_data.h"

triangles_shape make_bunny(float scale = 1, bool align_middle = true) {
  auto shape      = triangles_shape{};
  shape.triangles = bunny_triangles;
  shape.positions = bunny_positions;
  shape.normals   = bunny_normals;
  shape.texcoords = bunny_texcoords;
  // scale to height 1
  auto bbox = invalidb3f;
  for (auto& t : shape.triangles) {
    bbox = merge(bbox, triangle_bounds(shape.positions[t.x],
                           shape.positions[t.y], shape.positions[t.z]));
  }
  auto yscale = 2 / size(bbox).y;
  for (auto& p : shape.positions) p *= yscale;
  if (align_middle) {
    for (auto& p : shape.positions) p.y -= 1;
  }
  if (scale != 1) {
    for (auto& p : shape.positions) p *= scale;
  }
  return shape;
}

enum struct test_cameras_type { standard, wide };
enum struct test_environments_type { none, sky, sunsky };
enum struct test_arealights_type { none, standard, large };
enum struct test_floor_type { none, standard };
enum struct test_instance_name_type { material, shape };
enum struct test_shapes_type {
  // clang-format off
  features1, features2, rows, bunny_sphere,
  shapes1, shapes2, shapes3
  // clang-format off
};
enum struct test_materials_type {
  // clang-format off
  features1, features2, uvgrid, hair, plastic_metal,
  materials1, materials2, materials3, materials4, materials5,
  // clang-format on
};

struct test_params {
  test_cameras_type       cameras       = test_cameras_type::standard;
  test_environments_type  environments  = test_environments_type::sky;
  test_arealights_type    arealights    = test_arealights_type::standard;
  test_floor_type         floor         = test_floor_type::standard;
  test_shapes_type        shapes        = test_shapes_type::features1;
  test_materials_type     materials     = test_materials_type::features1;
  test_instance_name_type instance_name = test_instance_name_type::material;
};

// Scene test
void make_test(scene_scene& scene, const test_params& params) {
  // cameras
  switch (params.cameras) {
    case test_cameras_type::standard: {
      add_camera(scene, "default", {-0.75, 0.4, 0.9}, {-0.075, 0.05, -0.05},
          {0, 1, 0}, 0.05, 2.4, 0);
    } break;
    // TODO(fabio): fix wide camera
    case test_cameras_type::wide: {
      add_camera(scene, "default", {-0.75, 0.4, 0.9}, {-0.075, 0.05, -0.05},
          {0, 1, 0}, 0.05, 2.4, 0);
    } break;
  }
  // TODO(fabio): port other cameras
  switch (params.environments) {
    case test_environments_type::none: break;
    case test_environments_type::sky: {
      add_environment(scene, "sky", identity3x4f, {0.5, 0.5, 0.5},
          add_texture(scene, "sky",
              make_sunsky(
                  {2048, 1024}, pif / 4, 3.0, false, 1.0, 1.0, {0.7, 0.7, 0.7}),
              true));
    } break;
    case test_environments_type::sunsky: {
      add_environment(scene, "sunsky", identity3x4f, {0.5, 0.5, 0.5},
          add_texture(scene, "sky",
              make_sunsky(
                  {2048, 1024}, pif / 4, 3.0, true, 1.0, 1.0, {0.7, 0.7, 0.7}),
              true));
    } break;
  }
  switch (params.arealights) {
    case test_arealights_type::none: break;
    case test_arealights_type::standard: {
      add_instance(scene, "arealight1",
          lookat_frame({-0.4, 0.8, 0.8}, {0, 0.1, 0}, {0, 1, 0}, true),
          add_shape(scene, "arealight1", make_rect({1, 1}, {0.2, 0.2})),
          add_emission_material(
              scene, "arealight1", {20, 20, 20}, invalid_handle));
      add_instance(scene, "arealight2",
          lookat_frame({+0.4, 0.8, 0.8}, {0, 0.1, 0}, {0, 1, 0}, true),
          add_shape(scene, "arealight2", make_rect({1, 1}, {0.2, 0.2})),
          add_emission_material(
              scene, "arealight2", {20, 20, 20}, invalid_handle));
    } break;
    case test_arealights_type::large: {
      add_instance(scene, "largearealight1",
          lookat_frame({-0.8, 1.6, 1.6}, {0, 0.1, 0}, {0, 1, 0}, true),
          add_shape(scene, "largearealight1", make_rect({1, 1}, {0.4, 0.4})),
          add_emission_material(
              scene, "largearealight1", {10, 10, 10}, invalid_handle));
      add_instance(scene, "largearealight2",
          lookat_frame({+0.8, 1.6, 1.6}, {0, 0.1, 0}, {0, 1, 0}, true),
          add_shape(scene, "largearealight2", make_rect({1, 1}, {0.4, 0.4})),
          add_emission_material(
              scene, "largearealight2", {10, 10, 10}, invalid_handle));
    } break;
  }
  switch (params.floor) {
    case test_floor_type::none: break;
    case test_floor_type::standard: {
      add_instance(scene, "floor", identity3x4f,
          add_shape(scene, "floor", make_floor({1, 1}, {2, 2}, {20, 20})),
          add_matte_material(scene, "floor", {1, 1, 1},
              add_texture(scene, "floor", make_grid({1024, 1024}))));
    } break;
  }
  auto shapes = vector<shape_handle>{}, shapesi = vector<shape_handle>{};
  auto subdivs   = vector<subdiv_handle>{};
  auto materials = vector<material_handle>{};
  switch (params.shapes) {
    case test_shapes_type::features1: {
      auto bunny  = add_shape(scene, "bunny", make_bunny(0.075));
      auto sphere = add_shape(scene, "sphere", make_sphere(32, 0.075, 1));
      shapes      = {bunny, sphere, bunny, sphere, bunny};
    } break;
    case test_shapes_type::features2: {
      shapes  = {add_shape(scene, "sphere", make_sphere(32, 0.075, 1)),
          add_shape(scene, "suzanne", make_monkey(0.075f * 0.8f)),
          add_shape(scene, "hair",
              make_hair(make_sphere(32, 0.075f * 0.8f, 1), {4, 65536},
                  {0.1f * 0.15f, 0.1f * 0.15f},
                  {0.001f * 0.15f, 0.0005f * 0.15f}, {0.03, 100})),
          add_shape(scene, "displaced", make_sphere(128, 0.075f, 1)),
          add_shape(scene, "cube",
              make_rounded_box({32, 32, 32}, {0.075, 0.075, 0.075}, {1, 1, 1},
                  0.3 * 0.075f))};
      shapesi = {invalid_handle, invalid_handle,
          add_shape(scene, "hairi", make_sphere(32, 0.075f * 0.8f, 1)),
          invalid_handle, invalid_handle};
      subdivs = {add_subdiv(scene, "suzanne", make_monkey(0.075f * 0.8f),
                     shapes[1], 2),
          add_subdiv(scene, "displaced", make_sphere(128, 0.075f, 1), shapes[3],
              0, 0.025,
              add_texture(scene, "bumps-displacement", make_bumps({1024, 1024}),
                  false, true))};
    } break;
    case test_shapes_type::rows: {
      auto bunny  = add_shape(scene, "bunny", make_bunny(0.075));
      auto sphere = add_shape(scene, "sphere", make_sphere(32, 0.075, 1));
      shapes      = {bunny, bunny, bunny, bunny, bunny, sphere, sphere, sphere,
          sphere, sphere};
    } break;
    case test_shapes_type::bunny_sphere: {
      auto bunny  = add_shape(scene, "bunny", make_bunny(0.075));
      auto sphere = add_shape(scene, "sphere", make_sphere(32, 0.075, 1));
      shapes      = {bunny, sphere, bunny, sphere, bunny};
    } break;
    case test_shapes_type::shapes1: {
      shapes = {
          add_shape(scene, "sphere", make_sphere(32, 0.075, 1)),
          add_shape(scene, "uvsphere-flipcap",
              make_capped_uvsphere({32, 32}, 0.075, {1, 1}, 0.3 * 0.075)),
          add_shape(scene, "disk", make_disk(32, 0.075f, 1)),
          add_shape(scene, "uvcylinder",
              make_rounded_uvcylinder(
                  {32, 32, 32}, {0.075, 0.075}, {1, 1, 1}, 0.3 * 0.075)),
          add_shape(scene, "cube",
              make_rounded_box({32, 32, 32}, {0.075, 0.075, 0.075}, {1, 1, 1},
                  0.3 * 0.075f)),
      };
    } break;
    case test_shapes_type::shapes2: {
      shapes = {
          add_shape(scene, "cube-subdiv", make_fvcube(0.075)),
          add_shape(scene, "suzanne-subdiv", make_monkey(0.075)),
          add_shape(scene, "displaced", make_sphere(128, 0.075f, 1)),
          add_shape(scene, "bunny", make_bunny(0.075)),
          add_shape(scene, "teapot", make_sphere(32, 0.075, 1)),
      };
      subdivs = {
          add_subdiv(scene, "cube-subdiv", make_fvcube(0.075), shapes[0], 4),
          add_subdiv(scene, "suzanne-subdiv", make_monkey(0.075), shapes[1], 2),
          add_subdiv(scene, "displaced", make_sphere(128, 0.075f, 1), shapes[2],
              0, 0.025,
              add_texture(scene, "bumps-displacement", make_bumps({1024, 1024}),
                  false, true))};
    } break;
    case test_shapes_type::shapes3: {
      shapes = {
          invalid_handle,
          add_shape(scene, "hair1",
              make_hair(make_sphere(32, 0.075f * 0.8f, 1), {4, 65536},
                  {0.1f * 0.15f, 0.1f * 0.15f},
                  {0.001f * 0.15f, 0.0005f * 0.15f}, {0.03, 100})),
          add_shape(scene, "hair2",
              make_hair(make_sphere(32, 0.075f * 0.8f, 1), {4, 65536},
                  {0.1f * 0.15f, 0.1f * 0.15f},
                  {0.001f * 0.15f, 0.0005f * 0.15f})),
          add_shape(scene, "hair3",
              make_hair(make_sphere(32, 0.075f * 0.8f, 1), {4, 65536},
                  {0.1f * 0.15f, 0.1f * 0.15f},
                  {0.001f * 0.15f, 0.0005f * 0.15f}, {0, 0}, {0.5, 128})),
          invalid_handle,
      };
    } break;
  }
  switch (params.materials) {
    case test_materials_type::features1: {
      materials = {
          add_specular_coated_material(scene, "coated", {1, 1, 1},
              add_texture(scene, "uvgrid", make_uvgrid({1024, 1024})), 0.2),
          add_volumetric_material(
              scene, "glass", {1, 0.5, 0.5}, invalid_handle, 0),
          add_volumetric_material(scene, "jade", {0.5, 0.5, 0.5},
              invalid_handle, 0, invalid_handle, {0.3, 0.6, 0.3}),
          add_specular_material(scene, "bumped", {0.5, 0.7, 0.5},
              invalid_handle, 0.2, invalid_handle,
              add_texture(scene, "bumps-normal",
                  bump_to_normal(make_bumps({1024, 1024}), 0.05), false, true)),
          add_metallic_material(
              scene, "metal", {0.66, 0.45, 0.34}, invalid_handle, 0.2),
      };
    } break;
    case test_materials_type::features2: {
      auto uvgrid  = add_specular_material(scene, "uvgrid", {1, 1, 1},
          add_texture(scene, "uvgrid", make_uvgrid({1024, 1024})), 0.2);
      auto plastic = add_specular_material(
          scene, "plastic", {0.5, 0.7, 0.5}, invalid_handle, 0.2);
      auto hair = add_matte_material(
          scene, "hair", {0.7, 0.7, 0.7}, invalid_handle);
      materials = {uvgrid, plastic, hair, plastic, uvgrid};
    } break;
    case test_materials_type::uvgrid: {
      auto uvgrid = add_specular_material(scene, "uvgrid", {1, 1, 1},
          add_texture(scene, "uvgrid", make_uvgrid({1024, 1024})), 0.2);
      materials   = {uvgrid, uvgrid, uvgrid, uvgrid, uvgrid};
    } break;
    case test_materials_type::hair: {
      auto hair = add_matte_material(
          scene, "hair", {0.7, 0.7, 0.7}, invalid_handle);
      materials = {hair, hair, hair, hair, hair};
    } break;
    case test_materials_type::plastic_metal: {
      materials = {
          add_specular_material(
              scene, "plastic1", {0.5, 0.5, 0.7}, invalid_handle, 0.01),
          add_specular_material(
              scene, "plastic2", {0.5, 0.7, 0.5}, invalid_handle, 0.2),
          add_matte_material(scene, "matte", {0.7, 0.7, 0.7}, invalid_handle),
          add_metallic_material(
              scene, "metal1", {0.7, 0.7, 0.7}, invalid_handle, 0),
          add_metallic_material(
              scene, "metal2", {0.66, 0.45, 0.34}, invalid_handle, 0.2),
      };
    } break;
    case test_materials_type::materials1: {
      materials = {
          add_specular_material(
              scene, "plastic1", {0.5, 0.5, 0.7}, invalid_handle, 0.01),
          add_specular_material(
              scene, "plastic2", {0.5, 0.7, 0.5}, invalid_handle, 0.2),
          add_matte_material(scene, "matte", {0.7, 0.7, 0.7}, invalid_handle),
          add_metallic_material(
              scene, "metal1", {0.7, 0.7, 0.7}, invalid_handle, 0),
          add_metallic_material(
              scene, "metal2", {0.66, 0.45, 0.34}, invalid_handle, 0.2),
      };
    } break;
    case test_materials_type::materials2: {
      materials = {
          add_volumetric_material(
              scene, "glass1", {1, 1, 1}, invalid_handle, 0),
          add_volumetric_material(
              scene, "glass2", {1, 0.7, 0.7}, invalid_handle, 0.1),
          add_transparent_material(
              scene, "transparent", {0.7, 0.5, 0.5}, invalid_handle, 0.2),
          add_transmission_material(
              scene, "tglass1", {1, 1, 1}, invalid_handle, 0),
          add_transmission_material(
              scene, "tglass2", {1, 0.7, 0.7}, invalid_handle, 0.1),
      };
    } break;
    case test_materials_type::materials3: {
      auto bumps_normal = add_texture(scene, "bumps-normal",
          bump_to_normal(make_bumps({1024, 1024}), 0.05), false, true);
      materials         = {
          add_specular_material(scene, "plastic1", {0.5, 0.5, 0.7},
              invalid_handle, 0.01, invalid_handle, bumps_normal),
          add_specular_coated_material(
              scene, "plastic2", {0.5, 0.7, 0.5}, invalid_handle, 0.2),
          add_metallic_material(scene, "metal1", {0.7, 0.7, 0.7},
              invalid_handle, 0, invalid_handle, bumps_normal),
          add_metallic_coated_material(
              scene, "metal2", {0.66, 0.45, 0.34}, invalid_handle, 0.2),
          add_metallic_material(
              scene, "metal3", {0.66, 0.45, 0.34}, invalid_handle, 0.2),
      };
    } break;
    case test_materials_type::materials4: {
      materials = {
          add_volumetric_material(scene, "cloud", {0.65, 0.65, 0.65},
              invalid_handle, 0, invalid_handle, {0.9, 0.9, 0.9},
              invalid_handle, invalid_handle, 1),
          add_volumetric_material(
              scene, "glass", {1, 0.5, 0.5}, invalid_handle, 0),
          add_volumetric_material(scene, "jade", {0.5, 0.5, 0.5},
              invalid_handle, 0, invalid_handle, {0.3, 0.6, 0.3}),
          add_volumetrict_material(scene, "jade2", {0.5, 0.5, 0.5},
              invalid_handle, 0, invalid_handle, {0.3, 0.6, 0.3}),
          add_volumetric_material(scene, "smoke", {0.5, 0.5, 0.5},
              invalid_handle, 0.2, invalid_handle, {0.2, 0.2, 0.2}),
      };
    } break;
    case test_materials_type::materials5: {
      materials = {
          add_volumetric_material(scene, "skin1a", {0.76, 0.48, 0.23},
              invalid_handle, 0.25, invalid_handle, {0.436, 0.227, 0.131},
              invalid_handle, invalid_handle, 1.5, -0.8, 0.001),
          add_volumetric_material(scene, "skin2a", {0.82, 0.55, 0.4},
              invalid_handle, 0.25, invalid_handle, {0.623, 0.433, 0.343},
              invalid_handle, invalid_handle, 1.5, -0.8, 0.001),
          add_volumetric_material(scene, "skins", {0.76, 0.48, 0.23},
              invalid_handle, 0, invalid_handle, {0.436, 0.227, 0.131},
              invalid_handle, invalid_handle, 1.5, -0.8, 0.001),
          add_volumetrict_material(scene, "skin1b", {0.76, 0.48, 0.23},
              invalid_handle, 0.25, invalid_handle, {0.436, 0.227, 0.131},
              invalid_handle, invalid_handle, 1.5, -0.8, 0.001),
          add_volumetrict_material(scene, "skin2b", {0.82, 0.55, 0.4},
              invalid_handle, 0.25, invalid_handle, {0.623, 0.433, 0.343},
              invalid_handle, invalid_handle, 1.5, -0.8, 0.001),
      };
    } break;
  }
  for (auto idx = 0; idx < shapes.size(); idx++) {
    if (!shapes[idx]) continue;
    if (shapes.size() > 5) {
      add_instance(scene,
          scene.shape_names[idx] + "-" + scene.shape_names[idx % 5],
          {{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
              {0.2f * (idx % 5 - 2), 0.075, -0.4f * (idx / 5)}},
          shapes[idx], materials[idx % 5]);
    } else {
      auto name = params.instance_name == test_instance_name_type::material
                      ? scene.material_names[idx]
                      : scene.shape_names[idx];
      add_instance(scene, name,
          {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.2f * (idx % 5 - 2), 0.075, 0}},
          shapes[idx], materials[idx]);
    }
    if (!shapesi.empty() && shapesi[idx]) {
      // TODO(fabio): fix name
      add_instance(scene, "",
          {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.2f * (idx - 2), 0.075, 0}},
          shapesi[idx], materials[idx]);
    }
  }
}

// Scene presets used ofr testing.
bool make_preset(scene_scene& scene, const string& type, string& error) {
  if (type == "cornellbox") {
    make_cornellbox(scene);
    return true;
  } else if (type == "features1") {
    make_test(
        scene, {test_cameras_type::standard, test_environments_type::sky,
                   test_arealights_type::standard, test_floor_type::standard,
                   test_shapes_type::features1, test_materials_type::features1,
                   test_instance_name_type::material});
    return true;
  } else if (type == "features2") {
    make_test(
        scene, {test_cameras_type::standard, test_environments_type::sky,
                   test_arealights_type::standard, test_floor_type::standard,
                   test_shapes_type::features2, test_materials_type::features2,
                   test_instance_name_type::shape});
    return true;
  } else if (type == "materials1") {
    make_test(
        scene, {test_cameras_type::wide, test_environments_type::sky,
                   test_arealights_type::large, test_floor_type::standard,
                   test_shapes_type::rows, test_materials_type::materials1,
                   test_instance_name_type::material});
    return true;
  } else if (type == "materials2") {
    make_test(
        scene, {test_cameras_type::wide, test_environments_type::sky,
                   test_arealights_type::large, test_floor_type::standard,
                   test_shapes_type::rows, test_materials_type::materials2,
                   test_instance_name_type::material});
    return true;
  } else if (type == "materials3") {
    make_test(
        scene, {test_cameras_type::wide, test_environments_type::sky,
                   test_arealights_type::large, test_floor_type::standard,
                   test_shapes_type::rows, test_materials_type::materials3,
                   test_instance_name_type::material});
    return true;
  } else if (type == "materials4") {
    make_test(
        scene, {test_cameras_type::wide, test_environments_type::sky,
                   test_arealights_type::large, test_floor_type::standard,
                   test_shapes_type::rows, test_materials_type::materials4,
                   test_instance_name_type::material});
    return true;
  } else if (type == "materials5") {
    make_test(
        scene, {test_cameras_type::wide, test_environments_type::sky,
                   test_arealights_type::large, test_floor_type::standard,
                   test_shapes_type::rows, test_materials_type::materials5,
                   test_instance_name_type::material});
    return true;
  } else if (type == "shapes1") {
    make_test(scene, {test_cameras_type::standard, test_environments_type::sky,
                         test_arealights_type::large, test_floor_type::standard,
                         test_shapes_type::shapes1, test_materials_type::uvgrid,
                         test_instance_name_type::shape});
    return true;
  } else if (type == "shapes2") {
    make_test(scene, {test_cameras_type::standard, test_environments_type::sky,
                         test_arealights_type::large, test_floor_type::standard,
                         test_shapes_type::shapes2, test_materials_type::uvgrid,
                         test_instance_name_type::shape});
    return true;
  } else if (type == "shapes3") {
    make_test(scene, {test_cameras_type::standard, test_environments_type::sky,
                         test_arealights_type::large, test_floor_type::standard,
                         test_shapes_type::shapes3, test_materials_type::hair,
                         test_instance_name_type::shape});
    return true;
  } else if (type == "environments1") {
    make_test(scene,
        {test_cameras_type::standard, test_environments_type::sky,
            test_arealights_type::none, test_floor_type::standard,
            test_shapes_type::bunny_sphere, test_materials_type::plastic_metal,
            test_instance_name_type::material});
    return true;
  } else if (type == "environments2") {
    make_test(scene,
        {test_cameras_type::standard, test_environments_type::sunsky,
            test_arealights_type::none, test_floor_type::standard,
            test_shapes_type::bunny_sphere, test_materials_type::plastic_metal,
            test_instance_name_type::material});
    return true;
  } else if (type == "arealights1") {
    make_test(scene,
        {test_cameras_type::standard, test_environments_type::none,
            test_arealights_type::standard, test_floor_type::standard,
            test_shapes_type::bunny_sphere, test_materials_type::plastic_metal,
            test_instance_name_type::material});
    return true;
  } else {
    error = "unknown preset";
    return false;
  }
  return true;
}

// convert params
struct convert_params {
  string scene     = "scene.ply";
  string output    = "out.ply";
  bool   info      = false;
  bool   validate  = false;
  string copyright = "";
};

// Json IO
void serialize_value(json_mode mode, json_value& json, convert_params& value,
    const string& description) {
  serialize_object(mode, json, value, description);
  serialize_property(mode, json, value.scene, "scene", "Input scene.", true);
  serialize_property(mode, json, value.output, "output", "Output scene.");
  serialize_property(mode, json, value.info, "info", "Print info.");
  serialize_property(mode, json, value.validate, "validate", "Validate scene.");
  serialize_property(
      mode, json, value.copyright, "copyright", "Set scene copyright.");
  serialize_clipositionals(mode, json, {"scene"});
  serialize_clialternates(mode, json, {{"output", "o"}});
}

// convert images
int run_convert(const convert_params& params) {
  // load scene
  auto scene   = scene_scene{};
  auto ioerror = ""s;
  if (path_extension(params.scene) == ".ypreset") {
    print_progress("make preset", 0, 1);
    if (!make_preset(scene, path_basename(params.scene), ioerror))
      print_fatal(ioerror);
    print_progress("make preset", 1, 1);
  } else {
    if (!load_scene(params.scene, scene, ioerror, print_progress))
      print_fatal(ioerror);
  }

  // copyright
  if (params.copyright != "") {
    scene.asset.copyright = params.copyright;
  }

  // validate scene
  if (params.validate) {
    for (auto& error : scene_validation(scene)) print_info("error: " + error);
  }

  // print info
  if (params.info) {
    print_info("scene stats ------------");
    for (auto stat : scene_stats(scene)) print_info(stat);
  }

  // tesselate if needed
  if (path_extension(params.output) != ".json") {
    tesselate_shapes(scene, print_progress);
  }

  // make a directory if needed
  if (!make_directory(path_dirname(params.output), ioerror))
    print_fatal(ioerror);
  if (!scene.shapes.empty()) {
    if (!make_directory(
            path_join(path_dirname(params.output), "shapes"), ioerror))
      print_fatal(ioerror);
  }
  if (!scene.textures.empty()) {
    if (!make_directory(
            path_join(path_dirname(params.output), "textures"), ioerror))
      print_fatal(ioerror);
  }

  // save scene
  if (!save_scene(params.output, scene, ioerror, print_progress))
    print_fatal(ioerror);

  // done
  return 0;
}

// convert params
struct view_params {
  string scene  = "scene.json";
  string output = "out.png";
  string camera = "";
  bool   addsky = false;
};

// Json IO
void serialize_value(json_mode mode, json_value& json, view_params& value,
    const string& description) {
  serialize_object(mode, json, value, description);
  serialize_property(mode, json, value.scene, "scene", "Scene filename.", true);
  serialize_property(mode, json, value.output, "output", "Output filename.");
  serialize_property(mode, json, value.camera, "camera", "Camera name.");
  serialize_property(mode, json, value.addsky, "addsky", "Add sky.");
  serialize_clipositionals(mode, json, {"scene"});
  serialize_clialternates(mode, json, {{"output", "o"}});
}

#ifndef YOCTO_OPENGL

// view scene
int run_view(const view_params& params) {
  return print_fatal("Opengl not compiled");
}

#else

// view scene
int run_view(const view_params& params) {
  // load scene
  auto scene   = scene_scene{};
  auto ioerror = ""s;
  if (path_extension(params.scene) == ".ypreset") {
    print_progress("make preset", 0, 1);
    if (!make_preset(scene, path_basename(params.scene), ioerror))
      print_fatal(ioerror);
    print_progress("make preset", 1, 1);
  } else {
    if (!load_scene(params.scene, scene, ioerror, print_progress))
      print_fatal(ioerror);
  }

  // add sky
  if (params.addsky) add_sky(scene);

  // tesselation
  tesselate_shapes(scene, print_progress);

  // run view
  view_scene("yscene", params.scene, scene, find_camera(scene, params.camera),
      print_progress);

  // done
  return 0;
}

#endif

struct app_params {
  string         command = "convert";
  convert_params convert = {};
  view_params    view    = {};
};

// Json IO
void serialize_value(json_mode mode, json_value& json, app_params& value,
    const string& description) {
  serialize_object(mode, json, value, description);
  serialize_command(mode, json, value.command, "command", "Command.");
  serialize_property(mode, json, value.convert, "convert", "Convert shapes.");
  serialize_property(mode, json, value.view, "view", "View shapes.");
}

int main(int argc, const char* argv[]) {
  // command line parameters
  auto params = app_params{};
  parse_cli(params, "Process and view scenes", argc, argv);

  // dispatch commands
  if (params.command == "convert") {
    return run_convert(params.convert);
  } else if (params.command == "view") {
    return run_view(params.view);
  } else {
    return print_fatal("unknown command " + params.command);
  }
}
