//
// Utilities to display a scene graph using ImGui.
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

#ifndef _YOCTO_SCENEUI_H_
#define _YOCTO_SCENEUI_H_

#include "../yocto/yocto_scene.h"
#include "yocto_opengl.h"
using namespace yocto;

inline const unordered_map<int, string>& animation_type_names() {
    static auto names = unordered_map<int, string>{
        {(int)yocto_interpolation_type::linear, "linear"},
        {(int)yocto_interpolation_type::step, "step"},
        {(int)yocto_interpolation_type::bezier, "bezier"},
    };
    return names;
}

template <typename T>
inline void draw_opengl_widgets_scene_tree(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, int index, const vector<T>& vals,
    pair<string, int>& sel, const string& sel_type);

template <typename T>
inline void draw_opengl_widgets_scene_tree(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, int index, const vector<T*>& vals,
    pair<string, int>& sel, const string& sel_type);

inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_camera& value,
    pair<string, int>& sel) {}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_texture& value,
    pair<string, int>& sel) {}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_voltexture& value,
    pair<string, int>& sel) {}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_material& value,
    pair<string, int>& sel) {
    draw_opengl_widgets_scene_tree(win, "emission", scene,
        value.emission_texture, scene.textures, sel, "texture");
    draw_opengl_widgets_scene_tree(win, "diffuse", scene, value.diffuse_texture,
        scene.textures, sel, "texture");
    draw_opengl_widgets_scene_tree(win, "specular", scene,
        value.specular_texture, scene.textures, sel, "texture");
    draw_opengl_widgets_scene_tree(
        win, "bump", scene, value.bump_texture, scene.textures, sel, "texture");
    draw_opengl_widgets_scene_tree(win, "displament", scene,
        value.displacement_texture, scene.textures, sel, "texture");
    draw_opengl_widgets_scene_tree(win, "normal", scene, value.normal_texture,
        scene.textures, sel, "texture");
}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_shape& value,
    pair<string, int>& sel) {
    draw_opengl_widgets_scene_tree(win, "material", scene, value.material,
        scene.materials, sel, "material");
}

inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_instance& value,
    pair<string, int>& sel) {
    draw_opengl_widgets_scene_tree(
        win, "shape", scene, value.shape, scene.shapes, sel, "shape");
}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_environment& value,
    pair<string, int>& sel) {
    draw_opengl_widgets_scene_tree(win, "emission", scene,
        value.emission_texture, scene.textures, sel, "texture");
}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_scene_node& value,
    pair<string, int>& sel) {
    draw_opengl_widgets_scene_tree(win, "instance", scene, value.instance,
        scene.instances, sel, "instance");
    draw_opengl_widgets_scene_tree(
        win, "camera", scene, value.camera, scene.cameras, sel, "camera");
    draw_opengl_widgets_scene_tree(win, "environment", scene, value.environment,
        scene.environments, sel, "environment");
    draw_opengl_widgets_scene_tree(
        win, "parent", scene, value.parent, scene.nodes, sel, "node");
    auto cid = 0;
    for (auto ch : value.children) {
        draw_opengl_widgets_scene_tree(win, "child" + to_string(cid++), scene,
            ch, scene.nodes, sel, "node");
    }
}
inline void draw_scene_tree_opengl_widgets_rec(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, const yocto_animation& value,
    pair<string, int>& sel) {
    auto tid = 0;
    for (auto tg : value.node_targets) {
        draw_opengl_widgets_scene_tree(win, "target" + to_string(tid++), scene,
            tg, scene.nodes, sel, "node");
    }
}

template <typename T>
inline void draw_opengl_widgets_scene_tree(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, int index, const vector<T>& vals,
    pair<string, int>& sel, const string& sel_type) {
    if (index < 0) return;
    auto lbl = vals[index].name;
    if (!lbl_.empty()) lbl = lbl_ + ": " + vals[index].name;
    auto selected = sel == pair<string, int>{sel_type, index};
    if (begin_selectabletreenode_opengl_widget(win, lbl.c_str(), selected)) {
        draw_scene_tree_opengl_widgets_rec(win, lbl_, scene, vals[index], sel);
        end_treenode_opengl_widget(win);
    }
    if (selected) sel = {sel_type, index};
}

template <typename T>
inline void draw_opengl_widgets_scene_tree(const opengl_window& win,
    const string& lbl_, yocto_scene& scene, int index, const vector<T*>& vals,
    pair<string, int>& sel, const string& sel_type) {
    if (index < 0) return;
    auto lbl = vals[index]->name;
    if (!lbl_.empty()) lbl = lbl_ + ": " + vals[index]->name;
    auto selected = sel == pair<string, int>{sel_type, index};
    if (begin_selectabletreenode_opengl_widget(win, lbl.c_str(), selected)) {
        draw_scene_tree_opengl_widgets_rec(win, lbl_, scene, vals[index], sel);
        end_treenode_opengl_widget(win);
    }
    if (selected) sel = {sel_type, index};
}

inline void draw_opengl_widgets_scene_tree(
    const opengl_window& win, yocto_scene& scene, pair<string, int>& sel) {
    if (!scene.cameras.empty() && begin_treenode_opengl_widget(win, "cameras")) {
        for (auto v = 0; v < scene.cameras.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.cameras, sel, "camera");
        end_treenode_opengl_widget(win);
    }
    if (!scene.shapes.empty() && begin_treenode_opengl_widget(win, "shapes")) {
        for (auto v = 0; v < scene.shapes.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.shapes, sel, "shape");
        end_treenode_opengl_widget(win);
    }
    if (!scene.instances.empty() &&
        begin_treenode_opengl_widget(win, "instances")) {
        for (auto v = 0; v < scene.instances.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.instances, sel, "instance");
        end_treenode_opengl_widget(win);
    }
    if (!scene.materials.empty() &&
        begin_treenode_opengl_widget(win, "materials")) {
        for (auto v = 0; v < scene.materials.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.materials, sel, "material");
        end_treenode_opengl_widget(win);
    }
    if (!scene.textures.empty() && begin_treenode_opengl_widget(win, "textures")) {
        for (auto v = 0; v < scene.textures.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.textures, sel, "texture");
        end_treenode_opengl_widget(win);
    }
    if (!scene.environments.empty() &&
        begin_treenode_opengl_widget(win, "environments")) {
        for (auto v = 0; v < scene.environments.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.environments, sel, "environment");
        end_treenode_opengl_widget(win);
    }
    if (!scene.nodes.empty() && begin_treenode_opengl_widget(win, "nodes")) {
        for (auto v = 0; v < scene.nodes.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.nodes, sel, "node");
        end_treenode_opengl_widget(win);
    }
    if (!scene.animations.empty() &&
        begin_treenode_opengl_widget(win, "animations")) {
        for (auto v = 0; v < scene.animations.size(); v++)
            draw_opengl_widgets_scene_tree(
                win, "", scene, v, scene.animations, sel, "animation");
        end_treenode_opengl_widget(win);
    }
}

/// Visit struct elements.
inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_camera& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_slider_opengl_widget(win, "frame.x", value.frame.x.x, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.y", value.frame.y.x, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.z", value.frame.z.x, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.o", value.frame.o.x, -10, 10);
    edited += draw_checkbox_opengl_widget(win, "ortho", value.orthographic);
    edited += draw_slider_opengl_widget(win, "film", value.film_size, 0.01f, 1);
    edited += draw_slider_opengl_widget(
        win, "focal", value.focal_length, 0.01f, 1);
    edited += draw_slider_opengl_widget(
        win, "focus", value.focus_distance, 0.01f, 1000);
    edited += draw_slider_opengl_widget(
        win, "aperture", value.lens_aperture, 0, 5);
    return edited;
}

/// Visit struct elements.
inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_texture& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_textinput_opengl_widget(win, "path", value.filename);
    edited += draw_checkbox_opengl_widget(
        win, "clamp_to_edge", value.clamp_to_edge);
    edited += draw_slider_opengl_widget(win, "scale", value.height_scale, 0, 1);
    edited += draw_checkbox_opengl_widget(
        win, "ldr_as_linear", value.ldr_as_linear);
    draw_label_opengl_widget(win, "hdr_image", "%d x %d",
        value.hdr_image.size().x, value.hdr_image.size().y);
    draw_label_opengl_widget(win, "ldr_image", "%d x %d",
        value.ldr_image.size().x, value.ldr_image.size().y);
    return edited;
}

inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_material& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_coloredit_opengl_widget(
        win, "emission", value.emission);  // TODO:
                                           // HDR
    edited += draw_coloredit_opengl_widget(win, "diffuse", value.diffuse);
    edited += draw_coloredit_opengl_widget(win, "specular", value.specular);
    edited += draw_coloredit_opengl_widget(
        win, "transmission", value.transmission);
    edited += draw_slider_opengl_widget(win, "roughness", value.roughness, 0, 1);
    edited += draw_slider_opengl_widget(win, "opacity", value.opacity, 0, 1);
    edited += draw_checkbox_opengl_widget(
        win, "double sided", value.double_sided);
    continue_opengl_widget_line(win);
    edited += draw_checkbox_opengl_widget(win, "fresnel", value.fresnel);
    continue_opengl_widget_line(win);
    edited += draw_checkbox_opengl_widget(win, "refract", value.refract);
    edited += draw_coloredit_opengl_widget(
        win, "volume_density", value.volume_density);  // 0, 10
    edited += draw_coloredit_opengl_widget(
        win, "volume_albedo", value.volume_albedo);  // 0, 1
    edited += draw_slider_opengl_widget(
        win, "volume_phaseg", value.volume_phaseg, -1, 1);
    edited += draw_combobox_opengl_widget(
        win, "emission_texture", value.emission_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(
        win, "diffuse_texture", value.diffuse_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(
        win, "specular_texture", value.specular_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(win, "transmission_texture",
        value.transmission_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(
        win, "opacity_texture", value.opacity_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(win, "roughness_texture",
        value.roughness_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(
        win, "bump_texture", value.bump_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(win, "displacement_texture",
        value.displacement_texture, scene.textures, true);
    edited += draw_combobox_opengl_widget(
        win, "normal_texture", value.normal_texture, scene.textures, true);
    edited += draw_checkbox_opengl_widget(
        win, "base metallic", value.base_metallic);
    edited += draw_checkbox_opengl_widget(
        win, "glTF textures", value.gltf_textures);
    return edited;
}

inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_shape& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_textinput_opengl_widget(win, "path", value.filename);
    edited += draw_combobox_opengl_widget(
        win, "material", value.material, scene.materials, true);
    draw_label_opengl_widget(win, "lines", "%ld", value.lines.size());
    draw_label_opengl_widget(win, "triangles", "%ld", value.triangles.size());
    draw_label_opengl_widget(win, "quads", "%ld", value.quads.size());
    draw_label_opengl_widget(win, "pos", "%ld", value.positions.size());
    draw_label_opengl_widget(win, "norm", "%ld", value.normals.size());
    draw_label_opengl_widget(win, "texcoord", "%ld", value.texturecoords.size());
    draw_label_opengl_widget(win, "color", "%ld", value.colors.size());
    draw_label_opengl_widget(win, "radius", "%ld", value.radius.size());
    draw_label_opengl_widget(win, "tangsp", "%ld", value.tangentspaces.size());
    return edited;
}

inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_instance& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_slider_opengl_widget(win, "frame.x", value.frame.x, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.y", value.frame.y, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.z", value.frame.z, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.o", value.frame.o, -10, 10);
    edited += draw_combobox_opengl_widget(
        win, "shape", value.shape, scene.shapes, true);
    return edited;
}

inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_environment& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_slider_opengl_widget(win, "frame.x", value.frame.x, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.y", value.frame.y, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.z", value.frame.z, -1, 1);
    edited += draw_slider_opengl_widget(win, "frame.o", value.frame.o, -10, 10);
    edited += draw_coloredit_opengl_widget(win, "ke", value.emission);  // TODO:
                                                                        // HDR
    edited += draw_combobox_opengl_widget(
        win, "ke texture", value.emission_texture, scene.textures, true);
    return edited;
}

inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_scene_node& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_combobox_opengl_widget(
        win, "parent", value.parent, scene.nodes, true);
    edited += draw_slider_opengl_widget(win, "local.x", value.local.x, -1, 1);
    edited += draw_slider_opengl_widget(win, "local.y", value.local.y, -1, 1);
    edited += draw_slider_opengl_widget(win, "local.z", value.local.z, -1, 1);
    edited += draw_slider_opengl_widget(win, "local.o", value.local.o, -10, 10);
    edited += draw_slider_opengl_widget(
        win, "translation", value.translation, -10, 10);
    edited += draw_slider_opengl_widget(win, "rotation", value.rotation, -1, 1);
    edited += draw_slider_opengl_widget(win, "scale", value.scale, 0, 10);
    edited += draw_combobox_opengl_widget(
        win, "camera", value.camera, scene.cameras, true);
    edited += draw_combobox_opengl_widget(
        win, "instance", value.instance, scene.instances, true);
    edited += draw_combobox_opengl_widget(
        win, "environment", value.environment, scene.environments, true);
    return edited;
}

inline bool draw_opengl_widgets_scene_inspector(
    const opengl_window& win, yocto_animation& value, yocto_scene& scene) {
    auto edited = 0;
    edited += draw_textinput_opengl_widget(win, "name", value.name);
    edited += draw_textinput_opengl_widget(win, "path", value.filename);
    edited += draw_textinput_opengl_widget(win, "group", value.animation_group);
    // edited += draw_combobox_opengl_widget(win, "type", &value.type,
    // animation_type_names());
    draw_label_opengl_widget(win, "times", "%ld", value.keyframes_times.size());
    draw_label_opengl_widget(
        win, "translation", "%ld", value.translation_keyframes.size());
    draw_label_opengl_widget(
        win, "rotation", "%ld", value.rotation_keyframes.size());
    draw_label_opengl_widget(win, "scale", "%ld", value.scale_keyframes.size());
    draw_label_opengl_widget(
        win, "weights", "%ld", value.morph_weights_keyframes.size());
    draw_label_opengl_widget(win, "targets", "%ld", value.node_targets.size());
    return edited;
}

inline bool draw_opengl_widgets_scene_tree(const opengl_window& win,
    const string& lbl, yocto_scene& scene, pair<string, int>& sel,
    vector<pair<string, int>>& update_list, int height) {
    draw_opengl_widgets_scene_tree(win, scene, sel);
    auto update_len = update_list.size();
#if 0
    if (test_scn) {
        draw_add_elem_opengl_widgets(
            scene, "camera", scene.cameras, test_scn->cameras, sel, update_list);
        draw_add_elem_opengl_widgets(scene, "texture", scene.textures,
            test_scn->textures, sel, update_list);
        draw_add_elem_opengl_widgets(scene, "mat", scene.materials,
            test_scn->materials, sel, update_list);
        draw_add_elem_opengl_widgets(
            scene, "shape", scene.shapes, test_scn->shapes, sel, update_list);
        draw_add_elem_opengl_widgets(scene, "instance", scene.instances,
            test_scn->instances, sel, update_list);
        draw_add_elem_opengl_widgets(
            scene, "node", scene.nodes, test_scn->nodes, sel, update_list);
        draw_add_elem_opengl_widgets(scene, "environment", scene.environments,
            test_scn->environments, sel, update_list);
        draw_add_elem_opengl_widgets(scene, "anim", scene.animations,
            test_scn->animations, sel, update_list);
    }
#endif
    return update_list.size() != update_len;
}

inline bool draw_opengl_widgets_scene_inspector(const opengl_window& win,
    const string& lbl, yocto_scene& scene, pair<string, int>& sel,
    vector<pair<string, int>>& update_list, int height) {
    if (sel.first == "") return false;
    begin_child_opengl_widget(win, "scrolling scene inspector", {0, height});

    auto update_len = update_list.size();

    if (sel.first == "camera")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.cameras[sel.second], scene))
            update_list.push_back({"camera", sel.second});
    if (sel.first == "shape")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.shapes[sel.second], scene))
            update_list.push_back({"shape", sel.second});
    if (sel.first == "texture")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.textures[sel.second], scene))
            update_list.push_back({"texture", sel.second});
    if (sel.first == "material")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.materials[sel.second], scene))
            update_list.push_back({"material", sel.second});
    if (sel.first == "environment")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.environments[sel.second], scene))
            update_list.push_back({"environment", sel.second});
    if (sel.first == "instance")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.instances[sel.second], scene))
            update_list.push_back({"instance", sel.second});
    if (sel.first == "node")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.nodes[sel.second], scene))
            update_list.push_back({"node", sel.second});
    if (sel.first == "animation")
        if (draw_opengl_widgets_scene_inspector(
                win, scene.animations[sel.second], scene))
            update_list.push_back({"animation", sel.second});

    end_child_opengl_widget(win);
    return update_list.size() != update_len;
}

#endif
