//
// Implementation for Yocto/Obj. See yocto_gl.h for documentation.
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

#include "yocto_obj.h"

#include "yocto_image.h"

#include <algorithm>
#include <array>

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR WAVEFRONT OBJ
// -----------------------------------------------------------------------------
namespace ygl {

// skip whitespace
static inline void obj_skipws(char*& s) {
    while (*s == ' ') s++;
}

// skip a string if matched
static inline bool obj_streq(const char* s, const char* str) {
    while (*s == *str && *s && *str) {
        s++;
        str++;
    }
    return *s == *str;
}

#if YGL_FASTOBJ

// parse base value
static inline void obj_parse_base(char*& s, int& val) {
    while (*s == ' ') s++;
    val = 0;
    auto sn = (*s == '-') ? -1 : 1;
    if (*s == '-' || *s == '+') s++;
    while (*s >= '0' && *s <= '9') val = val * 10 + (*s++ - '0');
    val *= sn;
}

// parse base value
static inline void obj_parse_base(char*& s, float& val) {
    while (*s == ' ') s++;
    //    auto ss = s; auto sss = ss;
    auto mantissa = 0, fractional = 0, fractional_length = 0, exponent = 0;
    auto sn = (*s == '-') ? -1 : 1;
    if (*s == '-' || *s == '+') s++;
    while (*s >= '0' && *s <= '9') mantissa = mantissa * 10 + (*s++ - '0');
    if (*s == '.') {
        s++;
        while (*s >= '0' && *s <= '9') {
            fractional = fractional * 10 + (*s++ - '0');
            fractional_length++;
        }
    }
    mantissa *= sn;
    fractional *= sn;
    if (*s == 'e' || *s == 'E') {
        s++;
        auto en = (*s == '-') ? -1 : 1;
        if (*s == '-' || *s == '+') s++;
        while (*s >= '0' && *s <= '9') exponent = exponent * 10 + (*s++ - '0');
        exponent *= en;
    }
    auto dval = (double)mantissa;
    if (fractional)
        dval += fractional * std::pow(10.0, -(double)fractional_length);
    if (exponent) dval *= std::pow(10.0, (double)exponent);
    val = (float)dval;
#if 0
    auto cval = val;
    sscanf(ss, "%f", &cval);
    if(abs(val - cval) > 0.01f) {
        printf("- %g %g %s\n", val, cval, sss);
    }
    auto len = 0;
    sscanf(s, "%f%n", &val, &len);
    s += len;
#endif
}

// parse base value
static inline void obj_parse_base(char*& s, char* val) {
    while (*s == ' ') s++;
    while (*s && *s != ' ') *val++ = *s++;
    *val = 0;
}
// parse base value
static inline void obj_parse_base(char*& s, std::string& val) {
    char buf[4096];
    obj_parse_base(s, buf);
    val = buf;
}

#else

// parse base value
inline void obj_parse_base(char*& s, int& val) {
    auto len = 0;
    sscanf(s, "%d%n", &val, &len);
    s += len;
}

// parse base value
inline void obj_parse_base(char*& s, float& val) {
    auto len = 0;
    sscanf(s, "%f%n", &val, &len);
    s += len;
}

// parse base value
inline void obj_parse_base(char*& s, std::string& val) {
    char buf[4096];
    auto len = 0;
    sscanf(s, "%s%n", buf, &len);
    if (len) {
        s += len;
        val = buf;
    } else {
        val = "";
    }
}

// parse base value
inline void obj_parse_base(char*& s, char* val) {
    auto len = 0;
    sscanf(s, "%s%n", val, &len);
    s += len;
}

#endif

// parse value
static inline void obj_parse(char*& s, int& val) { obj_parse_base(s, val); }
static inline void obj_parse(char*& s, float& val) { obj_parse_base(s, val); }
static inline void obj_parse(char*& s, bool& val) {
    auto i = 0;
    obj_parse(s, i);
    val = i;
}
static inline void obj_parse(char*& s, std::string& val) {
    obj_parse_base(s, val);
}
static inline void obj_parse(char*& s, char* val) { obj_parse_base(s, val); }
static inline void obj_parse(char*& s, vec2i& val) {
    for (auto i = 0; i < 2; i++) obj_parse(s, (&val.x)[i]);
}
static inline void obj_parse(char*& s, vec2f& val) {
    for (auto i = 0; i < 2; i++) obj_parse(s, (&val.x)[i]);
}
static inline void obj_parse(char*& s, vec3f& val) {
    for (auto i = 0; i < 3; i++) obj_parse(s, (&val.x)[i]);
}
static inline void obj_parse(char*& s, frame3f& val) {
    for (auto i = 0; i < 12; i++) obj_parse(s, (&val.x.x)[i]);
}
static inline void obj_parse(char*& s, vec3i& val, vec3i vert_size) {
    char buf[1024];
    obj_parse_base(s, buf);
    val = {-1, -1, -1};
    auto i = 0;
    auto sb = buf;
    while (i < 3 && *sb) {
        obj_parse_base(sb, (&val.x)[i]);
        (&val.x)[i] = ((&val.x)[i] < 0) ? (&vert_size.x)[i] + (&val.x)[i] :
                                          (&val.x)[i] - 1;
        if (*sb != '/') break;
        while (*sb == '/') {
            sb++;
            i++;
        }
    }
}
static inline void obj_parse(
    char*& s, int& num, vec3i* vert_buf, vec3i vert_size) {
    num = 0;
    while (*s) {
        obj_parse(s, vert_buf[num], vert_size);
        num += 1;
    }
}

// clear the whitespace
static inline void obj_convertws(char* s) {
    while (*s) {
        if (*s == '\t' || *s == '\r' || *s == '\n') *s = ' ';
        s++;
    }
}

// Parse texture options and name
inline void obj_parse(char*& s, obj_texture_info& info) {
    // initialize
    info = obj_texture_info();

    // get tokens
    auto tokens = std::vector<std::string>();
    obj_skipws(s);
    while (*s) {
        tokens.push_back("");
        obj_parse(s, tokens.back());
        obj_skipws(s);
    }

    // exit if no tokens
    if (tokens.empty()) return;

    // texture name
    info.path = tokens.back();
    for (auto& c : info.path)
        if (c == '\\') c = '/';

    // texture options
    auto last = std::string();
    for (auto& tok : tokens) {
        if (tok == tokens.back()) break;
        if (tok[0] == '-') {
            last = tok;
            info.props[last] = {};
        } else {
            info.props[last].push_back(atof(tok.c_str()));
        }
    }

    // clamp
    if (info.props.find("clamp") != info.props.end()) {
        info.clamp = info.props.at("clamp").front();
        info.props.erase("clamp");
    }

    if (info.props.find("bm") != info.props.end()) {
        info.scale = info.props.at("bm").front();
        info.props.erase("bm");
    }
}

// Load MTL
std::vector<std::shared_ptr<obj_material>> load_mtl(
    const std::string& filename, bool flip_tr) {
    // clear materials
    auto materials = std::vector<std::shared_ptr<obj_material>>();

    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) throw std::runtime_error("cannot open filename " + filename);

    // add a material preemptively to avoid crashes
    materials.push_back(std::make_shared<obj_material>());
    auto mat = materials.back();

    // read the file line by line
    char line[4096];
    char cmd[1024];
    auto linenum = 0;
    while (fgets(line, sizeof(line), fs)) {
        // prepare to parse
        linenum += 1;
        auto ss = line;
        obj_convertws(ss);
        obj_skipws(ss);

        // skip empty and comments
        if (!ss[0] || ss[0] == '#') continue;

        // get command
        obj_parse(ss, cmd);

        // possible token values
        if (obj_streq(cmd, "newmtl")) {
            materials.push_back(std::make_shared<obj_material>());
            mat = materials.back();
            obj_parse(ss, mat->name);
        } else if (obj_streq(cmd, "illum")) {
            obj_parse(ss, mat->illum);
        } else if (obj_streq(cmd, "Ke")) {
            obj_parse(ss, mat->ke);
        } else if (obj_streq(cmd, "Ka")) {
            obj_parse(ss, mat->ka);
        } else if (obj_streq(cmd, "Kd")) {
            obj_parse(ss, mat->kd);
        } else if (obj_streq(cmd, "Ks")) {
            obj_parse(ss, mat->ks);
        } else if (obj_streq(cmd, "Kr")) {
            obj_parse(ss, mat->kr);
        } else if (obj_streq(cmd, "Kt")) {
            obj_parse(ss, mat->kt);
        } else if (obj_streq(cmd, "Tf")) {
            auto nchan = 0;
            obj_skipws(ss);
            while (*ss && nchan < 3) {
                obj_parse(ss, (&mat->kt.x)[nchan++]);
                obj_skipws(ss);
            }
            if (nchan < 3) mat->kt = {mat->kt.x, mat->kt.x, mat->kt.x};
            if (flip_tr)
                materials.back()->kt = vec3f{1, 1, 1} - materials.back()->kt;
        } else if (obj_streq(cmd, "Tr")) {
            auto nchan = 0;
            auto tr = zero3f;
            obj_skipws(ss);
            while (*ss && nchan < 3) {
                obj_parse(ss, (&tr.x)[nchan++]);
                obj_skipws(ss);
            }
            if (nchan < 3) tr = {tr.x, tr.x, tr.x};
            materials.back()->op = (tr.x + tr.y + tr.z) / 3;
            if (flip_tr) materials.back()->op = 1 - materials.back()->op;
        } else if (obj_streq(cmd, "Ns")) {
            obj_parse(ss, mat->ns);
        } else if (obj_streq(cmd, "d")) {
            obj_parse(ss, mat->op);
        } else if (obj_streq(cmd, "Ni")) {
            obj_parse(ss, mat->ior);
        } else if (obj_streq(cmd, "Pr") || obj_streq(cmd, "rs")) {
            obj_parse(ss, mat->rs);
        } else if (obj_streq(cmd, "Pm") || obj_streq(cmd, "Km")) {
            obj_parse(ss, mat->km);
        } else if (obj_streq(cmd, "map_Ke")) {
            obj_parse(ss, mat->ke_txt);
        } else if (obj_streq(cmd, "map_Ka")) {
            obj_parse(ss, mat->ka_txt);
        } else if (obj_streq(cmd, "map_Kd")) {
            obj_parse(ss, mat->kd_txt);
        } else if (obj_streq(cmd, "map_Ks")) {
            obj_parse(ss, mat->ks_txt);
        } else if (obj_streq(cmd, "map_Kr")) {
            obj_parse(ss, mat->kr_txt);
        } else if (obj_streq(cmd, "map_Tr")) {
            obj_parse(ss, mat->kt_txt);
        } else if (obj_streq(cmd, "map_Ns")) {
            obj_parse(ss, mat->ns_txt);
        } else if (obj_streq(cmd, "map_d") || obj_streq(cmd, "map_Tr")) {
            obj_parse(ss, mat->op_txt);
        } else if (obj_streq(cmd, "map_Ni")) {
            obj_parse(ss, mat->ior_txt);
        } else if (obj_streq(cmd, "map_Pr") || obj_streq(cmd, "map_rs")) {
            obj_parse(ss, mat->rs_txt);
        } else if (obj_streq(cmd, "map_Pm") || obj_streq(cmd, "map_Km")) {
            obj_parse(ss, mat->km_txt);
        } else if (obj_streq(cmd, "map_occ") || obj_streq(cmd, "occ")) {
            obj_parse(ss, mat->occ_txt);
        } else if (obj_streq(cmd, "map_bump") || obj_streq(cmd, "bump")) {
            obj_parse(ss, mat->bump_txt);
        } else if (obj_streq(cmd, "map_disp") || obj_streq(cmd, "disp")) {
            obj_parse(ss, mat->disp_txt);
        } else if (obj_streq(cmd, "map_norm") || obj_streq(cmd, "norm")) {
            obj_parse(ss, mat->norm_txt);
        } else {
            // copy into strings
            obj_skipws(ss);
            while (*ss) {
                mat->props[cmd].push_back("");
                obj_parse(ss, mat->props[cmd].back());
                obj_skipws(ss);
            }
        }
    }

    // remove first fake material
    materials.erase(materials.begin());

    // clone
    fclose(fs);

    // done
    return materials;
}

// Loads an OBJ
std::shared_ptr<obj_scene> load_obj(const std::string& filename,
    bool split_shapes, bool flip_texcoord, bool flip_tr) {
    // clear obj
    auto obj = std::make_shared<obj_scene>();

    // splitting policy
    auto split_material = split_shapes;
    auto split_group = split_shapes;
    auto split_smoothing = split_shapes;

    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) throw std::runtime_error("cannot open filename " + filename);

    // add object if needed
    auto add_object = [&](std::shared_ptr<obj_scene> obj, std::string name,
                          std::string matname, std::string groupname,
                          bool smoothing) {
        if (obj->objects.empty() || !obj->objects.back()->elems.empty())
            obj->objects.push_back(std::make_shared<obj_object>());
        auto oobj = obj->objects.back();
        oobj->name = name;
        if (oobj->materials.empty()) oobj->materials.push_back("");
        if (oobj->groups.empty()) oobj->groups.push_back("");
        if (oobj->smoothing.empty()) oobj->smoothing.push_back(true);
        oobj->materials.back() = matname;
        oobj->groups.back() = groupname;
        oobj->smoothing.back() = smoothing;
        return oobj;
    };

    // current parsing values
    auto matname = std::string();
    auto oname = std::string();
    auto gname = std::string();
    auto smoothing = true;
    auto oobj = add_object(obj, oname, matname, gname, smoothing);

    // properties
    auto oprops = std::unordered_map<std::string,
        std::unordered_map<std::string, std::vector<float>>>();

    // keep track of array lengths
    auto vert_size = zero3i;

    // elem type map
    static auto elem_type_map =
        std::unordered_map<std::string, obj_element_type>{
            {"f", obj_element_type::face}, {"l", obj_element_type::line},
            {"p", obj_element_type::point}, {"b", obj_element_type::bezier}};

    // read the file line by line
    char line[4096];
    char cmd[1024];
    auto linenum = 0;
    while (fgets(line, sizeof(line), fs)) {
        // prepare to parse
        linenum += 1;
        auto ss = line;
        obj_convertws(ss);
        obj_skipws(ss);

        // skip empty and comments
        if (!ss[0] || ss[0] == '#') continue;

        // get command
        obj_parse(ss, cmd);

        // possible token values
        if (obj_streq(cmd, "v")) {
            vert_size.x += 1;
            obj->pos.push_back(zero3f);
            obj_parse(ss, obj->pos.back());
        } else if (obj_streq(cmd, "vn")) {
            vert_size.z += 1;
            obj->norm.push_back(zero3f);
            obj_parse(ss, obj->norm.back());
        } else if (obj_streq(cmd, "vt")) {
            vert_size.y += 1;
            obj->texcoord.push_back(zero2f);
            obj_parse(ss, obj->texcoord.back());
            if (flip_texcoord)
                obj->texcoord.back().y = 1 - obj->texcoord.back().y;
        } else if (obj_streq(cmd, "f") || obj_streq(cmd, "l") ||
                   obj_streq(cmd, "p") || obj_streq(cmd, "b")) {
            auto elem = obj_element();
            elem.type = elem_type_map.at(cmd);
            elem.start = (uint32_t)oobj->verts_pos.size();
            elem.size = 0;
            elem.material = (int)oobj->materials.size() - 1;
            elem.group = (int)oobj->groups.size() - 1;
            elem.smoothing = (int)oobj->smoothing.size() - 1;
            oobj->elems.push_back(elem);
            obj_skipws(ss);
            while (*ss) {
                auto vert = vec3i{-1, -1, -1};
                obj_parse(ss, vert, vert_size);
                obj_skipws(ss);
                oobj->verts_pos.push_back(vert.x);
                oobj->verts_norm.push_back(vert.z);
                oobj->verts_texcoord.push_back(vert.y);
                oobj->elems.back().size += 1;
            }
        } else if (obj_streq(cmd, "o")) {
            obj_parse(ss, oname);
            gname = "";
            matname = "";
            smoothing = true;
            oobj = add_object(obj, oname, matname, gname, smoothing);
        } else if (obj_streq(cmd, "usemtl")) {
            obj_parse(ss, matname);
            if (split_material) {
                oobj = add_object(obj, oname, matname, gname, smoothing);
            } else {
                if (oobj->elems.empty()) {
                    oobj->materials.back() = matname;
                } else {
                    oobj->materials.push_back(matname);
                }
            }
        } else if (obj_streq(cmd, "g")) {
            obj_parse(ss, gname);
            if (split_group) {
                oobj = add_object(obj, oname, matname, gname, smoothing);
            } else {
                if (oobj->elems.empty()) {
                    oobj->groups.back() = gname;
                } else {
                    oobj->groups.push_back(gname);
                }
            }
        } else if (obj_streq(cmd, "s")) {
            auto name = std::string();
            obj_parse(ss, name);
            smoothing = (name == "on");
            if (split_smoothing) {
                oobj = add_object(obj, oname, matname, gname, smoothing);
            } else {
                if (oobj->elems.empty()) {
                    oobj->smoothing.back() = smoothing;
                } else {
                    oobj->smoothing.push_back(smoothing);
                }
            }
        } else if (obj_streq(cmd, "of")) {
            obj_parse(ss, oobj->frame);
        } else if (obj_streq(cmd, "os")) {
            obj_parse(ss, oobj->subdiv);
        } else if (obj_streq(cmd, "op")) {
            auto rname = std::string(), pname = std::string();
            obj_parse(ss, rname);
            obj_skipws(ss);
            obj_parse(ss, pname);
            obj_skipws(ss);
            auto& pvalues = oprops[rname][pname];
            while (*ss) {
                auto tok = std::string();
                obj_parse(ss, tok);
                obj_skipws(ss);
                pvalues.push_back((float)std::atof(tok.c_str()));
            }
        } else if (obj_streq(cmd, "mtllib")) {
            auto mtlname = std::string();
            obj_parse(ss, mtlname);
            auto mtlpath = get_dirname(filename) + "/" + mtlname;
            auto mats = load_mtl(mtlpath, flip_tr);
            obj->materials.insert(
                obj->materials.end(), mats.begin(), mats.end());
        } else if (obj_streq(cmd, "c")) {
            auto cam = std::make_shared<obj_camera>();
            obj_parse(ss, cam->name);
            obj_parse(ss, cam->ortho);
            obj_parse(ss, cam->width);
            obj_parse(ss, cam->height);
            obj_parse(ss, cam->focal);
            obj_parse(ss, cam->focus);
            obj_parse(ss, cam->aperture);
            obj_parse(ss, cam->frame);
            obj->cameras.push_back(cam);
        } else if (obj_streq(cmd, "e")) {
            auto env = std::make_shared<obj_environment>();
            obj_parse(ss, env->name);
            obj_parse(ss, env->ke);
            obj_parse(ss, env->ke_txt.path);
            if (env->ke_txt.path == "\"\"") env->ke_txt.path = "";
            obj_parse(ss, env->frame);
            obj->environments.push_back(env);
        } else {
            // unused
        }
    }

    // cleanup empty
    auto clear_vert_if_unused = [](std::vector<int>& vert) {
        if (vert.empty()) return;
        auto used = false;
        for (auto v : vert)
            if (v >= 0) used = true;
        if (!used) vert.clear();
    };
    for (auto idx = 0; idx < obj->objects.size(); idx++) {
        auto oobj = obj->objects[idx];
        clear_vert_if_unused(oobj->verts_pos);
        clear_vert_if_unused(oobj->verts_norm);
        clear_vert_if_unused(oobj->verts_texcoord);
        if (!oobj->elems.empty() || !oobj->verts_pos.empty()) continue;
        obj->objects.erase(obj->objects.begin() + idx);
        idx--;
    }
    auto end = std::remove_if(obj->objects.begin(), obj->objects.end(),
        [](const std::shared_ptr<obj_object>& x) { return !x; });
    obj->objects.erase(end, obj->objects.end());

    // apply properties
    for (auto oobj : obj->objects) oobj->props = oprops[oobj->name];

    // create textures
    auto txt_set =
        std::unordered_map<std::string, std::shared_ptr<obj_texture>>();
    auto add_texture = [](std::shared_ptr<obj_scene> obj, auto& txt_set,
                           const obj_texture_info& info) {
        if (info.path == "") return;
        if (txt_set.find(info.path) != txt_set.end()) return;
        auto txt = std::make_shared<obj_texture>();
        txt->path = info.path;
        txt_set[txt->path] = txt;
        obj->textures.push_back(txt);
    };
    for (auto mat : obj->materials) {
        add_texture(obj, txt_set, mat->ke_txt);
        add_texture(obj, txt_set, mat->kd_txt);
        add_texture(obj, txt_set, mat->ks_txt);
        add_texture(obj, txt_set, mat->kr_txt);
        add_texture(obj, txt_set, mat->kt_txt);
        add_texture(obj, txt_set, mat->rs_txt);
        add_texture(obj, txt_set, mat->bump_txt);
        add_texture(obj, txt_set, mat->norm_txt);
        add_texture(obj, txt_set, mat->disp_txt);
        add_texture(obj, txt_set, mat->occ_txt);
    }
    for (auto env : obj->environments) {
        add_texture(obj, txt_set, env->ke_txt);
    }

    // close file
    fclose(fs);

    // done
    return obj;
}

// Load MTL
void load_mtl(const std::string& filename, const obj_callbacks& callbacks,
    void* ctx, bool flip_tr) {
    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) throw std::runtime_error("cannot open filename " + filename);

    // add a material preemptively to avoid crashes
    auto mat = obj_material();
    auto parsed_one = false;

    // read the file line by line
    char line[4096];
    char cmd[1024];
    auto linenum = 0;
    auto done = false;
    while (!done && fgets(line, sizeof(line), fs)) {
        // prepare to parse
        linenum += 1;
        auto ss = line;
        obj_convertws(ss);
        obj_skipws(ss);

        // skip empty and comments
        if (!ss[0] || ss[0] == '#') continue;

        // get command
        obj_parse(ss, cmd);

        // possible token values
        if (obj_streq(cmd, "newmtl")) {
            if (parsed_one && callbacks.material)
                done = callbacks.material(ctx, mat);
            parsed_one = true;
            mat = obj_material();
            obj_parse(ss, mat.name);
        } else if (obj_streq(cmd, "illum")) {
            obj_parse(ss, mat.illum);
        } else if (obj_streq(cmd, "Ke")) {
            obj_parse(ss, mat.ke);
        } else if (obj_streq(cmd, "Ka")) {
            obj_parse(ss, mat.ka);
        } else if (obj_streq(cmd, "Kd")) {
            obj_parse(ss, mat.kd);
        } else if (obj_streq(cmd, "Ks")) {
            obj_parse(ss, mat.ks);
        } else if (obj_streq(cmd, "Kr")) {
            obj_parse(ss, mat.kr);
        } else if (obj_streq(cmd, "Kt")) {
            obj_parse(ss, mat.kt);
        } else if (obj_streq(cmd, "Tf")) {
            auto nchan = 0;
            obj_skipws(ss);
            while (*ss && nchan < 3) {
                obj_parse(ss, (&mat.kt.x)[nchan++]);
                obj_skipws(ss);
            }
            if (nchan < 3) mat.kt = {mat.kt.x, mat.kt.x, mat.kt.x};
            if (flip_tr) mat.kt = vec3f{1, 1, 1} - mat.kt;
        } else if (obj_streq(cmd, "Tr")) {
            auto nchan = 0;
            auto tr = zero3f;
            obj_skipws(ss);
            while (*ss && nchan < 3) {
                obj_parse(ss, (&tr.x)[nchan++]);
                obj_skipws(ss);
            }
            if (nchan < 3) tr = {tr.x, tr.x, tr.x};
            mat.op = (tr.x + tr.y + tr.z) / 3;
            if (flip_tr) mat.op = 1 - mat.op;
        } else if (obj_streq(cmd, "Ns")) {
            obj_parse(ss, mat.ns);
        } else if (obj_streq(cmd, "d")) {
            obj_parse(ss, mat.op);
        } else if (obj_streq(cmd, "Ni")) {
            obj_parse(ss, mat.ior);
        } else if (obj_streq(cmd, "Pr") || obj_streq(cmd, "rs")) {
            obj_parse(ss, mat.rs);
        } else if (obj_streq(cmd, "Pm") || obj_streq(cmd, "Km")) {
            obj_parse(ss, mat.km);
        } else if (obj_streq(cmd, "map_Ke")) {
            obj_parse(ss, mat.ke_txt);
        } else if (obj_streq(cmd, "map_Ka")) {
            obj_parse(ss, mat.ka_txt);
        } else if (obj_streq(cmd, "map_Kd")) {
            obj_parse(ss, mat.kd_txt);
        } else if (obj_streq(cmd, "map_Ks")) {
            obj_parse(ss, mat.ks_txt);
        } else if (obj_streq(cmd, "map_Kr")) {
            obj_parse(ss, mat.kr_txt);
        } else if (obj_streq(cmd, "map_Tr")) {
            obj_parse(ss, mat.kt_txt);
        } else if (obj_streq(cmd, "map_Ns")) {
            obj_parse(ss, mat.ns_txt);
        } else if (obj_streq(cmd, "map_d") || obj_streq(cmd, "map_Tr")) {
            obj_parse(ss, mat.op_txt);
        } else if (obj_streq(cmd, "map_Ni")) {
            obj_parse(ss, mat.ior_txt);
        } else if (obj_streq(cmd, "map_Pr") || obj_streq(cmd, "map_rs")) {
            obj_parse(ss, mat.rs_txt);
        } else if (obj_streq(cmd, "map_Pm") || obj_streq(cmd, "map_Km")) {
            obj_parse(ss, mat.km_txt);
        } else if (obj_streq(cmd, "map_occ") || obj_streq(cmd, "occ")) {
            obj_parse(ss, mat.occ_txt);
        } else if (obj_streq(cmd, "map_bump") || obj_streq(cmd, "bump")) {
            obj_parse(ss, mat.bump_txt);
        } else if (obj_streq(cmd, "map_disp") || obj_streq(cmd, "disp")) {
            obj_parse(ss, mat.disp_txt);
        } else if (obj_streq(cmd, "map_norm") || obj_streq(cmd, "norm")) {
            obj_parse(ss, mat.norm_txt);
        } else {
            // copy into strings
            obj_skipws(ss);
            while (*ss) {
                mat.props[cmd].push_back("");
                obj_parse(ss, mat.props[cmd].back());
                obj_skipws(ss);
            }
        }
    }

    // emit last
    if (parsed_one && callbacks.material) done = callbacks.material(ctx, mat);

    // clone
    fclose(fs);
}

// Loads an OBJ
void load_obj(const std::string& filename, const obj_callbacks& callbacks,
    void* ctx, bool flip_texcoord, bool flip_tr) {
    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) throw std::runtime_error("cannot open filename " + filename);

    // keep track of array lengths
    auto vert_size = zero3i;

    // read the file line by line
    char line[4096];
    char cmd[1024];
    auto linenum = 0;
    auto done = false;
    vec3i vert_buf[128];
    while (!done && fgets(line, sizeof(line), fs)) {
        // prepare to parse
        linenum += 1;
        auto ss = line;
        obj_convertws(ss);
        obj_skipws(ss);

        // skip empty and comments
        if (!ss[0] || ss[0] == '#') continue;

        // get command
        obj_parse(ss, cmd);

        // possible token values
        if (obj_streq(cmd, "v")) {
            vert_size.x += 1;
            if (callbacks.vertex) {
                auto val = zero3f;
                obj_parse(ss, val);
                done = callbacks.vertex(ctx, val);
            }
        } else if (obj_streq(cmd, "vn")) {
            vert_size.z += 1;
            if (callbacks.normal) {
                auto val = zero3f;
                obj_parse(ss, val);
                done = callbacks.normal(ctx, val);
            }
        } else if (obj_streq(cmd, "vt")) {
            vert_size.y += 1;
            if (callbacks.texcoord) {
                auto val = zero2f;
                if (flip_texcoord) val.y = 1 - val.y;
                done = callbacks.texcoord(ctx, val);
            }
        } else if (obj_streq(cmd, "f")) {
            if (callbacks.face) {
                obj_skipws(ss);
                auto num = 0;
                obj_parse(ss, num, vert_buf, vert_size);
                callbacks.face(ctx, num, vert_buf);
            }
        } else if (obj_streq(cmd, "l")) {
            if (callbacks.line) {
                obj_skipws(ss);
                auto num = 0;
                obj_parse(ss, num, vert_buf, vert_size);
                callbacks.line(ctx, num, vert_buf);
            }
        } else if (obj_streq(cmd, "p")) {
            if (callbacks.point) {
                obj_skipws(ss);
                auto num = 0;
                obj_parse(ss, num, vert_buf, vert_size);
                callbacks.point(ctx, num, vert_buf);
            }
        } else if (obj_streq(cmd, "o")) {
            if (callbacks.object) {
                auto name = std::string();
                obj_parse(ss, name);
                done = callbacks.object(ctx, name);
            }
        } else if (obj_streq(cmd, "usemtl")) {
            if (callbacks.usemat) {
                auto name = std::string();
                obj_parse(ss, name);
                done = callbacks.usemat(ctx, name);
            }
        } else if (obj_streq(cmd, "g")) {
            if (callbacks.group) {
                auto name = std::string();
                obj_parse(ss, name);
                done = callbacks.group(ctx, name);
            }
        } else if (obj_streq(cmd, "s")) {
            if (callbacks.smoothing) {
                auto name = std::string();
                obj_parse(ss, name);
                done = callbacks.smoothing(ctx, name != "off");
            }
        } else if (obj_streq(cmd, "mtllib")) {
            auto mtlname = std::string();
            obj_parse(ss, mtlname);
            auto mtlpath = get_dirname(filename) + "/" + mtlname;
            load_mtl(mtlpath, callbacks, ctx, flip_tr);
        } else if (obj_streq(cmd, "c")) {
            if (callbacks.camera) {
                auto cam = obj_camera();
                obj_parse(ss, cam.name);
                obj_parse(ss, cam.ortho);
                obj_parse(ss, cam.width);
                obj_parse(ss, cam.height);
                obj_parse(ss, cam.focal);
                obj_parse(ss, cam.focus);
                obj_parse(ss, cam.aperture);
                obj_parse(ss, cam.frame);
                done = callbacks.camera(ctx, cam);
            }
        } else if (obj_streq(cmd, "e")) {
            if (callbacks.environment) {
                auto env = obj_environment();
                obj_parse(ss, env.name);
                obj_parse(ss, env.ke);
                obj_parse(ss, env.ke_txt.path);
                if (env.ke_txt.path == "\"\"") env.ke_txt.path = "";
                obj_parse(ss, env.frame);
                done = callbacks.environment(ctx, env);
            }
        } else {
            // unused
        }
    }

    // close file
    fclose(fs);
}

// Dumps a value
inline void obj_dump(char*& s, char* val) {
    while (*val) *s++ = *val++;
}
inline void obj_dump(char*& s, const char* val) {
    while (*val) *s++ = *val++;
}
inline void obj_dump(char*& s, const std::string& val) {
    auto val_ = val.c_str();
    while (*val_) *s++ = *val_++;
}
inline void obj_dump(char*& s, int val) { s += sprintf(s, "%d", val); }
inline void obj_dump(char*& s, float val) { s += sprintf(s, "%g", val); }
inline void obj_dump(char*& s, const int* val, int num) {
    for (auto i = 0; i < num; i++) {
        if (i) *s++ = ' ';
        obj_dump(s, val[i]);
    }
}
inline void obj_dump(char*& s, const float* val, int num) {
    for (auto i = 0; i < num; i++) {
        if (i) *s++ = ' ';
        obj_dump(s, val[i]);
    }
}
inline void obj_dump(char*& s, vec2i val) { obj_dump(s, &val.x, 2); }
inline void obj_dump(char*& s, vec2f val) { obj_dump(s, &val.x, 2); }
inline void obj_dump(char*& s, vec3f val) { obj_dump(s, &val.x, 3); }
inline void obj_dump(char*& s, vec4f val) { obj_dump(s, &val.x, 4); }
inline void obj_dump(char*& s, const frame3f& val) {
    obj_dump(s, &val.x.x, 12);
}
inline void obj_dump(char*& s, const std::array<int, 3>& val) {
    auto nto_write = 0;
    for (auto i = 0; i < 3; i++) {
        if (val[i] >= 0) nto_write = i + 1;
    }
    for (auto i = 0; i < nto_write; i++) {
        if (i) *s++ = '/';
        if (val[i] >= 0) s += sprintf(s, "%d", val[i] + 1);
    }
}
inline void obj_dump(char*& s, const std::vector<std::string>& vals) {
    for (auto i = 0; i < vals.size(); i++) {
        if (i) *s++ = ' ';
        obj_dump(s, vals[i]);
    }
}
inline void obj_dump(char*& s, const obj_texture_info& v) {
    for (auto&& kv : v.props) {
        obj_dump(s, kv.first);
        *s++ = ' ';
        for (auto&& vv : kv.second) {
            obj_dump(s, vv);
            *s++ = ' ';
        }
    }
    if (v.clamp) obj_dump(s, "-clamp on ");
    obj_dump(s, v.path);
}

// Dumps a line
template <typename T>
inline void obj_dump_line(FILE* fs, const char* lbl, const T& val) {
    char buf[4096];
    buf[0] = 0;
    auto s = buf;
    obj_dump(s, lbl);
    *s++ = ' ';
    obj_dump(s, val);
    *s++ = '\n';
    *s = 0;
    fputs(buf, fs);
}

// Dumps a line
template <typename T>
inline void obj_dump_sp(FILE* fs, const T& val) {
    char buf[4096];
    buf[0] = 0;
    auto s = buf;
    obj_dump(s, val);
    *s++ = ' ';
    *s = 0;
    fputs(buf, fs);
}

// Dumps a newline
inline void obj_dump_nl(FILE* fs) { fputs("\n", fs); }

// Save an MTL file
void save_mtl(const std::string& filename,
    const std::vector<std::shared_ptr<obj_material>>& materials, bool flip_tr) {
    // open file
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) throw std::runtime_error("cannot open filename " + filename);

    // for each material, dump all the values
    for (auto mat : materials) {
        obj_dump_line(fs, "newmtl", mat->name);
        obj_dump_line(fs, "  illum", mat->illum);
        if (mat->ke != zero3f) obj_dump_line(fs, "  Ke", mat->ke);
        if (mat->ka != zero3f) obj_dump_line(fs, "  Ka", mat->ka);
        if (mat->kd != zero3f) obj_dump_line(fs, "  Kd", mat->kd);
        if (mat->ks != zero3f) obj_dump_line(fs, "  Ks", mat->ks);
        if (mat->kr != zero3f) obj_dump_line(fs, "  Kr", mat->kr);
        if (mat->kt != zero3f) obj_dump_line(fs, "  Kt", mat->kt);
        if (mat->ns != 0.0f)
            obj_dump_line(fs, "  Ns", (int)clamp(mat->ns, 0.0f, 1000000000.0f));
        if (mat->op != 1.0f) obj_dump_line(fs, "  d", mat->op);
        if (mat->ior != 1.0f) obj_dump_line(fs, "  Ni", mat->ior);
        if (mat->km != -1.0f) obj_dump_line(fs, "  Pm", mat->km);
        if (mat->rs != -1.0f) obj_dump_line(fs, "  Pr", mat->rs);
        if (mat->ke_txt.path != "") obj_dump_line(fs, "  map_Ke", mat->ke_txt);
        if (mat->ka_txt.path != "") obj_dump_line(fs, "  map_Ka", mat->ka_txt);
        if (mat->kd_txt.path != "") obj_dump_line(fs, "  map_Kd", mat->kd_txt);
        if (mat->ks_txt.path != "") obj_dump_line(fs, "  map_Ks", mat->ks_txt);
        if (mat->kr_txt.path != "") obj_dump_line(fs, "  map_Kr", mat->kr_txt);
        if (mat->kt_txt.path != "") obj_dump_line(fs, "  map_Kt", mat->kt_txt);
        if (mat->ns_txt.path != "") obj_dump_line(fs, "  map_Ns", mat->ns_txt);
        if (mat->op_txt.path != "") obj_dump_line(fs, "  map_d ", mat->op_txt);
        if (mat->ior_txt.path != "")
            obj_dump_line(fs, "  map_Ni", mat->ior_txt);
        if (mat->km_txt.path != "") obj_dump_line(fs, "  map_Pm ", mat->km_txt);
        if (mat->rs_txt.path != "") obj_dump_line(fs, "  map_Pr ", mat->rs_txt);
        if (mat->occ_txt.path != "")
            obj_dump_line(fs, "  map_occ ", mat->occ_txt);
        if (mat->bump_txt.path != "")
            obj_dump_line(fs, "  map_bump", mat->bump_txt);
        if (mat->disp_txt.path != "")
            obj_dump_line(fs, "  map_disp", mat->disp_txt);
        if (mat->norm_txt.path != "")
            obj_dump_line(fs, "  map_norm", mat->norm_txt);
        for (auto&& kv : mat->props) {
            obj_dump_line(fs, ("  " + kv.first + ' ').c_str(), kv.second);
        }
        obj_dump_nl(fs);
    }

    fclose(fs);
}

// Save an OBJ
void save_obj(const std::string& filename,
    const std::shared_ptr<obj_scene>& obj, bool flip_texcoord, bool flip_tr) {
    // open file
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) throw std::runtime_error("cannot open filename " + filename);

    // linkup to mtl
    auto mtlname = replace_path_extension(filename, "mtl");
    if (!obj->materials.empty()) {
        obj_dump_line(fs, "mtllib", mtlname);
    }

    // save cameras
    for (auto cam : obj->cameras) {
        obj_dump_sp(fs, "c");
        obj_dump_sp(fs, cam->name);
        obj_dump_sp(fs, cam->ortho);
        obj_dump_sp(fs, cam->width);
        obj_dump_sp(fs, cam->height);
        obj_dump_sp(fs, cam->focal);
        obj_dump_sp(fs, cam->focus);
        obj_dump_sp(fs, cam->aperture);
        obj_dump_sp(fs, cam->frame);
        obj_dump_nl(fs);
    }

    // save envs
    for (auto env : obj->environments) {
        obj_dump_sp(fs, "e");
        obj_dump_sp(fs, env->name);
        obj_dump_sp(fs, env->ke);
        obj_dump_sp(fs, (env->ke_txt.path != "") ? env->ke_txt.path : "\"\"");
        obj_dump_sp(fs, env->frame);
        obj_dump_nl(fs);
    }

    // save object properties
    for (auto oobj : obj->objects) {
        for (auto& kv : oobj->props) {
            obj_dump_sp(fs, "op");
            obj_dump_sp(fs, kv.first);
            for (auto v : kv.second) obj_dump_sp(fs, v);
            obj_dump_nl(fs);
        }
    }

    // save all vertex data
    for (auto& v : obj->pos) obj_dump_line(fs, "v", v);
    if (flip_texcoord) {
        for (auto& v : obj->texcoord)
            obj_dump_line(fs, "vt", vec2f{v.x, 1 - v.y});
    } else {
        for (auto& v : obj->texcoord) obj_dump_line(fs, "vt", v);
    }
    for (auto& v : obj->norm) obj_dump_line(fs, "vn", v);

    // save element data
    static auto elem_labels = std::unordered_map<obj_element_type, std::string>{
        {obj_element_type::point, "p"}, {obj_element_type::line, "l"},
        {obj_element_type::face, "f"}, {obj_element_type::bezier, "b"}};
    for (auto oobj : obj->objects) {
        obj_dump_line(fs, "o", oobj->name);
        if (oobj->frame != identity_frame3f)
            obj_dump_line(fs, "of", oobj->frame);
        if (oobj->subdiv != zero2i) obj_dump_line(fs, "os", oobj->subdiv);
        auto last_groupid = -1, last_materialid = -1, last_smoothingid = -1;
        for (auto& elem : oobj->elems) {
            if (last_materialid != elem.material && !oobj->materials.empty()) {
                auto matname = oobj->materials[elem.material];
                if (matname != "") obj_dump_line(fs, "usemtl", matname);
                last_materialid = elem.material;
            }
            if (last_groupid != elem.group && !oobj->groups.empty()) {
                auto groupname = oobj->groups[elem.group];
                if (groupname != "" || elem.group > 0)
                    obj_dump_line(fs, "g", groupname);
                last_groupid = elem.group;
            }
            if (last_smoothingid != elem.smoothing &&
                !oobj->smoothing.empty()) {
                auto smoothing = oobj->smoothing[elem.smoothing];
                if (!smoothing || elem.smoothing > 0)
                    obj_dump_line(fs, "s", smoothing ? "on" : "off");
                last_smoothingid = elem.smoothing;
            }
            obj_dump_sp(fs, elem_labels.at(elem.type).c_str());
            for (auto vid = elem.start; vid < elem.start + elem.size; vid++) {
                auto vert = std::array<int, 3>{{-1, -1, -1}};
                vert[0] = (oobj->verts_pos.empty()) ? -1 : oobj->verts_pos[vid];
                vert[1] = (oobj->verts_texcoord.empty()) ?
                              -1 :
                              oobj->verts_texcoord[vid];
                vert[2] =
                    (oobj->verts_norm.empty()) ? -1 : oobj->verts_norm[vid];
                obj_dump_sp(fs, vert);
            }
            obj_dump_nl(fs);
        }
    }

    fclose(fs);

    // save materials
    if (!obj->materials.empty())
        save_mtl(replace_path_extension(filename, "mtl"), obj->materials, flip_tr);
}

// Load OBJ texture images.
void load_obj_textures(const std::shared_ptr<obj_scene>& obj,
    const std::string& dirname, bool skip_missing) {
    // set gamma
    auto ldr_gamma = std::unordered_map<std::string, float>{{"", 1.0f}};
    for (auto txt : obj->textures) ldr_gamma[txt->path] = 2.2f;
    for (auto mat : obj->materials) {
        ldr_gamma[mat->ke_txt.path] = 2.2f;
        ldr_gamma[mat->kd_txt.path] = 2.2f;
        ldr_gamma[mat->ks_txt.path] = 2.2f;
        ldr_gamma[mat->kr_txt.path] = 2.2f;
        ldr_gamma[mat->kt_txt.path] = 2.2f;
        ldr_gamma[mat->rs_txt.path] = 1;
        ldr_gamma[mat->op_txt.path] = 1;
        ldr_gamma[mat->norm_txt.path] = 1;
        ldr_gamma[mat->disp_txt.path] = 1;
        ldr_gamma[mat->bump_txt.path] = 1;
    }
    for (auto env : obj->environments) { ldr_gamma[env->ke_txt.path] = 2.2f; }

    // load images
    for (auto txt : obj->textures) {
        auto filename = dirname + txt->path;
        for (auto& c : filename)
            if (c == '\\') c = '/';
        try {
            txt->img = load_image(filename, ldr_gamma.at(txt->path));
        } catch (std::exception&) {
            if (skip_missing) continue;
            throw;
        }
    }
}

// Save OBJ texture images.
void save_obj_textures(const std::shared_ptr<obj_scene>& obj,
    const std::string& dirname, bool skip_missing) {
    // set gamma
    auto ldr_gamma = std::unordered_map<std::string, float>{{"", 1.0f}};
    for (auto txt : obj->textures) ldr_gamma[txt->path] = 2.2f;
    for (auto mat : obj->materials) {
        ldr_gamma[mat->ke_txt.path] = 2.2f;
        ldr_gamma[mat->kd_txt.path] = 2.2f;
        ldr_gamma[mat->ks_txt.path] = 2.2f;
        ldr_gamma[mat->kr_txt.path] = 2.2f;
        ldr_gamma[mat->kt_txt.path] = 2.2f;
        ldr_gamma[mat->rs_txt.path] = 1;
        ldr_gamma[mat->op_txt.path] = 1;
        ldr_gamma[mat->norm_txt.path] = 1;
        ldr_gamma[mat->disp_txt.path] = 1;
        ldr_gamma[mat->bump_txt.path] = 1;
    }
    for (auto env : obj->environments) { ldr_gamma[env->ke_txt.path] = 2.2f; }

    // save images
    for (auto txt : obj->textures) {
        if (txt->img.empty()) continue;
        auto filename = dirname + txt->path;
        for (auto& c : filename)
            if (c == '\\') c = '/';
        try {
            save_image(filename, txt->img, ldr_gamma.at(txt->path));
        } catch (std::exception&) {
            if (skip_missing) continue;
            throw;
        }
    }
}

}  // namespace ygl
