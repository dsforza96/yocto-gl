//
// Implementation for Yocto/GL Input and Output functions.
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

#include "yglio.h"

#include <cstdlib>
#include <deque>
#include <regex>

#include <array>
#include <climits>
using namespace std::string_literals;

#include "ext/json.hpp"

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif

#ifndef __clang_analyzer__

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ext/stb_image_resize.h"

#define TINYEXR_IMPLEMENTATION
#include "ext/tinyexr.h"

#endif

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF STRING FORMAT UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Format duration string from nanoseconds
std::string format_duration(int64_t duration) {
    auto elapsed = duration / 1000000;  // milliseconds
    auto hours   = (int)(elapsed / 3600000);
    elapsed %= 3600000;
    auto mins = (int)(elapsed / 60000);
    elapsed %= 60000;
    auto secs  = (int)(elapsed / 1000);
    auto msecs = (int)(elapsed % 1000);
    char buf[256];
    sprintf(buf, "%02d:%02d:%02d.%03d", hours, mins, secs, msecs);
    return buf;
}
// Format a large integer number in human readable form
std::string format_num(uint64_t num) {
    auto rem = num % 1000;
    auto div = num / 1000;
    if (div > 0) return format_num(div) + "," + std::to_string(rem);
    return std::to_string(rem);
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PATH UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

std::string normalize_path(const std::string& filename_) {
    auto filename = filename_;
    for (auto& c : filename)
        if (c == '\\') c = '/';
    if (filename.size() > 1 && filename[0] == '/' && filename[1] == '/')
        throw std::runtime_error("no absolute paths");
    if (filename.size() > 3 && filename[1] == ':' && filename[2] == '/' &&
        filename[3] == '/')
        throw std::runtime_error("no absolute paths");
    auto pos = (size_t)0;
    while ((pos = filename.find("//")) != filename.npos)
        filename = filename.substr(0, pos) + filename.substr(pos + 1);
    return filename;
}

// Get directory name (not including '/').
std::string get_dirname(const std::string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('/');
    if (pos == std::string::npos) return "";
    return filename.substr(0, pos);
}

// Get extension (not including '.').
std::string get_extension(const std::string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

// Get filename without directory.
std::string get_filename(const std::string& filename_) {
    auto filename = normalize_path(filename_);
    auto pos      = filename.rfind('/');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

// Replace extension.
std::string replace_extension(
    const std::string& filename_, const std::string& ext_) {
    auto filename = normalize_path(filename_);
    auto ext      = normalize_path(ext_);
    if (ext.at(0) == '.') ext = ext.substr(1);
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return filename;
    return filename.substr(0, pos) + "." + ext;
}

// Check if a file can be opened for reading.
bool exists_file(const std::string& filename) {
    auto f = fopen(filename.c_str(), "r");
    if (!f) return false;
    fclose(f);
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF FILE READING
// -----------------------------------------------------------------------------
namespace ygl {

// Object that holds a FILE* and calls fclose() when it gets out of scope
struct fclose_guard {
    fclose_guard(FILE* fs) : fs_{fs} {}
    ~fclose_guard() { fclose(fs_); }
    FILE* fs_ = nullptr;
};

// Load a text file
std::string load_text(const std::string& filename) {
    auto str = std::string();
    if (!load_text(filename, str)) return {};
    return str;
}

// Load a text file
bool load_text(const std::string& filename, std::string& str) {
    // https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
    str     = {};
    auto fs = fopen(filename.c_str(), "rb");
    if (!fs) return false;
    fclose_guard fs_{fs};
    fseek(fs, 0, SEEK_END);
    auto fsize = ftell(fs);
    fseek(fs, 0, SEEK_SET);
    auto buf = std::vector<char>(fsize);
    if (fread(buf.data(), 1, fsize, fs) != fsize) return false;
    fclose(fs);
    str = {buf.begin(), buf.end()};
    return true;
}

// Save a text file
bool save_text(const std::string& filename, const std::string& str) {
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};
    fprintf(fs, "%s", str.c_str());
    fclose(fs);
    return true;
}

// Load a binary file
std::vector<byte> load_binary(const std::string& filename) {
    auto data = std::vector<byte>();
    if (!load_binary(filename, data)) return {};
    return data;
}

// Load a binary file
bool load_binary(const std::string& filename, std::vector<byte>& data) {
    // https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
    data    = {};
    auto fs = fopen(filename.c_str(), "rb");
    if (!fs) return false;
    fclose_guard fs_{fs};
    fseek(fs, 0, SEEK_END);
    auto fsize = ftell(fs);
    fseek(fs, 0, SEEK_SET);
    data.resize(fsize);
    if (fread((char*)data.data(), 1, fsize, fs) != fsize) return false;
    fclose(fs);
    return true;
}

// Save a binary file
bool save_binary(const std::string& filename, const std::vector<byte>& data) {
    auto fs = fopen(filename.c_str(), "wb");
    if (!fs) return false;
    fclose_guard fs_{fs};
    fwrite((char*)data.data(), 1, data.size(), fs);
    fclose(fs);
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// TRIVIAL COMMAND LINE PARSING
// -----------------------------------------------------------------------------
namespace ygl {

// initialize a command line parser
cmdline_parser make_cmdline_parser(
    int argc, char** argv, const std::string& usage, const std::string& cmd) {
    auto parser      = cmdline_parser();
    parser.args      = {argv + 1, argv + argc};
    parser.usage_cmd = (cmd.empty()) ? argv[0] : cmd;
    parser.usage_hlp = usage;
    return parser;
}

// check if option or argument
bool is_option(const std::string& name) {
    return name.size() > 1 && name.front() == '-';
}

// get names from string
std::vector<std::string> get_option_names(const std::string& name_) {
    auto names = std::vector<std::string>();
    auto name  = name_;
    while (name.find(',') != name.npos) {
        names.push_back(name.substr(0, name.find(',')));
        name = name.substr(name.find(',') + 1);
    }
    names.push_back(name);
    return names;
}

// add help
std::string get_option_usage(const std::string& name, const std::string& var,
    const std::string& usage, const std::string& def_,
    const std::vector<std::string>& choices) {
    auto def = def_;
    if (def != "") def = "[" + def + "]";
    auto namevar = name;
    if (var != "") namevar += " " + var;
    char buf[4096];
    sprintf(buf, "  %-24s %s %s\n", namevar.c_str(), usage.c_str(), def.c_str());
    auto usagelines = std::string(buf);
    if (!choices.empty()) {
        usagelines += "        accepted values:";
        for (auto& c : choices) usagelines += " " + c;
        usagelines += "\n";
    }
    return usagelines;
}

// print cmdline help
void print_cmdline_usage(const cmdline_parser& parser) {
    printf("%s: %s\n", parser.usage_cmd.c_str(), parser.usage_hlp.c_str());
    printf("usage: %s %s %s\n\n", parser.usage_cmd.c_str(),
        (parser.usage_opt.empty()) ? "" : "[options]",
        (parser.usage_arg.empty()) ? "" : "arguments");
    if (!parser.usage_opt.empty()) {
        printf("options:\n");
        printf("%s\n", parser.usage_opt.c_str());
    }
    if (!parser.usage_arg.empty()) {
        printf("arguments:\n");
        printf("%s\n", parser.usage_arg.c_str());
    }
}

// forward declaration
bool parse_flag(cmdline_parser& parser, const std::string& name, bool def,
    const std::string& usage);

// check if any error occurred and exit appropriately
void check_cmdline(cmdline_parser& parser) {
    if (parse_flag(parser, "--help,-?", false, "print help")) {
        print_cmdline_usage(parser);
        exit(0);
    }
    if (!parser.args.empty()) parser.error += "unmatched arguments remaining\n";
    if (!parser.error.empty()) {
        printf("error: %s", parser.error.c_str());
        print_cmdline_usage(parser);
        exit(1);
    }
}

// Parse a flag. Name should start with either "--" or "-".
bool parse_flag(cmdline_parser& parser, const std::string& name, bool def,
    const std::string& usage) {
    parser.usage_opt += get_option_usage(name, "", usage, "", {});
    if (parser.error != "") return def;
    auto names = get_option_names(name);
    auto pos   = parser.args.end();
    for (auto& name : names)
        pos = std::min(
            pos, std::find(parser.args.begin(), parser.args.end(), name));
    if (pos == parser.args.end()) return def;
    parser.args.erase(pos);
    return !def;
}

// Parse an option string. Name should start with "--" or "-".
std::string parse_option(cmdline_parser& parser, const std::string& name,
    const std::string& def, const std::string& usage, bool req,
    const std::vector<std::string>& choices) {
    parser.usage_opt += get_option_usage(name, "", usage, def, choices);
    if (parser.error != "") return def;
    auto names = get_option_names(name);
    auto pos   = parser.args.end();
    for (auto& name : names) {
        pos = std::min(
            pos, std::find(parser.args.begin(), parser.args.end(), name));
    }
    if (pos == parser.args.end()) {
        if (req) parser.error += "missing value for " + name;
        return def;
    }
    if (pos == parser.args.end() - 1) {
        parser.error += "missing value for " + name;
        return def;
    }
    auto val = *(pos + 1);
    parser.args.erase(pos, pos + 2);
    if (!choices.empty() &&
        std::find(choices.begin(), choices.end(), val) == choices.end()) {
        parser.error += "bad value for " + name;
        return def;
    }
    return val;
}

// Parse an argument string. Name should not start with "--" or "-".
std::string parse_argument(cmdline_parser& parser, const std::string& name,
    const std::string& def, const std::string& usage, bool req,
    const std::vector<std::string>& choices) {
    parser.usage_arg += get_option_usage(name, "", usage, def, choices);
    if (parser.error != "") return def;
    auto pos = std::find_if(parser.args.begin(), parser.args.end(),
        [](auto& v) { return v[0] != '-'; });
    if (pos == parser.args.end()) {
        if (req) parser.error += "missing value for " + name;
        return def;
    }
    auto val = *pos;
    parser.args.erase(pos);
    if (!choices.empty() &&
        std::find(choices.begin(), choices.end(), val) == choices.end()) {
        parser.error += "bad value for " + name;
        return def;
    }
    return val;
}

// Parse a string argument.
std::string parse_string(cmdline_parser& parser, const std::string& name,
    const std::string& def, const std::string& usage, bool req,
    const std::vector<std::string>& choices) {
    return is_option(name) ?
               parse_option(parser, name, def, usage, req, choices) :
               parse_argument(parser, name, def, usage, req, choices);
}

// Parse an integer, float, string. If name starts with "--" or "-", then it is
// an option, otherwise it is a position argument.
bool parse_arg(cmdline_parser& parser, const std::string& name, bool def,
    const std::string& usage) {
    return parse_flag(parser, name, def, usage);
}
std::string parse_arg(cmdline_parser& parser, const std::string& name,
    const std::string& def, const std::string& usage, bool req) {
    return parse_string(parser, name, def, usage, req, {});
}
std::string parse_arg(cmdline_parser& parser, const std::string& name,
    const char* def, const std::string& usage, bool req) {
    return parse_string(parser, name, def, usage, req, {});
}
int parse_arg(cmdline_parser& parser, const std::string& name, int def,
    const std::string& usage, bool req) {
    auto vals = parse_string(parser, name, std::to_string(def), usage, req, {});
    auto val  = def;
    if (sscanf(vals.c_str(), "%d", &val) != 1) {
        parser.error += "bad value for " + name;
        return def;
    }
    return val;
}
float parse_arg(cmdline_parser& parser, const std::string& name, float def,
    const std::string& usage, bool req) {
    auto vals = parse_string(parser, name, std::to_string(def), usage, req, {});
    auto val  = def;
    if (sscanf(vals.c_str(), "%f", &val) != 1) {
        parser.error += "bad value for " + name;
        return def;
    }
    return val;
}
vec2f parse_arg(cmdline_parser& parser, const std::string& name,
    const vec2f& def, const std::string& usage, bool req) {
    auto vals = parse_string(parser, name,
        std::to_string(def.x) + " " + std::to_string(def.y), usage, req, {});
    auto val  = def;
    if (sscanf(vals.c_str(), "%f %f", &val.x, &val.y) != 2) {
        parser.error += "bad value for " + name;
        return def;
    }
    return val;
}
vec3f parse_arg(cmdline_parser& parser, const std::string& name,
    const vec3f& def, const std::string& usage, bool req) {
    auto vals = parse_string(parser, name,
        std::to_string(def.x) + " " + std::to_string(def.y) + " " +
            std::to_string(def.z),
        usage, req, {});
    auto val  = def;
    if (sscanf(vals.c_str(), "%f %f %f", &val.x, &val.y, &val.z) != 3) {
        parser.error += "bad value for " + name;
        return def;
    }
    return val;
}
int parse_arge(cmdline_parser& parser, const std::string& name, int def,
    const std::string& usage, const std::vector<std::string>& labels, bool req) {
    auto val = parse_string(parser, name, labels.at(def), usage, req, labels);
    return (int)(std::find(labels.begin(), labels.end(), val) - labels.begin());
}

// Parser an argument
std::vector<std::string> parse_args(cmdline_parser& parser,
    const std::string& name, const std::vector<std::string>& def,
    const std::string& usage, bool req) {
    auto defs = std::string();
    for (auto& d : def) defs += " " + d;
    parser.usage_arg += get_option_usage(name, "", usage, defs, {});
    if (parser.error != "") return {};
    auto pos = std::find_if(parser.args.begin(), parser.args.end(),
        [](auto& v) { return v[0] != '-'; });
    if (pos == parser.args.end()) {
        if (req) parser.error += "missing value for " + name;
        return {};
    }
    auto val = std::vector<std::string>{pos, parser.args.end()};
    parser.args.erase(pos, parser.args.end());
    return val;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR IMAGEIO
// -----------------------------------------------------------------------------
namespace ygl {

// Split a string
std::vector<std::string> split_string(const std::string& str) {
    auto ret = std::vector<std::string>();
    if (str.empty()) return ret;
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
}

// Pfm load
float* load_pfm(const char* filename, int* w, int* h, int* nc, int req) {
    auto fs = fopen(filename, "rb");
    if (!fs) return nullptr;
    fclose_guard fs_{fs};

    // buffer
    char buf[256];
    auto toks = std::vector<std::string>();

    // read magic
    if (!fgets(buf, 256, fs)) return nullptr;
    toks = split_string(buf);
    if (toks[0] == "Pf")
        *nc = 1;
    else if (toks[0] == "PF")
        *nc = 3;
    else
        return nullptr;

    // read w, h
    if (!fgets(buf, 256, fs)) return nullptr;
    toks = split_string(buf);
    *w   = atoi(toks[0].c_str());
    *h   = atoi(toks[1].c_str());

    // read scale
    if (!fgets(buf, 256, fs)) return nullptr;
    toks   = split_string(buf);
    auto s = atof(toks[0].c_str());

    // read the data (flip y)
    auto npixels = (*w) * (*h);
    auto nvalues = (*w) * (*h) * (*nc);
    auto nrow    = (*w) * (*nc);
    auto pixels  = new float[nvalues];
    for (auto j = *h - 1; j >= 0; j--) {
        if (fread(pixels + j * nrow, sizeof(float), nrow, fs) != nrow) {
            delete[] pixels;
            return nullptr;
        }
    }

    // done reading
    fclose(fs);

    // endian conversion
    if (s > 0) {
        for (auto i = 0; i < nvalues; ++i) {
            auto dta = (uint8_t*)(pixels + i);
            std::swap(dta[0], dta[3]);
            std::swap(dta[1], dta[2]);
        }
    }

    // scale
    auto scl = (s > 0) ? s : -s;
    if (scl != 1) {
        for (auto i = 0; i < nvalues; i++) pixels[i] *= scl;
    }

    // proper number of channels
    if (!req || *nc == req) return pixels;

    // pack into channels
    if (req < 0 || req > 4) {
        delete[] pixels;
        return nullptr;
    }
    auto cpixels = new float[req * npixels];
    for (auto i = 0; i < npixels; i++) {
        auto vp = pixels + i * (*nc);
        auto cp = cpixels + i * req;
        if (*nc == 1) {
            switch (req) {
                case 1: cp[0] = vp[0]; break;
                case 2:
                    cp[0] = vp[0];
                    cp[1] = vp[0];
                    break;
                case 3:
                    cp[0] = vp[0];
                    cp[1] = vp[0];
                    cp[2] = vp[0];
                    break;
                case 4:
                    cp[0] = vp[0];
                    cp[1] = vp[0];
                    cp[2] = vp[0];
                    cp[3] = 1;
                    break;
            }
        } else {
            switch (req) {
                case 1: cp[0] = vp[0]; break;
                case 2:
                    cp[0] = vp[0];
                    cp[1] = vp[1];
                    break;
                case 3:
                    cp[0] = vp[0];
                    cp[1] = vp[1];
                    cp[2] = vp[2];
                    break;
                case 4:
                    cp[0] = vp[0];
                    cp[1] = vp[1];
                    cp[2] = vp[2];
                    cp[3] = 1;
                    break;
            }
        }
    }
    delete[] pixels;
    return cpixels;
}

// save pfm
bool save_pfm(const char* filename, int w, int h, int nc, const float* pixels) {
    auto fs = fopen(filename, "wb");
    if (!fs) return false;
    fclose_guard fs_{fs};

    fprintf(fs, "%s\n", (nc == 1) ? "Pf" : "PF");
    fprintf(fs, "%d %d\n", w, h);
    fprintf(fs, "-1\n");
    if (nc == 1 || nc == 3) {
        fwrite(pixels, sizeof(float), w * h * nc, fs);
    } else {
        for (auto i = 0; i < w * h; i++) {
            auto vz = 0.0f;
            auto v  = pixels + i * nc;
            fwrite(v + 0, sizeof(float), 1, fs);
            fwrite(v + 1, sizeof(float), 1, fs);
            if (nc == 2)
                fwrite(&vz, sizeof(float), 1, fs);
            else
                fwrite(v + 2, sizeof(float), 1, fs);
        }
    }

    fclose(fs);

    return true;
}

// check hdr extensions
bool is_hdr_filename(const std::string& filename) {
    auto ext = get_extension(filename);
    return ext == "hdr" || ext == "exr" || ext == "pfm";
}

// Loads an hdr image.
image<vec4f> load_image4f(const std::string& filename) {
    auto img = image<vec4f>{};
    if (!load_image4f(filename, img)) return {};
    return img;
}

// Loads an hdr image.
bool load_image4f(const std::string& filename, image<vec4f>& img) {
    auto ext  = get_extension(filename);
    auto size = zero2i;
    img       = {};
    if (ext == "exr") {
        auto pixels = (vec4f*)nullptr;
        if (LoadEXR((float**)&pixels, &size.x, &size.y, filename.c_str(),
                nullptr) < 0)
            return false;
        if (!pixels) return false;
        img = image<vec4f>{size, pixels};
        free(pixels);
        return true;
    } else if (ext == "pfm") {
        auto ncomp  = 0;
        auto pixels = (vec4f*)load_pfm(
            filename.c_str(), &size.x, &size.y, &ncomp, 4);
        if (!pixels) return false;
        img = image<vec4f>{size, pixels};
        free(pixels);
        return true;
    } else if (ext == "hdr") {
        auto ncomp  = 0;
        auto pixels = (vec4f*)stbi_loadf(
            filename.c_str(), &size.x, &size.y, &ncomp, 4);
        if (!pixels) return false;
        img = image<vec4f>{size, pixels};
        free(pixels);
        return true;
    } else {
        auto ncomp  = 0;
        auto pixels = (vec4f*)stbi_loadf(
            filename.c_str(), &size.x, &size.y, &ncomp, 4);
        if (!pixels) return false;
        img = image<vec4f>{size, pixels};
        free(pixels);
        return true;
    }
}

// Saves an hdr image.
bool save_image4f(const std::string& filename, const image<vec4f>& img) {
    auto ext  = get_extension(filename);
    auto img8 = (is_hdr_filename(filename)) ?
                    image<vec4b>{} :
                    float_to_byte(linear_to_srgb(img));
    if (ext == "png") {
        return stbi_write_png(filename.c_str(), img.size().x, img.size().y, 4,
            img8.data(), img.size().x * 4);
    } else if (ext == "jpg") {
        return stbi_write_jpg(
            filename.c_str(), img.size().x, img.size().y, 4, img8.data(), 75);
    } else if (ext == "tga") {
        return stbi_write_tga(
            filename.c_str(), img.size().x, img.size().y, 4, img8.data());
    } else if (ext == "bmp") {
        return stbi_write_bmp(
            filename.c_str(), img.size().x, img.size().y, 4, img8.data());
    } else if (ext == "hdr") {
        return stbi_write_hdr(filename.c_str(), img.size().x, img.size().y, 4,
            (float*)img.data());
    } else if (ext == "pfm") {
        return save_pfm(filename.c_str(), img.size().x, img.size().y, 4,
            (float*)img.data());
    } else if (ext == "exr") {
        return SaveEXR((float*)img.data(), img.size().x, img.size().y, 4,
            filename.c_str());
    } else {
        return false;
    }
}

// Loads an hdr image.
image<vec4f> load_image4f_from_memory(const byte* data, int data_size) {
    auto img = image<vec4f>{};
    if (!load_image4f_from_memory(data, data_size, img)) return {};
    return img;
}

// Loads an hdr image.
bool load_image4f_from_memory(
    const byte* data, int data_size, image<vec4f>& img) {
    img         = {};
    auto size   = zero2i;
    auto ncomp  = 0;
    auto pixels = (vec4f*)stbi_loadf_from_memory(
        data, data_size, &size.x, &size.y, &ncomp, 4);
    if (!pixels) return false;
    img = image<vec4f>{size, pixels};
    delete pixels;
    return true;
}

// Loads an hdr image.
image<vec4b> load_image4b(const std::string& filename) {
    auto img = image<vec4b>{};
    if (!load_image4b(filename, img)) return {};
    return img;
}

// Loads an hdr image.
bool load_image4b(const std::string& filename, image<vec4b>& img) {
    img       = {};
    auto ext  = get_extension(filename);
    auto size = zero2i;
    if (ext == "exr") {
        auto pixels = (vec4f*)nullptr;
        if (LoadEXR((float**)&pixels, &size.x, &size.y, filename.c_str(),
                nullptr) < 0)
            return false;
        if (!pixels) return false;
        auto imgf = image<vec4f>{size, pixels};
        img       = float_to_byte(linear_to_srgb(imgf));
        free(pixels);
        return true;
    } else if (ext == "pfm") {
        auto ncomp  = 0;
        auto pixels = (vec4f*)load_pfm(
            filename.c_str(), &size.x, &size.y, &ncomp, 4);
        if (!pixels) return false;
        auto imgf = image<vec4f>{size, pixels};
        img       = float_to_byte(linear_to_srgb(imgf));
        free(pixels);
        return true;
    } else if (ext == "hdr") {
        auto ncomp  = 0;
        auto pixels = (vec4b*)stbi_load(
            filename.c_str(), &size.x, &size.y, &ncomp, 4);
        if (!pixels) return false;
        img = image<vec4b>{size, pixels};
        free(pixels);
        return true;
    } else {
        auto ncomp  = 0;
        auto pixels = (vec4b*)stbi_load(
            filename.c_str(), &size.x, &size.y, &ncomp, 4);
        if (!pixels) return false;
        img = image<vec4b>{size, pixels};
        free(pixels);
        return true;
    }
}

// Saves an ldr image.
bool save_image4b(const std::string& filename, const image<vec4b>& img) {
    auto ext = get_extension(filename);
    if (ext == "png") {
        return stbi_write_png(filename.c_str(), img.size().x, img.size().y, 4,
            img.data(), img.size().x * 4);
    } else if (ext == "jpg") {
        return stbi_write_jpg(
            filename.c_str(), img.size().x, img.size().y, 4, img.data(), 75);
    } else if (ext == "tga") {
        return stbi_write_tga(
            filename.c_str(), img.size().x, img.size().y, 4, img.data());
    } else if (ext == "bmp") {
        return stbi_write_bmp(
            filename.c_str(), img.size().x, img.size().y, 4, img.data());
    } else if (ext == "hdr") {
        auto imgf = srgb_to_linear(byte_to_float(img));
        return stbi_write_hdr(filename.c_str(), img.size().x, img.size().y, 4,
            (float*)imgf.data());
    } else if (ext == "pfm") {
        auto imgf = srgb_to_linear(byte_to_float(img));
        return save_pfm(filename.c_str(), img.size().x, img.size().y, 4,
            (float*)imgf.data());
    } else if (ext == "exr") {
        auto imgf = srgb_to_linear(byte_to_float(img));
        return SaveEXR((float*)imgf.data(), img.size().x, img.size().y, 4,
            filename.c_str());
    } else {
        return false;
    }
}

// Loads an ldr image.
image<vec4b> load_image4b_from_memory(const byte* data, int data_size) {
    auto img = image<vec4b>{};
    if (!load_image4b_from_memory(data, data_size, img)) return {};
    return img;
}

// Loads an ldr image.
bool load_image4b_from_memory(
    const byte* data, int data_size, image<vec4b>& img) {
    img         = {};
    auto size   = zero2i;
    auto ncomp  = 0;
    auto pixels = (vec4b*)stbi_load_from_memory(
        data, data_size, &size.x, &size.y, &ncomp, 4);
    if (!pixels) return false;
    img = image<vec4b>{size, pixels};
    delete pixels;
    return true;
}

// Convenience helper that saves an HDR images as wither a linear HDR file or
// a tonemapped LDR file depending on file name
bool save_tonemapped_image(const std::string& filename, const image<vec4f>& hdr,
    float exposure, bool filmic, bool srgb) {
    if (is_hdr_filename(filename)) {
        return save_image4f(filename, hdr);
    } else {
        auto ldr = float_to_byte(tonemap_filmic(hdr, exposure, filmic, srgb));
        return save_image4b(filename, ldr);
    }
}

// Resize image.
image<vec4f> resize_image(const image<vec4f>& img, const vec2i& size_) {
    auto size = size_;
    if (!size.x && !size.y) throw std::runtime_error("bad image size");
    if (!size.x)
        size.x = (int)round(img.size().x * (size.y / (float)img.size().y));
    if (!size.y)
        size.y = (int)round(img.size().y * (size.x / (float)img.size().x));
    auto res_img = image<vec4f>{size};
    stbir_resize_float_generic((float*)img.data(), img.size().x, img.size().y,
        sizeof(vec4f) * img.size().x, (float*)res_img.data(), res_img.size().x,
        res_img.size().y, sizeof(vec4f) * res_img.size().x, 4, 3, 0,
        STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR,
        nullptr);
    return res_img;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR VOLUME IMAGE IO
// -----------------------------------------------------------------------------
namespace ygl {

// Loads volume data from binary format.
volume<float> load_volume1f(const std::string& filename) {
    auto vol = volume<float>();
    return vol;
}

// Loads volume data from binary format.
bool load_volume1f(const std::string& filename, volume<float>& vol) {
    vol     = {};
    auto fs = fopen(filename.c_str(), "r");
    if (!fs) return false;
    fclose_guard fs_{fs};
    auto         size = zero3i;
    if (fread(&size, sizeof(vec3i), 1, fs) != 1) return false;
    vol        = volume<float>{size};
    auto count = vol.size().x * vol.size().y * vol.size().z;
    if (fread(vol.data(), sizeof(float), count, fs) != count) return false;
    fclose(fs);
    return true;
}

// Saves volume data in binary format.
bool save_volume1f(const std::string& filename, const volume<float>& vol) {
    auto fs = fopen(filename.c_str(), "w");
    if (!fs) return false;
    fclose_guard fs_{fs};
    auto         size = vol.size();
    fwrite(&size, sizeof(vec3i), 1, fs);
    auto count = vol.size().x * vol.size().y * vol.size().z;
    if (fwrite(vol.data(), sizeof(float), count, fs) != count) return false;
    fclose(fs);
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// GENERIC IMAGE LOADING
// -----------------------------------------------------------------------------
namespace ygl {

// reset scene data pointer
void reset_scene_data(scene*& scn) {
    if (scn) delete scn;
    scn = nullptr;
}

// Load a scene
scene* load_scene(
    const std::string& filename, bool load_textures, bool skip_missing) {
    auto scn = (scene*)nullptr;
    if (!load_scene(filename, scn, load_textures, skip_missing)) return nullptr;
    return scn;
}

// Load a scene
bool load_scene(const std::string& filename, scene*& scn, bool load_textures,
    bool skip_missing) {
    auto ext = get_extension(filename);
    if (ext == "json" || ext == "JSON") {
        return load_json_scene(filename, scn, load_textures, skip_missing);
    } else if (ext == "obj" || ext == "OBJ") {
        return load_obj_scene(filename, scn, load_textures, skip_missing);
    } else if (ext == "gltf" || ext == "GLTF") {
        return load_gltf_scene(filename, scn, load_textures, skip_missing);
    } else if (ext == "pbrt" || ext == "PBRT") {
        return load_pbrt_scene(filename, scn, load_textures, skip_missing);
    } else if (ext == "ybin" || ext == "YBIN") {
        return load_ybin_scene(filename, scn, load_textures, skip_missing);
    } else {
        reset_scene_data(scn);
        return false;
    }
}

// Save a scene
bool save_scene(const std::string& filename, const scene* scn,
    bool save_textures, bool skip_missing) {
    auto ext = get_extension(filename);
    if (ext == "json" || ext == "JSON") {
        return save_json_scene(filename, scn, save_textures, skip_missing);
    } else if (ext == "obj" || ext == "OBJ") {
        return save_obj_scene(filename, scn, save_textures, skip_missing);
    } else if (ext == "gltf" || ext == "GLTF") {
        return save_gltf_scene(filename, scn, save_textures, skip_missing);
    } else if (ext == "pbrt" || ext == "PBRT") {
        return save_pbrt_scene(filename, scn, save_textures, skip_missing);
    } else if (ext == "ybin" || ext == "YBIN") {
        return save_ybin_scene(filename, scn, save_textures, skip_missing);
    } else {
        return false;
    }
}

bool load_scene_textures(scene* scn, const std::string& dirname,
    bool skip_missing, bool assign_opacity) {
    // load images
    for (auto txt : scn->textures) {
        if (txt->path == "" || !txt->imgf.empty() || !txt->imgb.empty())
            continue;
        auto filename = normalize_path(dirname + "/" + txt->path);
        if (is_hdr_filename(filename)) {
            if (!load_image4f(filename, txt->imgf)) {
                if (!skip_missing) return false;
            }
        } else {
            if (!load_image4b(filename, txt->imgb)) {
                if (!skip_missing) return false;
            }
        }
    }

    // load volumes
    for (auto txt : scn->voltextures) {
        if (txt->path == "" || !txt->vol.empty()) continue;
        auto filename = normalize_path(dirname + "/" + txt->path);
        if (!load_volume1f(filename, txt->vol)) {
            if (!skip_missing) return false;
        }
    }

    // assign opacity texture if needed
    if (assign_opacity) {
        auto has_opacity = std::unordered_map<texture*, bool>();
        for (auto& txt : scn->textures) {
            has_opacity[txt] = false;
            for (auto& p : txt->imgf) {
                if (p.w < 0.999f) {
                    has_opacity[txt] = true;
                    break;
                }
            }
            for (auto& p : txt->imgb) {
                if (p.w < 255) {
                    has_opacity[txt] = true;
                    break;
                }
            }
        }
        for (auto& mat : scn->materials) {
            if (mat->kd_txt && !mat->op_txt && has_opacity.at(mat->kd_txt))
                mat->op_txt = mat->kd_txt;
        }
    }

    // done
    return true;
}

// helper to save textures
bool save_scene_textures(
    const scene* scn, const std::string& dirname, bool skip_missing) {
    // save images
    for (auto txt : scn->textures) {
        if (txt->imgf.empty() && txt->imgb.empty()) continue;
        auto filename = normalize_path(dirname + "/" + txt->path);
        if (is_hdr_filename(filename)) {
            if (!save_image4f(filename, txt->imgf)) {
                if (!skip_missing) return false;
            }
        } else {
            if (!save_image4b(filename, txt->imgb)) {
                if (!skip_missing) return false;
            }
        }
    }

    // save volumes
    for (auto txt : scn->voltextures) {
        if (txt->vol.empty()) continue;
        auto filename = normalize_path(dirname + "/" + txt->path);
        if (!save_volume1f(filename, txt->vol)) {
            if (!skip_missing) return false;
        }
    }

    // done
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IO UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Encode in base64
std::string base64_encode(
    unsigned char const* bytes_to_encode, unsigned int in_len) {
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string   ret;
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
std::string base64_decode(std::string const& encoded_string) {
    static const std::string base64_chars =
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
    std::string   ret;

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

}  // namespace ygl

// -----------------------------------------------------------------------------
// JSON UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Json alias
using json = nlohmann::json;

// forward declaration
bool load_json(const std::string& filename, json& js);

// Load a JSON object
json load_json(const std::string& filename) {
    auto js = json();
    if (!load_json(filename, js)) return false;
    return js;
}

// Load a JSON object
bool load_json(const std::string& filename, json& js) {
    js       = {};
    auto txt = std::string();
    if (!load_text(filename, txt)) return false;
    try {
        js = json::parse(txt.begin(), txt.end());
        return true;
    } catch (...) { return false; }
}

// Save a JSON object
bool save_json(const std::string& filename, const json& js) {
    try {
        return save_text(filename, js.dump(4));
    } catch (...) { return false; }
}

template <typename T>
inline void to_json(json& js, const vec2<T>& val) {
    js = std::array<T, 2>{{val.x, val.y}};
}
template <typename T>
inline void from_json(const json& js, vec2<T>& val) {
    auto vala = js.get<std::array<T, 2>>();
    val       = {vala[0], vala[1]};
}
template <typename T>
inline void to_json(json& js, const vec3<T>& val) {
    js = std::array<T, 3>{{val.x, val.y, val.z}};
}
template <typename T>
inline void from_json(const json& js, vec3<T>& val) {
    auto vala = js.get<std::array<T, 3>>();
    val       = {vala[0], vala[1], vala[2]};
}
template <typename T>
inline void to_json(json& js, const vec4<T>& val) {
    js = std::array<T, 4>{{val.x, val.y, val.z, val.w}};
}
template <typename T>
inline void from_json(const json& js, vec4<T>& val) {
    auto vala = js.get<std::array<T, 4>>();
    val       = {vala[0], vala[1], vala[2], vala[3]};
}

template <typename T>
inline void to_json(json& js, const frame2<T>& val) {
    js = std::array<vec2<T>, 3>{{val.x, val.y, val.o}};
}
template <typename T>
inline void from_json(const json& js, frame2<T>& val) {
    auto vala = js.get<std::array<vec2<T>, 3>>();
    val       = {vala[0], vala[1], vala[2]};
}
template <typename T>
inline void to_json(json& js, const frame3<T>& val) {
    js = std::array<vec3<T>, 4>{{val.x, val.y, val.z, val.o}};
}
template <typename T>
inline void from_json(const json& js, frame3<T>& val) {
    auto vala = js.get<std::array<vec3<T>, 4>>();
    val       = {vala[0], vala[1], vala[2], vala[3]};
}

template <typename T>
inline void to_json(json& js, const mat2<T>& val) {
    js = std::array<vec2<T>, 2>{{val.x, val.y}};
}
template <typename T>
inline void from_json(const json& js, mat2<T>& val) {
    auto vala = js.get<std::array<vec2<T>, 2>>();
    val       = {vala[0], vala[1]};
}
template <typename T>
inline void to_json(json& js, const mat3<T>& val) {
    js = std::array<vec3<T>, 3>{{val.x, val.y, val.z}};
}
template <typename T>
inline void from_json(const json& js, mat3<T>& val) {
    auto vala = js.get<std::array<vec3<T>, 3>>();
    val       = {vala[0], vala[1], vala[2]};
}
template <typename T>
inline void to_json(json& js, const mat4<T>& val) {
    js = std::array<vec4<T>, 4>{{val.x, val.y, val.z, val.w}};
}
template <typename T>
inline void from_json(const json& js, mat4<T>& val) {
    auto vala = js.get<std::array<vec4<T>, 4>>();
    val       = {vala[0], vala[1], vala[2], vala[3]};
}

template <typename T>
inline void to_json(json& js, const bbox1<T>& val) {
    js = std::array<T, 2>{{val.min, val.max}};
}
template <typename T>
inline void from_json(const json& js, bbox1<T>& val) {
    auto vala = js.get<std::array<T, 2>>();
    val       = {vala[0], vala[1]};
}
template <typename T>
inline void to_json(json& js, const bbox2<T>& val) {
    js = std::array<vec2<T>, 2>{{val.min, val.max}};
}
template <typename T>
inline void from_json(const json& js, bbox2<T>& val) {
    auto vala = js.get<std::array<vec2<T>, 2>>();
    val       = {vala[0], vala[1]};
}
template <typename T>
inline void to_json(json& js, const bbox3<T>& val) {
    js = std::array<vec3<T>, 2>{{val.min, val.max}};
}
template <typename T>
inline void from_json(const json& js, bbox3<T>& val) {
    auto vala = js.get<std::array<vec3<T>, 2>>();
    val       = {vala[0], vala[1]};
}
template <typename T>
inline void to_json(json& js, const bbox4<T>& val) {
    js = std::array<vec4<T>, 2>{{val.min, val.max}};
}
template <typename T>
inline void from_json(const json& js, bbox4<T>& val) {
    auto vala = js.get<std::array<vec4<T>, 2>>();
    val       = {vala[0], vala[1]};
}

template <typename T>
inline void to_json(json& js, const image<T>& val) {
    js         = json::object();
    js["size"] = val.size();
    js["data"] = val.dataref();
}
template <typename T>
inline void from_json(const json& js, image<T>& val) {
    auto size = js.at("size").get<vec2i>();
    auto data = js.at("data").get<std::vector<T>>();
    val       = image<T>{size, data.data()};
}
template <typename T>
inline void to_json(json& js, const volume<T>& val) {
    js         = json::object();
    js["size"] = val.size();
    js["data"] = val.dataref();
}
template <typename T>
inline void from_json(const json& js, volume<T>& val) {
    auto size = js.at("size").get<vec3i>();
    auto data = js.at("data").get<std::vector<T>>();
    val       = volume<T>{size, data.data()};
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// BUILTIN JSON FORMAT
// -----------------------------------------------------------------------------
namespace ygl {

// Serialize struct
void to_json(json& js, const camera& val) {
    static const auto def = camera();
    if (val.name != def.name) js["name"] = val.name;
    if (val.frame != def.frame) js["frame"] = val.frame;
    if (val.ortho != def.ortho) js["ortho"] = val.ortho;
    if (val.film != def.film) js["film"] = val.film;
    if (val.focal != def.focal) js["focal"] = val.focal;
    if (val.focus != def.focus) js["focus"] = val.focus;
    if (val.aperture != def.aperture) js["aperture"] = val.aperture;
}

// Procedural commands for cameras
void from_json_proc(const json& js, camera& val) {
    if (js.count("from") || js.count("to")) {
        auto from = js.value("from", zero3f);
        auto to   = js.value("to", zero3f);
        auto up   = js.value("up", vec3f{0, 1, 0});
        val.frame = lookat_frame(from, to, up);
        val.focus = length(from - to);
    }
}

// Serialize struct
void from_json(const json& js, camera& val) {
    static const auto def = camera();
    val.name              = js.value("name", def.name);
    val.frame             = js.value("frame", def.frame);
    val.film              = js.value("film", def.film);
    val.focal             = js.value("focal", def.focal);
    val.focus             = js.value("focus", def.focus);
    val.aperture          = js.value("aperture", def.aperture);
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const texture& val) {
    static const auto def = texture();
    if (val.name != def.name) js["name"] = val.name;
    if (val.path != def.path) js["path"] = val.path;
    if (val.clamp != def.clamp) js["clamp"] = val.clamp;
    if (val.scale != def.scale) js["scale"] = val.scale;
    if (val.srgb != def.srgb) js["srgb"] = val.srgb;
    if (val.path == "") {
        if (!val.imgf.empty()) js["imgf"] = val.imgf;
        if (!val.imgb.empty()) js["imgb"] = val.imgb;
    }
}

// Procedural commands for textures
void from_json_proc(const json& js, texture& val) {
    auto type = js.value("type", ""s);
    if (type == "") return;
    auto is_hdr = false;
    auto size   = js.value("size", vec2i{512, 512});
    if (type == "grid") {
        val.imgf = make_grid_image4f(size, js.value("tile", 8),
            js.value("c0", vec4f{0.2f, 0.2f, 0.2f, 1}),
            js.value("c1", vec4f{0.8f, 0.8f, 0.8f, 1}));
    } else if (type == "checker") {
        val.imgf = make_checker_image4f(size, js.value("tile", 8),
            js.value("c0", vec4f{0.2f, 0.2f, 0.2f, 1}),
            js.value("c1", vec4f{0.8f, 0.8f, 0.8f, 1}));
    } else if (type == "bump") {
        val.imgf = make_bumpdimple_image4f(size, js.value("tile", 8));
    } else if (type == "uvramp") {
        val.imgf = make_uvramp_image4f(size);
    } else if (type == "uvgrid") {
        val.imgf = make_uvgrid_image4f(size);
    } else if (type == "sky") {
        if (size.x < size.y * 2) size.x = size.y * 2;
        val.imgf = make_sunsky_image4f(size, js.value("sun_angle", pif / 4),
            js.value("turbidity", 3.0f), js.value("has_sun", false),
            js.value("ground_albedo", vec3f{0.7f, 0.7f, 0.7f}));
        is_hdr   = true;
    } else if (type == "noise") {
        val.imgf = make_noise_image4f(
            size, js.value("scale", 1.0f), js.value("wrap", true));
    } else if (type == "fbm") {
        val.imgf = make_fbm_image4f(size, js.value("scale", 1.0f),
            js.value("lacunarity", 2.0f), js.value("gain", 0.5f),
            js.value("octaves", 6), js.value("wrap", true));
    } else if (type == "ridge") {
        val.imgf = make_ridge_image4f(size, js.value("scale", 1.0f),
            js.value("lacunarity", 2.0f), js.value("gain", 0.5f),
            js.value("offset", 1.0f), js.value("octaves", 6),
            js.value("wrap", true));
    } else if (type == "turbulence") {
        val.imgf = make_turbulence_image4f(size, js.value("scale", 1.0f),
            js.value("lacunarity", 2.0f), js.value("gain", 0.5f),
            js.value("octaves", 6), js.value("wrap", true));
    } else {
        throw std::runtime_error("unknown texture type " + type);
    }
    if (js.value("bump_to_normal", false)) {
        val.imgf = bump_to_normal_map(val.imgf, js.value("bump_scale", 1.0f));
        val.srgb = false;
    }
    if (!is_hdr) {
        if (val.srgb) {
            val.imgb = float_to_byte(linear_to_srgb(val.imgf));
        } else {
            val.imgb = float_to_byte(val.imgf);
        }
        val.imgf = {};
    }
    if (val.path == "") {
        auto ext = (is_hdr) ? std::string("hdr") : std::string("png");
        val.path = "textures/" + val.name + "." + ext;
    }
}

// Serialize struct
void from_json(const json& js, texture& val) {
    static const auto def = texture();
    val.name              = js.value("name", def.name);
    val.path              = js.value("path", def.path);
    val.clamp             = js.value("clamp", def.clamp);
    val.scale             = js.value("scale", def.scale);
    val.srgb              = js.value("srgb", def.srgb);
    val.imgf              = js.value("imgf", def.imgf);
    val.imgb              = js.value("imgb", def.imgb);
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const voltexture& val) {
    static const auto def = voltexture();
    if (val.name != def.name) js["name"] = val.name;
    if (val.path != def.path) js["path"] = val.path;
    if (val.clamp != def.clamp) js["clamp"] = val.clamp;
    if (val.path == "") {
        if (!val.vol.empty()) js["vol"] = val.vol;
    }
}

// Procedural commands for textures
void from_json_proc(const json& js, voltexture& val) {
    auto type = js.value("type", ""s);
    if (type == "") return;
    auto size = js.value("size", vec3i{512, 512, 512});
    if (type == "test_volume") {
        val.vol = make_test_volume1f(
            size, js.value("scale", 10.0f), js.value("exponent", 6.0f));
    } else {
        throw std::runtime_error("unknown texture type " + type);
    }
    if (val.path == "") {
        auto ext = std::string("vol");
        val.path = "textures/" + val.name + "." + ext;
    }
}

// Serialize struct
void from_json(const json& js, voltexture& val) {
    static const auto def = voltexture();
    val.name              = js.value("name", def.name);
    val.path              = js.value("path", def.path);
    val.vol               = js.value("vol", def.vol);
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const material& val) {
    static const auto def = material();
    if (val.name != def.name) js["name"] = val.name;
    if (val.base_metallic != def.base_metallic)
        js["base_metallic"] = val.base_metallic;
    if (val.gltf_textures != def.gltf_textures)
        js["gltf_textures"] = val.gltf_textures;
    if (val.double_sided != def.double_sided)
        js["double_sided"] = val.double_sided;
    if (val.ke != def.ke) js["ke"] = val.ke;
    if (val.kd != def.kd) js["kd"] = val.kd;
    if (val.ks != def.ks) js["ks"] = val.ks;
    if (val.kt != def.kt) js["kt"] = val.kt;
    if (val.rs != def.rs) js["rs"] = val.rs;
    if (val.op != def.op) js["op"] = val.op;
    if (val.fresnel != def.fresnel) js["fresnel"] = val.fresnel;
    if (val.refract != def.refract) js["refract"] = val.refract;
    if (val.ke_txt != def.ke_txt) js["ke_txt"] = val.ke_txt->name;
    if (val.kd_txt != def.kd_txt) js["kd_txt"] = val.kd_txt->name;
    if (val.ks_txt != def.ks_txt) js["ks_txt"] = val.ks_txt->name;
    if (val.kt_txt != def.kt_txt) js["kt_txt"] = val.kt_txt->name;
    if (val.rs_txt != def.rs_txt) js["rs_txt"] = val.rs_txt->name;
    if (val.op_txt != def.op_txt) js["op_txt"] = val.op_txt->name;
    if (val.occ_txt != def.occ_txt) js["occ_txt"] = val.occ_txt->name;
    if (val.bump_txt != def.bump_txt) js["bump_txt"] = val.bump_txt->name;
    if (val.disp_txt != def.disp_txt) js["disp_txt"] = val.disp_txt->name;
    if (val.norm_txt != def.norm_txt) js["norm_txt"] = val.norm_txt->name;
    if (val.vd_txt != def.vd_txt) js["vd_txt"] = val.vd_txt->name;
}

// Procedural commands for materials
void from_json_proc(const json& js, material& val) {}

// Serialize struct
void from_json(const json& js, material& val) {
    static const auto def = material();
    val.name              = js.value("name", def.name);
    val.base_metallic     = js.value("base_metallic", def.base_metallic);
    val.gltf_textures     = js.value("gltf_textures", def.gltf_textures);
    val.double_sided      = js.value("double_sided", def.double_sided);
    val.ke                = js.value("ke", def.ke);
    val.kd                = js.value("kd", def.kd);
    val.ks                = js.value("ks", def.ks);
    val.kt                = js.value("kt", def.kt);
    val.rs                = js.value("rs", def.rs);
    val.op                = js.value("op", def.op);
    val.ve                = js.value("ve", def.ve);
    val.va                = js.value("va", def.va);
    val.vd                = js.value("vd", def.vd);
    val.vg                = js.value("vg", def.vg);
    val.fresnel           = js.value("fresnel", def.fresnel);
    val.refract           = js.value("refract", def.refract);
    if (js.count("ke_txt")) {
        val.ke_txt       = new texture();
        val.ke_txt->name = js.at("ke_txt").get<std::string>();
    }
    if (js.count("kd_txt")) {
        val.kd_txt       = new texture();
        val.kd_txt->name = js.at("kd_txt").get<std::string>();
    }
    if (js.count("ks_txt")) {
        val.ks_txt       = new texture();
        val.ks_txt->name = js.at("ks_txt").get<std::string>();
    }
    if (js.count("kt_txt")) {
        val.kt_txt       = new texture();
        val.kt_txt->name = js.at("kt_txt").get<std::string>();
    }
    if (js.count("rs_txt")) {
        val.rs_txt       = new texture();
        val.rs_txt->name = js.at("rs_txt").get<std::string>();
    }
    if (js.count("op_txt")) {
        val.op_txt       = new texture();
        val.op_txt->name = js.at("op_txt").get<std::string>();
    }
    if (js.count("occ_txt")) {
        val.occ_txt       = new texture();
        val.occ_txt->name = js.at("occ_txt").get<std::string>();
    }
    if (js.count("bump_txt")) {
        val.bump_txt       = new texture();
        val.bump_txt->name = js.at("bump_txt").get<std::string>();
    }
    if (js.count("disp_txt")) {
        val.disp_txt       = new texture();
        val.disp_txt->name = js.at("disp_txt").get<std::string>();
    }
    if (js.count("norm_txt")) {
        val.norm_txt       = new texture();
        val.norm_txt->name = js.at("norm_txt").get<std::string>();
    }
    if (js.count("vd_txt")) {
        val.vd_txt       = new voltexture();
        val.vd_txt->name = js.at("vd_txt").get<std::string>();
    }
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const shape& val) {
    static const auto def = shape();
    if (val.name != def.name) js["name"] = val.name;
    if (val.path != def.path) js["path"] = val.path;
    if (val.path == "") {
        if (val.points != def.points) js["points"] = val.points;
        if (val.lines != def.lines) js["lines"] = val.lines;
        if (val.triangles != def.triangles) js["triangles"] = val.triangles;
        if (val.pos != def.pos) js["pos"] = val.pos;
        if (val.norm != def.norm) js["norm"] = val.norm;
        if (val.texcoord != def.texcoord) js["texcoord"] = val.texcoord;
        if (val.color != def.color) js["color"] = val.color;
        if (val.radius != def.radius) js["radius"] = val.radius;
        if (val.tangsp != def.tangsp) js["tangsp"] = val.tangsp;
    }
}

// Procedural commands for materials
void from_json_proc(const json& js, shape& val) {
    auto type = js.value("type", ""s);
    if (type == "") return;
    auto shp = make_shape_data();
    if (type == "quad") {
        shp = make_quad(js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{2, 2}), js.value("uvsize", vec2f{1, 1}),
            true);
    } else if (type == "quady") {
        shp = make_quad(js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{2, 2}), js.value("uvsize", vec2f{1, 1}),
            true);
    } else if (type == "quad_stack") {
        shp = make_quad_stack(js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}), js.value("uvsize", vec2f{1, 1}),
            true);
    } else if (type == "cube") {
        shp = make_cube(js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}), true);
    } else if (type == "cube_rounded") {
        shp = make_cube_rounded(js.value("steps", vec3i{32, 32, 32}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}), js.value("radius", 0.3f), true);
    } else if (type == "sphere") {
        shp = make_sphere(js.value("steps", vec2i{64, 32}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}), true);
    } else if (type == "sphere_cube") {
        shp = make_sphere_cube(js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f), true);
    } else if (type == "sphere_flipcap") {
        shp = make_sphere_flipcap(js.value("steps", vec2i{64, 32}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}),
            js.value("zflip", vec2f{-0.75f, +0.75f}), true);
    } else if (type == "disk") {
        shp = make_disk(js.value("steps", vec2i{32, 16}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}), true);
    } else if (type == "disk_quad") {
        shp = make_disk_quad(js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f), true);
    } else if (type == "disk_bulged") {
        shp = make_disk_bulged(js.value("steps", 32), js.value("size", 2.0f),
            js.value("uvsize", 1.0f), js.value("height", 0.25f), true);
    } else if (type == "cylinder_side") {
        shp = make_cylinder_side(js.value("steps", vec2i{64, 32}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec2f{1, 1}), true);
    } else if (type == "cylinder") {
        shp = make_cylinder(js.value("steps", vec3i{64, 32, 16}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec3f{1, 1, 1}), true);
    } else if (type == "cylinder_rounded") {
        shp = make_cylinder_rounded(js.value("steps", vec3i{64, 32, 16}),
            js.value("size", vec2f{2.0f, 2.0f}),
            js.value("uvsize", vec3f{1, 1, 1}), js.value("radius", 0.15f), true);
    } else if (type == "sphere_geodesic") {
        shp = make_geodesic_sphere(
            js.value("tesselation", 4), js.value("size", 2.0f), true);
    } else if (type == "floor") {
        shp = make_floor(js.value("steps", vec2i{1, 1}),
            js.value("size", vec2f{40, 40}), js.value("uvsize", vec2f{20, 20}),
            true);
    } else if (type == "matball") {
        shp = make_sphere(js.value("steps", vec2i{64, 32}),
            js.value("size", 2.0f), js.value("uvsize", vec2f{1, 1}), true);
    } else if (type == "hairball") {
        auto base = make_sphere_cube(32, js.value("size", 2.0f) * 0.8f, 1, true);
        shp = make_hair(js.value("steps", vec2i{4, 65536}), base.triangles,
            base.pos, base.norm, base.texcoord,
            js.value("length", vec2f{0.2f, 0.2f}),
            js.value("radius", vec2f{0.001f, 0.001f}),
            js.value("noise", vec2f{0, 0}), js.value("clump", vec2f{0, 0}));
    } else if (type == "hairball_interior") {
        shp = make_sphere_cube(32, js.value("size", 2.0f) * 0.8f, 1, true);
    } else if (type == "suzanne") {
        shp = make_suzanne(js.value("size", 2.0f), true);
    } else {
        throw std::runtime_error("unknown shape type " + type);
    }
    if (js.value("flipyz", false)) {
        for (auto& p : shp.pos) p = {p.x, p.z, p.y};
        for (auto& n : shp.norm) n = {n.x, n.z, n.y};
    }
    val.points    = shp.points;
    val.lines     = shp.lines;
    val.triangles = shp.triangles;
    val.pos       = shp.pos;
    val.norm      = shp.norm;
    val.texcoord  = shp.texcoord;
    val.radius    = shp.radius;
    if (val.path == "") val.path = "meshes/" + val.name + ".ply";
}

// Serialize struct
void from_json(const json& js, shape& val) {
    static const auto def = shape();
    val.name              = js.value("name", def.name);
    val.path              = js.value("path", def.path);
    val.points            = js.value("points", def.points);
    val.lines             = js.value("lines", def.lines);
    val.triangles         = js.value("triangles", def.triangles);
    val.pos               = js.value("pos", def.pos);
    val.norm              = js.value("norm", def.norm);
    val.texcoord          = js.value("texcoord", def.texcoord);
    val.color             = js.value("color", def.color);
    val.radius            = js.value("radius", def.radius);
    val.tangsp            = js.value("tangsp", def.tangsp);
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const subdiv& val) {
    static const auto def = subdiv();
    if (val.name != def.name) js["name"] = val.name;
    if (val.path != def.path) js["path"] = val.path;
    if (val.level != def.level) js["level"] = val.level;
    if (val.catmull_clark != def.catmull_clark)
        js["catmull_clark"] = val.catmull_clark;
    if (val.compute_normals != def.compute_normals)
        js["compute_normals"] = val.compute_normals;
    if (val.path == "") {
        if (val.quads_pos != def.quads_pos) js["quads_pos"] = val.quads_pos;
        if (val.quads_texcoord != def.quads_texcoord)
            js["quads_texcoord"] = val.quads_texcoord;
        if (val.quads_color != def.quads_color)
            js["quads_color"] = val.quads_color;
        if (val.pos != def.pos) js["pos"] = val.pos;
        if (val.texcoord != def.texcoord) js["texcoord"] = val.texcoord;
        if (val.color != def.color) js["color"] = val.color;
    }
}

// Procedural commands for subdivs
void from_json_proc(const json& js, subdiv& val) {
    auto type = js.value("type", ""s);
    if (type == "") return;
    auto shp = make_shape_data();
    if (type == "cube") {
        shp = make_fvcube(js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
    } else if (type == "cube_open") {
        shp = make_fvcube(js.value("steps", vec3i{1, 1, 1}),
            js.value("size", vec3f{2, 2, 2}),
            js.value("uvsize", vec3f{1, 1, 1}));
        shp.quads_pos.pop_back();
        shp.quads_norm.pop_back();
        shp.quads_texcoord.pop_back();
    } else if (type == "suzanne") {
        shp = make_suzanne(js.value("size", 2.0f), false);
        std::swap(shp.quads_pos, shp.quads);
    } else {
        throw std::runtime_error("unknown shape type " + type);
    }
    val.quads_pos      = shp.quads_pos;
    val.pos            = shp.pos;
    val.quads_texcoord = shp.quads_texcoord;
    val.texcoord       = shp.texcoord;
    if (val.path == "") val.path = "meshes/" + val.name + ".obj";
}

// Serialize struct
void from_json(const json& js, subdiv& val) {
    static const auto def = subdiv();
    val.name              = js.value("name", def.name);
    val.path              = js.value("path", def.path);
    val.level             = js.value("level", def.level);
    val.catmull_clark     = js.value("catmull_clark", def.catmull_clark);
    val.compute_normals   = js.value("compute_normals", def.compute_normals);
    val.quads_pos         = js.value("quads_pos", def.quads_pos);
    val.quads_texcoord    = js.value("quads_texcoord", def.quads_texcoord);
    val.quads_color       = js.value("quads_color", def.quads_color);
    val.pos               = js.value("pos", def.pos);
    val.texcoord          = js.value("texcoord", def.texcoord);
    val.color             = js.value("color", def.color);
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const instance& val) {
    static const auto def = instance();
    if (val.name != def.name) js["name"] = val.name;
    if (val.frame != def.frame) js["frame"] = val.frame;
    if (val.shp != def.shp) js["shp"] = val.shp->name;
    if (val.mat != def.mat) js["mat"] = val.mat->name;
    if (val.sbd != def.sbd) js["sbd"] = val.sbd->name;
}

// Procedural commands for instances
void from_json_proc(const json& js, instance& val) {
    if (js.count("from")) {
        auto from = js.value("from", zero3f);
        auto to   = js.value("to", zero3f);
        auto up   = js.value("up", vec3f{0, 1, 0});
        val.frame = lookat_frame(from, to, up, true);
    }
    if (js.count("translation") || js.count("rotation") || js.count("scale")) {
        auto translation = js.value("translation", zero3f);
        auto rotation    = js.value("rotation", zero4f);
        auto scaling     = js.value("scale", vec3f{1, 1, 1});
        val.frame = translation_frame(translation) * scaling_frame(scaling) *
                    rotation_frame(xyz(rotation), rotation.w);
    }
}

// Serialize struct
void from_json(const json& js, instance& val) {
    static const auto def = instance();
    val.name              = js.value("name", def.name);
    val.frame             = js.value("frame", def.frame);
    if (js.count("shp")) {
        val.shp       = new shape();
        val.shp->name = js.at("shp").get<std::string>();
    }
    if (js.count("mat")) {
        val.mat       = new material();
        val.mat->name = js.at("mat").get<std::string>();
    }
    if (js.count("sbd")) {
        val.sbd       = new subdiv();
        val.sbd->name = js.at("sbd").get<std::string>();
    }
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const environment& val) {
    static const auto def = environment();
    if (val.name != def.name) js["name"] = val.name;
    if (val.frame != def.frame) js["frame"] = val.frame;
    if (val.ke != def.ke) js["ke"] = val.ke;
    if (val.ke_txt != def.ke_txt) js["ke_txt"]["name"] = val.ke_txt->name;
}

// Procedural commands for materials
void from_json_proc(const json& js, environment& val) {
    if (js.count("rotation")) {
        auto rotation = js.value("rotation", zero4f);
        val.frame     = rotation_frame(xyz(rotation), rotation.w);
    }
}

// Serialize struct
void from_json(const json& js, environment& val) {
    static const auto def = environment();
    val.name              = js.value("name", def.name);
    val.frame             = js.value("frame", def.frame);
    val.ke                = js.value("ke", def.ke);
    if (js.count("ke_txt")) {
        val.ke_txt       = new texture();
        val.ke_txt->name = js.at("ke_txt").get<std::string>();
    }
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const node& val) {
    static const auto def = node();
    if (val.name != def.name) js["name"] = val.name;
    if (val.local != def.local) js["local"] = val.local;
    if (val.translation != def.translation) js["translation"] = val.translation;
    if (val.rotation != def.rotation) js["rotation"] = val.rotation;
    if (val.scale != def.scale) js["scale"] = val.scale;
    if (val.weights != def.weights) js["weights"] = val.weights;
    if (val.parent != def.parent) js["parent"] = val.parent->name;
    if (val.cam != def.cam) js["cam"] = val.cam->name;
    if (val.ist != def.ist) js["ist"] = val.ist->name;
    if (val.env != def.env) js["env"] = val.env->name;
}

// Procedural commands for nodes
void from_json_proc(const json& js, node& val) {
    if (js.count("from")) {
        auto from = js.value("from", zero3f);
        auto to   = js.value("to", zero3f);
        auto up   = js.value("up", vec3f{0, 1, 0});
        val.local = lookat_frame(from, to, up, true);
    }
}

// Serialize struct
void from_json(const json& js, node& val) {
    static const auto def = node();
    val.name              = js.value("name", def.name);
    val.local             = js.value("local", def.local);
    val.translation       = js.value("translation", def.translation);
    val.rotation          = js.value("rotation", def.rotation);
    val.scale             = js.value("scale", def.scale);
    val.weights           = js.value("weights", def.weights);
    if (js.count("parent")) {
        val.parent       = new node();
        val.parent->name = js.at("parent").get<std::string>();
    }
    if (js.count("cam")) {
        val.cam       = new camera();
        val.cam->name = js.at("cam").get<std::string>();
    }
    if (js.count("ist")) {
        val.ist       = new instance();
        val.ist->name = js.at("ist").get<std::string>();
    }
    if (js.count("env")) {
        val.env       = new environment();
        val.env->name = js.at("env").get<std::string>();
    }
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize enum
void to_json(json& js, const animation_type& val) {
    static auto names = std::map<animation_type, std::string>{
        {animation_type::linear, "linear"},
        {animation_type::step, "step"},
        {animation_type::bezier, "bezier"},
    };
    js = names.at(val);
}

// Serialize enum
void from_json(const json& js, animation_type& val) {
    static auto names = std::map<std::string, animation_type>{
        {"linear", animation_type::linear},
        {"step", animation_type::step},
        {"bezier", animation_type::bezier},
    };
    val = names.at(js.get<std::string>());
}

// Serialize struct
void to_json(json& js, const animation& val) {
    static const auto def = animation();
    if (val.name != def.name) js["name"] = val.name;
    if (val.path != def.path) js["path"] = val.path;
    if (val.group != def.group) js["group"] = val.group;
    if (val.type != def.type) js["type"] = val.type;
    if (val.path == "") {
        if (val.times != def.times) js["times"] = val.times;
        if (val.translation != def.translation)
            js["translation"] = val.translation;
        if (val.rotation != def.rotation) js["rotation"] = val.rotation;
        if (val.scale != def.scale) js["scale"] = val.scale;
    }
    if (val.targets != def.targets) {
        js["targets"] = json::array();
        for (auto v : val.targets) js["targets"].push_back(v->name);
    }
}

// Procedural commands for animations
void from_json_proc(const json& js, animation& val) {
    if (js.count("rotation_axisangle")) {
        for (auto& j : js.at("rotation_axisangle")) {
            val.rotation.push_back(rotation_quat(j.get<vec4f>()));
        }
    }
}

// Serialize struct
void from_json(const json& js, animation& val) {
    static const auto def = animation();
    val.name              = js.value("name", def.name);
    val.path              = js.value("path", def.path);
    val.group             = js.value("group", def.group);
    val.type              = js.value("type", def.type);
    val.times             = js.value("times", def.times);
    val.translation       = js.value("translation", def.translation);
    val.rotation          = js.value("rotation", def.rotation);
    val.scale             = js.value("scale", def.scale);
    for (auto& j : js.value("targets", json::array())) {
        val.targets.push_back(new node());
        val.targets.back()->name = j.get<std::string>();
    }
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), val);
}

// Serialize struct
void to_json(json& js, const scene& val) {
    static const auto def = scene();
    if (val.name != def.name) js["name"] = val.name;
    if (!val.cameras.empty()) {
        js["cameras"] = json::array();
        for (auto v : val.cameras) js["cameras"].push_back(json(*v));
    }
    if (!val.textures.empty()) {
        js["textures"] = json::array();
        for (auto v : val.textures) js["textures"].push_back(json(*v));
    }
    if (!val.materials.empty()) {
        js["materials"] = json::array();
        for (auto v : val.materials) js["materials"].push_back(json(*v));
    }
    if (!val.shapes.empty()) {
        js["shapes"] = json::array();
        for (auto v : val.shapes) js["shapes"].push_back(json(*v));
    }
    if (!val.subdivs.empty()) {
        js["subdivs"] = json::array();
        for (auto v : val.subdivs) js["subdivs"].push_back(json(*v));
    }
    if (!val.instances.empty()) {
        js["instances"] = json::array();
        for (auto v : val.instances) js["instances"].push_back(json(*v));
    }
    if (!val.environments.empty()) {
        js["environments"] = json::array();
        for (auto v : val.environments) js["environments"].push_back(json(*v));
    }
    if (!val.nodes.empty()) {
        js["nodes"] = json::array();
        for (auto v : val.nodes) js["nodes"].push_back(json(*v));
    }
    if (!val.animations.empty()) {
        js["animations"] = json::array();
        for (auto v : val.animations) js["animations"].push_back(*v);
    }
}
void to_json(json& js, const scene* val) {
    if (!val) {
        js = json();
        return;
    }
    to_json(js, *val);
}

template <typename T>
static std::unordered_map<std::string, T*> make_named_map(
    const std::vector<T*>& elems) {
    auto map = std::unordered_map<std::string, T*>();
    for (auto elem : elems) map[elem->name] = elem;
    return map;
};

// Procedural commands for scenes
void from_json_proc(const json& js, scene& val) {
    if (js.count("random_instances")) {
        auto& jjs  = js.at("random_instances");
        auto  num  = jjs.value("num", 100);
        auto  seed = jjs.value("seed", 13);
        auto  base = new instance();
        from_json(jjs.at("base"), *base);
        auto ists = std::vector<instance*>();
        for (auto& j : jjs.at("instances")) {
            ists.push_back(new instance());
            from_json(j, *ists.back());
        }

        auto pos                      = std::vector<vec3f>();
        auto norm                     = std::vector<vec3f>();
        auto texcoord                 = std::vector<vec2f>();
        std::tie(pos, norm, texcoord) = sample_triangles_points(
            base->shp->triangles, base->shp->pos, base->shp->norm,
            base->shp->texcoord, num, seed);

        auto nmap = std::unordered_map<instance*, int>();
        for (auto ist : ists) nmap[ist] = 0;
        auto rng = make_rng(seed, 17);
        for (auto i = 0; i < num; i++) {
            auto ist = ists.at(rand1i(rng, (int)ists.size() - 1));
            nmap[ist] += 1;
            val.instances.push_back(new instance());
            val.instances.back()->name  = ist->name + std::to_string(nmap[ist]);
            val.instances.back()->frame = base->frame *
                                          translation_frame(pos[i]) * ist->frame;
            val.instances.back()->shp = ist->shp;
            val.instances.back()->mat = ist->mat;
            val.instances.back()->sbd = ist->sbd;
        }
    }
}

// Load a scene
scene* load_json_scene(
    const std::string& filename, bool load_textures, bool skip_missing) {
    auto scn = (scene*)nullptr;
    if (!load_json_scene(filename, scn, load_textures, skip_missing))
        return nullptr;
    return scn;
}

// Load a scene in the builtin JSON format.
bool load_json_scene(const std::string& filename, scene*& scn,
    bool load_textures, bool skip_missing) {
    // reset
    reset_scene_data(scn);
    // initialize
    scn = new scene();

    // load jsonz
    auto js = json();
    if (!load_json(filename, js)) return false;

    // parse json scene
    scn->name = js.value("name", ""s);
    for (auto& j : js.value("cameras", json::array())) {
        scn->cameras.push_back(new camera());
        from_json(j, *scn->cameras.back());
    }
    for (auto& j : js.value("textures", json::array())) {
        scn->textures.push_back(new texture());
        from_json(j, *scn->textures.back());
    }
    for (auto& j : js.value("materials", json::array())) {
        scn->materials.push_back(new material());
        from_json(j, *scn->materials.back());
    }
    for (auto& j : js.value("shapes", json::array())) {
        scn->shapes.push_back(new shape());
        from_json(j, *scn->shapes.back());
    }
    for (auto& j : js.value("subdivs", json::array())) {
        scn->subdivs.push_back(new subdiv());
        from_json(j, *scn->subdivs.back());
    }
    for (auto& j : js.value("instances", json::array())) {
        scn->instances.push_back(new instance());
        from_json(j, *scn->instances.back());
    }
    for (auto& j : js.value("environments", json::array())) {
        scn->environments.push_back(new environment());
        from_json(j, *scn->environments.back());
    }
    for (auto& j : js.value("voltextures", json::array())) {
        scn->voltextures.push_back(new voltexture());
        from_json(j, *scn->voltextures.back());
    }
    for (auto& j : js.value("nodes", json::array())) {
        scn->nodes.push_back(new node());
        from_json(j, *scn->nodes.back());
    }
    for (auto& j : js.value("animations", json::array())) {
        scn->animations.push_back(new animation());
        from_json(j, *scn->animations.back());
    }
    if (js.count("!!proc")) from_json_proc(js.at("!!proc"), *scn);

    // fix references
    auto cmap    = make_named_map(scn->cameras);
    auto tmap    = make_named_map(scn->textures);
    auto vmap    = make_named_map(scn->voltextures);
    auto mmap    = make_named_map(scn->materials);
    auto smap    = make_named_map(scn->shapes);
    auto rmap    = make_named_map(scn->subdivs);
    auto imap    = make_named_map(scn->instances);
    auto emap    = make_named_map(scn->environments);
    auto nmap    = make_named_map(scn->nodes);
    auto fix_ref = [](auto& map, auto& elems, auto& ref) {
        if (!ref) return;
        auto name = ref->name;
        if (map.find(ref->name) != map.end()) {
            ref = map.at(name);
        } else {
            map[ref->name] = ref;
            elems.push_back(ref);
        }
    };
    for (auto anm : scn->animations) {
        for (auto& nde : anm->targets) fix_ref(nmap, scn->nodes, nde);
    }
    for (auto nde : scn->nodes) {
        fix_ref(nmap, scn->nodes, nde->parent);
        fix_ref(cmap, scn->cameras, nde->cam);
        fix_ref(imap, scn->instances, nde->ist);
        fix_ref(emap, scn->environments, nde->env);
    }
    for (auto env : scn->environments) {
        fix_ref(tmap, scn->textures, env->ke_txt);
    }
    for (auto ist : scn->instances) {
        fix_ref(mmap, scn->materials, ist->mat);
        fix_ref(smap, scn->shapes, ist->shp);
        fix_ref(rmap, scn->subdivs, ist->sbd);
    }
    for (auto mat : scn->materials) {
        fix_ref(tmap, scn->textures, mat->ke_txt);
        fix_ref(tmap, scn->textures, mat->kd_txt);
        fix_ref(tmap, scn->textures, mat->ks_txt);
        fix_ref(tmap, scn->textures, mat->kt_txt);
        fix_ref(tmap, scn->textures, mat->op_txt);
        fix_ref(tmap, scn->textures, mat->rs_txt);
        fix_ref(tmap, scn->textures, mat->occ_txt);
        fix_ref(tmap, scn->textures, mat->norm_txt);
        fix_ref(tmap, scn->textures, mat->bump_txt);
        fix_ref(tmap, scn->textures, mat->disp_txt);
        fix_ref(vmap, scn->voltextures, mat->vd_txt);
    }

    // load meshes
    auto dirname = get_dirname(filename);
    for (auto shp : scn->shapes) {
        if (shp->path == "" || !shp->pos.empty()) continue;
        auto filename = normalize_path(dirname + "/" + shp->path);
        if (!load_mesh(filename, shp->points, shp->lines, shp->triangles,
                shp->pos, shp->norm, shp->texcoord, shp->color, shp->radius)) {
            if (!skip_missing) return false;
        }
    }

    // load suddivs
    for (auto sbd : scn->subdivs) {
        if (sbd->path == "" || !sbd->pos.empty()) continue;
        auto filename   = normalize_path(dirname + "/" + sbd->path);
        auto quads_norm = std::vector<vec4i>();
        auto norm       = std::vector<vec3f>();
        if (!load_fvmesh(filename, sbd->quads_pos, sbd->pos, quads_norm, norm,
                sbd->quads_texcoord, sbd->texcoord, sbd->quads_color,
                sbd->color)) {
            if (!skip_missing) return false;
        }
    }

    // skip textures
    if (load_textures) {
        if (!load_scene_textures(scn, dirname, skip_missing, false))
            return false;
    }

    // fix scene
    if (scn->name == "") scn->name = get_filename(filename);
    add_missing_cameras(scn);
    add_missing_materials(scn);
    add_missing_names(scn);
    update_transforms(scn);

    // done
    return true;
}

// Save a scene in the builtin JSON format.
bool save_json_scene(const std::string& filename, const scene* scn,
    bool save_textures, bool skip_missing) {
    // save json
    auto js = json();
    try {
        js = json(scn);
    } catch (...) { return false; }
    if (!save_json(filename, js)) return false;

    // save meshes
    auto dirname = get_dirname(filename);
    for (auto& shp : scn->shapes) {
        if (shp->path == "") continue;
        auto filename = normalize_path(dirname + "/" + shp->path);
        if (!save_mesh(filename, shp->points, shp->lines, shp->triangles,
                shp->pos, shp->norm, shp->texcoord, shp->color, shp->radius)) {
            if (!skip_missing) return false;
        }
    }

    // save subdivs
    for (auto& sbd : scn->subdivs) {
        if (sbd->path == "") continue;
        auto filename = normalize_path(dirname + "/" + sbd->path);
        if (!save_fvmesh(filename, sbd->quads_pos, sbd->pos, {}, {},
                sbd->quads_texcoord, sbd->texcoord, sbd->quads_color,
                sbd->color)) {
            if (!skip_missing) return false;
        }
    }

    // skip textures
    if (save_textures) {
        if (!save_scene_textures(scn, dirname, skip_missing)) return false;
    }

    // done
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// OBJ CONVERSION
// -----------------------------------------------------------------------------
namespace ygl {

inline bool operator==(obj_vertex a, obj_vertex b) {
    return a.pos == b.pos && a.texcoord == b.texcoord && a.norm == b.norm;
}

struct obj_vertex_hash {
    size_t operator()(const obj_vertex& v) const {
        auto vh = std::hash<int>();
        auto h  = (size_t)0;
        for (auto i = 0; i < 3; i++)
            h ^= vh((&v.pos)[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// prepare obj line (remove comments and normalize whitespace)
void normalize_obj_line(char* s) {
    while (*s) {
        if (*s == '\t' || *s == '\r' || *s == '\n') {
            *s++ = ' ';
        } else if (*s == '#') {
            *s = 0;
            break;
        } else {
            s++;
        }
    }
}

// parse stream
inline int parse_int(char*& s) {
    if (!*s) return 0;
    while (*s == ' ') s++;
    if (!*s) return 0;
    auto val = 0;
    auto sn  = (*s == '-') ? -1 : 1;
    if (*s == '-' || *s == '+') s++;
    while (*s >= '0' && *s <= '9') val = val * 10 + (*s++ - '0');
    val *= sn;
    return val;
}
inline bool   parse_bool(char*& s) { return (bool)parse_int(s); }
inline double parse_double(char*& s) {
    if (!*s) return 0;
    while (*s == ' ') s++;
    auto val      = 0.0;
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
    val = (double)mantissa;
    if (fractional)
        val += fractional * std::pow(10.0, -(double)fractional_length);
    if (exponent) val *= std::pow(10.0, (double)exponent);
    return val;
}
inline float       parse_float(char*& s) { return (float)parse_double(s); }
inline std::string parse_string(char*& s) {
    if (!*s) return "";
    char buf[4096];
    auto valb = buf;
    while (*s == ' ') s++;
    while (*s && *s != ' ') *valb++ = *s++;
    *valb = 0;
    return buf;
}
inline vec2f parse_vec2f(char*& s) { return {parse_float(s), parse_float(s)}; }
inline vec3f parse_vec3f(char*& s) {
    return {parse_float(s), parse_float(s), parse_float(s)};
}
inline vec4f parse_vec4f(char*& s) {
    return {parse_float(s), parse_float(s), parse_float(s), parse_float(s)};
}
inline vec2i parse_vec2i(char*& s) { return {parse_int(s), parse_int(s)}; }
inline vec3i parse_vec3i(char*& s) {
    return {parse_int(s), parse_int(s), parse_int(s)};
}
inline vec4i parse_vec4i(char*& s) {
    return {parse_int(s), parse_int(s), parse_int(s), parse_int(s)};
}
inline frame3f parse_frame3f(char*& s) {
    if (!*s) return identity_frame3f;
    return {parse_vec3f(s), parse_vec3f(s), parse_vec3f(s), parse_vec3f(s)};
}

inline obj_vertex parse_obj_vertex(char*& s) {
    auto val = obj_vertex{0, 0, 0};
    val.pos  = parse_int(s);
    if (*s == '/') {
        s++;
        if (*s == '/') {
            s++;
            val.norm = parse_int(s);
        } else {
            val.texcoord = parse_int(s);
            if (*s == '/') {
                s++;
                val.norm = parse_int(s);
            }
        }
    }
    return val;
}

// Input for OBJ textures
inline obj_texture_info parse_obj_texture_info(char*& s) {
    // initialize
    auto info = obj_texture_info();

    // get tokens
    auto tokens = std::vector<std::string>();
    while (true) {
        auto v = parse_string(s);
        if (v == "") break;
        tokens.push_back(v);
    }
    if (tokens.empty()) return info;

    // texture name
    info.path = normalize_path(tokens.back());

    // texture options
    auto last = std::string();
    for (auto i = 0; i < tokens.size() - 1; i++) {
        if (tokens[i] == "-bm") info.scale = atof(tokens[i + 1].c_str());
        if (tokens[i] == "-clamp") info.clamp = true;
    }

    return info;
}

// Load obj materials
bool load_mtl(
    const std::string& filename, const obj_callbacks& cb, bool flip_tr) {
    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // currently parsed material
    auto mat   = obj_material();
    auto first = true;

    // read the file line by line
    char buf[4096];
    while (fgets(buf, 4096, fs)) {
        // line
        normalize_obj_line(buf);
        auto ss = buf;

        // get command
        auto cmd = parse_string(ss);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "newmtl") {
            if (!first && cb.material) cb.material(mat);
            first    = false;
            mat      = obj_material();
            mat.name = parse_string(ss);
        } else if (cmd == "illum") {
            mat.illum = parse_int(ss);
        } else if (cmd == "Ke") {
            mat.ke = parse_vec3f(ss);
        } else if (cmd == "Kd") {
            mat.kd = parse_vec3f(ss);
        } else if (cmd == "Ks") {
            mat.ks = parse_vec3f(ss);
        } else if (cmd == "Kt") {
            mat.kt = parse_vec3f(ss);
        } else if (cmd == "Tf") {
            mat.kt = {-1, -1, -1};
            mat.kt = parse_vec3f(ss);
            if (mat.kt.y < 0) mat.kt = {mat.kt.x, mat.kt.x, mat.kt.x};
            if (flip_tr) mat.kt = vec3f{1, 1, 1} - mat.kt;
        } else if (cmd == "Tr") {
            auto tr = vec3f{-1, -1, -1};
            tr      = parse_vec3f(ss);
            if (tr.y < 0) tr = {tr.x, tr.x, tr.x};
            mat.op = (tr.x + tr.y + tr.z) / 3;
            if (flip_tr) mat.op = 1 - mat.op;
        } else if (cmd == "Ns") {
            mat.ns = parse_float(ss);
            mat.rs = pow(2 / (mat.ns + 2), 1 / 4.0f);
            if (mat.rs < 0.01f) mat.rs = 0;
            if (mat.rs > 0.99f) mat.rs = 1;
        } else if (cmd == "d") {
            mat.op = parse_float(ss);
        } else if (cmd == "Pr" || cmd == "rs") {
            mat.rs = parse_float(ss);
        } else if (cmd == "map_Ke") {
            mat.ke_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_Kd") {
            mat.kd_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_Ks") {
            mat.ks_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_Tr") {
            mat.kt_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_d" || cmd == "map_Tr") {
            mat.op_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_Pr" || cmd == "map_rs") {
            mat.rs_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_occ" || cmd == "occ") {
            mat.occ_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_bump" || cmd == "bump") {
            mat.bump_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_disp" || cmd == "disp") {
            mat.disp_txt = parse_obj_texture_info(ss);
        } else if (cmd == "map_norm" || cmd == "norm") {
            mat.norm_txt = parse_obj_texture_info(ss);
        } else if (cmd == "Ve") {
            mat.ve = parse_vec3f(ss);
        } else if (cmd == "Va") {
            mat.va = parse_vec3f(ss);
        } else if (cmd == "Vd") {
            mat.vd = parse_vec3f(ss);
        } else if (cmd == "Vg") {
            mat.vg = parse_float(ss);
        } else if (cmd == "map_Vd") {
            mat.vd_txt = parse_obj_texture_info(ss);
        }
    }

    // issue current material
    if (!first && cb.material) cb.material(mat);

    // clone
    fclose(fs);

    // done
    return true;
}

// Load obj extensions
bool load_objx(const std::string& filename, const obj_callbacks& cb) {
    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // read the file line by line
    char buf[4096];
    while (fgets(buf, 4096, fs)) {
        // line
        normalize_obj_line(buf);
        auto ss = buf;

        // get command
        auto cmd = parse_string(ss);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "c") {
            auto cam     = obj_camera();
            cam.name     = parse_string(ss);
            cam.ortho    = parse_bool(ss);
            cam.film     = parse_vec2f(ss);
            cam.focal    = parse_float(ss);
            cam.focus    = parse_float(ss);
            cam.aperture = parse_float(ss);
            cam.frame    = parse_frame3f(ss);
            if (cb.camera) cb.camera(cam);
        } else if (cmd == "e") {
            auto env        = obj_environment();
            env.name        = parse_string(ss);
            env.ke          = parse_vec3f(ss);
            env.ke_txt.path = parse_string(ss);
            if (env.ke_txt.path == "\"\"") env.ke_txt.path = "";
            if (cb.environmnet) cb.environmnet(env);
        } else {
            // unused
        }
    }

    // close file
    fclose(fs);

    // done
    return true;
}

// Load obj scene
bool load_obj(const std::string& filename, const obj_callbacks& cb,
    bool flip_texcoord, bool flip_tr) {
    // open file
    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // track vertex size
    auto vert_size = obj_vertex();
    auto verts     = std::vector<obj_vertex>();  // buffer to avoid reallocation

    // read the file line by line
    char buf[4096];
    while (fgets(buf, 4096, fs)) {
        // prepare to parse
        normalize_obj_line(buf);
        auto ss = buf;

        // get command
        auto cmd = parse_string(ss);
        if (cmd == "") continue;

        // possible token values
        if (cmd == "v") {
            if (cb.vert) cb.vert(parse_vec3f(ss));
            vert_size.pos += 1;
        } else if (cmd == "vn") {
            if (cb.norm) cb.norm(parse_vec3f(ss));
            vert_size.norm += 1;
        } else if (cmd == "vt") {
            auto v = parse_vec2f(ss);
            if (flip_texcoord) v.y = 1 - v.y;
            if (cb.texcoord) cb.texcoord(v);
            vert_size.texcoord += 1;
        } else if (cmd == "f" || cmd == "l" || cmd == "p") {
            verts.clear();
            while (true) {
                auto vert = parse_obj_vertex(ss);
                if (!vert.pos) break;
                if (vert.pos < 0) vert.pos = vert_size.pos + vert.pos + 1;
                if (vert.texcoord < 0)
                    vert.texcoord = vert_size.texcoord + vert.texcoord + 1;
                if (vert.norm < 0) vert.norm = vert_size.norm + vert.norm + 1;
                verts.push_back(vert);
            }
            if (cmd == "f" && cb.face) cb.face(verts);
            if (cmd == "l" && cb.line) cb.line(verts);
            if (cmd == "p" && cb.point) cb.point(verts);
        } else if (cmd == "o") {
            if (cb.object) cb.object(parse_string(ss));
        } else if (cmd == "usemtl") {
            if (cb.usemtl) cb.usemtl(parse_string(ss));
        } else if (cmd == "g") {
            if (cb.group) cb.group(parse_string(ss));
        } else if (cmd == "s") {
            if (cb.smoothing) cb.smoothing(parse_string(ss));
        } else if (cmd == "mtllib") {
            auto mtlname = parse_string(ss);
            if (cb.mtllib) cb.mtllib(mtlname);
            auto mtlpath = get_dirname(filename) + "/" + mtlname;
            if (!load_mtl(mtlpath, cb, flip_tr)) {
                fclose(fs);
                return false;
            }
        } else {
            // unused
        }
    }

    // close file
    fclose(fs);

    // parse extensions if presents
    auto extname    = replace_extension(filename, "objx");
    auto ext_exists = exists_file(extname);
    if (ext_exists) {
        if (!load_objx(extname, cb)) return false;
    }

    // done
    return true;
}

// Load a scene
scene* load_obj_scene(const std::string& filename, bool load_textures,
    bool skip_missing, bool split_shapes) {
    auto scn = (scene*)nullptr;
    if (!load_obj_scene(
            filename, scn, load_textures, skip_missing, split_shapes))
        return nullptr;
    return scn;
}

// Loads an OBJ
bool load_obj_scene(const std::string& filename, scene*& scn,
    bool load_textures, bool skip_missing, bool split_shapes) {
    reset_scene_data(scn);
    scn = new scene();

    // splitting policy
    auto split_material  = split_shapes;
    auto split_group     = split_shapes;
    auto split_smoothing = split_shapes;

    // current parsing values
    auto matname   = std::string();
    auto oname     = std::string();
    auto gname     = std::string();
    auto smoothing = true;
    auto ist       = (instance*)nullptr;

    // vertices
    auto opos      = std::deque<vec3f>();
    auto onorm     = std::deque<vec3f>();
    auto otexcoord = std::deque<vec2f>();

    // object maps
    auto tmap = std::unordered_map<std::string, texture*>();
    auto vmap = std::unordered_map<std::string, voltexture*>();
    auto mmap = std::unordered_map<std::string, material*>();

    // vertex maps
    auto name_map     = std::unordered_map<std::string, int>();
    auto vert_map     = std::unordered_map<obj_vertex, int, obj_vertex_hash>();
    auto pos_map      = std::unordered_map<int, int>();
    auto norm_map     = std::unordered_map<int, int>();
    auto texcoord_map = std::unordered_map<int, int>();

    // add object if needed
    auto is_instance_empty = [](instance* ist) {
        if (ist->sbd) {
            return ist->sbd->pos.empty();
        } else if (ist->shp) {
            return ist->shp->pos.empty();
        } else {
            return true;
        }
    };
    auto add_instance = [&](scene* scn, const std::string& objname,
                            const std::string& matname,
                            const std::string& groupname, bool smoothing) {
        if (scn->instances.empty() || objname != scn->instances.back()->name ||
            !is_instance_empty(scn->instances.back())) {
            auto ist = new instance();
            scn->instances.push_back(ist);
            ist->shp = new shape();
            scn->shapes.push_back(ist->shp);
        }
        name_map[objname] += 1;
        auto name = (name_map[objname] == 1) ?
                        objname :
                        objname + "_" + std::to_string(name_map[objname] - 1);
        if (objname == "") name = "object" + name;
        auto ist  = scn->instances.back();
        ist->name = name;
        if (ist->shp) ist->shp->name = ist->name;
        if (ist->sbd) ist->sbd->name = ist->name;
        if (matname != "") {
            auto it = mmap.find(matname);
            if (it == mmap.end())
                throw std::runtime_error("missing material " + matname);
            ist->mat = it->second;
        }
        vert_map.clear();
        pos_map.clear();
        norm_map.clear();
        texcoord_map.clear();
        return ist;
    };
    // Parse texture options and name
    auto add_texture = [scn, &tmap](const obj_texture_info& info, bool srgb) {
        if (info.path == "") return (texture*)nullptr;
        if (tmap.find(info.path) != tmap.end()) { return tmap.at(info.path); }

        // create texture
        auto txt   = new texture();
        txt->name  = info.path;
        txt->path  = info.path;
        txt->clamp = info.clamp;
        txt->scale = info.scale;
        txt->srgb  = srgb && !is_hdr_filename(info.path);
        scn->textures.push_back(txt);
        tmap[info.path] = txt;

        return txt;
    };
    // Parse texture options and name
    auto add_voltexture = [scn, &vmap](const obj_texture_info& info, bool srgb) {
        if (info.path == "") return (voltexture*)nullptr;
        if (vmap.find(info.path) != vmap.end()) { return vmap.at(info.path); }

        // create texture
        auto txt  = new voltexture();
        txt->name = info.path;
        txt->path = info.path;
        scn->voltextures.push_back(txt);
        vmap[info.path] = txt;

        return txt;
    };
    // Add  vertices to the current shape
    auto add_verts = [&](const std::vector<obj_vertex>& verts) {
        for (auto& vert : verts) {
            auto it = vert_map.find(vert);
            if (it != vert_map.end()) continue;
            auto nverts = (int)ist->shp->pos.size();
            vert_map.insert(it, {vert, nverts});
            if (vert.pos) ist->shp->pos.push_back(opos.at(vert.pos - 1));
            if (vert.texcoord)
                ist->shp->texcoord.push_back(otexcoord.at(vert.texcoord - 1));
            if (vert.norm) ist->shp->norm.push_back(onorm.at(vert.norm - 1));
        }
    };

    // current objet
    ist = add_instance(scn, "", "", "", true);

    // callbacks
    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 2; i < verts.size(); i++)
            ist->shp->triangles.push_back({vert_map.at(verts[0]),
                vert_map.at(verts[i - 1]), vert_map.at(verts[i])});
    };
    cb.line = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 1; i < verts.size(); i++)
            ist->shp->lines.push_back(
                {vert_map.at(verts[i - 1]), vert_map.at(verts[i])});
    };
    cb.point = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 0; i < verts.size(); i++)
            ist->shp->points.push_back(vert_map.at(verts[i]));
    };
    cb.object = [&](const std::string& name) {
        oname     = name;
        gname     = "";
        matname   = "";
        smoothing = true;
        ist       = add_instance(scn, oname, matname, gname, smoothing);
    };
    cb.group = [&](const std::string& name) {
        gname = name;
        if (split_group) {
            ist = add_instance(scn, oname, matname, gname, smoothing);
        }
    };
    cb.smoothing = [&](const std::string& name) {
        smoothing = (name == "on");
        if (split_smoothing) {
            ist = add_instance(scn, oname, matname, gname, smoothing);
        }
    };
    cb.usemtl = [&](const std::string& name) {
        matname = name;
        if (split_material) {
            ist = add_instance(scn, oname, matname, gname, smoothing);
        } else {
            if (matname != "") ist->mat = mmap.at(matname);
        }
    };
    cb.material = [&](const obj_material& omat) {
        auto mat      = new material();
        mat->name     = omat.name;
        mat->ke       = omat.ke;
        mat->kd       = omat.kd;
        mat->ks       = omat.ks;
        mat->kt       = omat.kt;
        mat->rs       = omat.rs;
        mat->op       = omat.op;
        mat->ke_txt   = add_texture(omat.ke_txt, true);
        mat->kd_txt   = add_texture(omat.kd_txt, true);
        mat->ks_txt   = add_texture(omat.ks_txt, true);
        mat->kt_txt   = add_texture(omat.kt_txt, true);
        mat->op_txt   = add_texture(omat.op_txt, false);
        mat->rs_txt   = add_texture(omat.rs_txt, false);
        mat->occ_txt  = add_texture(omat.occ_txt, false);
        mat->bump_txt = add_texture(omat.bump_txt, false);
        mat->disp_txt = add_texture(omat.disp_txt, false);
        mat->norm_txt = add_texture(omat.norm_txt, false);
        mat->ve       = omat.ve;
        mat->va       = omat.va;
        mat->vd       = omat.vd;
        mat->vg       = omat.vg;
        mat->vd_txt   = add_voltexture(omat.vd_txt, false);
        scn->materials.push_back(mat);
        mmap[mat->name] = mat;
    };
    cb.camera = [&](const obj_camera& ocam) {
        auto cam      = new camera();
        cam->name     = ocam.name;
        cam->ortho    = ocam.ortho;
        cam->film     = ocam.film;
        cam->focal    = ocam.focal;
        cam->focus    = ocam.focus;
        cam->aperture = ocam.aperture;
        cam->frame    = ocam.frame;
        scn->cameras.push_back(cam);
    };
    cb.environmnet = [&](const obj_environment& oenv) {
        auto env    = new environment();
        env->name   = oenv.name;
        env->ke     = oenv.ke;
        env->ke_txt = add_texture(oenv.ke_txt, true);
        scn->environments.push_back(env);
    };

    // Parse obj
    if (!load_obj(filename, cb)) return false;

    // cleanup empty
    // TODO: delete unused
    for (auto idx = 0; idx < scn->instances.size(); idx++) {
        if (!is_instance_empty(scn->instances[idx])) continue;
        auto ist = scn->instances[idx];
        if (ist->shp) {
            scn->shapes.erase(
                std::find(scn->shapes.begin(), scn->shapes.end(), ist->shp));
        }
        if (ist->sbd) {
            scn->subdivs.erase(
                std::find(scn->subdivs.begin(), scn->subdivs.end(), ist->sbd));
        }
        scn->instances.erase(scn->instances.begin() + idx);
        idx--;
    }

    // load textures
    auto dirname = get_dirname(filename);
    if (load_textures) {
        if (!load_scene_textures(scn, dirname, skip_missing, false))
            return false;
    }

    // fix scene
    scn->name = get_filename(filename);
    add_missing_cameras(scn);
    add_missing_materials(scn);
    add_missing_names(scn);
    update_transforms(scn);

    // done
    return true;
}

bool save_mtl(
    const std::string& filename, const scene* scn, bool flip_tr = true) {
    // open
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // for each material, dump all the values
    for (auto mat : scn->materials) {
        print(fs, "newmtl {}\n", mat->name);
        print(fs, "  illum 2\n");
        if (mat->ke != zero3f) print(fs, "  Ke {}\n", mat->ke);
        if (mat->kd != zero3f) print(fs, "  Kd {}\n", mat->kd);
        if (mat->ks != zero3f) print(fs, "  Ks {}\n", mat->ks);
        if (mat->kt != zero3f) print(fs, "  Kt {}\n", mat->kt);
        if (mat->rs != 1.0f)
            print(fs, "  Ns {}\n",
                (int)clamp(2 / pow(mat->rs + 1e-10f, 4.0f) - 2, 0.0f, 1.0e12f));
        if (mat->op != 1.0f) print(fs, "  d {}\n", mat->op);
        if (mat->rs != -1.0f) print(fs, "  Pr {}\n", mat->rs);
        if (mat->ke_txt) print(fs, "  map_Ke {}\n", mat->ke_txt->path);
        if (mat->kd_txt) print(fs, "  map_Kd {}\n", mat->kd_txt->path);
        if (mat->ks_txt) print(fs, "  map_Ks {}\n", mat->ks_txt->path);
        if (mat->kt_txt) print(fs, "  map_Kt {}\n", mat->kt_txt->path);
        if (mat->op_txt && mat->op_txt != mat->kd_txt)
            print(fs, "  map_d  {}\n", mat->op_txt->path);
        if (mat->rs_txt) print(fs, "  map_Pr {}\n", mat->rs_txt->path);
        if (mat->occ_txt) print(fs, "  map_occ {}\n", mat->occ_txt->path);
        if (mat->bump_txt) print(fs, "  map_bump {}\n", mat->bump_txt->path);
        if (mat->disp_txt) print(fs, "  map_disp {}\n", mat->disp_txt->path);
        if (mat->norm_txt) print(fs, "  map_norm {}\n", mat->norm_txt->path);
        if (mat->ve != zero3f) print(fs, "  Ve {}\n", mat->ve);
        if (mat->vd != zero3f) print(fs, "  Vd {}\n", mat->vd);
        if (mat->va != zero3f) print(fs, "  Va {}\n", mat->va);
        if (mat->vg != 0) print(fs, "  Vg {}\n", mat->vg);
        if (mat->vd_txt) print(fs, "  map_Vd {}\n", mat->vd_txt->path);
        print(fs, "\n");
    }

    // done
    fclose(fs);
    return true;
}

bool save_objx(const std::string& filename, const scene* scn) {
    // scene
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // cameras
    for (auto cam : scn->cameras) {
        print(fs, "c {} {} {} {} {} {} {}\n", cam->name, (int)cam->ortho,
            cam->film, cam->focal, cam->focus, cam->aperture, cam->frame);
    }

    // environments
    for (auto env : scn->environments) {
        print(fs, "e {} {} {} {}\n", env->name.c_str(), env->ke,
            ((env->ke_txt) ? env->ke_txt->path.c_str() : "\"\""), env->frame);
    }

    // done
    fclose(fs);
    return true;
}

std::string to_string(const obj_vertex& v) {
    auto s = std::to_string(v.pos);
    if (v.texcoord) {
        s += "/" + std::to_string(v.texcoord);
        if (v.norm) s += "/" + std::to_string(v.norm);
    } else {
        if (v.norm) s += "//" + std::to_string(v.norm);
    }
    return s;
}

bool save_obj(
    const std::string& filename, const scene* scn, bool flip_texcoord = true) {
    // scene
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // material library
    if (!scn->materials.empty()) {
        auto mtlname = replace_extension(get_filename(filename), "mtl");
        print(fs, "mtllib {}\n", mtlname);
    }

    // shapes
    auto offset = obj_vertex{0, 0, 0};
    for (auto ist : scn->instances) {
        if (!ist->sbd) {
            print(fs, "o {}\n", ist->name);
            if (ist->mat) print(fs, "usemtl {}\n", ist->mat->name);
            if (ist->frame == identity_frame3f) {
                for (auto& p : ist->shp->pos) print(fs, "v {}\n", p);
                for (auto& n : ist->shp->norm) print(fs, "vn {}\n", n);
                for (auto& t : ist->shp->texcoord)
                    print(fs, "vt {}\n",
                        vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            } else {
                for (auto& pp : ist->shp->pos) {
                    print(fs, "v {}\n", transform_point(ist->frame, pp));
                }
                for (auto& nn : ist->shp->norm) {
                    print(fs, "vn {}\n", transform_direction(ist->frame, nn));
                }
                for (auto& t : ist->shp->texcoord)
                    print(fs, "vt {}\n",
                        vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            }
            auto mask = obj_vertex{1, ist->shp->texcoord.empty() ? 0 : 1,
                ist->shp->norm.empty() ? 0 : 1};
            auto vert = [mask, offset](int i) {
                return obj_vertex{(i + offset.pos + 1) * mask.pos,
                    (i + offset.texcoord + 1) * mask.texcoord,
                    (i + offset.norm + 1) * mask.norm};
            };
            for (auto& t : ist->shp->triangles) {
                print(fs, "f {} {} {}\n", to_string(vert(t.x)),
                    to_string(vert(t.y)), to_string(vert(t.z)));
            }
            for (auto& l : ist->shp->lines) {
                print(fs, "l {} {}\n", to_string(vert(l.x)),
                    to_string(vert(l.y)));
            }
            offset.pos += ist->shp->pos.size();
            offset.texcoord += ist->shp->texcoord.size();
            offset.norm += ist->shp->norm.size();
        } else {
            print(fs, "o {}\n", ist->name);
            if (ist->mat) print(fs, "usemtl {}\n", ist->mat->name);
            if (ist->frame == identity_frame3f) {
                for (auto& p : ist->sbd->pos) print(fs, "v {}\n", p);
                for (auto& t : ist->sbd->texcoord)
                    print(fs, "vt {}\n",
                        vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            } else {
                for (auto& pp : ist->sbd->pos) {
                    auto p = transform_point(ist->frame, pp);
                    print(fs, "v {}\n", p);
                }
                for (auto& t : ist->sbd->texcoord)
                    print(fs, "vt {}\n",
                        vec2f{t.x, (flip_texcoord) ? 1 - t.y : t.y});
            }
            if (!ist->sbd->texcoord.empty()) {
                auto vert = [offset](int ip, int it) {
                    return obj_vertex{
                        ip + offset.pos + 1, it + offset.texcoord + 1, 0};
                };
                for (auto i = 0; i < ist->sbd->quads_pos.size(); i++) {
                    auto qp = ist->sbd->quads_pos[i];
                    auto qt = ist->sbd->quads_texcoord[i];
                    if (qp.z == qp.w) {
                        print(fs, "f {} {} {}\n", to_string(vert(qp.x, qt.x)),
                            to_string(vert(qp.y, qt.y)),
                            to_string(vert(qp.z, qt.z)));
                    } else {
                        print(fs, "f {} {} {} {}\n", to_string(vert(qp.x, qt.x)),
                            to_string(vert(qp.y, qt.y)),
                            to_string(vert(qp.z, qt.z)),
                            to_string(vert(qp.w, qt.w)));
                    }
                }
            } else {
                auto vert = [offset](int ip) {
                    return obj_vertex{ip + offset.pos + 1, 0, 0};
                };
                for (auto& q : ist->sbd->quads_pos) {
                    if (q.z == q.w) {
                        print(fs, "f {} {} {}\n", to_string(vert(q.x)),
                            to_string(vert(q.y)), to_string(vert(q.z)));
                    } else {
                        print(fs, "f {} {} {} {}\n", to_string(vert(q.x)),
                            to_string(vert(q.y)), to_string(vert(q.z)),
                            to_string(vert(q.w)));
                    }
                }
            }
            offset.pos += ist->sbd->pos.size();
            offset.texcoord += ist->sbd->texcoord.size();
        }
    }

    fclose(fs);
    return true;
}

bool save_obj_scene(const std::string& filename, const scene* scn,
    bool save_textures, bool skip_missing) {
    if (!save_obj(filename, scn, true)) return false;
    if (!scn->materials.empty()) {
        if (!save_mtl(replace_extension(filename, ".mtl"), scn, true))
            return false;
    }
    if (!scn->cameras.empty() || !scn->environments.empty()) {
        if (!save_objx(replace_extension(filename, ".objx"), scn)) return false;
    }

    // skip textures if needed
    auto dirname = get_dirname(filename);
    if (save_textures) {
        if (!save_scene_textures(scn, dirname, skip_missing)) return false;
    }

    // done
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// GLTF CONVESION
// -----------------------------------------------------------------------------
namespace ygl {

static bool startswith(const std::string& str, const std::string& substr) {
    if (str.length() < substr.length()) return false;
    for (auto i = 0; i < substr.length(); i++)
        if (str[i] != substr[i]) return false;
    return true;
}

// Load a scene
scene* load_gltf_scene(
    const std::string& filename, bool load_textures, bool skip_missing) {
    auto scn = (scene*)nullptr;
    if (!load_gltf_scene(filename, scn, load_textures, skip_missing))
        return nullptr;
    return scn;
}

// convert gltf to scene
bool gltf_to_scene(scene* scn, const json& gltf, const std::string& dirname) {
    // convert textures
    if (gltf.count("images")) {
        for (auto iid = 0; iid < gltf.at("images").size(); iid++) {
            auto& gimg = gltf.at("images").at(iid);
            auto  txt  = new texture();
            txt->name  = gimg.value("name", ""s);
            txt->path  = (startswith(gimg.value("uri", ""s), "data:")) ?
                            std::string("[glTF-inline].png") :
                            gimg.value("uri", ""s);
            scn->textures.push_back(txt);
        }
    }

    // load buffers
    auto bmap = std::vector<std::vector<byte>>();
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
                if (pos == uri.npos) { return false; }
                // decode
                auto data_char = base64_decode(uri.substr(pos + 1));
                data           = std::vector<unsigned char>(
                    (unsigned char*)data_char.c_str(),
                    (unsigned char*)data_char.c_str() + data_char.length());
            } else {
                auto filename = normalize_path(dirname + "/" + uri);
                if (!load_binary(filename, data)) return false;
            }
            if (gbuf.value("byteLength", -1) != data.size()) { return false; }
        }
    }

    // add a texture
    auto add_texture = [scn, &gltf](const json& ginfo, bool srgb) {
        if (!gltf.count("images") || !gltf.count("textures"))
            return (texture*)nullptr;
        if (ginfo.is_null() || ginfo.empty()) return (texture*)nullptr;
        if (ginfo.value("index", -1) < 0) return (texture*)nullptr;
        auto& gtxt = gltf.at("textures").at(ginfo.value("index", -1));
        if (gtxt.empty() || gtxt.value("source", -1) < 0)
            return (texture*)nullptr;
        auto txt = scn->textures.at(gtxt.value("source", -1));
        if (!gltf.count("samplers") || gtxt.value("sampler", -1) < 0)
            return txt;
        auto& gsmp = gltf.at("samplers").at(gtxt.value("sampler", -1));
        txt->clamp = gsmp.value("wrapS", ""s) == "ClampToEdge" ||
                     gsmp.value("wrapT", ""s) == "ClampToEdge";
        txt->scale = gsmp.value("scale", 1.0f) * gsmp.value("strength", 1.0f);
        txt->srgb  = srgb && !is_hdr_filename(txt->path);
        return txt;
    };

    // convert materials
    if (gltf.count("materials")) {
        for (auto mid = 0; mid < gltf.at("materials").size(); mid++) {
            auto& gmat = gltf.at("materials").at(mid);
            auto  mat  = new material();
            mat->name  = gmat.value("name", ""s);
            mat->ke    = gmat.value("emissiveFactor", zero3f);
            if (gmat.count("emissiveTexture"))
                mat->ke_txt = add_texture(gmat.at("emissiveTexture"), true);
            if (gmat.count("extensions") && gmat.at("extensions")
                                                .count("KHR_materials_"
                                                       "pbrSpecularGlossines"
                                                       "s")) {
                mat->base_metallic = false;
                mat->gltf_textures = true;
                auto& gsg          = gmat.at("extensions")
                                .at("KHR_materials_pbrSpecularGlossiness");
                auto kb = gsg.value("diffuseFactor", vec4f{1, 1, 1, 1});
                mat->kd = {kb.x, kb.y, kb.z};
                mat->op = kb.w;
                mat->ks = gsg.value("specularFactor", vec3f{1, 1, 1});
                mat->rs = 1 - gsg.value("glossinessFactor", 1.0f);
                if (gsg.count("diffuseTexture"))
                    mat->kd_txt = add_texture(gsg.at("diffuseTexture"), true);
                if (gsg.count("specularGlossinessTexture"))
                    mat->ks_txt = add_texture(
                        gsg.at("specularGlossinessTexture"), true);
                mat->rs_txt = mat->ks_txt;
            } else if (gmat.count("pbrMetallicRoughness")) {
                mat->base_metallic = true;
                mat->gltf_textures = true;
                auto& gmr          = gmat.at("pbrMetallicRoughness");
                auto  kb = gmr.value("baseColorFactor", vec4f{1, 1, 1, 1});
                mat->kd  = {kb.x, kb.y, kb.z};
                mat->op  = kb.w;
                auto km  = gmr.value("metallicFactor", 1.0f);
                mat->ks  = {km, km, km};
                mat->rs  = gmr.value("roughnessFactor", 1.0f);
                if (gmr.count("baseColorTexture"))
                    mat->kd_txt = add_texture(gmr.at("baseColorTexture"), true);
                if (gmr.count("metallicRoughnessTexture"))
                    mat->ks_txt = add_texture(
                        gmr.at("metallicRoughnessTexture"), false);
                mat->rs_txt = mat->ks_txt;
            }
            if (gmat.count("occlusionTexture"))
                mat->occ_txt = add_texture(gmat.at("occlusionTexture"), false);
            if (gmat.count("normalTexture"))
                mat->norm_txt = add_texture(gmat.at("normalTexture"), false);
            mat->double_sided = gmat.value("doubleSided", false);
            scn->materials.push_back(mat);
        }
    }

    // get values from accessors
    auto accessor_values =
        [&gltf, &bmap](const json& gacc,
            bool normalize = false) -> std::vector<std::array<double, 4>> {
        auto gview  = gltf.at("bufferViews").at(gacc.value("bufferView", -1));
        auto data   = bmap.at(gview.value("buffer", -1)).data();
        auto offset = gacc.value("byteOffset", 0) + gview.value("byteOffset", 0);
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
        if (compTypeNum == 5122 || compTypeNum == 5123) { compSize = 2; }
        if (compTypeNum == 5124 || compTypeNum == 5125 || compTypeNum == 5126) {
            compSize = 4;
        }
        if (!stride) stride = compSize * ncomp;
        auto vals = std::vector<std::array<double, 4>>(
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
    auto meshes = std::vector<std::vector<std::pair<shape*, material*>>>();
    if (gltf.count("meshes")) {
        for (auto mid = 0; mid < gltf.at("meshes").size(); mid++) {
            auto& gmesh = gltf.at("meshes").at(mid);
            meshes.push_back({});
            auto sid = 0;
            for (auto& gprim : gmesh.value("primitives", json::array())) {
                if (!gprim.count("attributes")) continue;
                auto shp  = new shape();
                shp->name = gmesh.value("name", ""s) +
                            ((sid) ? std::to_string(sid) : std::string());
                sid++;
                for (json::iterator gattr_it = gprim.at("attributes").begin();
                     gattr_it != gprim.at("attributes").end(); ++gattr_it) {
                    auto  semantic = gattr_it.key();
                    auto& gacc     = gltf.at("accessors")
                                     .at(gattr_it.value().get<int>());
                    auto vals = accessor_values(gacc);
                    if (semantic == "POSITION") {
                        shp->pos.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shp->pos.push_back({(float)vals[i][0],
                                (float)vals[i][1], (float)vals[i][2]});
                    } else if (semantic == "NORMAL") {
                        shp->norm.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shp->norm.push_back({(float)vals[i][0],
                                (float)vals[i][1], (float)vals[i][2]});
                    } else if (semantic == "TEXCOORD" ||
                               semantic == "TEXCOORD_0") {
                        shp->texcoord.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shp->texcoord.push_back(
                                {(float)vals[i][0], (float)vals[i][1]});
                    } else if (semantic == "COLOR" || semantic == "COLOR_0") {
                        shp->color.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shp->color.push_back(
                                {(float)vals[i][0], (float)vals[i][1],
                                    (float)vals[i][2], (float)vals[i][3]});
                    } else if (semantic == "TANGENT") {
                        shp->tangsp.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shp->tangsp.push_back(
                                {(float)vals[i][0], (float)vals[i][1],
                                    (float)vals[i][2], (float)vals[i][3]});
                        for (auto& t : shp->tangsp) t.w = -t.w;
                    } else if (semantic == "RADIUS") {
                        shp->radius.reserve(vals.size());
                        for (auto i = 0; i < vals.size(); i++)
                            shp->radius.push_back((float)vals[i][0]);
                    } else {
                        // ignore
                    }
                }
                // indices
                auto mode = gprim.value("mode", 4);
                if (!gprim.count("indices")) {
                    if (mode == 4) {
                        // triangles
                        shp->triangles.reserve(shp->pos.size() / 3);
                        for (auto i = 0; i < shp->pos.size() / 3; i++)
                            shp->triangles.push_back(
                                {i * 3 + 0, i * 3 + 1, i * 3 + 2});
                    } else if (mode == 6) {
                        // triangle fan
                        shp->triangles.reserve(shp->pos.size() - 2);
                        for (auto i = 2; i < shp->pos.size(); i++)
                            shp->triangles.push_back({0, i - 1, i});
                    } else if (mode == 5) {
                        // triangle strip
                        shp->triangles.reserve(shp->pos.size() - 2);
                        for (auto i = 2; i < shp->pos.size(); i++)
                            shp->triangles.push_back({i - 2, i - 1, i});
                    } else if (mode == 1) {
                        // lines
                        shp->lines.reserve(shp->pos.size() / 2);
                        for (auto i = 0; i < shp->pos.size() / 2; i++)
                            shp->lines.push_back({i * 2 + 0, i * 2 + 1});
                    } else if (mode == 2) {
                        // line loop
                        shp->lines.reserve(shp->pos.size());
                        for (auto i = 1; i < shp->pos.size(); i++)
                            shp->lines.push_back({i - 1, i});
                        shp->lines.back() = {(int)shp->pos.size() - 1, 0};
                    } else if (mode == 3) {
                        // line strip
                        shp->lines.reserve(shp->pos.size() - 1);
                        for (auto i = 1; i < shp->pos.size(); i++)
                            shp->lines.push_back({i - 1, i});
                    } else if (mode == -1 || mode == 0) {
                        // points
                        printf("points not supported\n");
                    } else {
                        throw std::runtime_error("unknown primitive type");
                    }
                } else {
                    auto indices = accessor_values(
                        gltf.at("accessors").at(gprim.value("indices", -1)),
                        false);
                    if (mode == 4) {
                        // triangles
                        shp->triangles.reserve(indices.size());
                        for (auto i = 0; i < indices.size() / 3; i++)
                            shp->triangles.push_back({(int)indices[i * 3 + 0][0],
                                (int)indices[i * 3 + 1][0],
                                (int)indices[i * 3 + 2][0]});
                    } else if (mode == 6) {
                        // triangle fan
                        shp->triangles.reserve(indices.size() - 2);
                        for (auto i = 2; i < indices.size(); i++)
                            shp->triangles.push_back({(int)indices[0][0],
                                (int)indices[i - 1][0], (int)indices[i][0]});
                    } else if (mode == 5) {
                        // triangle strip
                        shp->triangles.reserve(indices.size() - 2);
                        for (auto i = 2; i < indices.size(); i++)
                            shp->triangles.push_back({(int)indices[i - 2][0],
                                (int)indices[i - 1][0], (int)indices[i][0]});
                    } else if (mode == 1) {
                        // lines
                        shp->lines.reserve(indices.size() / 2);
                        for (auto i = 0; i < indices.size() / 2; i++)
                            shp->lines.push_back({(int)indices[i * 2 + 0][0],
                                (int)indices[i * 2 + 1][0]});
                    } else if (mode == 2) {
                        // line loop
                        shp->lines.reserve(indices.size());
                        for (auto i = 1; i < indices.size(); i++)
                            shp->lines.push_back(
                                {(int)indices[i - 1][0], (int)indices[i][0]});
                        shp->lines.back() = {(int)indices[indices.size() - 1][0],
                            (int)indices[0][0]};
                    } else if (mode == 3) {
                        // line strip
                        shp->lines.reserve(indices.size() - 1);
                        for (auto i = 1; i < indices.size(); i++)
                            shp->lines.push_back(
                                {(int)indices[i - 1][0], (int)indices[i][0]});
                    } else if (mode == -1 || mode == 0) {
                        // points
                        printf("points not supported\n");
                    } else {
                        throw std::runtime_error("unknown primitive type");
                    }
                }
                auto mat = (gprim.count("material")) ?
                               scn->materials.at(gprim.value("material", -1)) :
                               nullptr;
                meshes.back().push_back({shp, mat});
                scn->shapes.push_back(shp);
            }
        }
    }

    // convert cameras
    if (gltf.count("cameras")) {
        for (auto cid = 0; cid < gltf.at("cameras").size(); cid++) {
            auto& gcam = gltf.at("cameras").at(cid);
            auto  cam  = new camera();
            cam->name  = gcam.value("name", ""s);
            cam->ortho = gcam.value("type", ""s) == "orthographic";
            if (cam->ortho) {
                printf("orthographic not supported well\n");
                auto ortho = gcam.value("orthographic", json::object());
                set_camera_fovy(cam, ortho.value("ymag", 0.0f),
                    ortho.value("xmag", 0.0f) / ortho.value("ymag", 0.0f));
                cam->focus    = maxf;
                cam->aperture = 0;
            } else {
                auto persp = gcam.value("perspective", json::object());
                set_camera_fovy(cam, persp.value("yfov", 1.0f),
                    persp.value("aspectRatio", 1.0f));
                cam->focus    = maxf;
                cam->aperture = 0;
            }
            scn->cameras.push_back(cam);
        }
    }

    // convert nodes
    if (gltf.count("nodes")) {
        for (auto nid = 0; nid < gltf.at("nodes").size(); nid++) {
            auto& gnde = gltf.at("nodes").at(nid);
            auto  nde  = new node();
            nde->name  = gnde.value("name", ""s);
            if (gnde.count("camera"))
                nde->cam = scn->cameras[gnde.value("camera", 0)];
            nde->translation = gnde.value("translation", zero3f);
            nde->rotation    = gnde.value("rotation", vec4f{0, 0, 0, 1});
            nde->scale       = gnde.value("scale", vec3f{1, 1, 1});
            nde->local = mat_to_frame(gnde.value("matrix", identity_mat4f));
            scn->nodes.push_back(nde);
        }

        // set up parent pointers
        for (auto nid = 0; nid < gltf.at("nodes").size(); nid++) {
            auto& gnde = gltf.at("nodes").at(nid);
            if (!gnde.count("children")) continue;
            auto nde = scn->nodes[nid];
            for (auto& cid : gnde.at("children"))
                scn->nodes[cid.get<int>()]->parent = nde;
        }

        // set up instances
        for (auto nid = 0; nid < gltf.at("nodes").size(); nid++) {
            auto& gnde = gltf.at("nodes").at(nid);
            if (!gnde.count("mesh")) continue;
            auto  nde  = scn->nodes[nid];
            auto& shps = meshes.at(gnde.value("mesh", 0));
            if (shps.empty()) continue;
            if (shps.size() == 1) {
                nde->ist       = new instance();
                nde->ist->name = nde->name;
                nde->ist->shp  = shps[0].first;
                nde->ist->mat  = shps[0].second;
                scn->instances.push_back(nde->ist);
            } else {
                for (auto shp : shps) {
                    auto child       = new node();
                    child->name      = nde->name + "_" + shp.first->name;
                    child->parent    = nde;
                    child->ist       = new instance();
                    child->ist->name = child->name;
                    child->ist->shp  = shp.first;
                    child->ist->mat  = shp.second;
                    scn->instances.push_back(child->ist);
                }
            }
        }
    }

    // convert animations
    if (gltf.count("animations")) {
        for (auto& ganm : gltf.at("animations")) {
            auto aid         = 0;
            auto sampler_map = std::unordered_map<vec2i, int>();
            for (auto& gchannel : ganm.at("channels")) {
                auto path_ = gchannel.at("target").at("path").get<std::string>();
                auto path  = -1;
                if (path_ == "translation") path = 0;
                if (path_ == "rotation") path = 1;
                if (path_ == "scale") path = 2;
                if (path_ == "weights") path = 3;
                if (sampler_map.find({gchannel.at("sampler").get<int>(),
                        path}) == sampler_map.end()) {
                    auto& gsampler = ganm.at("samplers")
                                         .at(gchannel.at("sampler").get<int>());
                    auto anm  = new animation();
                    anm->name = (ganm.count("name") ? ganm.value("name", ""s) :
                                                      "anim") +
                                std::to_string(aid++);
                    anm->group      = ganm.value("name", ""s);
                    auto input_view = accessor_values(
                        gltf.at("accessors").at(gsampler.value("input", -1)));
                    anm->times.resize(input_view.size());
                    for (auto i = 0; i < input_view.size(); i++)
                        anm->times[i] = input_view[i][0];
                    auto type = gsampler.value("interpolation", "LINEAR");
                    if (type == "LINEAR") anm->type = animation_type::linear;
                    if (type == "STEP") anm->type = animation_type::step;
                    if (type == "CUBICSPLINE")
                        anm->type = animation_type::bezier;
                    auto output_view = accessor_values(
                        gltf.at("accessors").at(gsampler.value("output", -1)));
                    switch (path) {
                        case 0: {  // translation
                            anm->translation.reserve(output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                anm->translation.push_back(
                                    {(float)output_view[i][0],
                                        (float)output_view[i][1],
                                        (float)output_view[i][2]});
                        } break;
                        case 1: {  // rotation
                            anm->rotation.reserve(output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                anm->rotation.push_back(
                                    {(float)output_view[i][0],
                                        (float)output_view[i][1],
                                        (float)output_view[i][2],
                                        (float)output_view[i][3]});
                        } break;
                        case 2: {  // scale
                            anm->scale.reserve(output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                anm->scale.push_back({(float)output_view[i][0],
                                    (float)output_view[i][1],
                                    (float)output_view[i][2]});
                        } break;
                        case 3: {  // weights
                            printf("weights not supported for now\n");
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
                            auto values = std::vector<float>();
                            values.reserve(output_view.size());
                            for (auto i = 0; i < output_view.size(); i++)
                                values.push_back(output_view.get(i));
                            anm->weights.resize(values.size() / ncomp);
                            for (auto i = 0; i < anm->weights.size(); i++) {
                                anm->weights[i].resize(ncomp);
                                for (auto j = 0; j < ncomp; j++)
                                    anm->weights[i][j] = values[i * ncomp + j];
                            }
                        }
#endif
                        } break;
                        default: { return false; }
                    }
                    sampler_map[{gchannel.at("sampler").get<int>(), path}] =
                        (int)scn->animations.size();
                    scn->animations.push_back(anm);
                }
                scn->animations[sampler_map.at(
                                    {gchannel.at("sampler").get<int>(), path})]
                    ->targets.push_back(
                        scn->nodes
                            [(int)gchannel.at("target").at("node").get<int>()]);
            }
        }
    }

    return true;
}

// Load a scene
bool load_gltf_scene(const std::string& filename, scene*& scn,
    bool load_textures, bool skip_missing) {
    // reset
    reset_scene_data(scn);

    // initialization
    scn = new scene();

    // convert json
    auto js = json();
    if (!load_json(filename, js)) return false;
    try {
        if (!gltf_to_scene(scn, js, get_dirname(filename))) return false;
    } catch (...) { return false; }

    // load textures
    auto dirname = get_dirname(filename);
    if (load_textures) {
        if (!load_scene_textures(scn, dirname, skip_missing, false))
            return false;
    }

    // fix scene
    scn->name = get_filename(filename);
    add_missing_cameras(scn);
    add_missing_materials(scn);
    add_missing_names(scn);
    update_transforms(scn);

    // fix cameras
    auto bbox = compute_bbox(scn);
    for (auto cam : scn->cameras) {
        auto center = (bbox.min + bbox.max) / 2;
        auto dist   = dot(-cam->frame.z, center - cam->frame.o);
        if (dist > 0) cam->focus = dist;
    }

    // done
    return true;
}

// convert gltf scene to json
bool scene_to_gltf(const scene* scn, json& js) {
    // init to emprt object
    js = json::object();

    // start creating json
    js["asset"]["version"] = "2.0";

    // prepare top level nodes
    if (!scn->cameras.empty()) js["cameras"] = json::array();
    if (!scn->textures.empty()) {
        js["textures"] = json::array();
        js["images"]   = json::array();
    }
    if (!scn->materials.empty()) js["materials"] = json::array();
    if (!scn->shapes.empty()) {
        js["meshes"]      = json::array();
        js["buffers"]     = json::array();
        js["bufferViews"] = json::array();
        js["accessors"]   = json::array();
    }
    if (!scn->instances.empty()) js["nodes"] = json::array();
    if (!scn->nodes.empty()) js["nodes"] = json::array();

    // convert cameras
    auto cmap = std::unordered_map<camera*, int>();
    for (auto cam : scn->cameras) {
        auto cjs    = json();
        cjs["name"] = cam->name;
        if (!cam->ortho) {
            cjs["type"]                       = "perspective";
            cjs["perspective"]["aspectRatio"] = cam->film.x / cam->film.y;
            cjs["perspective"]["znear"]       = 0.01f;
        } else {
            cjs["type"]                  = "orthographic";
            cjs["orthographic"]["xmag"]  = cam->film.x / 2;
            cjs["orthographic"]["ymag"]  = cam->film.y / 2;
            cjs["orthographic"]["znear"] = 0.01f;
        }
        cmap[cam] = (int)js["cameras"].size();
        js["cameras"].push_back(cjs);
    }

    // textures
    auto tmap = std::unordered_map<texture*, int>();
    for (auto& txt : scn->textures) {
        auto tjs = json(), ijs = json();
        tjs["source"] = (int)js["images"].size();
        ijs["uri"]    = txt->path;
        js["images"].push_back(ijs);
        js["textures"].push_back(tjs);
        tmap[txt] = (int)js["textures"].size() - 1;
    }

    // material
    auto mmap = std::unordered_map<material*, int>();
    for (auto mat : scn->materials) {
        auto mjs           = json();
        mjs["name"]        = mat->name;
        mjs["doubleSided"] = mat->double_sided;
        if (mat->ke != zero3f) mjs["emissiveFactor"] = mat->ke;
        if (mat->ke_txt) mjs["emissiveTexture"]["index"] = tmap.at(mat->ke_txt);
        auto kd = vec4f{mat->kd.x, mat->kd.y, mat->kd.z, mat->op};
        if (mat->base_metallic) {
            auto mmjs               = json();
            mmjs["baseColorFactor"] = kd;
            mmjs["metallicFactor"]  = mat->ks.x;
            mmjs["roughnessFactor"] = mat->rs;
            if (mat->kd_txt)
                mmjs["baseColorTexture"]["index"] = tmap.at(mat->kd_txt);
            if (mat->ks_txt)
                mmjs["metallicRoughnessTexture"]["index"] = tmap.at(mat->ks_txt);
            mjs["pbrMetallicRoughness"] = mmjs;
        } else {
            auto mmjs                = json();
            mmjs["diffuseFactor"]    = kd;
            mmjs["specularFactor"]   = mat->ks;
            mmjs["glossinessFactor"] = 1 - mat->rs;
            if (mat->kd_txt)
                mmjs["diffuseTexture"]["index"] = tmap.at(mat->kd_txt);
            if (mat->ks_txt)
                mmjs["specularGlossinessTexture"]["index"] = tmap.at(
                    mat->ks_txt);
            mjs["extensions"]["KHR_materials_pbrSpecularGlossiness"] = mmjs;
        }
        if (mat->norm_txt)
            mjs["normalTexture"]["index"] = tmap.at(mat->norm_txt);
        if (mat->occ_txt)
            mjs["occlusionTexture"]["index"] = tmap.at(mat->occ_txt);
        js["materials"].push_back(mjs);
        mmap[mat] = (int)js["materials"].size() - 1;
    }

    // determine shape materials
    auto shape_mats = std::unordered_map<shape*, int>();
    for (auto ist : scn->instances)
        if (ist->mat) shape_mats[ist->shp] = mmap.at(ist->mat);

    // shapes
    auto smap = std::unordered_map<shape*, int>();
    for (auto shp : scn->shapes) {
        auto mjs = json(), bjs = json(), pjs = json();
        auto bid          = js["buffers"].size();
        mjs["name"]       = shp->name;
        mjs["primitives"] = json::array();
        bjs["name"]       = shp->name;
        bjs["byteLength"] = 0;
        bjs["uri"]        = replace_extension(shp->path, ".bin");
        auto mat_it       = shape_mats.find(shp);
        if (mat_it != shape_mats.end()) pjs["material"] = mat_it->second;
        auto add_accessor = [&js, &bjs, bid](int count, std::string type,
                                bool indices = false) {
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
        auto nverts = (int)shp->pos.size();
        if (!shp->pos.empty())
            pjs["attributes"]["POSITION"] = add_accessor(nverts, "VEC3");
        if (!shp->norm.empty())
            pjs["attributes"]["NORMAL"] = add_accessor(nverts, "VEC3");
        if (!shp->texcoord.empty())
            pjs["attributes"]["TEXCOORD_0"] = add_accessor(nverts, "VEC2");
        if (!shp->color.empty())
            pjs["attributes"]["COLOR_0"] = add_accessor(nverts, "VEC4");
        if (!shp->radius.empty())
            pjs["attributes"]["RADIUS"] = add_accessor(nverts, "SCALAR");
        if (!shp->lines.empty()) {
            pjs["indices"] = add_accessor(
                (int)shp->lines.size() * 2, "SCALAR", true);
            pjs["mode"] = 1;
        }
        if (!shp->triangles.empty()) {
            pjs["indices"] = add_accessor(
                (int)shp->triangles.size() * 3, "SCALAR", true);
            pjs["mode"] = 4;
        }
        mjs["primitives"].push_back(pjs);
        js["meshes"].push_back(mjs);
        js["buffers"].push_back(bjs);
        smap[shp] = (int)js["meshes"].size() - 1;
    }

    // nodes
    auto nmap = std::unordered_map<node*, int>();
    for (auto& nde : scn->nodes) {
        auto njs           = json();
        njs["name"]        = nde->name;
        njs["matrix"]      = frame_to_mat(nde->local);
        njs["translation"] = nde->translation;
        njs["rotation"]    = nde->rotation;
        njs["scale"]       = nde->scale;
        if (nde->cam) njs["camera"] = cmap.at(nde->cam);
        if (nde->ist) njs["mesh"] = smap.at(nde->ist->shp);
        if (!nde->children.empty()) {
            njs["children"] = json::array();
            for (auto& c : nde->children) njs["children"].push_back(nmap.at(c));
        }
        js["nodes"].push_back(njs);
        nmap[nde] = (int)js["nodes"].size() - 1;
    }

    // animations not supported yet
    if (!scn->animations.empty()) printf("animation not supported yet\n");

    // nodes from instances
    if (scn->nodes.empty()) {
        for (auto cam : scn->cameras) {
            auto njs      = json();
            njs["name"]   = cam->name;
            njs["camera"] = cmap.at(cam);
            njs["matrix"] = frame_to_mat(cam->frame);
            js["nodes"].push_back(njs);
        }
        for (auto ist : scn->instances) {
            auto njs      = json();
            njs["name"]   = ist->name;
            njs["mesh"]   = smap.at(ist->shp);
            njs["matrix"] = frame_to_mat(ist->frame);
            js["nodes"].push_back(njs);
        }
    }

    // done
    return true;
}

// save gltf mesh
bool save_gltf_mesh(const std::string& filename, const shape* shp) {
    auto fs = fopen(filename.c_str(), "wb");
    if (!fs) return false;
    fclose_guard fs_{fs};

    fwrite((char*)shp->pos.data(), 3 * 4, shp->pos.size(), fs);
    fwrite((char*)shp->norm.data(), 3 * 4, shp->norm.size(), fs);
    fwrite((char*)shp->texcoord.data(), 2 * 4, shp->texcoord.size(), fs);
    fwrite((char*)shp->color.data(), 4 * 4, shp->color.size(), fs);
    fwrite((char*)shp->radius.data(), 1 * 4, shp->radius.size(), fs);
    fwrite((char*)shp->lines.data(), 2 * 4, shp->lines.size(), fs);
    fwrite((char*)shp->triangles.data(), 3 * 4, shp->triangles.size(), fs);

    fclose(fs);
    return true;
}

// Save gltf json
bool save_gltf_scene(const std::string& filename, const scene* scn,
    bool save_textures, bool skip_missing) {
    // save json
    auto js = json();
    try {
        if (!scene_to_gltf(scn, js)) return false;
    } catch (...) { return false; }
    if (!save_json(filename, js)) return false;

    // meshes
    auto dirname = get_dirname(filename);
    for (auto& shp : scn->shapes) {
        if (shp->path == "") continue;
        auto filename = normalize_path(dirname + "/" + shp->path);
        filename      = replace_extension(filename, ".bin");
        if (!save_gltf_mesh(filename, shp)) {
            if (!skip_missing) return false;
        }
    }

    // save textures
    if (save_textures) {
        if (!save_scene_textures(scn, dirname, skip_missing)) return false;
    }

    // done
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF PBRT
// -----------------------------------------------------------------------------
namespace ygl {

// convert pbrt to json
bool pbrt_to_json(const std::string& filename, json& js) {
    auto split = [](const std::string& str) {
        auto ret = std::vector<std::string>();
        if (str.empty()) return ret;
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

    auto is_cmd = [](const std::vector<std::string>& tokens, int i) -> bool {
        auto& tok = tokens.at(i);
        return !(tok[0] == '[' || tok[0] == ']' || tok[0] == '\"' ||
                 tok[0] == '-' || tok[0] == '+' || tok[0] == '.' ||
                 std::isdigit(tok[0]));
    };
    auto is_number = [](const std::vector<std::string>& tokens, int i) -> bool {
        auto& tok = tokens.at(i);
        return tok[0] == '-' || tok[0] == '+' || tok[0] == '.' ||
               std::isdigit(tok[0]);
    };
    auto parse_string = [](const std::vector<std::string>& tokens,
                            int&                           i) -> std::string {
        if (tokens[i][0] != '"') throw std::runtime_error("string expected");
        auto tok = tokens[i++];
        tok      = tok.substr(1, tok.size() - 2);
        if (tok.find('|') != tok.npos) tok = tok.substr(tok.find('|') + 1);
        return tok;
    };
    auto parse_param = [&](const std::vector<std::string>& tokens, int& i,
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
                if (!first && !list) throw std::runtime_error("bad params");
                js.push_back(atof(tokens[i].c_str()));
                i++;
                if (!list) break;
            }
        }
    };
    auto parse_param_list = [&](const std::vector<std::string>& tokens, int& i,
                                json& js) -> void {
        while (i < tokens.size()) {
            if (is_cmd(tokens, i)) break;
            auto name = parse_string(tokens, i);
            js[name]  = json::array();
            parse_param(tokens, i, js.at(name));
            if (js.at(name).size() == 1) { js.at(name) = js.at(name).at(0); }
        }
    };
    auto parse_param_numbers = [&](const std::vector<std::string>& tokens,
                                   int& i, json& js) -> void {
        js["values"] = json::array();
        if (tokens[i][0] == '[') i++;
        while (is_number(tokens, i)) {
            js.at("values").push_back((float)atof(tokens[i++].c_str()));
        }
        if (tokens[i][0] == ']') i++;
    };

    auto fs = fopen(filename.c_str(), "rt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    auto pbrt = std::string();
    char buf[4096];
    while (fgets(buf, 4096, fs)) {
        auto line = std::string(buf);
        if (line.find('#') == line.npos)
            pbrt += line + "\n";
        else
            pbrt += line.substr(0, line.find('#')) + "\n";
    }
    fclose(fs);

    auto re = std::regex("\"(\\w+)\\s+(\\w+)\"");
    pbrt    = std::regex_replace(pbrt, re, "\"$1|$2\"");
    pbrt    = std::regex_replace(pbrt, std::regex("\\["), " [ ");
    pbrt    = std::regex_replace(pbrt, std::regex("\\]"), " ] ");
    js      = json::array();

    auto tokens = split(pbrt);
    auto i      = 0;
    while (i < tokens.size()) {
        if (!is_cmd(tokens, i)) throw std::runtime_error("command expected");
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
            throw std::runtime_error("unsupported command " + tok);
        }
        js.push_back(jcmd);
    }
    // auto fstr = std::fstream(filename + ".json");
    // fstr << js;
    return true;
}

// Load a scene
scene* load_pbrt_scene(
    const std::string& filename, bool load_textures, bool skip_missing) {
    auto scn = (scene*)nullptr;
    if (!load_pbrt_scene(filename, scn, load_textures, skip_missing))
        return nullptr;
    return scn;
}

// load pbrt scenes
bool load_pbrt_scene(const std::string& filename, scene*& scn,
    bool load_textures, bool skip_missing) {
    // reset
    reset_scene_data(scn);

    // convert to json
    auto js = json();
    try {
        if (!pbrt_to_json(filename, js)) return false;
    } catch (...) { return false; }

    auto dirname_ = get_dirname(filename);

    struct stack_item {
        frame3f   frame     = identity_frame3f;
        material* mat       = nullptr;
        material* light_mat = nullptr;
        float     focus = 1, aspect = 1;
        bool      reverse = false;
    };

    // parse
    scn        = new scene();
    auto stack = std::vector<stack_item>();
    stack.push_back(stack_item());
    auto txt_map = std::map<std::string, texture*>();
    auto mat_map = std::map<std::string, material*>();
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
        printf("cannot handle vec3f\n");
        return zero3f;
    };

    auto get_vec4f = [](const json& js) -> vec4f {
        if (js.is_number())
            return {js.get<float>(), js.get<float>(), js.get<float>(),
                js.get<float>()};
        if (js.is_array() && js.size() == 4)
            return {js.at(0).get<float>(), js.at(1).get<float>(),
                js.at(2).get<float>(), js.at(3).get<float>()};
        printf("cannot handle vec4f\n");
        return zero4f;
    };

    auto get_mat4f = [](const json& js) -> frame3f {
        if (!js.is_array() || js.size() != 16) {
            printf("cannot handle vec4f\n");
            return identity_frame3f;
        }
        float m[16] = {0};
        for (auto i = 0; i < 16; i++) m[i] = js.at(i).get<float>();
        return {{m[0], m[1], m[2]}, {m[4], m[5], m[6]}, {m[8], m[9], m[10]},
            {m[12], m[13], m[14]}};
    };

    auto get_mat3f = [](const json& js) -> frame3f {
        if (!js.is_array() || js.size() != 9) {
            printf("cannot handle mat3f\n");
            return identity_frame3f;
        }
        auto m = identity_frame3f;
        for (auto i = 0; i < 9; i++) (&m.x.x)[i] = js.at(i).get<float>();
        return m;
    };

    auto get_vector_vec3i = [](const json& js) -> std::vector<vec3i> {
        if (!js.is_array() || js.size() % 3) {
            printf("cannot handle vector<vec3f>\n");
            return {};
        }
        auto vals = std::vector<vec3i>(js.size() / 3);
        for (auto i = 0; i < vals.size(); i++) {
            vals[i].x = (int)std::round(js.at(i * 3 + 0).get<float>());
            vals[i].y = (int)std::round(js.at(i * 3 + 1).get<float>());
            vals[i].z = (int)std::round(js.at(i * 3 + 2).get<float>());
        }
        return vals;
    };

    auto get_vector_vec3f = [](const json& js) -> std::vector<vec3f> {
        if (!js.is_array() || js.size() % 3) {
            printf("cannot handle vector<vec3f>\n");
            return {};
        }
        auto vals = std::vector<vec3f>(js.size() / 3);
        for (auto i = 0; i < vals.size(); i++) {
            vals[i].x = js.at(i * 3 + 0).get<float>();
            vals[i].y = js.at(i * 3 + 1).get<float>();
            vals[i].z = js.at(i * 3 + 2).get<float>();
        }
        return vals;
    };

    auto get_vector_vec2f = [](const json& js) -> std::vector<vec2f> {
        if (!js.is_array() || js.size() % 2) {
            printf("cannot handle vector<vec3f>\n");
            return {};
        }
        auto vals = std::vector<vec2f>(js.size() / 2);
        for (auto i = 0; i < vals.size(); i++) {
            vals[i].x = js.at(i * 2 + 0).get<float>();
            vals[i].y = js.at(i * 2 + 1).get<float>();
        }
        return vals;
    };

    auto get_scaled_texture =
        [&txt_map, &get_vec3f](const json& js) -> std::pair<vec3f, texture*> {
        if (js.is_string())
            return {{1, 1, 1}, txt_map.at(js.get<std::string>())};
        return {get_vec3f(js), nullptr};
    };

    auto use_hierarchy = false;

    std::map<std::string, std::vector<instance*>> objects;
    for (auto& jcmd : js) {
        auto cmd = jcmd.at("cmd").get<std::string>();
        if (cmd == "ObjectInstance") {
            use_hierarchy = true;
            break;
        }
    }

    auto lid = 0, sid = 0, cid = 0;
    auto cur_object = ""s;
    for (auto& jcmd : js) {
        auto cmd = jcmd.at("cmd").get<std::string>();
        if (cmd == "Integrator" || cmd == "Sampler" || cmd == "PixelFilter") {
        } else if (cmd == "Transform") {
            stack.back().frame = get_mat4f(jcmd.at("values"));
        } else if (cmd == " ConcatTransform") {
            stack.back().frame = stack.back().frame * get_mat4f(jcmd.at("value"
                                                                        "s"));
        } else if (cmd == "Scale") {
            auto v             = get_vec3f(jcmd.at("values"));
            stack.back().frame = stack.back().frame * scaling_frame(v);
        } else if (cmd == "Translate") {
            auto v             = get_vec3f(jcmd.at("values"));
            stack.back().frame = stack.back().frame * translation_frame(v);
        } else if (cmd == "Rotate") {
            auto v             = get_vec4f(jcmd.at("values"));
            stack.back().frame = stack.back().frame *
                                 rotation_frame(
                                     vec3f{v.y, v.z, v.w}, v.x * pif / 180);
        } else if (cmd == "LookAt") {
            auto m             = get_mat3f(jcmd.at("values"));
            stack.back().frame = stack.back().frame *
                                 inverse(lookat_frame(m.x, m.y, m.z, true));
            stack.back().focus = length(m.x - m.y);
        } else if (cmd == "ReverseOrientation") {
            stack.back().reverse = !stack.back().reverse;
        } else if (cmd == "Film") {
            stack.back().aspect = jcmd.at("xresolution").get<float>() /
                                  jcmd.at("yresolution").get<float>();
        } else if (cmd == "Camera") {
            auto cam     = new camera();
            cam->name    = "cam" + std::to_string(cid++);
            cam->frame   = inverse(stack.back().frame);
            cam->frame.z = -cam->frame.z;
            cam->focus   = stack.back().focus;
            auto aspect  = stack.back().aspect;
            auto fovy    = 1.0f;
            auto type    = jcmd.at("type").get<std::string>();
            if (type == "perspective") {
                fovy = jcmd.at("fov").get<float>() * pif / 180;
            } else {
                printf("%s camera not supported\n", type.c_str());
            }
            set_camera_fovy(cam, fovy, aspect);
            scn->cameras.push_back(cam);
        } else if (cmd == "Texture") {
            auto found = false;
            auto name  = jcmd.at("name").get<std::string>();
            for (auto& txt : scn->textures) {
                if (txt->name == name) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                auto txt = new texture();
                scn->textures.push_back(txt);
                txt->name          = jcmd.at("name").get<std::string>();
                txt_map[txt->name] = txt;
                auto type          = jcmd.at("type").get<std::string>();
                if (type == "imagemap") {
                    txt->path = jcmd.at("filename").get<std::string>();
                    if (get_extension(txt->path) == "pfm")
                        txt->path = replace_extension(txt->path, ".hdr");
                } else {
                    printf("%s texture not supported\n", type.c_str());
                }
            }
        } else if (cmd == "MakeNamedMaterial" || cmd == "Material") {
            auto found = false;
            if (cmd == "MakeNamedMaterial") {
                auto name = jcmd.at("name").get<std::string>();
                for (auto mat : scn->materials) {
                    if (mat->name == name) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                auto mat = new material();
                scn->materials.push_back(mat);
                if (cmd == "Material") {
                    mat->name        = "unnamed_mat" + std::to_string(mid++);
                    stack.back().mat = mat;
                } else {
                    mat->name          = jcmd.at("name").get<std::string>();
                    mat_map[mat->name] = mat;
                }
                auto type = "uber"s;
                if (jcmd.count("type"))
                    type = jcmd.at("type").get<std::string>();
                if (type == "uber") {
                    if (jcmd.count("Kd"))
                        std::tie(mat->kd, mat->kd_txt) = get_scaled_texture(
                            jcmd.at("Kd"));
                    if (jcmd.count("Ks"))
                        std::tie(mat->ks, mat->ks_txt) = get_scaled_texture(
                            jcmd.at("Ks"));
                    if (jcmd.count("Kt"))
                        std::tie(mat->kt, mat->kt_txt) = get_scaled_texture(
                            jcmd.at("Kt"));
                    if (jcmd.count("opacity")) {
                        auto op              = vec3f{0, 0, 0};
                        auto op_txt          = (texture*)nullptr;
                        std::tie(op, op_txt) = get_scaled_texture(
                            jcmd.at("opacity"));
                        mat->op     = (op.x + op.y + op.z) / 3;
                        mat->op_txt = op_txt;
                    }
                    mat->rs = 0;
                } else if (type == "matte") {
                    mat->kd = {1, 1, 1};
                    if (jcmd.count("Kd"))
                        std::tie(mat->kd, mat->kd_txt) = get_scaled_texture(
                            jcmd.at("Kd"));
                    mat->rs = 1;
                } else if (type == "mirror") {
                    mat->kd = {0, 0, 0};
                    mat->ks = {1, 1, 1};
                    mat->rs = 0;
                } else if (type == "metal") {
                    auto eta = get_vec3f(jcmd.at("eta"));
                    auto k   = get_vec3f(jcmd.at("k"));
                    mat->ks  = fresnel_metal(1, eta, k);
                    mat->rs  = 0;
                } else if (type == "substrate") {
                    if (jcmd.count("Kd"))
                        std::tie(mat->kd, mat->kd_txt) = get_scaled_texture(
                            jcmd.at("Kd"));
                    mat->ks = {0.04f, 0.04f, 0.04f};
                    if (jcmd.count("Ks"))
                        std::tie(mat->ks, mat->ks_txt) = get_scaled_texture(
                            jcmd.at("Ks"));
                    mat->rs = 0;
                } else if (type == "glass") {
                    mat->ks = {0.04f, 0.04f, 0.04f};
                    mat->kt = {1, 1, 1};
                    if (jcmd.count("Ks"))
                        std::tie(mat->ks, mat->ks_txt) = get_scaled_texture(
                            jcmd.at("Ks"));
                    if (jcmd.count("Kt"))
                        std::tie(mat->kt, mat->kt_txt) = get_scaled_texture(
                            jcmd.at("Kt"));
                    mat->rs = 0;
                } else if (type == "mix") {
                    printf("mix material not properly supported\n");
                    if (jcmd.count("namedmaterial1")) {
                        auto mat1 = jcmd.at("namedmaterial1").get<std::string>();
                        auto saved_name = mat->name;
                        *mat            = *mat_map.at(mat1);
                        mat->name       = saved_name;
                    } else {
                        printf("mix material missing front material\n");
                    }
                } else {
                    mat->kd = {1, 0, 0};
                    printf("%s material not supported\n", type.c_str());
                }
                if (jcmd.count("uroughness")) {
                    auto remap = js.count("remaproughness") &&
                                 js.at("remaproughness").get<bool>();
                    if (jcmd.count("uroughness"))
                        mat->rs = jcmd.at("uroughness").get<float>();
                    // if (!remap) mat->rs = mat->rs * mat->rs;
                    if (remap) printf("remap roughness not supported\n");
                }
                if (jcmd.count("roughness")) {
                    auto remap = js.count("remaproughness") &&
                                 js.at("remaproughness").get<bool>();
                    if (jcmd.count("roughness"))
                        mat->rs = jcmd.at("roughness").get<float>();
                    // if (!remap) mat->rs = mat->rs * mat->rs;
                    if (remap) printf("remap roughness not supported\n");
                }
                if (stack.back().light_mat) {
                    mat->ke     = stack.back().light_mat->ke;
                    mat->ke_txt = stack.back().light_mat->ke_txt;
                }
            }
        } else if (cmd == "NamedMaterial") {
            stack.back().mat = mat_map.at(jcmd.at("name").get<std::string>());
            if (stack.back().light_mat) {
                auto mat = new material(*stack.back().mat);
                mat->name += "_" + std::to_string(lid++);
                mat->ke     = stack.back().light_mat->ke;
                mat->ke_txt = stack.back().light_mat->ke_txt;
                scn->materials.push_back(mat);
                stack.back().mat = mat;
            }
        } else if (cmd == "Shape") {
            auto shp  = new shape();
            auto type = jcmd.at("type").get<std::string>();
            if (type == "plymesh") {
                auto filename = jcmd.at("filename").get<std::string>();
                shp->name     = get_filename(filename);
                shp->path     = filename;
                if (!load_ply_mesh(dirname_ + "/" + filename, shp->points,
                        shp->lines, shp->triangles, shp->pos, shp->norm,
                        shp->texcoord, shp->color, shp->radius))
                    return false;
            } else if (type == "trianglemesh") {
                shp->name = "mesh" + std::to_string(sid++);
                shp->path = "models/" + shp->name + ".ply";
                if (jcmd.count("indices"))
                    shp->triangles = get_vector_vec3i(jcmd.at("indices"));
                if (jcmd.count("P")) shp->pos = get_vector_vec3f(jcmd.at("P"));
                if (jcmd.count("N")) shp->norm = get_vector_vec3f(jcmd.at("N"));
                if (jcmd.count("uv"))
                    shp->texcoord = get_vector_vec2f(jcmd.at("uv"));
            } else if (type == "sphere") {
                shp->name   = "sphere" + std::to_string(sid++);
                shp->path   = "models/" + shp->name + ".ply";
                auto radius = 1.0f;
                if (jcmd.count("radius"))
                    radius = jcmd.at("radius").get<float>();
                auto sshp     = make_sphere({64, 32}, 2 * radius, {1, 1}, true);
                shp->pos      = sshp.pos;
                shp->norm     = sshp.norm;
                shp->texcoord = sshp.texcoord;
                shp->triangles = sshp.triangles;
            } else if (type == "disk") {
                shp->name   = "disk" + std::to_string(sid++);
                shp->path   = "models/" + shp->name + ".ply";
                auto radius = 1.0f;
                if (jcmd.count("radius"))
                    radius = jcmd.at("radius").get<float>();
                auto sshp      = make_disk({32, 16}, 2 * radius, {1, 1}, true);
                shp->pos       = sshp.pos;
                shp->norm      = sshp.norm;
                shp->texcoord  = sshp.texcoord;
                shp->triangles = sshp.triangles;
            } else {
                printf("%s shape not supported\n", type.c_str());
            }
            auto frame = stack.back().frame;
            auto scl = vec3f{length(frame.x), length(frame.y), length(frame.z)};
            for (auto& p : shp->pos) p *= scl;
            frame = {normalize(frame.x), normalize(frame.y), normalize(frame.z),
                frame.o};
            if (stack.back().reverse) {
                for (auto& t : shp->triangles) std::swap(t.y, t.z);
            }
            scn->shapes.push_back(shp);
            auto ist   = new instance();
            ist->name  = shp->name;
            ist->frame = frame;
            ist->shp   = shp;
            ist->mat   = stack.back().mat;
            if (cur_object != "") {
                objects[cur_object].push_back(ist);
            } else {
                scn->instances.push_back(ist);
            }
        } else if (cmd == "ObjectInstance") {
            static auto instances = std::map<std::string, int>();
            auto        name      = jcmd.at("name").get<std::string>();
            auto&       object    = objects.at(name);
            for (auto shp : object) {
                instances[shp->name] += 1;
                auto ist  = new instance();
                ist->name = shp->name + "_ist" +
                            std::to_string(instances[shp->name]);
                ist->frame = stack.back().frame * shp->frame;
                ist->shp   = shp->shp;
                scn->instances.push_back(ist);
            }
        } else if (cmd == "AreaLightSource") {
            auto type = jcmd.at("type").get<std::string>();
            if (type == "diffuse") {
                auto lmat              = new material();
                lmat->ke               = get_vec3f(jcmd.at("L"));
                stack.back().light_mat = lmat;
            } else {
                printf("%s area light not supported\n", type.c_str());
            }
        } else if (cmd == "LightSource") {
            auto type = jcmd.at("type").get<std::string>();
            if (type == "infinite") {
                auto env  = new environment();
                env->name = "env" + std::to_string(lid++);
                // env->frame = frame3f{{1,0,0},{0,0,-1},{0,-1,0},{0,0,0}} *
                // stack.back().frame;
                env->frame = stack.back().frame * frame3f{{0, 0, 1}, {0, 1, 0},
                                                      {1, 0, 0}, {0, 0, 0}};
                env->ke    = {1, 1, 1};
                if (jcmd.count("scale")) env->ke *= get_vec3f(jcmd.at("scale"));
                if (jcmd.count("mapname")) {
                    auto txt  = new texture();
                    txt->path = jcmd.at("mapname").get<std::string>();
                    txt->name = env->name;
                    scn->textures.push_back(txt);
                    env->ke_txt = txt;
                }
                scn->environments.push_back(env);
            } else if (type == "distant") {
                auto distant_dist = 100;
                auto shp          = new shape();
                shp->name         = "distant" + std::to_string(lid++);
                auto from = vec3f{0, 0, 0}, to = vec3f{0, 0, 0};
                if (jcmd.count("from")) from = get_vec3f(jcmd.at("from"));
                if (jcmd.count("to")) to = get_vec3f(jcmd.at("to"));
                auto dir       = normalize(from - to);
                auto size      = distant_dist * sin(5 * pif / 180);
                auto sshp      = make_quad({1, 1}, {size, size}, {1, 1}, true);
                shp->pos       = sshp.pos;
                shp->norm      = sshp.norm;
                shp->texcoord  = sshp.texcoord;
                shp->triangles = sshp.triangles;
                scn->shapes.push_back(shp);
                auto mat  = new material();
                mat->name = shp->name;
                mat->ke   = {1, 1, 1};
                if (jcmd.count("L")) mat->ke *= get_vec3f(jcmd.at("L"));
                if (jcmd.count("scale")) mat->ke *= get_vec3f(jcmd.at("scale"));
                mat->ke *= (distant_dist * distant_dist) / (size * size);
                scn->materials.push_back(mat);
                auto ist   = new instance();
                ist->name  = shp->name;
                ist->shp   = shp;
                ist->mat   = mat;
                ist->frame = stack.back().frame *
                             lookat_frame(
                                 dir * distant_dist, zero3f, {0, 1, 0}, true);
                scn->instances.push_back(ist);
                printf("%s light not properly supported\n", type.c_str());
            } else {
                printf("%s light not supported\n", type.c_str());
            }
        } else if (cmd == "WorldBegin") {
            stack.push_back(stack_item());
        } else if (cmd == "AttributeBegin") {
            stack.push_back(stack.back());
        } else if (cmd == "ObjectBegin") {
            auto name     = jcmd.at("name").get<std::string>();
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
            printf("%s command not supported\n", cmd.c_str());
        }
    }
    if (use_hierarchy) {
        for (auto cam : scn->cameras) {
            auto nde   = new node();
            nde->name  = cam->name;
            nde->local = cam->frame;
            nde->cam   = cam;
            scn->nodes.insert(scn->nodes.begin(), nde);
        }
        for (auto env : scn->environments) {
            auto nde   = new node();
            nde->name  = env->name;
            nde->local = env->frame;
            nde->env   = env;
            scn->nodes.push_back(nde);
        }
    }

    // load textures
    auto dirname = get_dirname(filename);
    if (load_textures) {
        if (!load_scene_textures(scn, dirname, skip_missing, false))
            return false;
    }

    // fix scene
    scn->name = get_filename(filename);
    add_missing_cameras(scn);
    add_missing_materials(scn);
    add_missing_names(scn);
    update_transforms(scn);

    return scn;
}

// Convert a scene to pbrt format
bool save_pbrt(const std::string& filename, const scene* scn) {
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};

#if 0
WorldBegin

#uniform blue - ish illumination from all directions
LightSource "infinite" "rgb L" [.4 .45 .5]

#approximate the sun
LightSource "distant"  "point from" [ -30 40  100 ]
   "blackbody L" [3000 1.5]

AttributeBegin
  Material "glass"
  Shape "sphere" "float radius" 1
AttributeEnd

AttributeBegin
  Texture "checks" "spectrum" "checkerboard"
          "float uscale" [8] "float vscale" [8]
          "rgb tex1" [.1 .1 .1] "rgb tex2" [.8 .8 .8]
  Material "matte" "texture Kd" "checks"
  Translate 0 0 -1
  Shape "trianglemesh"
      "integer indices" [0 1 2 0 2 3]
      "point P" [ -20 -20 0   20 -20 0   20 20 0   -20 20 0 ]
      "float st" [ 0 0   1 0    1 1   0 1 ]
AttributeEnd

WorldEnd
#endif

    // convert camera and settings
    auto cam  = scn->cameras.front();
    auto from = cam->frame.o;
    auto to   = cam->frame.o - cam->frame.z;
    auto up   = cam->frame.y;
    print(fs, "LookAt {} {} {}\n", from, to, up);
    print(fs, "Camera \"perspective\" \"float fov\" {}\n",
        eval_camera_fovy(cam) * 180 / pif);

    // save renderer
    print(fs, "Sampler \"random\" \"integer pixelsamples\" [64]\n");
    // fprintf(f, "Sampler \"sobol\" \"interger pixelsamples\" [64]\n");
    print(fs, "Integrator \"path\"\n");
    print(fs,
        "Film \"image\" \"string filename\" [\"{}\"] "
        "\"integer xresolution\" [{}] \"integer yresolution\" [{}]\n",
        replace_extension(filename, "exr"), eval_image_size(cam, 512).x,
        eval_image_size(cam, 512).y);

    // start world
    print(fs, "WorldBegin\n");

    // convert textures
    for (auto txt : scn->textures) {
        print(fs,
            "Texture \"{}\" \"spectrum\" \"imagemap\" "
            "\"string filename\" [\"{}\"]\n",
            txt->name, txt->path);
    }

    // convert materials
    for (auto mat : scn->materials) {
        print(fs, "MakeNamedMaterial \"{}\" ", mat->name);
        print(fs, "\"string type\" \"{}\" ", "uber");
        if (mat->kd_txt)
            print(fs, "\"texture Kd\" [\"{}\"] ", mat->kd_txt->name);
        else
            print(fs, "\"rgb Kd\" [{}] ", mat->kd);
        if (mat->ks_txt)
            print(fs, "\"texture Ks\" [\"{}\"] ", mat->ks_txt->name);
        else
            print(fs, "\"rgb Ks\" [{}] ", mat->ks);
        print(fs, "\"float roughness\" [{}] ", mat->rs);
        print(fs, "\n");
    }

    // convert instances
    for (auto ist : scn->instances) {
        print(fs, "AttributeBegin\n");
        print(fs, "TransformBegin\n");
        print(fs, "ConcatTransform [{}]\n", frame_to_mat(ist->frame));
        if (ist->mat->ke != zero3f)
            print(fs, "AreaLightSource \"diffuse\" \"rgb L\" [ {} ]\n",
                ist->mat->ke);
        print(fs, "NamedMaterial \"{}\"\n", ist->mat->name);
        print(fs, "Shape \"plymesh\" \"string filename\" [\"{}\"]\n",
            ist->shp->path.c_str());
        print(fs, "TransformEnd\n");
        print(fs, "AttributeEnd\n");
    }

    // end world
    print(fs, "WorldEnd\n");

    // done
    fclose(fs);
    return true;
}

// Save a pbrt scene
bool save_pbrt_scene(const std::string& filename, const scene* scn,
    bool save_textures, bool skip_missing) {
    // save json
    if (!save_pbrt(filename, scn)) return false;

    // save meshes
    auto dirname = get_dirname(filename);
    for (auto& shp : scn->shapes) {
        if (shp->path == "") continue;
        auto filename = normalize_path(dirname + "/" + shp->path);
        if (!save_mesh(filename, shp->points, shp->lines, shp->triangles,
                shp->pos, shp->norm, shp->texcoord, shp->color, shp->radius)) {
            if (!skip_missing) return false;
        }
    }

    // skip textures
    if (save_textures) {
        if (!save_scene_textures(scn, dirname, skip_missing)) return false;
    }

    // done
    return true;
}

// Attempt to fix pbrt z-up.
void pbrt_flipyz_scene(const scene* scn) {
    // flip meshes
    for (auto shp : scn->shapes) {
        for (auto& p : shp->pos) std::swap(p.y, p.z);
        for (auto& n : shp->norm) std::swap(n.y, n.z);
    }
    for (auto ist : scn->instances) {
        ist->frame = ist->frame *
                     frame3f{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0, 0}};
    }
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF BINARY SCENE FORMAT
// -----------------------------------------------------------------------------
namespace ygl {

// serialize_bin( ) can both save/load data to/from a binary file. The behaviour
// is set by the boolean 'save'. serialize_bin(var, file, true) : writes var as
// binary into file serialize_bin(var, file, false): read file as binary and set
// var

// Serialize type or struct with no allocated resource
template <typename T>
bool serialize_bin_value(T& val, FILE* fs, bool save) {
    if (save) {
        if (fwrite(&val, sizeof(T), 1, fs) != 1) return false;
        return true;
    } else {
        if (fread(&val, sizeof(T), 1, fs) != 1) return false;
        return false;
    }
}

// Serialize std::vector
template <typename T>
bool serialize_bin_value(std::vector<T>& vec, FILE* fs, bool save) {
    if (save) {
        auto count = (size_t)vec.size();
        if (!serialize_bin_value(count, fs, save)) return false;
        if (fwrite(vec.data(), sizeof(T), count, fs) != count) return false;
        return true;
    } else {
        auto count = (size_t)0;
        if (!serialize_bin_value(count, fs, save)) return false;
        vec = std::vector<T>(count);
        if (fread(vec.data(), sizeof(T), count, fs) != count) return false;
        return true;
    }
}

// Serialize std::string
bool serialize_bin_value(std::string& vec, FILE* fs, bool save) {
    if (save) {
        auto count = (size_t)vec.size();
        if (!serialize_bin_value(count, fs, save)) return false;
        if (fwrite(vec.data(), sizeof(char), count, fs) != count) return false;
        return true;
    } else {
        auto count = (size_t)0;
        if (!serialize_bin_value(count, fs, save)) return false;
        vec = std::string(count, ' ');
        if (fread((void*)vec.data(), sizeof(char), count, fs) != count)
            return false;
        return true;
    }
}

// Serialize image
template <typename T>
bool serialize_bin_value(image<T>& img, FILE* fs, bool save) {
    if (save) {
        auto size = (vec2i)img.size();
        if (!serialize_bin_value(size, fs, save)) return false;
        auto count = size.x * size.y;
        if (fwrite(img.data(), sizeof(T), count, fs) != count) return false;
        return true;
    } else {
        auto size = zero2i;
        if (!serialize_bin_value(size, fs, save)) return false;
        auto count = size.x * size.y;
        img        = image<T>(size);
        if (fread(img.data(), sizeof(T), count, fs) != count) return false;
        return true;
    }
}

// Serialize image
template <typename T>
bool serialize_bin_value(volume<T>& vol, FILE* fs, bool save) {
    if (save) {
        auto size = (vec3i)vol.size();
        if (!serialize_bin_value(size, fs, save)) return false;
        auto count = size.x * size.y * size.z;
        if (fwrite(vol.data(), sizeof(T), count, fs) != count) return false;
        return true;
    } else {
        auto size = zero3i;
        if (!serialize_bin_value(size, fs, save)) return false;
        auto count = size.x * size.y * size.z;
        vol        = volume<T>(size);
        if (fread(vol.data(), sizeof(T), count, fs) != count) return false;
        return true;
    }
}

// Serialize std::vector of pointers
template <typename T>
bool serialize_bin_object(std::vector<T*>& vec, FILE* fs, bool save) {
    if (save) {
        auto count = (size_t)vec.size();
        if (!serialize_bin_value(count, fs, true)) return false;
        for (auto i = 0; i < vec.size(); ++i) {
            if (!serialize_bin_object(vec[i], fs, true)) return false;
        }
        return true;
    } else {
        auto count = (size_t)0;
        if (!serialize_bin_value(count, fs, false)) return false;
        vec = std::vector<T*>(count);
        for (auto i = 0; i < vec.size(); ++i) {
            vec[i] = new T();
            if (!serialize_bin_object(vec[i], fs, false)) return false;
        }
        return true;
    }
}

// Serialize std::vector of pointers
template <typename T>
bool serialize_bin_object(
    std::vector<T*>& vec, const scene* scn, FILE* fs, bool save) {
    if (save) {
        auto count = (size_t)vec.size();
        if (!serialize_bin_value(count, fs, true)) return false;
        for (auto i = 0; i < vec.size(); ++i) {
            if (!serialize_bin_object(vec[i], scn, fs, true)) return false;
        }
        return true;
    } else {
        auto count = (size_t)0;
        if (!serialize_bin_value(count, fs, false)) return false;
        vec = std::vector<T*>(count);
        for (auto i = 0; i < vec.size(); ++i) {
            vec[i] = new T();
            if (!serialize_bin_object(vec[i], scn, fs, false)) return false;
        }
        return true;
    }
}

// Serialize a pointer. It is saved as an integer index (handle) of the array of
// pointers vec. On loading, the handle is converted back into a pointer.
template <typename T>
bool serialize_bin_handle(
    T*& val, const std::vector<T*>& vec, FILE* fs, bool save) {
    if (save) {
        auto handle = -1;
        for (auto i = 0; i < vec.size(); ++i)
            if (vec[i] == val) {
                handle = i;
                break;
            }
        if (!serialize_bin_value(handle, fs, true)) return false;
        return true;
    } else {
        auto handle = -1;
        if (!serialize_bin_value(handle, fs, false)) return false;
        val = (handle == -1) ? nullptr : vec[handle];
        return true;
    }
}

// Serialize a pointer. It is saved as an integer index (handle) of the array of
// pointers vec. On loading, the handle is converted back into a pointer.
template <typename T>
bool serialize_bin_handle(
    std::vector<T*>& vals, const std::vector<T*>& vec_, FILE* fs, bool save) {
    if (save) {
        auto count = (size_t)vals.size();
        if (!serialize_bin_value(count, fs, true)) return false;
        for (auto i = 0; i < vals.size(); ++i) {
            if (!serialize_bin_handle(vals[i], vec_, fs, true)) return false;
        }
    } else {
        auto count = (size_t)0;
        if (!serialize_bin_value(count, fs, false)) return false;
        vals = std::vector<T*>(count);
        for (auto i = 0; i < vals.size(); ++i) {
            if (!serialize_bin_handle(vals[i], vec_, fs, false)) return false;
        }
    }
}

// Serialize yocto types. This is mostly boiler plate code.
bool serialize_bin_object(camera* cam, FILE* fs, bool save) {
    if (!serialize_bin_value(cam->name, fs, save)) return false;
    if (!serialize_bin_value(cam->frame, fs, save)) return false;
    if (!serialize_bin_value(cam->ortho, fs, save)) return false;
    if (!serialize_bin_value(cam->film, fs, save)) return false;
    if (!serialize_bin_value(cam->focal, fs, save)) return false;
    if (!serialize_bin_value(cam->focus, fs, save)) return false;
    if (!serialize_bin_value(cam->aperture, fs, save)) return false;
    return true;
}

bool serialize_bin_object(bvh_tree* bvh, FILE* fs, bool save) {
    if (!serialize_bin_value(bvh->pos, fs, save)) return false;
    if (!serialize_bin_value(bvh->radius, fs, save)) return false;
    if (!serialize_bin_value(bvh->points, fs, save)) return false;
    if (!serialize_bin_value(bvh->lines, fs, save)) return false;
    if (!serialize_bin_value(bvh->triangles, fs, save)) return false;
    if (!serialize_bin_value(bvh->quads, fs, save)) return false;
    if (!serialize_bin_value(bvh->nodes, fs, save)) return false;
    if (!serialize_bin_value(bvh->instances, fs, save)) return false;
    if (!serialize_bin_object(bvh->shape_bvhs, fs, save)) return false;
    if (!serialize_bin_value(bvh->nodes, fs, save)) return false;
    return true;
}

bool serialize_bin_object(shape* shp, const scene* scn, FILE* fs, bool save) {
    if (!serialize_bin_value(shp->name, fs, save)) return false;
    if (!serialize_bin_value(shp->path, fs, save)) return false;
    if (!serialize_bin_value(shp->points, fs, save)) return false;
    if (!serialize_bin_value(shp->lines, fs, save)) return false;
    if (!serialize_bin_value(shp->triangles, fs, save)) return false;
    if (!serialize_bin_value(shp->pos, fs, save)) return false;
    if (!serialize_bin_value(shp->norm, fs, save)) return false;
    if (!serialize_bin_value(shp->texcoord, fs, save)) return false;
    if (!serialize_bin_value(shp->color, fs, save)) return false;
    if (!serialize_bin_value(shp->radius, fs, save)) return false;
    if (!serialize_bin_value(shp->tangsp, fs, save)) return false;
    return true;
}

bool serialize_bin_object(subdiv* sbd, FILE* fs, bool save) {
    if (!serialize_bin_value(sbd->name, fs, save)) return false;
    if (!serialize_bin_value(sbd->path, fs, save)) return false;
    if (!serialize_bin_value(sbd->level, fs, save)) return false;
    if (!serialize_bin_value(sbd->catmull_clark, fs, save)) return false;
    if (!serialize_bin_value(sbd->compute_normals, fs, save)) return false;
    if (!serialize_bin_value(sbd->quads_pos, fs, save)) return false;
    if (!serialize_bin_value(sbd->quads_texcoord, fs, save)) return false;
    if (!serialize_bin_value(sbd->quads_color, fs, save)) return false;
    if (!serialize_bin_value(sbd->crease_pos, fs, save)) return false;
    if (!serialize_bin_value(sbd->crease_texcoord, fs, save)) return false;
    if (!serialize_bin_value(sbd->pos, fs, save)) return false;
    if (!serialize_bin_value(sbd->texcoord, fs, save)) return false;
    if (!serialize_bin_value(sbd->color, fs, save)) return false;
    return true;
}

bool serialize_bin_object(texture* tex, FILE* fs, bool save) {
    if (!serialize_bin_value(tex->name, fs, save)) return false;
    if (!serialize_bin_value(tex->path, fs, save)) return false;
    if (!serialize_bin_value(tex->imgf, fs, save)) return false;
    if (!serialize_bin_value(tex->imgb, fs, save)) return false;
    if (!serialize_bin_value(tex->clamp, fs, save)) return false;
    if (!serialize_bin_value(tex->scale, fs, save)) return false;
    if (!serialize_bin_value(tex->srgb, fs, save)) return false;
    if (!serialize_bin_value(tex->has_opacity, fs, save)) return false;
    return true;
}

bool serialize_bin_object(voltexture* tex, FILE* fs, bool save) {
    if (!serialize_bin_value(tex->name, fs, save)) return false;
    if (!serialize_bin_value(tex->path, fs, save)) return false;
    if (!serialize_bin_value(tex->vol, fs, save)) return false;
    if (!serialize_bin_value(tex->clamp, fs, save)) return false;
    return true;
}

bool serialize_bin_object(
    environment* env, const scene* scn, FILE* fs, bool save) {
    if (!serialize_bin_value(env->name, fs, save)) return false;
    if (!serialize_bin_value(env->frame, fs, save)) return false;
    if (!serialize_bin_value(env->ke, fs, save)) return false;
    if (!serialize_bin_handle(env->ke_txt, scn->textures, fs, save))
        return false;
    return true;
}

bool serialize_bin_object(material* mat, const scene* scn, FILE* fs, bool save) {
    if (!serialize_bin_value(mat->name, fs, save)) return false;
    if (!serialize_bin_value(mat->base_metallic, fs, save)) return false;
    if (!serialize_bin_value(mat->gltf_textures, fs, save)) return false;
    if (!serialize_bin_value(mat->double_sided, fs, save)) return false;
    if (!serialize_bin_value(mat->ke, fs, save)) return false;
    if (!serialize_bin_value(mat->kd, fs, save)) return false;
    if (!serialize_bin_value(mat->ks, fs, save)) return false;
    if (!serialize_bin_value(mat->kt, fs, save)) return false;
    if (!serialize_bin_value(mat->rs, fs, save)) return false;
    if (!serialize_bin_value(mat->op, fs, save)) return false;
    if (!serialize_bin_value(mat->fresnel, fs, save)) return false;
    if (!serialize_bin_value(mat->refract, fs, save)) return false;
    if (!serialize_bin_handle(mat->ke_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->kd_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->ks_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->kt_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->rs_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->op_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->occ_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->bump_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->disp_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_handle(mat->norm_txt, scn->textures, fs, save))
        return false;
    if (!serialize_bin_value(mat->ve, fs, save)) return false;
    if (!serialize_bin_value(mat->va, fs, save)) return false;
    if (!serialize_bin_value(mat->vd, fs, save)) return false;
    if (!serialize_bin_value(mat->vg, fs, save)) return false;
    if (!serialize_bin_handle(mat->vd_txt, scn->voltextures, fs, save))
        return false;
    return true;
};

bool serialize_bin_object(instance* ist, const scene* scn, FILE* fs, bool save) {
    if (!serialize_bin_value(ist->name, fs, save)) return false;
    if (!serialize_bin_value(ist->frame, fs, save)) return false;
    if (!serialize_bin_handle(ist->shp, scn->shapes, fs, save)) return false;
    if (!serialize_bin_handle(ist->mat, scn->materials, fs, save)) return false;
    if (!serialize_bin_handle(ist->sbd, scn->subdivs, fs, save)) return false;
    return true;
};

bool serialize_scene(scene* scn, FILE* fs, bool save) {
    if (!serialize_bin_value(scn->name, fs, save)) return false;
    if (!serialize_bin_object(scn->cameras, fs, save)) return false;
    if (!serialize_bin_object(scn->shapes, scn, fs, save)) return false;
    if (!serialize_bin_object(scn->subdivs, fs, save)) return false;
    if (!serialize_bin_object(scn->textures, fs, save)) return false;
    if (!serialize_bin_object(scn->voltextures, fs, save)) return false;
    if (!serialize_bin_object(scn->materials, scn, fs, save)) return false;
    if (!serialize_bin_object(scn->instances, scn, fs, save)) return false;
    if (!serialize_bin_object(scn->environments, scn, fs, save)) return false;
    return true;
}

// Load/save a binary dump useful for very fast scene IO.
scene* load_ybin_scene(
    const std::string& filename, bool load_textures, bool skip_missing) {
    auto scn = (scene*)nullptr;
    if (!load_ybin_scene(filename, scn, skip_missing)) return nullptr;
    return scn;
}

// Load/save a binary dump useful for very fast scene IO.
bool load_ybin_scene(const std::string& filename, scene*& scn,
    bool load_textures, bool skip_missing) {
    auto fs = fopen(filename.c_str(), "rb");
    if (!fs) return true;
    fclose_guard fs_{fs};
    scn = new scene();
    if (!serialize_scene(scn, fs, false)) return false;
    fclose(fs);
    return false;
}

// Load/save a binary dump useful for very fast scene IO.
bool save_ybin_scene(const std::string& filename, const scene* scn,
    bool save_textures, bool skip_missing) {
    auto fs = fopen(filename.c_str(), "wb");
    if (!fs) return false;
    fclose_guard fs_{fs};
    if (!serialize_scene((scene*)scn, fs, true)) return false;
    fclose(fs);
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHAPE IO
// -----------------------------------------------------------------------------
namespace ygl {

// Reset mesh data
void reset_mesh_data(std::vector<int>& points, std::vector<vec2i>& lines,
    std::vector<vec3i>& triangles, std::vector<vec3f>& pos,
    std::vector<vec3f>& norm, std::vector<vec2f>& texcoord,
    std::vector<vec4f>& color, std::vector<float>& radius) {
    points    = {};
    lines     = {};
    triangles = {};
    pos       = {};
    norm      = {};
    texcoord  = {};
    color     = {};
    radius    = {};
}

// Load ply mesh
bool load_mesh(const std::string& filename, std::vector<int>& points,
    std::vector<vec2i>& lines, std::vector<vec3i>& triangles,
    std::vector<vec3f>& pos, std::vector<vec3f>& norm,
    std::vector<vec2f>& texcoord, std::vector<vec4f>& color,
    std::vector<float>& radius) {
    auto ext = get_extension(filename);
    if (ext == "ply" || ext == "PLY") {
        return load_ply_mesh(filename, points, lines, triangles, pos, norm,
            texcoord, color, radius);
    } else if (ext == "obj" || ext == "OBJ") {
        return load_obj_mesh(
            filename, points, lines, triangles, pos, norm, texcoord);
    } else {
        reset_mesh_data(
            points, lines, triangles, pos, norm, texcoord, color, radius);
        return false;
    }
}

// Save ply mesh
bool save_mesh(const std::string& filename, const std::vector<int>& points,
    const std::vector<vec2i>& lines, const std::vector<vec3i>& triangles,
    const std::vector<vec3f>& pos, const std::vector<vec3f>& norm,
    const std::vector<vec2f>& texcoord, const std::vector<vec4f>& color,
    const std::vector<float>& radius, bool ascii) {
    auto ext = get_extension(filename);
    if (ext == "ply" || ext == "PLY") {
        return save_ply_mesh(filename, points, lines, triangles, pos, norm,
            texcoord, color, radius, ascii);
    } else if (ext == "obj" || ext == "OBJ") {
        return save_obj_mesh(
            filename, points, lines, triangles, pos, norm, texcoord);
    } else {
        return false;
    }
}

// prepare obj line (remove comments and normalize whitespace)
void normalize_ply_line(char* s) {
    while (*s) {
        if (*s == '\t' || *s == '\r' || *s == '\n') {
            *s++ = ' ';
        } else {
            s++;
        }
    }
}

// Load ply mesh
ply_data load_ply(const std::string& filename) {
    auto ply = ply_data{};
    if (!load_ply(filename, ply)) return {};
    return ply;
}

// Load ply mesh
bool load_ply(const std::string& filename, ply_data& ply) {
    // clear
    ply = {};

    // open file
    auto fs = fopen(filename.c_str(), "rb");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // parse header
    auto ascii = false;
    char line[4096];
    while (fgets(line, sizeof(line), fs)) {
        normalize_ply_line(line);
        auto ss  = line;
        auto cmd = parse_string(ss);
        if (cmd == "") continue;
        if (cmd == "ply") {
        } else if (cmd == "comment") {
        } else if (cmd == "format") {
            auto fmt = parse_string(ss);
            if (fmt != "ascii" && fmt != "binary_little_endian") return false;
            ascii = fmt == "ascii";
        } else if (cmd == "element") {
            auto elem  = ply_element();
            elem.name  = parse_string(ss);
            elem.count = parse_int(ss);
            ply.elements.push_back(elem);
        } else if (cmd == "property") {
            auto prop = ply_property();
            auto type = parse_string(ss);
            if (type == "list") {
                auto count_type = parse_string(ss);
                auto elem_type  = parse_string(ss);
                if (count_type != "uchar" && count_type != "uint8")
                    throw std::runtime_error("unsupported ply list type");
                if (elem_type != "int")
                    throw std::runtime_error("unsupported ply list type");
                prop.type = ply_type::ply_int_list;
            } else if (type == "float") {
                prop.type = ply_type::ply_float;
            } else if (type == "uchar" || type == "uint8") {
                prop.type = ply_type::ply_uchar;
            } else if (type == "int") {
                prop.type = ply_type::ply_int;
            } else {
                return false;
            }
            prop.name = parse_string(ss);
            prop.scalars.resize(ply.elements.back().count);
            if (prop.type == ply_type::ply_int_list)
                prop.lists.resize(ply.elements.back().count);
            ply.elements.back().properties.push_back(prop);
        } else if (cmd == "end_header") {
            break;
        } else {
            return false;
        }
    }

    // parse content
    for (auto& elem : ply.elements) {
        for (auto vid = 0; vid < elem.count; vid++) {
            auto ss = (char*)nullptr;
            if (ascii) {
                if (!fgets(line, sizeof(line), fs)) return false;
                ss = line;
            }
            for (auto pid = 0; pid < elem.properties.size(); pid++) {
                auto& prop = elem.properties[pid];
                if (prop.type == ply_type::ply_float) {
                    auto v = 0.0f;
                    if (ascii) {
                        v = parse_float(ss);
                    } else {
                        if (fread((char*)&v, 4, 1, fs) != 1) return false;
                    }
                    prop.scalars[vid] = v;
                } else if (prop.type == ply_type::ply_int) {
                    auto v = 0;
                    if (ascii) {
                        v = parse_int(ss);
                    } else {
                        if (fread((char*)&v, 4, 1, fs) != 1) return false;
                    }
                    prop.scalars[vid] = v;
                } else if (prop.type == ply_type::ply_uchar) {
                    auto vc = (unsigned char)0;
                    if (ascii) {
                        auto v = parse_int(ss);
                        vc     = (unsigned char)v;
                    } else {
                        if (fread((char*)&vc, 1, 1, fs) != 1) return false;
                    }
                    prop.scalars[vid] = vc / 255.0f;
                } else if (prop.type == ply_type::ply_int_list) {
                    auto vc = (unsigned char)0;
                    if (ascii) {
                        auto v = parse_int(ss);
                        vc     = (unsigned char)v;
                    } else {
                        if (fread((char*)&vc, 1, 1, fs) != 1) return false;
                    }
                    prop.scalars[vid] = vc;
                    for (auto i = 0; i < (int)prop.scalars[vid]; i++)
                        if (ascii) {
                            prop.lists[vid][i] = parse_int(ss);
                        } else {
                            if (fread((char*)&prop.lists[vid][i], 4, 1, fs) != 1)
                                return false;
                        }
                } else {
                    return false;
                }
            }
        }
    }

    fclose(fs);

    return true;
}

// Load ply mesh
bool load_ply_mesh(const std::string& filename, std::vector<int>& points,
    std::vector<vec2i>& lines, std::vector<vec3i>& triangles,
    std::vector<vec3f>& pos, std::vector<vec3f>& norm,
    std::vector<vec2f>& texcoord, std::vector<vec4f>& color,
    std::vector<float>& radius) {
    // clear
    reset_mesh_data(
        points, lines, triangles, pos, norm, texcoord, color, radius);

    // load ply
    auto ply = ply_data{};
    if (!load_ply(filename, ply)) return false;

    // copy vertex data
    for (auto& elem : ply.elements) {
        if (elem.name != "vertex") continue;
        auto count = elem.count;
        for (auto& prop : elem.properties) {
            auto vals        = prop.scalars.data();
            auto copy_floats = [vals, count](auto& vert, const auto& def,
                                   int stride, int offset) {
                if (vert.size() != count) vert.resize(count, def);
                auto dst = (float*)vert.data();
                for (auto i = 0; i < count; i++)
                    dst[i * stride + offset] = vals[i];
            };
            if (prop.name == "x") copy_floats(pos, zero3f, 3, 0);
            if (prop.name == "y") copy_floats(pos, zero3f, 3, 1);
            if (prop.name == "z") copy_floats(pos, zero3f, 3, 2);
            if (prop.name == "nx") copy_floats(norm, zero3f, 3, 0);
            if (prop.name == "ny") copy_floats(norm, zero3f, 3, 1);
            if (prop.name == "nz") copy_floats(norm, zero3f, 3, 2);
            if (prop.name == "u") copy_floats(texcoord, zero2f, 2, 0);
            if (prop.name == "v") copy_floats(texcoord, zero2f, 2, 1);
            if (prop.name == "red") copy_floats(color, vec4f{0, 0, 0, 1}, 4, 0);
            if (prop.name == "green")
                copy_floats(color, vec4f{0, 0, 0, 1}, 4, 1);
            if (prop.name == "blue")
                copy_floats(color, vec4f{0, 0, 0, 1}, 4, 2);
            if (prop.name == "alpha")
                copy_floats(color, vec4f{0, 0, 0, 1}, 4, 3);
            if (prop.name == "radius") copy_floats(radius, 0.0f, 1, 0);
        }
    }

    // copy triangle data
    for (auto& elem : ply.elements) {
        if (elem.name != "face") continue;
        auto count = elem.count;
        for (auto& prop : elem.properties) {
            if (prop.name == "vertex_indices") {
                for (auto fid = 0; fid < count; fid++) {
                    auto& list = prop.lists[fid];
                    for (auto i = 2; i < (int)prop.scalars[fid]; i++)
                        triangles.push_back({list[0], list[i - 1], list[i]});
                }
            }
        }
    }

    // done
    return true;
}

// Save ply mesh
bool save_ply_mesh(const std::string& filename, const std::vector<int>& points,
    const std::vector<vec2i>& lines, const std::vector<vec3i>& triangles,
    const std::vector<vec3f>& pos, const std::vector<vec3f>& norm,
    const std::vector<vec2f>& texcoord, const std::vector<vec4f>& color,
    const std::vector<float>& radius, bool ascii) {
    auto fs = fopen(filename.c_str(), "wb");
    if (!fs) return false;
    fclose_guard fs_{fs};

    // header
    print(fs, "ply\n");
    if (ascii)
        print(fs, "format ascii 1.0\n");
    else
        print(fs, "format binary_little_endian 1.0\n");
    print(fs, "element vertex {}\n", (int)pos.size());
    if (!pos.empty())
        print(fs, "property float x\nproperty float y\nproperty float z\n");
    if (!norm.empty())
        print(fs,
            "property float nx\nproperty float ny\nproperty float "
            "nz\n");
    if (!texcoord.empty()) print(fs, "property float u\nproperty float v\n");
    if (!color.empty())
        print(fs,
            "property float red\nproperty float green\nproperty float "
            "blue\nproperty float alpha\n");
    if (!radius.empty()) print(fs, "property float radius\n");
    if (!triangles.empty()) {
        print(fs, "element face {}\n", (int)triangles.size());
        print(fs, "property list uchar int vertex_indices\n");
    }
    if (!lines.empty()) {
        print(fs, "element line {}\n", (int)lines.size());
        print(fs, "property list uchar int vertex_indices\n");
    }
    print(fs, "end_header\n");

    // body
    if (ascii) {
        // write vertex data
        for (auto i = 0; i < pos.size(); i++) {
            if (!pos.empty()) print(fs, "{} ", pos[i]);
            if (!norm.empty()) print(fs, "{} ", norm[i]);
            if (!texcoord.empty()) print(fs, "{} ", texcoord[i]);
            if (!color.empty()) print(fs, "{} ", color[i]);
            if (!radius.empty()) print(fs, "{} ", radius[i]);
            print(fs, "\n");
        }

        // write face data
        for (auto i = 0; i < triangles.size(); i++)
            print(fs, "3 {}\n", triangles[i]);
        for (auto i = 0; i < lines.size(); i++) print(fs, "2 {}\n", lines[i]);
    } else {
        // write vertex data
        for (auto i = 0; i < pos.size(); i++) {
            if (!pos.empty()) fwrite((char*)&pos[i], 1, sizeof(pos[i]), fs);
            if (!norm.empty()) fwrite((char*)&norm[i], 1, sizeof(norm[i]), fs);
            if (!texcoord.empty())
                fwrite((char*)&texcoord[i], 1, sizeof(texcoord[i]), fs);
            if (!color.empty())
                fwrite((char*)&color[i], 1, sizeof(color[i]), fs);
            if (!radius.empty())
                fwrite((char*)&radius[i], 1, sizeof(radius[i]), fs);
        }

        // write face data
        for (auto i = 0; i < triangles.size(); i++) {
            auto n = (byte)3;
            fwrite((char*)&n, 1, sizeof(n), fs);
            fwrite((char*)&triangles[i], 1, sizeof(triangles[i]), fs);
        }
        for (auto i = 0; i < lines.size(); i++) {
            auto n = (byte)3;
            fwrite((char*)&n, 1, sizeof(n), fs);
            fwrite((char*)&lines[i], 1, sizeof(lines[i]), fs);
        }
    }

    // done
    fclose(fs);

    // done
    return true;
}

// Load ply mesh
bool load_obj_mesh(const std::string& filename, std::vector<int>& points,
    std::vector<vec2i>& lines, std::vector<vec3i>& triangles,
    std::vector<vec3f>& pos, std::vector<vec3f>& norm,
    std::vector<vec2f>& texcoord, bool flip_texcoord) {
    // clear
    auto color  = std::vector<vec4f>{};
    auto radius = std::vector<float>{};
    reset_mesh_data(
        points, lines, triangles, pos, norm, texcoord, color, radius);

    // obj vertices
    auto opos      = std::deque<vec3f>();
    auto onorm     = std::deque<vec3f>();
    auto otexcoord = std::deque<vec2f>();

    // vertex maps
    auto vert_map = std::unordered_map<obj_vertex, int, obj_vertex_hash>();

    // Add  vertices to the current shape
    auto add_verts = [&](const std::vector<obj_vertex>& verts) {
        for (auto& vert : verts) {
            auto it = vert_map.find(vert);
            if (it != vert_map.end()) continue;
            auto nverts = (int)pos.size();
            vert_map.insert(it, {vert, nverts});
            if (vert.pos) pos.push_back(opos.at(vert.pos - 1));
            if (vert.texcoord)
                texcoord.push_back(otexcoord.at(vert.texcoord - 1));
            if (vert.norm) norm.push_back(onorm.at(vert.norm - 1));
        }
    };

    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 2; i < verts.size(); i++)
            triangles.push_back({vert_map.at(verts[0]),
                vert_map.at(verts[i - 1]), vert_map.at(verts[i])});
    };
    cb.line = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 1; i < verts.size(); i++)
            lines.push_back({vert_map.at(verts[i - 1]), vert_map.at(verts[i])});
    };
    cb.point = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        for (auto i = 0; i < verts.size(); i++)
            points.push_back(vert_map.at(verts[i]));
    };

    // load obj
    return load_obj(filename, cb, flip_texcoord);
}

// Load ply mesh
bool save_obj_mesh(const std::string& filename, const std::vector<int>& points,
    const std::vector<vec2i>& lines, const std::vector<vec3i>& triangles,
    const std::vector<vec3f>& pos, const std::vector<vec3f>& norm,
    const std::vector<vec2f>& texcoord, bool flip_texcoord) {
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    for (auto& p : pos) print(fs, "v {}\n", p);
    for (auto& n : norm) print(fs, "vn {}\n", n);
    for (auto& t : texcoord)
        print(fs, "vt {}\n", t.x, (flip_texcoord) ? 1 - t.y : t.y);
    auto mask = obj_vertex{1, texcoord.empty() ? 0 : 1, norm.empty() ? 0 : 1};
    auto vert = [mask](int i) {
        return obj_vertex{
            (i + 1) * mask.pos, (i + 1) * mask.texcoord, (i + 1) * mask.norm};
    };
    for (auto& t : triangles)
        print(fs, "f {}\n", to_string(vert(t.x)).c_str(),
            to_string(vert(t.y)).c_str(), to_string(vert(t.z)).c_str());
    for (auto& l : lines)
        print(fs, "l {}\n", to_string(vert(l.x)).c_str(),
            to_string(vert(l.y)).c_str());
    for (auto& p : points) print(fs, "p {}\n", to_string(vert(p)).c_str());

    fclose(fs);
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHAPE IO
// -----------------------------------------------------------------------------
namespace ygl {

// Reset mesh data
void reset_fvmesh_data(std::vector<vec4i>& quads_pos, std::vector<vec3f>& pos,
    std::vector<vec4i>& quads_norm, std::vector<vec3f>& norm,
    std::vector<vec4i>& quads_texcoord, std::vector<vec2f>& texcoord,
    std::vector<vec4i>& quads_color, std::vector<vec4f>& color) {
    quads_pos      = {};
    pos            = {};
    quads_norm     = {};
    norm           = {};
    quads_texcoord = {};
    texcoord       = {};
    quads_color    = {};
    color          = {};
}

// Load mesh
bool load_fvmesh(const std::string& filename, std::vector<vec4i>& quads_pos,
    std::vector<vec3f>& pos, std::vector<vec4i>& quads_norm,
    std::vector<vec3f>& norm, std::vector<vec4i>& quads_texcoord,
    std::vector<vec2f>& texcoord, std::vector<vec4i>& quads_color,
    std::vector<vec4f>& color) {
    auto ext = get_extension(filename);
    if (ext == "obj" || ext == "OBJ") {
        return load_obj_fvmesh(filename, quads_pos, pos, quads_norm, norm,
            quads_texcoord, texcoord);
    } else {
        reset_fvmesh_data(quads_pos, pos, quads_norm, norm, quads_texcoord,
            texcoord, quads_color, color);
        return false;
    }
}

// Save mesh
bool save_fvmesh(const std::string& filename,
    const std::vector<vec4i>& quads_pos, const std::vector<vec3f>& pos,
    const std::vector<vec4i>& quads_norm, const std::vector<vec3f>& norm,
    const std::vector<vec4i>& quads_texcoord,
    const std::vector<vec2f>& texcoord, const std::vector<vec4i>& quads_color,
    const std::vector<vec4f>& color, bool ascii) {
    auto ext = get_extension(filename);
    if (ext == "obj" || ext == "OBJ") {
        return save_obj_fvmesh(filename, quads_pos, pos, quads_norm, norm,
            quads_texcoord, texcoord);
    } else {
        return false;
    }
}

// Load obj mesh
bool load_obj_fvmesh(const std::string& filename, std::vector<vec4i>& quads_pos,
    std::vector<vec3f>& pos, std::vector<vec4i>& quads_norm,
    std::vector<vec3f>& norm, std::vector<vec4i>& quads_texcoord,
    std::vector<vec2f>& texcoord, bool flip_texcoord) {
    // clear
    std::vector<vec4i> quads_color;
    std::vector<vec4f> color;
    reset_fvmesh_data(quads_pos, pos, quads_norm, norm, quads_texcoord,
        texcoord, quads_color, color);

    // obj vertex
    auto opos      = std::deque<vec3f>();
    auto onorm     = std::deque<vec3f>();
    auto otexcoord = std::deque<vec2f>();

    // vertex maps
    auto pos_map      = std::unordered_map<int, int>();
    auto texcoord_map = std::unordered_map<int, int>();
    auto norm_map     = std::unordered_map<int, int>();

    // add vertex
    auto add_verts = [&](const std::vector<obj_vertex>& verts) {
        for (auto& vert : verts) {
            if (!vert.pos) continue;
            auto pos_it = pos_map.find(vert.pos);
            if (pos_it != pos_map.end()) continue;
            auto nverts = (int)pos.size();
            pos_map.insert(pos_it, {vert.pos, nverts});
            pos.push_back(opos.at(vert.pos - 1));
        }
        for (auto& vert : verts) {
            if (!vert.texcoord) continue;
            auto texcoord_it = texcoord_map.find(vert.texcoord);
            if (texcoord_it != texcoord_map.end()) continue;
            auto nverts = (int)texcoord.size();
            texcoord_map.insert(texcoord_it, {vert.texcoord, nverts});
            texcoord.push_back(otexcoord.at(vert.texcoord - 1));
        }
        for (auto& vert : verts) {
            if (!vert.norm) continue;
            auto norm_it = norm_map.find(vert.norm);
            if (norm_it != norm_map.end()) continue;
            auto nverts = (int)norm.size();
            norm_map.insert(norm_it, {vert.norm, nverts});
            norm.push_back(onorm.at(vert.norm - 1));
        }
    };

    auto cb     = obj_callbacks();
    cb.vert     = [&](vec3f v) { opos.push_back(v); };
    cb.norm     = [&](vec3f v) { onorm.push_back(v); };
    cb.texcoord = [&](vec2f v) { otexcoord.push_back(v); };
    cb.face     = [&](const std::vector<obj_vertex>& verts) {
        add_verts(verts);
        if (verts.size() == 4) {
            if (verts[0].pos) {
                quads_pos.push_back(
                    {pos_map.at(verts[0].pos), pos_map.at(verts[1].pos),
                        pos_map.at(verts[2].pos), pos_map.at(verts[3].pos)});
            }
            if (verts[0].texcoord) {
                quads_texcoord.push_back({texcoord_map.at(verts[0].texcoord),
                    texcoord_map.at(verts[1].texcoord),
                    texcoord_map.at(verts[2].texcoord),
                    texcoord_map.at(verts[3].texcoord)});
            }
            if (verts[0].norm) {
                quads_norm.push_back({norm_map.at(verts[0].norm),
                    norm_map.at(verts[1].norm), norm_map.at(verts[2].norm),
                    norm_map.at(verts[3].norm)});
            }
        } else {
            if (verts[0].pos) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_pos.push_back({pos_map.at(verts[0].pos),
                        pos_map.at(verts[1].pos), pos_map.at(verts[i].pos),
                        pos_map.at(verts[i].pos)});
            }
            if (verts[0].texcoord) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_texcoord.push_back({texcoord_map.at(verts[0].texcoord),
                        texcoord_map.at(verts[1].texcoord),
                        texcoord_map.at(verts[i].texcoord),
                        texcoord_map.at(verts[i].texcoord)});
            }
            if (verts[0].norm) {
                for (auto i = 2; i < verts.size(); i++)
                    quads_norm.push_back({norm_map.at(verts[0].norm),
                        norm_map.at(verts[1].norm), norm_map.at(verts[i].norm),
                        norm_map.at(verts[i].norm)});
            }
        }
    };

    // load obj
    return load_obj(filename, cb, flip_texcoord);
}

// Load ply mesh
bool save_obj_fvmesh(const std::string& filename,
    const std::vector<vec4i>& quads_pos, const std::vector<vec3f>& pos,
    const std::vector<vec4i>& quads_norm, const std::vector<vec3f>& norm,
    const std::vector<vec4i>& quads_texcoord,
    const std::vector<vec2f>& texcoord, bool flip_texcoord) {
    auto fs = fopen(filename.c_str(), "wt");
    if (!fs) return false;
    fclose_guard fs_{fs};

    for (auto& p : pos) fprintf(fs, "v %g %g %g\n", p.x, p.y, p.z);
    for (auto& n : norm) fprintf(fs, "vn %g %g %g\n", n.x, n.y, n.z);
    for (auto& t : texcoord)
        fprintf(fs, "vt %g %g\n", t.x, (flip_texcoord) ? 1 - t.y : t.y);
    auto mask = obj_vertex{1, texcoord.empty() ? 0 : 1, norm.empty() ? 0 : 1};
    auto vert = [mask](int pif, int ti, int ni) {
        return obj_vertex{(pif + 1) * mask.pos, (ti + 1) * mask.texcoord,
            (ni + 1) * mask.norm};
    };
    for (auto i = 0; i < quads_pos.size(); i++) {
        auto qp = quads_pos.at(i);
        auto qt = !quads_texcoord.empty() ? quads_texcoord.at(i) :
                                            vec4i{-1, -1, -1, -1};
        auto qn = !quads_norm.empty() ? quads_norm.at(i) : vec4i{-1, -1, -1, -1};
        if (qp.z != qp.w)
            fprintf(fs, "f %s %s %s %s\n",
                to_string(vert(qp.x, qt.x, qn.x)).c_str(),
                to_string(vert(qp.y, qt.y, qn.y)).c_str(),
                to_string(vert(qp.z, qt.z, qn.z)).c_str(),
                to_string(vert(qp.w, qt.w, qn.w)).c_str());
        else
            fprintf(fs, "f %s %s %s\n",
                to_string(vert(qp.x, qt.x, qn.x)).c_str(),
                to_string(vert(qp.y, qt.y, qn.y)).c_str(),
                to_string(vert(qp.z, qt.z, qn.z)).c_str());
    }

    fclose(fs);
    return true;
}

}  // namespace ygl
