//
// Implementation for Yocto/GL.
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
// LICENCE OF INCLUDED SOFTWARE FOR PERLIN NOISE
// https://github.com/nothings/stb/blob/master/stb_perlin.h
//
// ------------------------------------------------------------------------------
// ALTERNATIVE B - Public Domain (www.unlicense.org)
// This is free and unencumbered software released into the public domain.
// Anyone is free to copy, modify, publish, use, compile, sell, or distribute
// this software, either in source code form or as a compiled binary, for any
// purpose, commercial or non-commercial, and by any means. In jurisdictions
// that recognize copyright laws, the author or authors of this software
// dedicate any and all copyright interest in the software to the public domain.
// We make this dedication for the benefit of the public at large and to the
// detriment of our heirs and successors. We intend this dedication to be an
// overt act of relinquishment in perpetuity of all present and future rights to
// this software under copyright law.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ------------------------------------------------------------------------------

#include "ygl.h"

#include <atomic>

#if YGL_EMBREE
#include <embree3/rtcore.h>
#endif

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PERLIN NOISE
// -----------------------------------------------------------------------------
namespace ygl {

// clang-format off
float stb__perlin_lerp(float a, float b, float t)
{
   return a + (b-a) * t;
}

int stb__perlin_fastfloor(float a)
{
    int ai = (int) a;
    return (a < ai) ? ai-1 : ai;
}

// different grad function from Perlin's, but easy to modify to match reference
float stb__perlin_grad(int hash, float x, float y, float z)
{
   static float basis[12][4] =
   {
      {  1, 1, 0 },
      { -1, 1, 0 },
      {  1,-1, 0 },
      { -1,-1, 0 },
      {  1, 0, 1 },
      { -1, 0, 1 },
      {  1, 0,-1 },
      { -1, 0,-1 },
      {  0, 1, 1 },
      {  0,-1, 1 },
      {  0, 1,-1 },
      {  0,-1,-1 },
   };

   // perlin's gradient has 12 cases so some get used 1/16th of the time
   // and some 2/16ths. We reduce bias by changing those fractions
   // to 5/64ths and 6/64ths, and the same 4 cases get the extra weight.
   static unsigned char indices[64] =
   {
      0,1,2,3,4,5,6,7,8,9,10,11,
      0,9,1,11,
      0,1,2,3,4,5,6,7,8,9,10,11,
      0,1,2,3,4,5,6,7,8,9,10,11,
      0,1,2,3,4,5,6,7,8,9,10,11,
      0,1,2,3,4,5,6,7,8,9,10,11,
   };

   // if you use reference permutation table, change 63 below to 15 to match reference
   // (this is why the ordering of the table above is funky)
   float *grad = basis[indices[hash & 63]];
   return grad[0]*x + grad[1]*y + grad[2]*z;
}

float stb_perlin_noise3(float x, float y, float z, int x_wrap, int y_wrap, int z_wrap)
{
    // not same permutation table as Perlin's reference to avoid copyright issues;
    // Perlin's table can be found at http://mrl.nyu.edu/~perlin/noise/
    // @OPTIMIZE: should this be unsigned char instead of int for cache?
    static unsigned char stb__perlin_randtab[512] =
    {
    23, 125, 161, 52, 103, 117, 70, 37, 247, 101, 203, 169, 124, 126, 44, 123,
    152, 238, 145, 45, 171, 114, 253, 10, 192, 136, 4, 157, 249, 30, 35, 72,
    175, 63, 77, 90, 181, 16, 96, 111, 133, 104, 75, 162, 93, 56, 66, 240,
    8, 50, 84, 229, 49, 210, 173, 239, 141, 1, 87, 18, 2, 198, 143, 57,
    225, 160, 58, 217, 168, 206, 245, 204, 199, 6, 73, 60, 20, 230, 211, 233,
    94, 200, 88, 9, 74, 155, 33, 15, 219, 130, 226, 202, 83, 236, 42, 172,
    165, 218, 55, 222, 46, 107, 98, 154, 109, 67, 196, 178, 127, 158, 13, 243,
    65, 79, 166, 248, 25, 224, 115, 80, 68, 51, 184, 128, 232, 208, 151, 122,
    26, 212, 105, 43, 179, 213, 235, 148, 146, 89, 14, 195, 28, 78, 112, 76,
    250, 47, 24, 251, 140, 108, 186, 190, 228, 170, 183, 139, 39, 188, 244, 246,
    132, 48, 119, 144, 180, 138, 134, 193, 82, 182, 120, 121, 86, 220, 209, 3,
    91, 241, 149, 85, 205, 150, 113, 216, 31, 100, 41, 164, 177, 214, 153, 231,
    38, 71, 185, 174, 97, 201, 29, 95, 7, 92, 54, 254, 191, 118, 34, 221,
    131, 11, 163, 99, 234, 81, 227, 147, 156, 176, 17, 142, 69, 12, 110, 62,
    27, 255, 0, 194, 59, 116, 242, 252, 19, 21, 187, 53, 207, 129, 64, 135,
    61, 40, 167, 237, 102, 223, 106, 159, 197, 189, 215, 137, 36, 32, 22, 5,

    // and a second copy so we don't need an extra mask or static initializer
    23, 125, 161, 52, 103, 117, 70, 37, 247, 101, 203, 169, 124, 126, 44, 123,
    152, 238, 145, 45, 171, 114, 253, 10, 192, 136, 4, 157, 249, 30, 35, 72,
    175, 63, 77, 90, 181, 16, 96, 111, 133, 104, 75, 162, 93, 56, 66, 240,
    8, 50, 84, 229, 49, 210, 173, 239, 141, 1, 87, 18, 2, 198, 143, 57,
    225, 160, 58, 217, 168, 206, 245, 204, 199, 6, 73, 60, 20, 230, 211, 233,
    94, 200, 88, 9, 74, 155, 33, 15, 219, 130, 226, 202, 83, 236, 42, 172,
    165, 218, 55, 222, 46, 107, 98, 154, 109, 67, 196, 178, 127, 158, 13, 243,
    65, 79, 166, 248, 25, 224, 115, 80, 68, 51, 184, 128, 232, 208, 151, 122,
    26, 212, 105, 43, 179, 213, 235, 148, 146, 89, 14, 195, 28, 78, 112, 76,
    250, 47, 24, 251, 140, 108, 186, 190, 228, 170, 183, 139, 39, 188, 244, 246,
    132, 48, 119, 144, 180, 138, 134, 193, 82, 182, 120, 121, 86, 220, 209, 3,
    91, 241, 149, 85, 205, 150, 113, 216, 31, 100, 41, 164, 177, 214, 153, 231,
    38, 71, 185, 174, 97, 201, 29, 95, 7, 92, 54, 254, 191, 118, 34, 221,
    131, 11, 163, 99, 234, 81, 227, 147, 156, 176, 17, 142, 69, 12, 110, 62,
    27, 255, 0, 194, 59, 116, 242, 252, 19, 21, 187, 53, 207, 129, 64, 135,
    61, 40, 167, 237, 102, 223, 106, 159, 197, 189, 215, 137, 36, 32, 22, 5,
    };

   float u,v,w;
   float n000,n001,n010,n011,n100,n101,n110,n111;
   float n00,n01,n10,n11;
   float n0,n1;

   unsigned int x_mask = (x_wrap-1) & 255;
   unsigned int y_mask = (y_wrap-1) & 255;
   unsigned int z_mask = (z_wrap-1) & 255;
   int px = stb__perlin_fastfloor(x);
   int py = stb__perlin_fastfloor(y);
   int pz = stb__perlin_fastfloor(z);
   int x0 = px & x_mask, x1 = (px+1) & x_mask;
   int y0 = py & y_mask, y1 = (py+1) & y_mask;
   int z0 = pz & z_mask, z1 = (pz+1) & z_mask;
   int r0,r1, r00,r01,r10,r11;

   #define stb__perlin_ease(a)   (((a*6-15)*a + 10) * a * a * a)

   x -= px; u = stb__perlin_ease(x);
   y -= py; v = stb__perlin_ease(y);
   z -= pz; w = stb__perlin_ease(z);

   r0 = stb__perlin_randtab[x0];
   r1 = stb__perlin_randtab[x1];

   r00 = stb__perlin_randtab[r0+y0];
   r01 = stb__perlin_randtab[r0+y1];
   r10 = stb__perlin_randtab[r1+y0];
   r11 = stb__perlin_randtab[r1+y1];

   n000 = stb__perlin_grad(stb__perlin_randtab[r00+z0], x  , y  , z   );
   n001 = stb__perlin_grad(stb__perlin_randtab[r00+z1], x  , y  , z-1 );
   n010 = stb__perlin_grad(stb__perlin_randtab[r01+z0], x  , y-1, z   );
   n011 = stb__perlin_grad(stb__perlin_randtab[r01+z1], x  , y-1, z-1 );
   n100 = stb__perlin_grad(stb__perlin_randtab[r10+z0], x-1, y  , z   );
   n101 = stb__perlin_grad(stb__perlin_randtab[r10+z1], x-1, y  , z-1 );
   n110 = stb__perlin_grad(stb__perlin_randtab[r11+z0], x-1, y-1, z   );
   n111 = stb__perlin_grad(stb__perlin_randtab[r11+z1], x-1, y-1, z-1 );

   n00 = stb__perlin_lerp(n000,n001,w);
   n01 = stb__perlin_lerp(n010,n011,w);
   n10 = stb__perlin_lerp(n100,n101,w);
   n11 = stb__perlin_lerp(n110,n111,w);

   n0 = stb__perlin_lerp(n00,n01,v);
   n1 = stb__perlin_lerp(n10,n11,v);

   return stb__perlin_lerp(n0,n1,u);
}

float stb_perlin_ridge_noise3(float x, float y, float z,float lacunarity, float gain, float offset, int octaves,int x_wrap, int y_wrap, int z_wrap)
{
   int i;
   float frequency = 1.0f;
   float prev = 1.0f;
   float amplitude = 0.5f;
   float sum = 0.0f;

   for (i = 0; i < octaves; i++) {
      float r = (float)(stb_perlin_noise3(x*frequency,y*frequency,z*frequency,x_wrap,y_wrap,z_wrap));
      r = r<0 ? -r : r; // fabs()
      r = offset - r;
      r = r*r;
      sum += r*amplitude*prev;
      prev = r;
      frequency *= lacunarity;
      amplitude *= gain;
   }
   return sum;
}

float stb_perlin_fbm_noise3(float x, float y, float z,float lacunarity, float gain, int octaves,int x_wrap, int y_wrap, int z_wrap)
{
   int i;
   float frequency = 1.0f;
   float amplitude = 1.0f;
   float sum = 0.0f;

   for (i = 0; i < octaves; i++) {
      sum += stb_perlin_noise3(x*frequency,y*frequency,z*frequency,x_wrap,y_wrap,z_wrap)*amplitude;
      frequency *= lacunarity;
      amplitude *= gain;
   }
   return sum;
}

float stb_perlin_turbulence_noise3(float x, float y, float z, float lacunarity, float gain, int octaves,int x_wrap, int y_wrap, int z_wrap)
{
   int i;
   float frequency = 1.0f;
   float amplitude = 1.0f;
   float sum = 0.0f;

   for (i = 0; i < octaves; i++) {
      float r = stb_perlin_noise3(x*frequency,y*frequency,z*frequency,x_wrap,y_wrap,z_wrap)*amplitude;
      r = r<0 ? -r : r; // fabs()
      sum += r;
      frequency *= lacunarity;
      amplitude *= gain;
   }
   return sum;
}
// clang-format on

// adapeted  stb_perlin.h
float perlin_noise(const vec3f& p, const vec3i& wrap) {
    return stb_perlin_noise3(p.x, p.y, p.z, wrap.x, wrap.y, wrap.z);
}

// adapeted  stb_perlin.h
float perlin_ridge_noise(const vec3f& p, float lacunarity, float gain,
    float offset, int octaves, const vec3i& wrap) {
    return stb_perlin_ridge_noise3(p.x, p.y, p.z, lacunarity, gain, offset,
        octaves, wrap.x, wrap.y, wrap.z);
}

// adapeted  stb_perlin.h
float perlin_fbm_noise(const vec3f& p, float lacunarity, float gain,
    int octaves, const vec3i& wrap) {
    return stb_perlin_fbm_noise3(
        p.x, p.y, p.z, lacunarity, gain, octaves, wrap.x, wrap.y, wrap.z);
}

// adapeted  stb_perlin.h
float perlin_turbulence_noise(const vec3f& p, float lacunarity, float gain,
    int octaves, const vec3i& wrap) {
    return stb_perlin_turbulence_noise3(
        p.x, p.y, p.z, lacunarity, gain, octaves, wrap.x, wrap.y, wrap.z);
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHAPE UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Compute per-vertex tangents for lines.
void compute_vertex_tangents(const vector<vec2i>& lines,
    const vector<vec3f>& positions, vector<vec3f>& tangents) {
    tangents = vector<vec3f>(positions.size(), zero3f);
    for (auto& l : lines) {
        auto tangent = line_tangent(positions[l.x], positions[l.y]);
        auto length  = line_length(positions[l.x], positions[l.y]);
        tangents[l.x] += tangent * length;
        tangents[l.y] += tangent * length;
    }
    for (auto& tangent : tangents) tangent = normalize(tangent);
}

// Compute per-vertex normals for triangles.
void compute_vertex_normals(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, vector<vec3f>& normals) {
    normals = vector<vec3f>(positions.size(), zero3f);
    for (auto& t : triangles) {
        auto normal = triangle_normal(
            positions[t.x], positions[t.y], positions[t.z]);
        auto area = triangle_area(positions[t.x], positions[t.y], positions[t.z]);
        normals[t.x] += normal * area;
        normals[t.y] += normal * area;
        normals[t.z] += normal * area;
    }
    for (auto& normal : normals) normal = normalize(normal);
}

// Compute per-vertex normals for quads.
void compute_vertex_normals(const vector<vec4i>& quads,
    const vector<vec3f>& positions, vector<vec3f>& normals) {
    normals = vector<vec3f>(positions.size(), zero3f);
    for (auto q : quads) {
        auto normal = quad_normal(
            positions[q.x], positions[q.y], positions[q.z], positions[q.w]);
        auto area = quad_area(
            positions[q.x], positions[q.y], positions[q.z], positions[q.w]);
        normals[q.x] += normal * area;
        normals[q.y] += normal * area;
        normals[q.z] += normal * area;
        if (q.z != q.w) normals[q.w] += normal * area;
    }
    for (auto& normal : normals) normal = normalize(normal);
}

// Compute per-vertex tangent frame for triangle meshes.
// Tangent space is defined by a four component vector.
// The first three components are the tangent with respect to the U texcoord.
// The fourth component is the sign of the tangent wrt the V texcoord.
// Tangent frame is useful in normal mapping.
void compute_tangent_spaces(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3f>& normals,
    const vector<vec2f>& texturecoords, vector<vec4f>& tangentspaces) {
    auto tangu = vector<vec3f>(positions.size(), zero3f);
    auto tangv = vector<vec3f>(positions.size(), zero3f);
    for (auto t : triangles) {
        auto tutv = triangle_tangents_fromuv(positions[t.x], positions[t.y],
            positions[t.z], texturecoords[t.x], texturecoords[t.y],
            texturecoords[t.z]);
        tutv      = {normalize(get<0>(tutv)), normalize(get<1>(tutv))};
        for (auto vid : {t.x, t.y, t.z}) tangu[vid] += get<0>(tutv);
        for (auto vid : {t.x, t.y, t.z}) tangv[vid] += get<1>(tutv);
    }
    for (auto& t : tangu) t = normalize(t);
    for (auto& t : tangv) t = normalize(t);
    tangentspaces = vector<vec4f>(positions.size(), zero4f);
    for (auto i = 0; i < positions.size(); i++) {
        tangu[i] = orthonormalize(tangu[i], normals[i]);
        auto s = (dot(cross(normals[i], tangu[i]), tangv[i]) < 0) ? -1.0f : 1.0f;
        tangentspaces[i] = {tangu[i].x, tangu[i].y, tangu[i].z, s};
    }
}

// Apply skinning
void compute_skinning(const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec4f>& weights,
    const vector<vec4i>& joints, const vector<frame3f>& xforms,
    vector<vec3f>& skinned_positions, vector<vec3f>& skinned_normals) {
    skinned_positions = vector<vec3f>(positions.size());
    skinned_normals   = vector<vec3f>(normals.size());
    for (auto i = 0; i < positions.size(); i++) {
        skinned_positions[i] = transform_point(xforms[joints[i].x], positions[i]) *
                                   weights[i].x +
                               transform_point(xforms[joints[i].y], positions[i]) *
                                   weights[i].y +
                               transform_point(xforms[joints[i].z], positions[i]) *
                                   weights[i].z +
                               transform_point(xforms[joints[i].w], positions[i]) *
                                   weights[i].w;
    }
    for (auto i = 0; i < positions.size(); i++) {
        skinned_normals[i] = normalize(
            transform_direction(xforms[joints[i].x], normals[i]) * weights[i].x +
            transform_direction(xforms[joints[i].y], normals[i]) * weights[i].y +
            transform_direction(xforms[joints[i].z], normals[i]) * weights[i].z +
            transform_direction(xforms[joints[i].w], normals[i]) * weights[i].w);
    }
}

// Apply skinning as specified in Khronos glTF
void compute_matrix_skinning(const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec4f>& weights,
    const vector<vec4i>& joints, const vector<mat4f>& xforms,
    vector<vec3f>& skinned_positions, vector<vec3f>& skinned_normals) {
    skinned_positions = vector<vec3f>(positions.size());
    skinned_normals   = vector<vec3f>(normals.size());
    for (auto i = 0; i < positions.size(); i++) {
        auto xform = xforms[joints[i].x] * weights[i].x +
                     xforms[joints[i].y] * weights[i].y +
                     xforms[joints[i].z] * weights[i].z +
                     xforms[joints[i].w] * weights[i].w;
        skinned_positions[i] = transform_point(xform, positions[i]);
        skinned_normals[i] = normalize(transform_direction(xform, normals[i]));
    }
}

// Initialize an edge map with elements.
void insert_edges(edge_map& emap, const vector<vec3i>& triangles) {
    for (auto& t : triangles) {
        insert_edge(emap, {t.x, t.y});
        insert_edge(emap, {t.y, t.z});
        insert_edge(emap, {t.z, t.x});
    }
}
void insert_edges(edge_map& emap, const vector<vec4i>& quads) {
    for (auto& q : quads) {
        insert_edge(emap, {q.x, q.y});
        insert_edge(emap, {q.y, q.z});
        if (q.z != q.w) insert_edge(emap, {q.z, q.w});
        insert_edge(emap, {q.w, q.x});
    }
}
// Insert an edge and return its index
int insert_edge(edge_map& emap, const vec2i& e) {
    auto es = vec2i{min(e.x, e.y), max(e.x, e.y)};
    auto it = emap.find(es);
    if (it == emap.end()) {
        auto idx = (int)emap.size();
        emap.insert(it, {es, {idx, 1}});
        return idx;
    } else {
        it->second.y += 1;
        return it->second.x;
    }
}
// Get the edge index
int get_edge_index(const edge_map& emap, const vec2i& e) {
    auto es = vec2i{min(e.x, e.y), max(e.x, e.y)};
    return emap.at(es).x;
}
// Get the edge index
int get_edge_count(const edge_map& emap, const vec2i& e) {
    auto es = vec2i{min(e.x, e.y), max(e.x, e.y)};
    return emap.at(es).y;
}
// Get a list of edges, boundary edges, boundary vertices
void get_edges(const edge_map& emap, vector<vec2i>& edges) {
    edges = vector<vec2i>(emap.size());
    for (auto& kv : emap) edges[kv.second.x] = kv.first;
}
void get_boundary(const edge_map& emap, vector<vec2i>& boundary) {
    boundary = vector<vec2i>();
    for (auto& kv : emap)
        if (kv.second.y < 2) boundary.push_back(kv.first);
}

// Convert quads to triangles
void convert_quads_to_triangles(
    const vector<vec4i>& quads, vector<vec3i>& triangles) {
    triangles = vector<vec3i>();
    triangles.reserve(quads.size() * 2);
    for (auto& q : quads) {
        triangles.push_back({q.x, q.y, q.w});
        if (q.z != q.w) triangles.push_back({q.z, q.w, q.y});
    }
}

// Convert quads to triangles with a diamond-like topology.
// Quads have to be consecutive one row after another.
void convert_quads_to_triangles(
    const vector<vec4i>& quads, vector<vec3i>& triangles, int row_length) {
    triangles = vector<vec3i>();
    triangles.reserve(quads.size() * 2);
    for (auto& q : quads) {
        triangles.push_back({q.x, q.y, q.w});
        if (q.z != q.w) triangles.push_back({q.z, q.w, q.y});
    }
#if 0
        triangles.resize(usteps * vsteps * 2);
        for (auto j = 0; j < vsteps; j++) {
            for (auto i = 0; i < usteps; i++) {
                auto f1 = triangles[(j * usteps + i) * 2 + 0];
                auto f2 = triangles[(j * usteps + i) * 2 + 1];
                if ((i + j) % 2) {
                    f1 = {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)};
                    f2 = {vid(i + 1, j + 1), vid(i, j + 1), vid(i, j)};
                } else {
                    f1 = {vid(i, j), vid(i + 1, j), vid(i, j + 1)};
                    f2 = {vid(i + 1, j + 1), vid(i, j + 1), vid(i + 1, j)};
                }
            }
        }
#endif
}

// Convert triangles to quads by creating degenerate quads
void convert_triangles_to_quads(
    const vector<vec3i>& triangles, vector<vec4i>& quads) {
    quads = vector<vec4i>();
    quads.reserve(triangles.size());
    for (auto& t : triangles) quads.push_back({t.x, t.y, t.z, t.z});
}

// Convert beziers to lines using 3 lines for each bezier.
void convert_bezier_to_lines(const vector<vec4i>& beziers, vector<vec2i>& lines) {
    lines = vector<vec2i>();
    lines.reserve(beziers.size() * 3);
    for (auto b : beziers) {
        lines.push_back({b.x, b.y});
        lines.push_back({b.y, b.z});
        lines.push_back({b.z, b.w});
    }
}

// Convert face varying data to single primitives. Returns the quads indices
// and filled vectors for pos, norm and texcoord.
void convert_face_varying(const vector<vec4i>& quads_positions,
    const vector<vec4i>&                       quads_normals,
    const vector<vec4i>& quads_texturecoords, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    vector<vec4i>& split_quads, vector<vec3f>& split_positions,
    vector<vec3f>& split_normals, vector<vec2f>& split_texcturecoords) {
    // make faces unique
    unordered_map<vec3i, int> vert_map;
    split_quads = vector<vec4i>(quads_positions.size());
    for (auto fid = 0; fid < quads_positions.size(); fid++) {
        for (auto c = 0; c < 4; c++) {
            auto v = vec3i{
                (&quads_positions[fid].x)[c],
                (!quads_normals.empty()) ? (&quads_normals[fid].x)[c] : -1,
                (!quads_texturecoords.empty()) ? (&quads_texturecoords[fid].x)[c] :
                                                 -1,
            };
            auto it = vert_map.find(v);
            if (it == vert_map.end()) {
                auto s = (int)vert_map.size();
                vert_map.insert(it, {v, s});
                (&split_quads[fid].x)[c] = s;
            } else {
                (&split_quads[fid].x)[c] = it->second;
            }
        }
    }

    // fill vert data
    split_positions.clear();
    if (!positions.empty()) {
        split_positions.resize(vert_map.size());
        for (auto kv : vert_map) {
            split_positions[kv.second] = positions[kv.first.x];
        }
    }
    split_normals.clear();
    if (!normals.empty()) {
        split_normals.resize(vert_map.size());
        for (auto kv : vert_map) {
            split_normals[kv.second] = normals[kv.first.y];
        }
    }
    split_texcturecoords.clear();
    if (!texturecoords.empty()) {
        split_texcturecoords.resize(vert_map.size());
        for (auto kv : vert_map) {
            split_texcturecoords[kv.second] = texturecoords[kv.first.z];
        }
    }
}

// Split primitives per id
template <typename T>
void ungroup_elems(const vector<T>& elems, const vector<int>& ids,
    vector<vector<T>>& split_elems) {
    auto max_id = *std::max_element(ids.begin(), ids.end());
    split_elems = vector<vector<T>>(max_id + 1);
    for (auto elem_id = 0; elem_id < elems.size(); elem_id++) {
        split_elems[ids[elem_id]].push_back(elems[elem_id]);
    }
}
void ungroup_lines(const vector<vec2i>& lines, const vector<int>& ids,
    vector<vector<vec2i>>& split_lines) {
    ungroup_elems(lines, ids, split_lines);
}
void ungroup_triangles(const vector<vec3i>& triangles, const vector<int>& ids,
    vector<vector<vec3i>>& split_triangles) {
    ungroup_elems(triangles, ids, split_triangles);
}
void ungroup_quads(const vector<vec4i>& quads, const vector<int>& ids,
    vector<vector<vec4i>>& split_quads) {
    ungroup_elems(quads, ids, split_quads);
}

// Subdivide lines.
template <typename T>
void subdivide_lines(vector<vec2i>& lines, vector<T>& vert, int level) {
    // early exit
    if (lines.empty() || vert.empty()) return;
    // iterate over levels
    for (auto l = 0; l < level; l++) {
        auto nverts = (int)vert.size();
        auto nlines = (int)lines.size();
        // create vertices
        auto tvert = vector<T>(nverts + nlines);
        for (auto i = 0; i < nverts; i++) tvert[i] = vert[i];
        for (auto i = 0; i < nlines; i++) {
            auto l            = lines[i];
            tvert[nverts + i] = (vert[l.x] + vert[l.y]) / 2;
        }
        // create lines
        auto tlines = vector<vec2i>(nlines * 2);
        for (auto i = 0; i < nlines; i++) {
            auto l            = lines[i];
            tlines[i * 2 + 0] = {l.x, nverts + i};
            tlines[i * 2 + 0] = {nverts + i, l.y};
        }
        // done
        swap(lines, tlines);
        swap(vert, tvert);
    }
}

template void subdivide_lines(vector<vec2i>&, vector<float>&, int);
template void subdivide_lines(vector<vec2i>&, vector<vec2f>&, int);
template void subdivide_lines(vector<vec2i>&, vector<vec3f>&, int);
template void subdivide_lines(vector<vec2i>&, vector<vec4f>&, int);

// Subdivide triangle.
template <typename T>
void subdivide_triangles(vector<vec3i>& triangles, vector<T>& vert, int level) {
    // early exit
    if (triangles.empty() || vert.empty()) return;
    // iterate over levels
    for (auto l = 0; l < level; l++) {
        // get edges
        auto emap = edge_map{};
        insert_edges(emap, triangles);
        auto edges = vector<vec2i>{};
        get_edges(emap, edges);
        // number of elements
        auto nverts = (int)vert.size();
        auto nedges = (int)edges.size();
        auto nfaces = (int)triangles.size();
        // create vertices
        auto tvert = vector<T>(nverts + nedges);
        for (auto i = 0; i < nverts; i++) tvert[i] = vert[i];
        for (auto i = 0; i < nedges; i++) {
            auto e            = edges[i];
            tvert[nverts + i] = (vert[e.x] + vert[e.y]) / 2;
        }
        // create triangles
        auto ttriangles = vector<vec3i>(nfaces * 4);
        for (auto i = 0; i < nfaces; i++) {
            auto t                = triangles[i];
            ttriangles[i * 4 + 0] = {t.x,
                nverts + get_edge_index(emap, {t.x, t.y}),
                nverts + get_edge_index(emap, {t.z, t.x})};
            ttriangles[i * 4 + 1] = {t.y,
                nverts + get_edge_index(emap, {t.y, t.z}),
                nverts + get_edge_index(emap, {t.x, t.y})};
            ttriangles[i * 4 + 2] = {t.z,
                nverts + get_edge_index(emap, {t.z, t.x}),
                nverts + get_edge_index(emap, {t.y, t.z})};
            ttriangles[i * 4 + 3] = {nverts + get_edge_index(emap, {t.x, t.y}),
                nverts + get_edge_index(emap, {t.y, t.z}),
                nverts + get_edge_index(emap, {t.z, t.x})};
        }
        // done
        swap(triangles, ttriangles);
        swap(vert, tvert);
    }
}

template void subdivide_triangles(vector<vec3i>&, vector<float>&, int);
template void subdivide_triangles(vector<vec3i>&, vector<vec2f>&, int);
template void subdivide_triangles(vector<vec3i>&, vector<vec3f>&, int);
template void subdivide_triangles(vector<vec3i>&, vector<vec4f>&, int);

// Subdivide quads.
template <typename T>
void subdivide_quads(vector<vec4i>& quads, vector<T>& vert, int level) {
    // early exit
    if (quads.empty() || vert.empty()) return;
    // iterate over levels
    for (auto l = 0; l < level; l++) {
        // get edges
        auto emap = edge_map{};
        insert_edges(emap, quads);
        auto edges = vector<vec2i>{};
        get_edges(emap, edges);
        // number of elements
        auto nverts = (int)vert.size();
        auto nedges = (int)edges.size();
        auto nfaces = (int)quads.size();
        // create vertices
        auto tvert = vector<T>(nverts + nedges + nfaces);
        for (auto i = 0; i < nverts; i++) tvert[i] = vert[i];
        for (auto i = 0; i < nedges; i++) {
            auto e            = edges[i];
            tvert[nverts + i] = (vert[e.x] + vert[e.y]) / 2;
        }
        for (auto i = 0; i < nfaces; i++) {
            auto q = quads[i];
            if (q.z != q.w) {
                tvert[nverts + nedges + i] = (vert[q.x] + vert[q.y] +
                                                 vert[q.z] + vert[q.w]) /
                                             4;
            } else {
                tvert[nverts + nedges + i] = (vert[q.x] + vert[q.y] + vert[q.y]) /
                                             3;
            }
        }
        // create quads
        auto tquads = vector<vec4i>(nfaces * 4);  // conservative allocation
        auto qi     = 0;
        for (auto i = 0; i < nfaces; i++) {
            auto q = quads[i];
            if (q.z != q.w) {
                tquads[qi++] = {q.x, nverts + get_edge_index(emap, {q.x, q.y}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.w, q.x})};
                tquads[qi++] = {q.y, nverts + get_edge_index(emap, {q.y, q.z}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.x, q.y})};
                tquads[qi++] = {q.z, nverts + get_edge_index(emap, {q.z, q.w}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.y, q.z})};
                tquads[qi++] = {q.w, nverts + get_edge_index(emap, {q.w, q.x}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.z, q.w})};
            } else {
                tquads[qi++] = {q.x, nverts + get_edge_index(emap, {q.x, q.y}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.z, q.x})};
                tquads[qi++] = {q.y, nverts + get_edge_index(emap, {q.y, q.z}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.x, q.y})};
                tquads[qi++] = {q.z, nverts + get_edge_index(emap, {q.z, q.x}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.y, q.z})};
            }
        }
        tquads.resize(qi);
        // done
        swap(quads, tquads);
        swap(vert, tvert);
    }
}

template void subdivide_quads(vector<vec4i>&, vector<float>&, int);
template void subdivide_quads(vector<vec4i>&, vector<vec2f>&, int);
template void subdivide_quads(vector<vec4i>&, vector<vec3f>&, int);
template void subdivide_quads(vector<vec4i>&, vector<vec4f>&, int);

// Subdivide beziers.
template <typename T>
void subdivide_beziers(vector<vec4i>& beziers, vector<T>& vert, int level) {
    // early exit
    if (beziers.empty() || vert.empty()) return;
    // iterate over levels
    for (auto l = 0; l < level; l++) {
        auto vmap     = unordered_map<int, int>();
        auto tvert    = vector<T>();
        auto tbeziers = vector<vec4i>();
        for (auto b : beziers) {
            if (vmap.find(b.x) == vmap.end()) {
                vmap[b.x] = (int)tvert.size();
                tvert.push_back(vert[b.x]);
            }
            if (vmap.find(b.w) == vmap.end()) {
                vmap[b.w] = (int)tvert.size();
                tvert.push_back(vert[b.w]);
            }
            auto bo = (int)tvert.size();
            tbeziers.push_back({vmap.at(b.x), bo + 0, bo + 1, bo + 2});
            tbeziers.push_back({bo + 2, bo + 3, bo + 4, vmap.at(b.w)});
            tvert.push_back(vert[b.x] / 2 + vert[b.y] / 2);
            tvert.push_back(vert[b.x] / 4 + vert[b.y] / 2 + vert[b.z] / 4);
            tvert.push_back(vert[b.x] / 8 + 3 * vert[b.y] / 8 +
                            3 * vert[b.z] / 8 + vert[b.w] / 8);
            tvert.push_back(vert[b.y] / 4 + vert[b.z] / 2 + vert[b.w] / 4);
            tvert.push_back(vert[b.z] / 2 + vert[b.w] / 2);
        }

        swap(beziers, tbeziers);
        swap(vert, tvert);
    }
}

template void subdivide_beziers(vector<vec4i>&, vector<float>&, int);
template void subdivide_beziers(vector<vec4i>&, vector<vec2f>&, int);
template void subdivide_beziers(vector<vec4i>&, vector<vec3f>&, int);
template void subdivide_beziers(vector<vec4i>&, vector<vec4f>&, int);

// Subdivide catmullclark.
template <typename T>
void subdivide_catmullclark(
    vector<vec4i>& quads, vector<T>& vert, int level, bool lock_boundary) {
    // early exit
    if (quads.empty() || vert.empty()) return;
    // iterate over levels
    for (auto l = 0; l < level; l++) {
        // get edges
        // get edges
        auto emap = edge_map{};
        insert_edges(emap, quads);
        auto edges    = vector<vec2i>{};
        auto boundary = vector<vec2i>{};
        get_edges(emap, edges);
        get_boundary(emap, boundary);
        // number of elements
        auto nverts    = (int)vert.size();
        auto nedges    = (int)edges.size();
        auto nboundary = (int)boundary.size();
        auto nfaces    = (int)quads.size();

        // split elements ------------------------------------
        // create vertices
        auto tvert = vector<T>(nverts + nedges + nfaces);
        for (auto i = 0; i < nverts; i++) tvert[i] = vert[i];
        for (auto i = 0; i < nedges; i++) {
            auto e            = edges[i];
            tvert[nverts + i] = (vert[e.x] + vert[e.y]) / 2;
        }
        for (auto i = 0; i < nfaces; i++) {
            auto q = quads[i];
            if (q.z != q.w) {
                tvert[nverts + nedges + i] = (vert[q.x] + vert[q.y] +
                                                 vert[q.z] + vert[q.w]) /
                                             4;
            } else {
                tvert[nverts + nedges + i] = (vert[q.x] + vert[q.y] + vert[q.y]) /
                                             3;
            }
        }
        // create quads
        auto tquads = vector<vec4i>(nfaces * 4);  // conservative allocation
        auto qi     = 0;
        for (auto i = 0; i < nfaces; i++) {
            auto q = quads[i];
            if (q.z != q.w) {
                tquads[qi++] = {q.x, nverts + get_edge_index(emap, {q.x, q.y}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.w, q.x})};
                tquads[qi++] = {q.y, nverts + get_edge_index(emap, {q.y, q.z}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.x, q.y})};
                tquads[qi++] = {q.z, nverts + get_edge_index(emap, {q.z, q.w}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.y, q.z})};
                tquads[qi++] = {q.w, nverts + get_edge_index(emap, {q.w, q.x}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.z, q.w})};
            } else {
                tquads[qi++] = {q.x, nverts + get_edge_index(emap, {q.x, q.y}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.z, q.x})};
                tquads[qi++] = {q.y, nverts + get_edge_index(emap, {q.y, q.z}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.x, q.y})};
                tquads[qi++] = {q.z, nverts + get_edge_index(emap, {q.z, q.x}),
                    nverts + nedges + i,
                    nverts + get_edge_index(emap, {q.y, q.z})};
            }
        }
        tquads.resize(qi);

        // split boundary
        auto tboundary = vector<vec2i>(nboundary * 2);
        for (auto i = 0; i < nboundary; i++) {
            auto e = boundary[i];
            tboundary.push_back({e.x, nverts + get_edge_index(emap, e)});
            tboundary.push_back({nverts + get_edge_index(emap, e), e.y});
        }

        // setup creases -----------------------------------
        auto tcrease_edges = vector<vec2i>();
        auto tcrease_verts = vector<int>();
        if (lock_boundary) {
            for (auto& b : tboundary) {
                tcrease_verts.push_back(b.x);
                tcrease_verts.push_back(b.y);
            }
        } else {
            for (auto& b : tboundary) tcrease_edges.push_back(b);
        }

        // define vertex valence ---------------------------
        auto tvert_val = vector<int>(tvert.size(), 2);
        for (auto& e : tboundary) {
            tvert_val[e.x] = (lock_boundary) ? 0 : 1;
            tvert_val[e.y] = (lock_boundary) ? 0 : 1;
        }

        // averaging pass ----------------------------------
        auto avert  = vector<T>(tvert.size(), T());
        auto acount = vector<int>(tvert.size(), 0);
        for (auto p : tcrease_verts) {
            if (tvert_val[p] != 0) continue;
            avert[p] += tvert[p];
            acount[p] += 1;
        }
        for (auto& e : tcrease_edges) {
            auto c = (tvert[e.x] + tvert[e.y]) / 2.0f;
            for (auto vid : {e.x, e.y}) {
                if (tvert_val[vid] != 1) continue;
                avert[vid] += c;
                acount[vid] += 1;
            }
        }
        for (auto& q : tquads) {
            auto c = (tvert[q.x] + tvert[q.y] + tvert[q.z] + tvert[q.w]) / 4.0f;
            for (auto vid : {q.x, q.y, q.z, q.w}) {
                if (tvert_val[vid] != 2) continue;
                avert[vid] += c;
                acount[vid] += 1;
            }
        }
        for (auto i = 0; i < tvert.size(); i++) avert[i] /= (float)acount[i];

        // correction pass ----------------------------------
        // p = p + (avg_p - p) * (4/avg_count)
        for (auto i = 0; i < tvert.size(); i++) {
            if (tvert_val[i] != 2) continue;
            avert[i] = tvert[i] + (avert[i] - tvert[i]) * (4.0f / acount[i]);
        }
        tvert = avert;

        swap(quads, tquads);
        swap(vert, tvert);
    }
}

template void subdivide_catmullclark(vector<vec4i>&, vector<float>&, int, bool);
template void subdivide_catmullclark(vector<vec4i>&, vector<vec2f>&, int, bool);
template void subdivide_catmullclark(vector<vec4i>&, vector<vec3f>&, int, bool);
template void subdivide_catmullclark(vector<vec4i>&, vector<vec4f>&, int, bool);

// Weld vertices within a threshold. For noe the implementation is O(n^2).
void weld_vertices(
    vector<vec3f>& positions, float threshold, vector<int>& welded_indices) {
    welded_indices        = vector<int>(positions.size());
    auto welded_positions = vector<vec3f>();
    for (auto i = 0; i < positions.size(); i++) {
        welded_indices[i] = (int)welded_positions.size();
        for (auto j = 0; j < welded_positions.size(); j++) {
            if (length(positions[i] - welded_positions[j]) < threshold) {
                welded_indices[i] = j;
                break;
            }
        }
        if (welded_indices[i] == (int)welded_positions.size())
            welded_positions.push_back(positions[i]);
    }
    swap(positions, welded_positions);
}
void weld_triangles(
    vector<vec3i>& triangles, vector<vec3f>& positions, float threshold) {
    auto welded_indices = vector<int>();
    weld_vertices(positions, threshold, welded_indices);
    for (auto& t : triangles) {
        t.x = welded_indices[t.x];
        t.y = welded_indices[t.y];
        t.z = welded_indices[t.z];
    }
}
void weld_quads(vector<vec4i>& quads, vector<vec3f>& positions, float threshold) {
    auto welded_indices = vector<int>();
    weld_vertices(positions, threshold, welded_indices);
    for (auto& q : quads) {
        q.x = welded_indices[q.x];
        q.y = welded_indices[q.y];
        q.z = welded_indices[q.z];
        q.w = welded_indices[q.w];
    }
}

// Samples a set of points over a triangle mesh uniformly. The rng function
// takes the point index and returns vec3f numbers uniform directibuted in
// [0,1]^3. unorm and texcoord are optional.
void sample_triangles_points(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3f>& normals,
    const vector<vec2f>& texturecoords, int npoints,
    vector<vec3f>& sampled_positions, vector<vec3f>& sampled_normals,
    vector<vec2f>& sampled_texturecoords, int seed) {
    sampled_positions     = vector<vec3f>(npoints);
    sampled_normals       = vector<vec3f>(npoints);
    sampled_texturecoords = vector<vec2f>(npoints);
    auto cdf              = vector<float>{};
    sample_triangles_element_cdf(triangles, positions, cdf);
    auto rng = make_rng(seed);
    for (auto i = 0; i < npoints; i++) {
        auto sample = sample_triangles_element(cdf, get_random_float(rng),
            {get_random_float(rng), get_random_float(rng)});
        auto t      = triangles[sample.first];
        sampled_positions[i] = interpolate_triangle(
            positions[t.x], positions[t.y], positions[t.z], sample.second);
        if (!sampled_normals.empty()) {
            sampled_normals[i] = normalize(interpolate_triangle(
                normals[t.x], normals[t.y], normals[t.z], sample.second));
        } else {
            sampled_normals[i] = triangle_normal(
                positions[t.x], positions[t.y], positions[t.z]);
        }
        if (!sampled_texturecoords.empty()) {
            sampled_texturecoords[i] = interpolate_triangle(texturecoords[t.x],
                texturecoords[t.y], texturecoords[t.z], sample.second);
        } else {
            sampled_texturecoords[i] = zero2f;
        }
    }
}

// Samples a set of points over a triangle mesh uniformly. The rng function
// takes the point index and returns vec3f numbers uniform directibuted in
// [0,1]^3. unorm and texcoord are optional.
void sample_quads_points(const vector<vec4i>& quads,
    const vector<vec3f>& positions, const vector<vec3f>& normals,
    const vector<vec2f>& texturecoords, int npoints,
    vector<vec3f>& sampled_positions, vector<vec3f>& sampled_normals,
    vector<vec2f>& sampled_texturecoords, int seed) {
    sampled_positions     = vector<vec3f>(npoints);
    sampled_normals       = vector<vec3f>(npoints);
    sampled_texturecoords = vector<vec2f>(npoints);
    auto cdf              = vector<float>{};
    sample_quads_element_cdf(quads, positions, cdf);
    auto rng = make_rng(seed);
    for (auto i = 0; i < npoints; i++) {
        auto sample          = sample_quads_element(cdf, get_random_float(rng),
            {get_random_float(rng), get_random_float(rng)});
        auto q               = quads[sample.first];
        sampled_positions[i] = interpolate_quad(positions[q.x], positions[q.y],
            positions[q.z], positions[q.w], sample.second);
        if (!sampled_normals.empty()) {
            sampled_normals[i] = normalize(interpolate_quad(normals[q.x],
                normals[q.y], normals[q.z], normals[q.w], sample.second));
        } else {
            sampled_normals[i] = quad_normal(
                positions[q.x], positions[q.y], positions[q.z], positions[q.w]);
        }
        if (!sampled_texturecoords.empty()) {
            sampled_texturecoords[i] = interpolate_quad(texturecoords[q.x],
                texturecoords[q.y], texturecoords[q.z], texturecoords[q.w],
                sample.second);
        } else {
            sampled_texturecoords[i] = zero2f;
        }
    }
}

// Merge shape elements
void merge_lines(
    vector<vec2i>& lines, const vector<vec2i>& merge_lines, int num_verts) {
    for (auto& l : merge_lines)
        lines.push_back({l.x + num_verts, l.y + num_verts});
}
void merge_triangles(vector<vec3i>& triangles,
    const vector<vec3i>& merge_triangles, int num_verts) {
    for (auto& t : merge_triangles)
        triangles.push_back(
            {t.x + num_verts, t.y + num_verts, t.z + num_verts});
}
void merge_quads(
    vector<vec4i>& quads, const vector<vec4i>& merge_quads, int num_verts) {
    for (auto& q : merge_quads)
        quads.push_back({q.x + num_verts, q.y + num_verts,
            q.z + num_verts, q.w + num_verts});
}
void merge_lines(vector<vec2i>& lines, vector<vec3f>& positions,
    vector<vec3f>& tangents, vector<vec2f>& texturecoords,
    vector<float>& radius, const vector<vec2i>& merge_lines,
    const vector<vec3f>& merge_positions, const vector<vec3f>& merge_tangents,
    const vector<vec2f>& merge_texturecoords, const vector<float>& merge_radius) {
    auto merge_verts = (int)positions.size();
    for (auto& l : merge_lines)
        lines.push_back({l.x + merge_verts, l.y + merge_verts});
    positions.insert(
        positions.end(), merge_positions.begin(), merge_positions.end());
    tangents.insert(tangents.end(), merge_tangents.begin(), merge_tangents.end());
    texturecoords.insert(texturecoords.end(), merge_texturecoords.begin(),
        merge_texturecoords.end());
    radius.insert(radius.end(), merge_radius.begin(), merge_radius.end());
}
void merge_triangles(vector<vec3i>& triangles, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    const vector<vec3i>& merge_triangles, const vector<vec3f>& merge_positions,
    const vector<vec3f>& merge_normals, const vector<vec2f>& merge_texturecoords) {
    auto merge_verts = (int)positions.size();
    for (auto& t : merge_triangles)
        triangles.push_back(
            {t.x + merge_verts, t.y + merge_verts, t.z + merge_verts});
    positions.insert(
        positions.end(), merge_positions.begin(), merge_positions.end());
    normals.insert(normals.end(), merge_normals.begin(), merge_normals.end());
    texturecoords.insert(texturecoords.end(), merge_texturecoords.begin(),
        merge_texturecoords.end());
}
void merge_quads(vector<vec4i>& quads, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    const vector<vec4i>& merge_quads, const vector<vec3f>& merge_positions,
    const vector<vec3f>& merge_normals, const vector<vec2f>& merge_texturecoords) {
    auto merge_verts = (int)positions.size();
    for (auto& q : merge_quads)
        quads.push_back({q.x + merge_verts, q.y + merge_verts,
            q.z + merge_verts, q.w + merge_verts});
    positions.insert(
        positions.end(), merge_positions.begin(), merge_positions.end());
    normals.insert(normals.end(), merge_normals.begin(), merge_normals.end());
    texturecoords.insert(texturecoords.end(), merge_texturecoords.begin(),
        merge_texturecoords.end());
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENRTATION OF RAY-PRIMITIVE INTERSECTION FUNCTIONS
// -----------------------------------------------------------------------------
namespace ygl {

// Intersect a ray with a point (approximate)
bool intersect_point(
    const ray3f& ray, const vec3f& p, float r, float& distance, vec2f& uv) {
    // find parameter for line-point minimum distance
    auto w = p - ray.o;
    auto t = dot(w, ray.d) / dot(ray.d, ray.d);

    // exit if not within bounds
    if (t < ray.tmin || t > ray.tmax) return false;

    // test for line-point distance vs point radius
    auto rp  = ray.o + ray.d * t;
    auto prp = p - rp;
    if (dot(prp, prp) > r * r) return false;

    // intersection occurred: set params and exit
    distance = t;
    uv       = {0, 0};

    return true;
}

// Intersect a ray with a line
bool intersect_line(const ray3f& ray, const vec3f& p0, const vec3f& p1,
    float r0, float r1, float& distance, vec2f& uv) {
    // setup intersection params
    auto u = ray.d;
    auto v = p1 - p0;
    auto w = ray.o - p0;

    // compute values to solve a linear system
    auto a   = dot(u, u);
    auto b   = dot(u, v);
    auto c   = dot(v, v);
    auto d   = dot(u, w);
    auto e   = dot(v, w);
    auto det = a * c - b * b;

    // check determinant and exit if lines are parallel
    // (could use EPSILONS if desired)
    if (det == 0) return false;

    // compute Parameters on both ray and segment
    auto t = (b * e - c * d) / det;
    auto s = (a * e - b * d) / det;

    // exit if not within bounds
    if (t < ray.tmin || t > ray.tmax) return false;

    // clamp segment param to segment corners
    s = clamp(s, (float)0, (float)1);

    // compute segment-segment distance on the closest points
    auto pr  = ray.o + ray.d * t;
    auto pl  = p0 + (p1 - p0) * s;
    auto prl = pr - pl;

    // check with the line radius at the same point
    auto d2 = dot(prl, prl);
    auto r  = r0 * (1 - s) + r1 * s;
    if (d2 > r * r) return false;

    // intersection occurred: set params and exit
    distance = t;
    uv       = {s, sqrt(d2) / r};

    return true;
}

// Intersect a ray with a triangle
bool intersect_triangle(const ray3f& ray, const vec3f& p0, const vec3f& p1,
    const vec3f& p2, float& distance, vec2f& uv) {
    // compute triangle edges
    auto edge1 = p1 - p0;
    auto edge2 = p2 - p0;

    // compute determinant to solve a linear system
    auto pvec = cross(ray.d, edge2);
    auto det  = dot(edge1, pvec);

    // check determinant and exit if triangle and ray are parallel
    // (could use EPSILONS if desired)
    if (det == 0) return false;
    auto inv_det = 1.0f / det;

    // compute and check first bricentric coordinated
    auto tvec = ray.o - p0;
    auto u    = dot(tvec, pvec) * inv_det;
    if (u < 0 || u > 1) return false;

    // compute and check second bricentric coordinated
    auto qvec = cross(tvec, edge1);
    auto v    = dot(ray.d, qvec) * inv_det;
    if (v < 0 || u + v > 1) return false;

    // compute and check ray parameter
    auto t = dot(edge2, qvec) * inv_det;
    if (t < ray.tmin || t > ray.tmax) return false;

    // intersection occurred: set params and exit
    distance = t;
    uv       = {u, v};

    return true;
}

// Intersect a ray with a quad.
bool intersect_quad(const ray3f& ray, const vec3f& p0, const vec3f& p1,
    const vec3f& p2, const vec3f& p3, float& distance, vec2f& uv) {
    auto hit  = false;
    auto tray = ray;
    if (intersect_triangle(tray, p0, p1, p3, distance, uv)) {
        tray.tmax = distance;
        hit       = true;
    }
    if (intersect_triangle(tray, p2, p3, p1, distance, uv)) {
        uv        = {1 - uv.x, 1 - uv.y};
        tray.tmax = distance;
        hit       = true;
    }
    return hit;
}

// Min/max used in BVH traversal. Copied here since the traversal code
// relies on the specific behaviour wrt NaNs.
static inline const float& _safemin(const float& a, const float& b) {
    return (a < b) ? a : b;
}
// Min/max used in BVH traversal. Copied here since the traversal code
// relies on the specific behaviour wrt NaNs.
static inline const float& _safemax(const float& a, const float& b) {
    return (a > b) ? a : b;
}

// Intersect a ray with a axis-aligned bounding box
bool intersect_bbox(const ray3f& ray, const bbox3f& bbox) {
    // determine intersection ranges
    auto invd = vec3f{1, 1, 1} / ray.d;
    auto t0   = (bbox.min - ray.o) * invd;
    auto t1   = (bbox.max - ray.o) * invd;
    // flip based on range directions
    if (invd.x < 0.0f) swap(t0.x, t1.x);
    if (invd.y < 0.0f) swap(t0.y, t1.y);
    if (invd.z < 0.0f) swap(t0.z, t1.z);
    auto tmin = _safemax(t0.z, _safemax(t0.y, _safemax(t0.x, ray.tmin)));
    auto tmax = _safemin(t1.z, _safemin(t1.y, _safemin(t1.x, ray.tmax)));
    tmax *= 1.00000024f;  // for double: 1.0000000000000004
    return tmin <= tmax;
}

// Intersect a ray with a axis-aligned bounding box
bool intersect_bbox(const ray3f& ray, const vec3f& ray_dinv,
    const vec3i& ray_dsign, const bbox3f& bbox_) {
    auto bbox  = &bbox_.min;
    auto txmin = (bbox[ray_dsign.x].x - ray.o.x) * ray_dinv.x;
    auto txmax = (bbox[1 - ray_dsign.x].x - ray.o.x) * ray_dinv.x;
    auto tymin = (bbox[ray_dsign.y].y - ray.o.y) * ray_dinv.y;
    auto tymax = (bbox[1 - ray_dsign.y].y - ray.o.y) * ray_dinv.y;
    auto tzmin = (bbox[ray_dsign.z].z - ray.o.z) * ray_dinv.z;
    auto tzmax = (bbox[1 - ray_dsign.z].z - ray.o.z) * ray_dinv.z;
    auto tmin  = _safemax(tzmin, _safemax(tymin, _safemax(txmin, ray.tmin)));
    auto tmax  = _safemin(tzmax, _safemin(tymax, _safemin(txmax, ray.tmax)));
    tmax *= 1.00000024f;  // for double: 1.0000000000000004
    return tmin <= tmax;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENRTATION OF POINT-PRIMITIVE DISTANCE FUNCTIONS
// -----------------------------------------------------------------------------
namespace ygl {

// TODO: documentation
bool overlap_point(const vec3f& pos, float dist_max, const vec3f& p, float r,
    float& distance, vec2f& uv) {
    auto d2 = dot(pos - p, pos - p);
    if (d2 > (dist_max + r) * (dist_max + r)) return false;
    distance = sqrt(d2);
    uv       = {0, 0};
    return true;
}

// TODO: documentation
float closestuv_line(const vec3f& pos, const vec3f& p0, const vec3f& p1) {
    auto ab = p1 - p0;
    auto d  = dot(ab, ab);
    // Project c onto ab, computing parameterized position d(t) = a + t*(b –
    // a)
    auto u = dot(pos - p0, ab) / d;
    u      = clamp(u, (float)0, (float)1);
    return u;
}

// TODO: documentation
bool overlap_line(const vec3f& pos, float dist_max, const vec3f& p0,
    const vec3f& p1, float r0, float r1, float& distance, vec2f& uv) {
    auto u = closestuv_line(pos, p0, p1);
    // Compute projected position from the clamped t d = a + t * ab;
    auto p  = p0 + (p1 - p0) * u;
    auto r  = r0 + (r1 - r0) * u;
    auto d2 = dot(pos - p, pos - p);
    // check distance
    if (d2 > (dist_max + r) * (dist_max + r)) return false;
    // done
    distance = sqrt(d2);
    uv       = {u, 0};
    return true;
}

// TODO: documentation
// this is a complicated test -> I probably "--"+prefix to use a sequence of
// test (triangle body, and 3 edges)
vec2f closestuv_triangle(
    const vec3f& pos, const vec3f& p0, const vec3f& p1, const vec3f& p2) {
    auto ab = p1 - p0;
    auto ac = p2 - p0;
    auto ap = pos - p0;

    auto d1 = dot(ab, ap);
    auto d2 = dot(ac, ap);

    // corner and edge cases
    if (d1 <= 0 && d2 <= 0) return {0, 0};

    auto bp = pos - p1;
    auto d3 = dot(ab, bp);
    auto d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return {1, 0};

    auto vc = d1 * d4 - d3 * d2;
    if ((vc <= 0) && (d1 >= 0) && (d3 <= 0)) return {d1 / (d1 - d3), 0};

    auto cp = pos - p2;
    auto d5 = dot(ab, cp);
    auto d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return {0, 1};

    auto vb = d5 * d2 - d1 * d6;
    if ((vb <= 0) && (d2 >= 0) && (d6 <= 0)) return {0, d2 / (d2 - d6)};

    auto va = d3 * d6 - d5 * d4;
    if ((va <= 0) && (d4 - d3 >= 0) && (d5 - d6 >= 0)) {
        auto w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {1 - w, w};
    }

    // face case
    auto denom = 1 / (va + vb + vc);
    auto u     = vb * denom;
    auto v     = vc * denom;
    return {u, v};
}

// TODO: documentation
bool overlap_triangle(const vec3f& pos, float dist_max, const vec3f& p0,
    const vec3f& p1, const vec3f& p2, float r0, float r1, float r2,
    float& distance, vec2f& uv) {
    uv      = closestuv_triangle(pos, p0, p1, p2);
    auto p  = interpolate_triangle(p0, p1, p2, uv);
    auto r  = interpolate_triangle(r0, r1, r2, uv);
    auto dd = dot(p - pos, p - pos);
    if (dd > (dist_max + r) * (dist_max + r)) return false;
    distance = sqrt(dd);
    return true;
}

// TODO: documentation
bool overlap_quad(const vec3f& pos, float dist_max, const vec3f& p0,
    const vec3f& p1, const vec3f& p2, const vec3f& p3, float r0, float r1,
    float r2, float r3, float& distance, vec2f& uv) {
    auto hit = false;
    if (overlap_triangle(pos, dist_max, p0, p1, p3, r0, r1, r3, distance, uv)) {
        dist_max = distance;
        hit      = true;
    }
    if (overlap_triangle(pos, dist_max, p2, p3, p1, r2, r3, r1, distance, uv)) {
        // dist_max = distance;
        uv  = {1 - uv.x, 1 - uv.y};
        hit = true;
    }
    return hit;
}

// TODO: documentation
bool distance_check_bbox(const vec3f& pos, float dist_max, const bbox3f& bbox) {
    // computing distance
    auto dd = 0.0f;

    // For each axis count any excess distance outside box extents
    if (pos.x < bbox.min.x) dd += (bbox.min.x - pos.x) * (bbox.min.x - pos.x);
    if (pos.x > bbox.max.x) dd += (pos.x - bbox.max.x) * (pos.x - bbox.max.x);
    if (pos.y < bbox.min.y) dd += (bbox.min.y - pos.y) * (bbox.min.y - pos.y);
    if (pos.y > bbox.max.y) dd += (pos.y - bbox.max.y) * (pos.y - bbox.max.y);
    if (pos.z < bbox.min.z) dd += (bbox.min.z - pos.z) * (bbox.min.z - pos.z);
    if (pos.z > bbox.max.z) dd += (pos.z - bbox.max.z) * (pos.z - bbox.max.z);

    // check distance
    return dd < dist_max * dist_max;
}

// TODO: doc
bool overlap_bbox(const bbox3f& bbox1, const bbox3f& bbox2) {
    if (bbox1.max.x < bbox2.min.x || bbox1.min.x > bbox2.max.x) return false;
    if (bbox1.max.y < bbox2.min.y || bbox1.min.y > bbox2.max.y) return false;
    if (bbox1.max.z < bbox2.min.z || bbox1.min.z > bbox2.max.z) return false;
    return true;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR BVH
// -----------------------------------------------------------------------------
namespace ygl {

// Cleanup
void clear_shape_bvh_embree(bvh_shape& bvh) {
#if YGL_EMBREE
    if (bvh.embree_bvh) {
        rtcReleaseScene((RTCScene)bvh.embree_bvh);
    }
#endif
}
void clear_scene_bvh_embree(bvh_scene& bvh) {
#if YGL_EMBREE
    if (bvh.embree_bvh) {
        for (auto i = 0; i < max(1, (int)bvh.instances.size()); i++) {
            auto geom = rtcGetGeometry((RTCScene)bvh.embree_bvh, i);
            rtcDetachGeometry((RTCScene)bvh.embree_bvh, i);
            rtcReleaseGeometry(geom);
        }
        rtcReleaseScene((RTCScene)bvh.embree_bvh);
    }
#endif
}

// BVH primitive with its bbox, its center and the index to the primitive
struct bvh_prim {
    bbox3f bbox   = invalid_bbox3f;
    vec3f  center = zero3f;
    int    primid = 0;
};

// Initializes the BVH node node that contains the primitives sorted_prims
// from start to end, by either splitting it into two other nodes,
// or initializing it as a leaf. When splitting, the heuristic heuristic is
// used and nodes added sequentially in the preallocated nodes array and
// the number of nodes nnodes is updated.
int make_bvh_node(vector<bvh_node>& nodes, vector<bvh_prim>& prims, int start,
    int end, bool high_quality) {
    // add a new node
    auto nodeid = (int)nodes.size();
    nodes.push_back({});
    auto& node = nodes.back();

    // compute bounds
    node.bbox = invalid_bbox3f;
    for (auto i = start; i < end; i++) node.bbox += prims[i].bbox;

    // split into two children
    if (end - start > bvh_max_prims) {
        // initialize split axis and position
        auto split_axis = 0;
        auto mid        = (start + end) / 2;

        // compute primintive bounds and size
        auto cbbox = invalid_bbox3f;
        for (auto i = start; i < end; i++) cbbox += prims[i].center;
        auto csize = cbbox.max - cbbox.min;

        // choose the split axis and position
        if (csize != zero3f) {
            // check heuristic
            if (high_quality) {
                // consider N bins, compute their cost and keep the minimum
                const int nbins     = 16;
                auto      middle    = 0.0f;
                auto      min_cost  = maxf;
                auto      bbox_area = [](auto& b) {
                    auto size = b.max - b.min;
                    return 1e-12f + 2 * size.x * size.y + 2 * size.x * size.z +
                           2 * size.y * size.z;
                };
                auto min_left = 0, min_right = 0;
                for (auto axis = 0; axis < 3; axis++) {
                    for (auto b = 1; b < nbins; b++) {
                        auto split = (&cbbox.min.x)[axis] +
                                     b * (&csize.x)[axis] / nbins;
                        auto left_bbox   = invalid_bbox3f,
                             right_bbox  = invalid_bbox3f;
                        auto left_nprims = 0, right_nprims = 0;
                        for (auto i = start; i < end; i++) {
                            if ((&prims[i].center.x)[axis] < split) {
                                left_bbox += prims[i].bbox;
                                left_nprims += 1;
                            } else {
                                right_bbox += prims[i].bbox;
                                right_nprims += 1;
                            }
                        }
                        auto cost = 1 +
                                    left_nprims * bbox_area(left_bbox) /
                                        bbox_area(cbbox) +
                                    right_nprims * bbox_area(right_bbox) /
                                        bbox_area(cbbox);
                        if (cost < min_cost) {
                            min_cost   = cost;
                            middle     = split;
                            split_axis = axis;
                            min_left   = left_nprims;
                            min_right  = right_nprims;
                        }
                    }
                }
                // split
                mid = (int)(std::partition(prims.data() + start,
                                prims.data() + end,
                                [split_axis, middle](auto& a) {
                                    return (&a.center.x)[split_axis] < middle;
                                }) -
                            prims.data());
                if (mid == start || mid == end) {
                    log_error("bad BVH build");
                }
            } else {
                // split along largest
                auto largest_axis = 0;
                if (csize.x >= csize.y && csize.x >= csize.z) largest_axis = 0;
                if (csize.y >= csize.x && csize.y >= csize.z) largest_axis = 1;
                if (csize.z >= csize.x && csize.z >= csize.y) largest_axis = 2;
                    // balanced tree split: find the largest axis of the
                    // bounding box and split along this one right in the middle
#if 1
                split_axis = largest_axis;
                mid        = (start + end) / 2;
                std::nth_element(prims.data() + start, prims.data() + mid,
                    prims.data() + end, [split_axis](auto& a, auto& b) {
                        return (&a.center.x)[split_axis] <
                               (&b.center.x)[split_axis];
                    });
#endif
#if 0
                // split the space in the middle along the largest axis
                split_axis = largest_axis;
                auto csize = (cbbox.max + cbbox.min) / 2;
                auto middle = (&csize.x)[largest_axis];
                mid = (int)(std::partition(prims.data() + start,
                                prims.data() + end,
                                [split_axis, middle](auto& a) {
                                    return (&a.center.x)[split_axis] < middle;
                                }) -
                            prims.data());
#endif
            }

            // if we were not able to split, just break the primitives in half
            if (mid == start || mid == end) {
                split_axis = 0;
                mid        = (start + end) / 2;
            }
        }

        // make an internal node
        node.is_internal      = true;
        node.split_axis       = split_axis;
        node.num_primitives   = 2;
        node.primitive_ids[0] = make_bvh_node(
            nodes, prims, start, mid, high_quality);
        node.primitive_ids[1] = make_bvh_node(
            nodes, prims, mid, end, high_quality);
    } else {
        // Make a leaf node
        node.is_internal    = false;
        node.num_primitives = end - start;
        for (auto i = 0; i < node.num_primitives; i++)
            node.primitive_ids[i] = prims[start + i].primid;
    }

    // return nodeid
    return nodeid;
}

// Build a BVH from a set of primitives.
void build_shape_bvh(bvh_shape& bvh, bool high_quality) {
    // get the number of primitives and the primitive type
    auto prims = vector<bvh_prim>();
    if (!bvh.points.empty()) {
        for (auto& p : bvh.points) {
            prims.push_back({point_bounds(bvh.positions[p], bvh.radius[p])});
        }
    } else if (!bvh.lines.empty()) {
        for (auto& l : bvh.lines) {
            prims.push_back({line_bounds(bvh.positions[l.x], bvh.positions[l.y],
                bvh.radius[l.x], bvh.radius[l.y])});
        }
    } else if (!bvh.triangles.empty()) {
        for (auto& t : bvh.triangles) {
            prims.push_back({triangle_bounds(
                bvh.positions[t.x], bvh.positions[t.y], bvh.positions[t.z])});
        }
    } else if (!bvh.quads.empty()) {
        for (auto& q : bvh.quads) {
            prims.push_back({quad_bounds(bvh.positions[q.x], bvh.positions[q.y],
                bvh.positions[q.z], bvh.positions[q.w])});
        }
    }

    // create an array of primitives to sort
    for (auto i = 0; i < prims.size(); i++) {
        prims[i].center = (prims[i].bbox.min + prims[i].bbox.max) / 2;
        prims[i].primid = i;
    }

    // build nodes
    bvh.nodes.clear();
    bvh.nodes.reserve(prims.size() * 2);
    make_bvh_node(bvh.nodes, prims, 0, (int)prims.size(), high_quality);
    bvh.nodes.shrink_to_fit();
}

// Build a BVH from a set of primitives.
void build_scene_bvh(bvh_scene& bvh, bool high_quality) {
    // get the number of primitives and the primitive type
    auto prims = vector<bvh_prim>();
    if (!bvh.instances.empty()) {
        for (auto& instance : bvh.instances) {
            if (instance.shape_id >= 0) {
                auto& sbvh = bvh.shape_bvhs[instance.shape_id];
                prims.push_back(
                    {transform_bbox(instance.frame, sbvh.nodes[0].bbox)});
            } else if (instance.surface_id >= 0) {
                auto& sbvh = bvh.surface_bvhs[instance.surface_id];
                prims.push_back(
                    {transform_bbox(instance.frame, sbvh.nodes[0].bbox)});
            } else {
                log_error("empty instance");
            }
        }
    }

    // create an array of primitives to sort
    for (auto i = 0; i < prims.size(); i++) {
        prims[i].center = (prims[i].bbox.min + prims[i].bbox.max) / 2;
        prims[i].primid = i;
    }

    // build nodes
    bvh.nodes.clear();
    bvh.nodes.reserve(prims.size() * 2);
    make_bvh_node(bvh.nodes, prims, 0, (int)prims.size(), high_quality);
    bvh.nodes.shrink_to_fit();
}

// Recursively recomputes the node bounds for a shape bvh
void refit_shape_bvh(bvh_shape& bvh, int nodeid) {
    // refit
    auto& node = bvh.nodes[nodeid];
    node.bbox  = invalid_bbox3f;
    if (node.is_internal) {
        for (auto i = 0; i < 2; i++) {
            refit_shape_bvh(bvh, node.primitive_ids[i]);
            node.bbox += bvh.nodes[node.primitive_ids[i]].bbox;
        }
    } else if (!bvh.triangles.empty()) {
        for (auto i = 0; i < node.num_primitives; i++) {
            auto& t = bvh.triangles[node.primitive_ids[i]];
            node.bbox += triangle_bounds(
                bvh.positions[t.x], bvh.positions[t.y], bvh.positions[t.z]);
        }
    } else if (!bvh.quads.empty()) {
        for (auto i = 0; i < node.num_primitives; i++) {
            auto& t = bvh.quads[node.primitive_ids[i]];
            node.bbox += quad_bounds(bvh.positions[t.x], bvh.positions[t.y],
                bvh.positions[t.z], bvh.positions[t.w]);
        }
    } else if (!bvh.lines.empty()) {
        for (auto i = 0; i < node.num_primitives; i++) {
            auto& l = bvh.lines[node.primitive_ids[i]];
            node.bbox += line_bounds(bvh.positions[l.x], bvh.positions[l.y],
                bvh.radius[l.x], bvh.radius[l.y]);
        }
    } else if (!bvh.points.empty()) {
        for (auto i = 0; i < node.num_primitives; i++) {
            auto& p = bvh.points[node.primitive_ids[i]];
            node.bbox += point_bounds(bvh.positions[p], bvh.radius[p]);
        }
    } else {
        log_error("empty bvh");
    }
}

// Recursively recomputes the node bounds for a shape bvh
void refit_scene_bvh(bvh_scene& bvh, int nodeid) {
    // refit
    auto& node = bvh.nodes[nodeid];
    node.bbox  = invalid_bbox3f;
    if (node.is_internal) {
        for (auto i = 0; i < 2; i++) {
            refit_scene_bvh(bvh, node.primitive_ids[i]);
            node.bbox += bvh.nodes[node.primitive_ids[i]].bbox;
        }
    } else if (!bvh.instances.empty()) {
        for (auto i = 0; i < node.num_primitives; i++) {
            auto& instance = bvh.instances[node.primitive_ids[i]];
            auto  sbvh     = bvh.shape_bvhs[instance.shape_id];
            node.bbox += transform_bbox(instance.frame, sbvh.nodes[0].bbox);
        }
    } else {
        log_error("empty bvh");
    }
}

// Recursively recomputes the node bounds for a shape bvh
void refit_shape_bvh(bvh_shape& bvh) { refit_shape_bvh(bvh, 0); }
void refit_scene_bvh(bvh_scene& bvh) { refit_scene_bvh(bvh, 0); }

#if YGL_EMBREE
void embree_error(void* ctx, RTCError code, const char* str) {
    switch (code) {
        case RTC_ERROR_UNKNOWN: printf("RTC_ERROR_UNKNOWN"); break;
        case RTC_ERROR_INVALID_ARGUMENT:
            printf("RTC_ERROR_INVALID_ARGUMENT");
            break;
        case RTC_ERROR_INVALID_OPERATION:
            printf("RTC_ERROR_INVALID_OPERATION");
            break;
        case RTC_ERROR_OUT_OF_MEMORY: printf("RTC_ERROR_OUT_OF_MEMORY"); break;
        case RTC_ERROR_UNSUPPORTED_CPU:
            printf("RTC_ERROR_UNSUPPORTED_CPU");
            break;
        case RTC_ERROR_CANCELLED: printf("RTC_ERROR_CANCELLED"); break;
        default: printf("invalid error code"); break;
    }
    printf(": %s\n", str);
}

// Get Embree device
RTCDevice get_embree_device() {
    static RTCDevice device = nullptr;
    if (!device) {
        device = rtcNewDevice("");
        rtcSetDeviceErrorFunction(device, embree_error, nullptr);
    }
    return device;
}

// Build a BVH using Embree. Calls `build_scene_bvh()` if Embree is not available.
void build_embree_bvh(bvh_shape& bvh) {
    auto embree_device = get_embree_device();
    auto embree_scene  = rtcNewScene(embree_device);
    if (!bvh.points.empty()) {
        log_error("embree does not support points");
    } else if (!bvh.lines.empty()) {
        log_error("not yet implemented");
    } else if (!bvh.triangles.empty()) {
        auto embree_geom = rtcNewGeometry(
            embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetGeometryVertexAttributeCount(embree_geom, 1);
        auto vert = rtcSetNewGeometryBuffer(embree_geom, RTC_BUFFER_TYPE_VERTEX,
            0, RTC_FORMAT_FLOAT3, 3 * 4, bvh.positions.size());
        auto triangles = rtcSetNewGeometryBuffer(embree_geom,
            RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * 4,
            bvh.triangles.size());
        memcpy(vert, bvh.positions.data(), bvh.positions.size() * 12);
        memcpy(triangles, bvh.triangles.data(), bvh.triangles.size() * 12);
        rtcCommitGeometry(embree_geom);
        rtcAttachGeometryByID(embree_scene, embree_geom, 0);
    } else if (!bvh.quads.empty()) {
        auto embree_geom = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_QUAD);
        rtcSetGeometryVertexAttributeCount(embree_geom, 1);
        auto vert = rtcSetNewGeometryBuffer(embree_geom, RTC_BUFFER_TYPE_VERTEX,
            0, RTC_FORMAT_FLOAT3, 3 * 4, bvh.positions.size());
        auto quads = rtcSetNewGeometryBuffer(embree_geom, RTC_BUFFER_TYPE_INDEX,
            0, RTC_FORMAT_UINT4, 4 * 4, bvh.quads.size());
        memcpy(vert, bvh.positions.data(), bvh.positions.size() * 12);
        memcpy(quads, bvh.quads.data(), bvh.quads.size() * 16);
        rtcCommitGeometry(embree_geom);
        rtcAttachGeometryByID(embree_scene, embree_geom, 0);
    } else {
        log_error("empty bvh");
    }
    rtcCommitScene(embree_scene);
    bvh.embree_bvh = embree_scene;
}
void build_embree_bvh(bvh_scene& bvh) {
    auto embree_device = get_embree_device();
    auto embree_scene  = rtcNewScene(embree_device);
    if (!bvh.instances.empty()) {
        for (auto instance_id = 0; instance_id < bvh.instances.size();
             instance_id++) {
            auto& instance    = bvh.instances[instance_id];
            auto  embree_geom = rtcNewGeometry(
                embree_device, RTC_GEOMETRY_TYPE_INSTANCE);
            if (instance.shape_id >= 0) {
                rtcSetGeometryInstancedScene(embree_geom,
                    (RTCScene)bvh.shape_bvhs[instance.shape_id].embree_bvh);
            } else if (instance.surface_id >= 0) {
                rtcSetGeometryInstancedScene(embree_geom,
                    (RTCScene)bvh.surface_bvhs[instance.surface_id].embree_bvh);
            } else {
                log_error("empty instance");
            }
            rtcSetGeometryTransform(embree_geom, 0,
                RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, &instance.frame);
            rtcCommitGeometry(embree_geom);
            rtcAttachGeometryByID(embree_scene, embree_geom, instance_id);
        }
    }
    rtcCommitScene(embree_scene);
    bvh.embree_bvh = embree_scene;
}
// Refit a BVH using Embree. Calls `refit_scene_bvh()` if Embree is not available.
void refit_embree_bvh(bvh_shape& bvh) { log_error("not yet implemented"); }
void refit_embree_bvh(bvh_scene& bvh) { log_error("not yet implemented"); }
bool intersect_embree_bvh(const bvh_shape& bvh, const ray3f& ray, bool find_any,
    float& distance, int& element_id, vec2f& uv) {
    RTCRayHit embree_ray;
    embree_ray.ray.org_x     = ray.o.x;
    embree_ray.ray.org_y     = ray.o.y;
    embree_ray.ray.org_z     = ray.o.z;
    embree_ray.ray.dir_x     = ray.d.x;
    embree_ray.ray.dir_y     = ray.d.y;
    embree_ray.ray.dir_z     = ray.d.z;
    embree_ray.ray.tnear     = ray.tmin;
    embree_ray.ray.tfar      = ray.tmax;
    embree_ray.ray.flags     = 0;
    embree_ray.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
    embree_ray.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    RTCIntersectContext embree_ctx;
    rtcInitIntersectContext(&embree_ctx);
    rtcIntersect1((RTCScene)bvh.embree_bvh, &embree_ctx, &embree_ray);
    if (embree_ray.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;
    distance   = embree_ray.ray.tfar;
    uv         = {embree_ray.hit.u, embree_ray.hit.v};
    element_id = embree_ray.hit.primID;
    return true;
}
bool intersect_embree_bvh(const bvh_scene& bvh, const ray3f& ray, bool find_any,
    float& distance, int& instance_id, int& element_id, vec2f& uv) {
    RTCRayHit embree_ray;
    embree_ray.ray.org_x     = ray.o.x;
    embree_ray.ray.org_y     = ray.o.y;
    embree_ray.ray.org_z     = ray.o.z;
    embree_ray.ray.dir_x     = ray.d.x;
    embree_ray.ray.dir_y     = ray.d.y;
    embree_ray.ray.dir_z     = ray.d.z;
    embree_ray.ray.tnear     = ray.tmin;
    embree_ray.ray.tfar      = ray.tmax;
    embree_ray.ray.flags     = 0;
    embree_ray.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
    embree_ray.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    RTCIntersectContext embree_ctx;
    rtcInitIntersectContext(&embree_ctx);
    rtcIntersect1((RTCScene)bvh.embree_bvh, &embree_ctx, &embree_ray);
    if (embree_ray.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;
    distance    = embree_ray.ray.tfar;
    uv          = {embree_ray.hit.u, embree_ray.hit.v};
    element_id  = embree_ray.hit.primID;
    instance_id = embree_ray.hit.instID[0];
    return true;
}
#else
// Build a BVH using Embree. Calls `build_scene_bvh()` if Embree is not available.
void build_embree_bvh(bvh_shape& bvh) { return build_shape_bvh(bvh, true); }
void build_embree_bvh(bvh_scene& bvh) { return build_scene_bvh(bvh, true); }
// Refit a BVH using Embree. Calls `refit_scene_bvh()` if Embree is not available.
void refit_embree_bvh(bvh_shape& bvh) { return refit_shape_bvh(bvh); }
void refit_embree_bvh(bvh_scene& bvh) { return refit_scene_bvh(bvh); }
// Intersect BVH using Embree
bool intersect_embree_bvh(const bvh_shape& bvh, const ray3f& ray_,
    bool find_any, float& distance, int& element_id, vec2f& element_uv) {
    log_error("this should not have been called");
    return false;
}
bool intersect_embree_bvh(const bvh_scene& bvh, const ray3f& ray_, bool find_any,
    float& distance, int& instance_id, int& element_id, vec2f& element_uv) {
    log_error("this should not have been called");
    return false;
}
#endif

// Build a BVH from a set of primitives.
void build_shape_bvh_embree(bvh_shape& bvh, bool high_quality, bool embree) {
#if YGL_EMBREE
    if (embree) return build_embree_bvh(bvh);
#endif
    build_shape_bvh(bvh, high_quality);
}
void build_scene_bvh_embree(bvh_scene& bvh, bool high_quality, bool embree) {
#if YGL_EMBREE
    if (embree) return build_embree_bvh(bvh);
#endif
    build_scene_bvh(bvh, high_quality);
}

// Intersect ray with a bvh.
bool intersect_shape_bvh(const bvh_shape& bvh, const ray3f& ray_, bool find_any,
    float& distance, int& element_id, vec2f& element_uv) {
    // call Embree if needed
    if (bvh.embree_bvh)
        return intersect_embree_bvh(
            bvh, ray_, find_any, distance, element_id, element_uv);

    // node stack
    int  node_stack[128];
    auto node_cur          = 0;
    node_stack[node_cur++] = 0;

    // shared variables
    auto hit = false;

    // copy ray to modify it
    auto ray = ray_;

    // prepare ray for fast queries
    auto ray_dinv  = vec3f{1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
    auto ray_dsign = vec3i{(ray_dinv.x < 0) ? 1 : 0, (ray_dinv.y < 0) ? 1 : 0,
        (ray_dinv.z < 0) ? 1 : 0};

    // walking stack
    while (node_cur) {
        // grab node
        auto& node = bvh.nodes[node_stack[--node_cur]];

        // intersect bbox
        if (!intersect_bbox(ray, ray_dinv, ray_dsign, node.bbox)) continue;

        // intersect node, switching based on node type
        // for each type, iterate over the the primitive list
        if (node.is_internal) {
            // for internal nodes, attempts to proceed along the
            // split axis from smallest to largest nodes
            if ((&ray_dsign.x)[node.split_axis]) {
                node_stack[node_cur++] = node.primitive_ids[0];
                node_stack[node_cur++] = node.primitive_ids[1];
            } else {
                node_stack[node_cur++] = node.primitive_ids[1];
                node_stack[node_cur++] = node.primitive_ids[0];
            }
        } else if (!bvh.triangles.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& t = bvh.triangles[node.primitive_ids[i]];
                if (intersect_triangle(ray, bvh.positions[t.x], bvh.positions[t.y],
                        bvh.positions[t.z], distance, element_uv)) {
                    hit        = true;
                    ray.tmax   = distance;
                    element_id = node.primitive_ids[i];
                }
            }
        } else if (!bvh.quads.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& t = bvh.quads[node.primitive_ids[i]];
                if (intersect_quad(ray, bvh.positions[t.x], bvh.positions[t.y],
                        bvh.positions[t.z], bvh.positions[t.w], distance,
                        element_uv)) {
                    hit        = true;
                    ray.tmax   = distance;
                    element_id = node.primitive_ids[i];
                }
            }
        } else if (!bvh.lines.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& l = bvh.lines[node.primitive_ids[i]];
                if (intersect_line(ray, bvh.positions[l.x], bvh.positions[l.y],
                        bvh.radius[l.x], bvh.radius[l.y], distance, element_uv)) {
                    hit        = true;
                    ray.tmax   = distance;
                    element_id = node.primitive_ids[i];
                }
            }
        } else if (!bvh.points.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& p = bvh.points[node.primitive_ids[i]];
                if (intersect_point(ray, bvh.positions[p], bvh.radius[p],
                        distance, element_uv)) {
                    hit        = true;
                    ray.tmax   = distance;
                    element_id = node.primitive_ids[i];
                }
            }
        } else {
            log_error("empty bvh");
        }

        // check for early exit
        if (find_any && hit) return true;
    }

    return hit;
}

// Intersect ray with a bvh.
bool intersect_scene_bvh(const bvh_scene& bvh, const ray3f& ray_, bool find_any,
    float& distance, int& instance_id, int& element_id, vec2f& element_uv) {
    // call Embree if needed
    if (bvh.embree_bvh)
        return intersect_embree_bvh(
            bvh, ray_, find_any, distance, instance_id, element_id, element_uv);

    // node stack
    int  node_stack[128];
    auto node_cur          = 0;
    node_stack[node_cur++] = 0;

    // shared variables
    auto hit = false;

    // copy ray to modify it
    auto ray = ray_;

    // prepare ray for fast queries
    auto ray_dinv  = vec3f{1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z};
    auto ray_dsign = vec3i{(ray_dinv.x < 0) ? 1 : 0, (ray_dinv.y < 0) ? 1 : 0,
        (ray_dinv.z < 0) ? 1 : 0};

    // walking stack
    while (node_cur) {
        // grab node
        auto& node = bvh.nodes[node_stack[--node_cur]];

        // intersect bbox
        if (!intersect_bbox(ray, ray_dinv, ray_dsign, node.bbox)) continue;

        // intersect node, switching based on node type
        // for each type, iterate over the the primitive list
        if (node.is_internal) {
            // for internal nodes, attempts to proceed along the
            // split axis from smallest to largest nodes
            if ((&ray_dsign.x)[node.split_axis]) {
                node_stack[node_cur++] = node.primitive_ids[0];
                node_stack[node_cur++] = node.primitive_ids[1];
            } else {
                node_stack[node_cur++] = node.primitive_ids[1];
                node_stack[node_cur++] = node.primitive_ids[0];
            }
        } else if (!bvh.instances.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& instance = bvh.instances[node.primitive_ids[i]];
                if (instance.shape_id >= 0) {
                    if (intersect_shape_bvh(bvh.shape_bvhs[instance.shape_id],
                            transform_ray(instance.frame_inverse, ray),
                            find_any, distance, element_id, element_uv)) {
                        hit         = true;
                        ray.tmax    = distance;
                        instance_id = node.primitive_ids[i];
                    }
                } else if (instance.surface_id >= 0) {
                    if (intersect_shape_bvh(bvh.surface_bvhs[instance.surface_id],
                            transform_ray(instance.frame_inverse, ray),
                            find_any, distance, element_id, element_uv)) {
                        hit         = true;
                        ray.tmax    = distance;
                        instance_id = node.primitive_ids[i];
                    }
                } else {
                    log_error("empty instance");
                }
            }
        } else {
            log_error("empty bvh");
        }

        // check for early exit
        if (find_any && hit) return true;
    }

    return hit;
}

// Finds the closest element with a bvh.
bool overlap_shape_bvh(const bvh_shape& bvh, const vec3f& pos,
    float max_distance, bool find_any, float& distance, int& instance_id,
    int& element_id, vec2f& element_uv) {
    // node stack
    int  node_stack[64];
    auto node_cur          = 0;
    node_stack[node_cur++] = 0;

    // hit
    auto hit = false;

    // walking stack
    while (node_cur) {
        // grab node
        auto node = bvh.nodes[node_stack[--node_cur]];

        // intersect bbox
        if (!distance_check_bbox(pos, max_distance, node.bbox)) continue;

        // intersect node, switching based on node type
        // for each type, iterate over the the primitive list
        if (node.is_internal) {
            // internal node
            node_stack[node_cur++] = node.primitive_ids[0];
            node_stack[node_cur++] = node.primitive_ids[1];
        } else if (!bvh.triangles.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& t = bvh.triangles[node.primitive_ids[i]];
                if (overlap_triangle(pos, max_distance, bvh.positions[t.x],
                        bvh.positions[t.y], bvh.positions[t.z], bvh.radius[t.x],
                        bvh.radius[t.y], bvh.radius[t.z], distance, element_uv)) {
                    hit          = true;
                    max_distance = distance;
                    element_id   = node.primitive_ids[i];
                }
            }
        } else if (!bvh.quads.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& q = bvh.quads[node.primitive_ids[i]];
                if (overlap_quad(pos, max_distance, bvh.positions[q.x],
                        bvh.positions[q.y], bvh.positions[q.z],
                        bvh.positions[q.w], bvh.radius[q.x], bvh.radius[q.y],
                        bvh.radius[q.z], bvh.radius[q.w], distance, element_uv)) {
                    hit          = true;
                    max_distance = distance;
                    element_id   = node.primitive_ids[i];
                }
            }
        } else if (!bvh.lines.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& l = bvh.lines[node.primitive_ids[i]];
                if (overlap_line(pos, max_distance, bvh.positions[l.x],
                        bvh.positions[l.y], bvh.radius[l.x], bvh.radius[l.y],
                        distance, element_uv)) {
                    hit          = true;
                    max_distance = distance;
                    element_id   = node.primitive_ids[i];
                }
            }
        } else if (!bvh.points.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& p = bvh.points[node.primitive_ids[i]];
                if (overlap_point(pos, max_distance, bvh.positions[p],
                        bvh.radius[p], distance, element_uv)) {
                    hit          = true;
                    max_distance = distance;
                    element_id   = node.primitive_ids[i];
                }
            }
        } else {
            log_error("empty bvh");
        }

        // check for early exit
        if (find_any && hit) return true;
    }

    return hit;
}

// Finds the closest element with a bvh.
bool overlap_scene_bvh(const bvh_scene& bvh, const vec3f& pos,
    float max_distance, bool find_any, float& distance, int& instance_id,
    int& element_id, vec2f& element_uv) {
    // node stack
    int  node_stack[64];
    auto node_cur          = 0;
    node_stack[node_cur++] = 0;

    // hit
    auto hit = false;

    // walking stack
    while (node_cur) {
        // grab node
        auto node = bvh.nodes[node_stack[--node_cur]];

        // intersect bbox
        if (!distance_check_bbox(pos, max_distance, node.bbox)) continue;

        // intersect node, switching based on node type
        // for each type, iterate over the the primitive list
        if (node.is_internal) {
            // internal node
            node_stack[node_cur++] = node.primitive_ids[0];
            node_stack[node_cur++] = node.primitive_ids[1];
        } else if (!bvh.instances.empty()) {
            for (auto i = 0; i < node.num_primitives; i++) {
                auto& instance = bvh.instances[node.primitive_ids[i]];
                if (overlap_shape_bvh(bvh.shape_bvhs[instance.shape_id],
                        transform_point(instance.frame_inverse, pos),
                        max_distance, find_any, distance, instance_id,
                        element_id, element_uv)) {
                    hit          = true;
                    max_distance = distance;
                    instance_id  = node.primitive_ids[i];
                }
            }
        } else {
            log_error("empty bvh");
        }

        // check for early exit
        if (find_any && hit) return true;
    }

    return hit;
}

#if 0
    // Finds the overlap between BVH leaf nodes.
    template <typename OverlapElem>
    void overlap_bvh_elems(const bvh_scene& bvh1, const bvh_scene& bvh2,
                           bool skip_duplicates, bool skip_self, vector<vec2i>& overlaps,
                           const OverlapElem& overlap_elems) {
        // node stack
        vec2i node_stack[128];
        auto node_cur = 0;
        node_stack[node_cur++] = {0, 0};

        // walking stack
        while (node_cur) {
            // grab node
            auto node_idx = node_stack[--node_cur];
            const auto node1 = bvh1->nodes[node_idx.x];
            const auto node2 = bvh2->nodes[node_idx.y];

            // intersect bbox
            if (!overlap_bbox(node1.bbox, node2.bbox)) continue;

            // check for leaves
            if (node1.isleaf && node2.isleaf) {
                // collide primitives
                for (auto i1 = node1.start; i1 < node1.start + node1.count; i1++) {
                    for (auto i2 = node2.start; i2 < node2.start + node2.count;
                         i2++) {
                        auto idx1 = bvh1->sorted_prim[i1];
                        auto idx2 = bvh2->sorted_prim[i2];
                        if (skip_duplicates && idx1 > idx2) continue;
                        if (skip_self && idx1 == idx2) continue;
                        if (overlap_elems(idx1, idx2))
                            overlaps.push_back({idx1, idx2});
                    }
                }
            } else {
                // descend
                if (node1.isleaf) {
                    for (auto idx2 = node2.start; idx2 < node2.start + node2.count;
                         idx2++) {
                        node_stack[node_cur++] = {node_idx.x, (int)idx2};
                    }
                } else if (node2.isleaf) {
                    for (auto idx1 = node1.start; idx1 < node1.start + node1.count;
                         idx1++) {
                        node_stack[node_cur++] = {(int)idx1, node_idx.y};
                    }
                } else {
                    for (auto idx2 = node2.start; idx2 < node2.start + node2.count;
                         idx2++) {
                        for (auto idx1 = node1.start;
                             idx1 < node1.start + node1.count; idx1++) {
                            node_stack[node_cur++] = {(int)idx1, (int)idx2};
                        }
                    }
                }
            }
        }
    }
#endif

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHAPE EXAMPLES
// -----------------------------------------------------------------------------
namespace ygl {

// Make a quad.
void make_quad_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec2i& steps, const vec2f& size, const vec2f& uvsize) {
    positions.resize((steps.x + 1) * (steps.y + 1));
    normals.resize((steps.x + 1) * (steps.y + 1));
    texturecoords.resize((steps.x + 1) * (steps.y + 1));
    for (auto j = 0; j <= steps.y; j++) {
        for (auto i = 0; i <= steps.x; i++) {
            auto uv = vec2f{i / (float)steps.x, j / (float)steps.y};
            positions[j * (steps.x + 1) + i] = {
                (uv.x - 0.5f) * size.x, (uv.y - 0.5f) * size.y, 0};
            normals[j * (steps.x + 1) + i]       = {0, 0, 1};
            texturecoords[j * (steps.x + 1) + i] = uv * uvsize;
        }
    }

    quads.resize(steps.x * steps.y);
    for (auto j = 0; j < steps.y; j++) {
        for (auto i = 0; i < steps.x; i++) {
            quads[j * steps.x + i] = {j * (steps.x + 1) + i,
                j * (steps.x + 1) + i + 1, (j + 1) * (steps.x + 1) + i + 1,
                (j + 1) * (steps.x + 1) + i};
        }
    }
    // shape.triangles.resize(steps.x * steps.y * 2);
    // for (auto j = 0; j < steps.y; j++) {
    //     for (auto i = 0; i < steps.x; i++) {
    //         shape.triangles[(j * steps.x + i) * 2 + 0] = {
    //             j * (steps.x + 1) + i, j * (steps.x + 1) + i + 1,
    //             (j + 1) * (steps.x + 1) + i + 1};
    //         shape.triangles[(j * steps.x + i) * 2 + 1] = {
    //             j * (steps.x + 1) + i, (j + 1) * (steps.x + 1) + i + 1,
    //             (j + 1) * (steps.x + 1) + i};
    //     }
    // }
}

void make_floor_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec2i& steps, const vec2f& size, const vec2f& uvsize) {
    make_quad_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    for (auto& p : positions) p = {p.x, p.z, p.y};
    for (auto& normal : normals) normal = {normal.x, normal.z, normal.y};
}

// Make a stack of quads
void make_quad_stack_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec3i& steps, const vec3f& size, const vec2f& uvsize) {
    auto qquads = vector<vec4i>{};
    auto qpositions = vector<vec3f>{};
    auto qnormals = vector<vec3f>{};
    auto qtexturecoords = vector<vec2f>{};
    for (auto i = 0; i <= steps.z; i++) {
        make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, {steps.x, steps.y}, {size.x, size.y}, uvsize);
        for (auto& p : qpositions)
            p.z = (-0.5f + (float)i / steps.z) * size.z;
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    }
}

// Make a cube.
void make_cube_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec3i& steps, const vec3f& size, const vec3f& uvsize) {
    auto qquads = vector<vec4i>{};
    auto qpositions = vector<vec3f>{};
    auto qnormals = vector<vec3f>{};
    auto qtexturecoords = vector<vec2f>{};
    // + z
    make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, 
        {steps.x, steps.y}, {size.x, size.y}, {uvsize.x, uvsize.y});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i] = {
            qpositions[i].x, qpositions[i].y, size.z / 2};
        qnormals[i] = {0, 0, 1};
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // - z
    make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, 
        {steps.x, steps.y}, {size.x, size.y}, {uvsize.x, uvsize.y});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i] = {
            -qpositions[i].x, qpositions[i].y, -size.z / 2};
        qnormals[i] = {0, 0, -1};
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // + x
    make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, 
        {steps.y, steps.z}, {size.y, size.z}, {uvsize.y, uvsize.z});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i] = {
            size.x / 2, qpositions[i].y, -qpositions[i].x};
        qnormals[i] = {1, 0, 0};
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // - x
    make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, 
        {steps.y, steps.z}, {size.y, size.z}, {uvsize.y, uvsize.z});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i] = {
            -size.x / 2, qpositions[i].y, qpositions[i].x};
        qnormals[i] = {-1, 0, 0};
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // + y
    make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, 
        {steps.x, steps.y}, {size.x, size.y}, {uvsize.x, uvsize.y});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i] = {
            qpositions[i].x, size.y / 2, -qpositions[i].y};
        qnormals[i] = {0, 1, 0};
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // - y
    make_quad_shape(qquads, qpositions, qnormals, qtexturecoords, 
        {steps.x, steps.y}, {size.x, size.y}, {uvsize.x, uvsize.y});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i] = {
            qpositions[i].x, -size.y / 2, qpositions[i].y};
        qnormals[i] = {0, -1, 0};
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
}

// Make a rounded cube.
void make_cube_rounded_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec3i& steps, const vec3f& size, const vec3f& uvsize, float radius) {
    make_cube_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    auto c     = size / 2 - vec3f{radius, radius, radius};
    for (auto i = 0; i < positions.size(); i++) {
        auto pc = vec3f{fabs(positions[i].x), fabs(positions[i].y),
            fabs(positions[i].z)};
        auto ps = vec3f{positions[i].x < 0 ? -1.0f : 1.0f,
            positions[i].y < 0 ? -1.0f : 1.0f,
            positions[i].z < 0 ? -1.0f : 1.0f};
        if (pc.x >= c.x && pc.y >= c.y && pc.z >= c.z) {
            auto pn            = normalize(pc - c);
            positions[i] = c + radius * pn;
            normals[i]   = pn;
        } else if (pc.x >= c.x && pc.y >= c.y) {
            auto pn            = normalize((pc - c) * vec3f{1, 1, 0});
            positions[i] = {c.x + radius * pn.x, c.y + radius * pn.y, pc.z};
            normals[i]   = pn;
        } else if (pc.x >= c.x && pc.z >= c.z) {
            auto pn            = normalize((pc - c) * vec3f{1, 0, 1});
            positions[i] = {c.x + radius * pn.x, pc.y, c.z + radius * pn.z};
            normals[i]   = pn;
        } else if (pc.y >= c.y && pc.z >= c.z) {
            auto pn            = normalize((pc - c) * vec3f{0, 1, 1});
            positions[i] = {pc.x, c.y + radius * pn.y, c.z + radius * pn.z};
            normals[i]   = pn;
        } else {
            continue;
        }
        positions[i] *= ps;
        normals[i] *= ps;
    }
}

// Make a sphere.
void make_sphere_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec2i& steps, float size, const vec2f& uvsize) {
    make_quad_shape(quads, positions, normals, texturecoords, steps, {1, 1}, {1, 1});
    for (auto i = 0; i < positions.size(); i++) {
        auto uv = texturecoords[i];
        auto a  = vec2f{2 * pif * uv.x, pif * (1 - uv.y)};
        auto p  = vec3f{cos(a.x) * sin(a.y), sin(a.x) * sin(a.y), cos(a.y)};
        positions[i]     = p * (size / 2);
        normals[i]       = normalize(p);
        texturecoords[i] = uv * uvsize;
    }
}

// Make a spherecube.
void make_sphere_cube_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,int steps, float size, float uvsize) {
    make_cube_shape(quads, positions, normals, texturecoords, 
        {steps, steps, steps}, {1, 1, 1}, {uvsize, uvsize, uvsize});
    for (auto i = 0; i < positions.size(); i++) {
        auto p             = positions[i];
        positions[i] = normalize(p) * (size / 2);
        normals[i]   = normalize(p);
    }
}

// Make a flipped sphere. This is not watertight.
void make_sphere_flipcap_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec2i& steps, float size, const vec2f& uvsize, const vec2f& zflip) {
    make_sphere_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    for (auto i = 0; i < positions.size(); i++) {
        if (positions[i].z > zflip.y) {
            positions[i].z = 2 * zflip.y - positions[i].z;
            normals[i].x   = -normals[i].x;
            normals[i].y   = -normals[i].y;
        } else if (positions[i].z < zflip.x) {
            positions[i].z = 2 * zflip.x - positions[i].z;
            normals[i].x   = -normals[i].x;
            normals[i].y   = -normals[i].y;
        }
    }
}

// Make a disk.
void make_disk_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec2i& steps, float size, const vec2f& uvsize) {
    make_quad_shape(quads, positions, normals, texturecoords, steps, {1, 1}, {1, 1});
    for (auto i = 0; i < positions.size(); i++) {
        auto uv            = texturecoords[i];
        auto phi           = 2 * pif * uv.x;
        positions[i] = {
            cos(phi) * uv.y * size / 2, sin(phi) * uv.y * size / 2, 0};
        normals[i]       = {0, 0, 1};
        texturecoords[i] = uv * uvsize;
    }
}

// Make a disk from a quad.
void make_disk_quad_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,int steps, float size, float uvsize) {
    make_quad_shape(quads, positions, normals, texturecoords, {steps, steps}, {2, 2}, {uvsize, uvsize});
    for (auto i = 0; i < positions.size(); i++) {
        // Analytical Methods for Squaring the Disc, by C. Fong
        // https://arxiv.org/abs/1509.06344
        auto xy = vec2f{positions[i].x, positions[i].y};
        auto uv = vec2f{
            xy.x * sqrt(1 - xy.y * xy.y / 2), xy.y * sqrt(1 - xy.x * xy.x / 2)};
        positions[i] = {uv.x * size / 2, uv.y * size / 2, 0};
    }
}

// Make a bulged disk from a quad.
void make_disk_bulged_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    int steps, float size, float uvsize, float height) {
    make_disk_quad_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    if (height == 0) return;
    auto radius = (size * size / 4 + height * height) / (2 * height);
    auto center = vec3f{0, 0, -radius + height};
    for (auto i = 0; i < positions.size(); i++) {
        auto pn            = normalize(positions[i] - center);
        positions[i] = center + pn * radius;
        normals[i]   = pn;
    }
}

// Make a cylinder (side-only).
void make_cylinder_side_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec2i& steps, const vec2f& size, const vec2f& uvsize) {
    make_quad_shape(quads, positions, normals, texturecoords, steps, {1, 1}, {1, 1});
    for (auto i = 0; i < positions.size(); i++) {
        auto uv                = texturecoords[i];
        auto phi               = 2 * pif * uv.x;
        positions[i]     = {cos(phi) * size.x / 2, sin(phi) * size.x / 2,
            (uv.y - 0.5f) * size.y};
        normals[i]       = {cos(phi), sin(phi), 0};
        texturecoords[i] = uv * uvsize;
    }
}

// Make a cylinder.
void make_cylinder_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec3i& steps, const vec2f& size, const vec3f& uvsize) {
    auto qquads = vector<vec4i>{};
    auto qpositions = vector<vec3f>{};
    auto qnormals = vector<vec3f>{};
    auto qtexturecoords = vector<vec2f>{};
    // side
    make_cylinder_side_shape(qquads, qpositions, qnormals, qtexturecoords,
        {steps.x, steps.y}, {size.x, size.y}, {uvsize.x, uvsize.y});
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // top
    make_disk_shape(qquads, qpositions, qnormals, qtexturecoords,{steps.x, steps.z}, size.x, {uvsize.x, uvsize.z});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i].z = size.y / 2;
    }
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
    // bottom
    make_disk_shape(qquads, qpositions, qnormals, qtexturecoords,{steps.x, steps.z}, size.x, {uvsize.x, uvsize.z});
    for (auto i = 0; i < qpositions.size(); i++) {
        qpositions[i].z = -size.y / 2;
        qnormals[i]     = -qnormals[i];
    }
    for (auto i = 0; i < qquads.size(); i++)
        swap(qquads[i].x, qquads[i].z);
        merge_quads(quads, positions, normals, texturecoords, qquads, qpositions, qnormals, qtexturecoords);
}

// Make a rounded cylinder.
void make_cylinder_rounded_shape(vector<vec4i>& quads, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords,
    const vec3i& steps, const vec2f& size, const vec3f& uvsize, float radius) {
    make_cylinder_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    auto c     = size / 2 - vec2f{radius, radius};
    for (auto i = 0; i < positions.size(); i++) {
        auto phi = atan2(positions[i].y, positions[i].x);
        auto r   = length(vec2f{positions[i].x, positions[i].y});
        auto z   = positions[i].z;
        auto pc  = vec2f{r, fabs(z)};
        auto ps  = (z < 0) ? -1.0f : 1.0f;
        if (pc.x >= c.x && pc.y >= c.y) {
            auto pn            = normalize(pc - c);
            positions[i] = {cos(phi) * c.x + radius * pn.x,
                sin(phi) * c.x + radius * pn.x, ps * (c.y + radius * pn.y)};
            normals[i]   = {cos(phi) * pn.x, sin(phi) * pn.x, ps * pn.y};
        } else {
            continue;
        }
    }
}

// Make a geodesic sphere.
void make_geodesic_sphere_shape(vector<vec3i>& triangles, vector<vec3f>& positions, vector<vec3f>& normals, int tesselation, float size) {
    // https://stackoverflow.com/questions/17705621/algorithm-for-a-geodesic-sphere
    const float X   = 0.525731112119133606f;
    const float Z   = 0.850650808352039932f;
    static auto sphere_pos = vector<vec3f>{{-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z},
        {X, 0.0, -Z}, {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
        {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}};
    static auto sphere_triangles = vector<vec3i>{{0, 1, 4}, {0, 4, 9}, {9, 4, 5},
        {4, 8, 5}, {4, 1, 8}, {8, 1, 10}, {8, 10, 3}, {5, 8, 3}, {5, 3, 2},
        {2, 3, 7}, {7, 3, 10}, {7, 10, 6}, {7, 6, 11}, {11, 6, 0}, {0, 6, 1},
        {6, 10, 1}, {9, 11, 0}, {9, 2, 11}, {9, 5, 2}, {7, 11, 2}};
    positions       = sphere_pos;
    triangles       = sphere_triangles;
    subdivide_triangles(
        triangles, positions, max(0, tesselation - 2));
    for (auto& p : positions) p = normalize(p) * size / 2;
    normals = positions;
}

// Make a facevarying cube with unique vertices but different texture
// coordinates.
void make_cube_facevarying_shape(vector<vec4i>& quads_positions,
    vector<vec4i>&                       quads_normals,
    vector<vec4i>& quads_texturecoords, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    const vec3i& steps, const vec3f& size, const vec3f& uvsize) {
    auto quads = vector<vec4i>{};
    make_cube_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    quads_positions     = quads;
    quads_normals       = quads;
    quads_texturecoords = quads;
    weld_quads(quads_positions, positions,
        min(0.1f * size / vec3f{(float)steps.x, (float)steps.y, (float)steps.z}));
}
void make_cube_multiplematerials_shape(vector<vec4i>& quads_positions,
    vector<vec4i>&                       quads_normals,
    vector<vec4i>& quads_texturecoords, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords, vector<int>& quads_materials,
    const vec3i& steps, const vec3f& size, const vec3f& uvsize) {
    make_cube_facevarying_shape(quads_positions, quads_normals, quads_texturecoords, positions, normals, texturecoords, steps, size, uvsize);
    quads_materials = vector<int>(quads_positions.size());
    auto quads_per_face = (int)quads_positions.size() / 6;
    for (auto i = 0; i < quads_positions.size(); i++) {
        quads_materials[i] = i / quads_per_face;
    }
}
void make_cube_posonly_shape(vector<vec4i>& quads,
vector<vec3f>& positions,
    const vec3i& steps, const vec3f& size, const vec3f& uvsize) {
        auto normals = vector<vec3f>{};
        auto texturecoords = vector<vec2f>{};
    make_cube_shape(quads, positions, normals, texturecoords, steps, size, uvsize);
    weld_quads(quads, positions,
        min(0.1f * size / vec3f{(float)steps.x, (float)steps.y, (float)steps.z}));
}

// Make a suzanne monkey model for testing. Note that some quads are
// degenerate.
void make_suzanne_shape(vector<vec4i>& quads,
vector<vec3f>& positions, float size) {
    static auto suzanne_pos       = vector<vec3f>{{0.4375, 0.1640625, 0.765625},
        {-0.4375, 0.1640625, 0.765625}, {0.5, 0.09375, 0.6875},
        {-0.5, 0.09375, 0.6875}, {0.546875, 0.0546875, 0.578125},
        {-0.546875, 0.0546875, 0.578125}, {0.3515625, -0.0234375, 0.6171875},
        {-0.3515625, -0.0234375, 0.6171875}, {0.3515625, 0.03125, 0.71875},
        {-0.3515625, 0.03125, 0.71875}, {0.3515625, 0.1328125, 0.78125},
        {-0.3515625, 0.1328125, 0.78125}, {0.2734375, 0.1640625, 0.796875},
        {-0.2734375, 0.1640625, 0.796875}, {0.203125, 0.09375, 0.7421875},
        {-0.203125, 0.09375, 0.7421875}, {0.15625, 0.0546875, 0.6484375},
        {-0.15625, 0.0546875, 0.6484375}, {0.078125, 0.2421875, 0.65625},
        {-0.078125, 0.2421875, 0.65625}, {0.140625, 0.2421875, 0.7421875},
        {-0.140625, 0.2421875, 0.7421875}, {0.2421875, 0.2421875, 0.796875},
        {-0.2421875, 0.2421875, 0.796875}, {0.2734375, 0.328125, 0.796875},
        {-0.2734375, 0.328125, 0.796875}, {0.203125, 0.390625, 0.7421875},
        {-0.203125, 0.390625, 0.7421875}, {0.15625, 0.4375, 0.6484375},
        {-0.15625, 0.4375, 0.6484375}, {0.3515625, 0.515625, 0.6171875},
        {-0.3515625, 0.515625, 0.6171875}, {0.3515625, 0.453125, 0.71875},
        {-0.3515625, 0.453125, 0.71875}, {0.3515625, 0.359375, 0.78125},
        {-0.3515625, 0.359375, 0.78125}, {0.4375, 0.328125, 0.765625},
        {-0.4375, 0.328125, 0.765625}, {0.5, 0.390625, 0.6875},
        {-0.5, 0.390625, 0.6875}, {0.546875, 0.4375, 0.578125},
        {-0.546875, 0.4375, 0.578125}, {0.625, 0.2421875, 0.5625},
        {-0.625, 0.2421875, 0.5625}, {0.5625, 0.2421875, 0.671875},
        {-0.5625, 0.2421875, 0.671875}, {0.46875, 0.2421875, 0.7578125},
        {-0.46875, 0.2421875, 0.7578125}, {0.4765625, 0.2421875, 0.7734375},
        {-0.4765625, 0.2421875, 0.7734375}, {0.4453125, 0.3359375, 0.78125},
        {-0.4453125, 0.3359375, 0.78125}, {0.3515625, 0.375, 0.8046875},
        {-0.3515625, 0.375, 0.8046875}, {0.265625, 0.3359375, 0.8203125},
        {-0.265625, 0.3359375, 0.8203125}, {0.2265625, 0.2421875, 0.8203125},
        {-0.2265625, 0.2421875, 0.8203125}, {0.265625, 0.15625, 0.8203125},
        {-0.265625, 0.15625, 0.8203125}, {0.3515625, 0.2421875, 0.828125},
        {-0.3515625, 0.2421875, 0.828125}, {0.3515625, 0.1171875, 0.8046875},
        {-0.3515625, 0.1171875, 0.8046875}, {0.4453125, 0.15625, 0.78125},
        {-0.4453125, 0.15625, 0.78125}, {0.0, 0.4296875, 0.7421875},
        {0.0, 0.3515625, 0.8203125}, {0.0, -0.6796875, 0.734375},
        {0.0, -0.3203125, 0.78125}, {0.0, -0.1875, 0.796875},
        {0.0, -0.7734375, 0.71875}, {0.0, 0.40625, 0.6015625},
        {0.0, 0.5703125, 0.5703125}, {0.0, 0.8984375, -0.546875},
        {0.0, 0.5625, -0.8515625}, {0.0, 0.0703125, -0.828125},
        {0.0, -0.3828125, -0.3515625}, {0.203125, -0.1875, 0.5625},
        {-0.203125, -0.1875, 0.5625}, {0.3125, -0.4375, 0.5703125},
        {-0.3125, -0.4375, 0.5703125}, {0.3515625, -0.6953125, 0.5703125},
        {-0.3515625, -0.6953125, 0.5703125}, {0.3671875, -0.890625, 0.53125},
        {-0.3671875, -0.890625, 0.53125}, {0.328125, -0.9453125, 0.5234375},
        {-0.328125, -0.9453125, 0.5234375}, {0.1796875, -0.96875, 0.5546875},
        {-0.1796875, -0.96875, 0.5546875}, {0.0, -0.984375, 0.578125},
        {0.4375, -0.140625, 0.53125}, {-0.4375, -0.140625, 0.53125},
        {0.6328125, -0.0390625, 0.5390625}, {-0.6328125, -0.0390625, 0.5390625},
        {0.828125, 0.1484375, 0.4453125}, {-0.828125, 0.1484375, 0.4453125},
        {0.859375, 0.4296875, 0.59375}, {-0.859375, 0.4296875, 0.59375},
        {0.7109375, 0.484375, 0.625}, {-0.7109375, 0.484375, 0.625},
        {0.4921875, 0.6015625, 0.6875}, {-0.4921875, 0.6015625, 0.6875},
        {0.3203125, 0.7578125, 0.734375}, {-0.3203125, 0.7578125, 0.734375},
        {0.15625, 0.71875, 0.7578125}, {-0.15625, 0.71875, 0.7578125},
        {0.0625, 0.4921875, 0.75}, {-0.0625, 0.4921875, 0.75},
        {0.1640625, 0.4140625, 0.7734375}, {-0.1640625, 0.4140625, 0.7734375},
        {0.125, 0.3046875, 0.765625}, {-0.125, 0.3046875, 0.765625},
        {0.203125, 0.09375, 0.7421875}, {-0.203125, 0.09375, 0.7421875},
        {0.375, 0.015625, 0.703125}, {-0.375, 0.015625, 0.703125},
        {0.4921875, 0.0625, 0.671875}, {-0.4921875, 0.0625, 0.671875},
        {0.625, 0.1875, 0.6484375}, {-0.625, 0.1875, 0.6484375},
        {0.640625, 0.296875, 0.6484375}, {-0.640625, 0.296875, 0.6484375},
        {0.6015625, 0.375, 0.6640625}, {-0.6015625, 0.375, 0.6640625},
        {0.4296875, 0.4375, 0.71875}, {-0.4296875, 0.4375, 0.71875},
        {0.25, 0.46875, 0.7578125}, {-0.25, 0.46875, 0.7578125},
        {0.0, -0.765625, 0.734375}, {0.109375, -0.71875, 0.734375},
        {-0.109375, -0.71875, 0.734375}, {0.1171875, -0.8359375, 0.7109375},
        {-0.1171875, -0.8359375, 0.7109375}, {0.0625, -0.8828125, 0.6953125},
        {-0.0625, -0.8828125, 0.6953125}, {0.0, -0.890625, 0.6875},
        {0.0, -0.1953125, 0.75}, {0.0, -0.140625, 0.7421875},
        {0.1015625, -0.1484375, 0.7421875}, {-0.1015625, -0.1484375, 0.7421875},
        {0.125, -0.2265625, 0.75}, {-0.125, -0.2265625, 0.75},
        {0.0859375, -0.2890625, 0.7421875}, {-0.0859375, -0.2890625, 0.7421875},
        {0.3984375, -0.046875, 0.671875}, {-0.3984375, -0.046875, 0.671875},
        {0.6171875, 0.0546875, 0.625}, {-0.6171875, 0.0546875, 0.625},
        {0.7265625, 0.203125, 0.6015625}, {-0.7265625, 0.203125, 0.6015625},
        {0.7421875, 0.375, 0.65625}, {-0.7421875, 0.375, 0.65625},
        {0.6875, 0.4140625, 0.7265625}, {-0.6875, 0.4140625, 0.7265625},
        {0.4375, 0.546875, 0.796875}, {-0.4375, 0.546875, 0.796875},
        {0.3125, 0.640625, 0.8359375}, {-0.3125, 0.640625, 0.8359375},
        {0.203125, 0.6171875, 0.8515625}, {-0.203125, 0.6171875, 0.8515625},
        {0.1015625, 0.4296875, 0.84375}, {-0.1015625, 0.4296875, 0.84375},
        {0.125, -0.1015625, 0.8125}, {-0.125, -0.1015625, 0.8125},
        {0.2109375, -0.4453125, 0.7109375}, {-0.2109375, -0.4453125, 0.7109375},
        {0.25, -0.703125, 0.6875}, {-0.25, -0.703125, 0.6875},
        {0.265625, -0.8203125, 0.6640625}, {-0.265625, -0.8203125, 0.6640625},
        {0.234375, -0.9140625, 0.6328125}, {-0.234375, -0.9140625, 0.6328125},
        {0.1640625, -0.9296875, 0.6328125}, {-0.1640625, -0.9296875, 0.6328125},
        {0.0, -0.9453125, 0.640625}, {0.0, 0.046875, 0.7265625},
        {0.0, 0.2109375, 0.765625}, {0.328125, 0.4765625, 0.7421875},
        {-0.328125, 0.4765625, 0.7421875}, {0.1640625, 0.140625, 0.75},
        {-0.1640625, 0.140625, 0.75}, {0.1328125, 0.2109375, 0.7578125},
        {-0.1328125, 0.2109375, 0.7578125}, {0.1171875, -0.6875, 0.734375},
        {-0.1171875, -0.6875, 0.734375}, {0.078125, -0.4453125, 0.75},
        {-0.078125, -0.4453125, 0.75}, {0.0, -0.4453125, 0.75},
        {0.0, -0.328125, 0.7421875}, {0.09375, -0.2734375, 0.78125},
        {-0.09375, -0.2734375, 0.78125}, {0.1328125, -0.2265625, 0.796875},
        {-0.1328125, -0.2265625, 0.796875}, {0.109375, -0.1328125, 0.78125},
        {-0.109375, -0.1328125, 0.78125}, {0.0390625, -0.125, 0.78125},
        {-0.0390625, -0.125, 0.78125}, {0.0, -0.203125, 0.828125},
        {0.046875, -0.1484375, 0.8125}, {-0.046875, -0.1484375, 0.8125},
        {0.09375, -0.15625, 0.8125}, {-0.09375, -0.15625, 0.8125},
        {0.109375, -0.2265625, 0.828125}, {-0.109375, -0.2265625, 0.828125},
        {0.078125, -0.25, 0.8046875}, {-0.078125, -0.25, 0.8046875},
        {0.0, -0.2890625, 0.8046875}, {0.2578125, -0.3125, 0.5546875},
        {-0.2578125, -0.3125, 0.5546875}, {0.1640625, -0.2421875, 0.7109375},
        {-0.1640625, -0.2421875, 0.7109375}, {0.1796875, -0.3125, 0.7109375},
        {-0.1796875, -0.3125, 0.7109375}, {0.234375, -0.25, 0.5546875},
        {-0.234375, -0.25, 0.5546875}, {0.0, -0.875, 0.6875},
        {0.046875, -0.8671875, 0.6875}, {-0.046875, -0.8671875, 0.6875},
        {0.09375, -0.8203125, 0.7109375}, {-0.09375, -0.8203125, 0.7109375},
        {0.09375, -0.7421875, 0.7265625}, {-0.09375, -0.7421875, 0.7265625},
        {0.0, -0.78125, 0.65625}, {0.09375, -0.75, 0.6640625},
        {-0.09375, -0.75, 0.6640625}, {0.09375, -0.8125, 0.640625},
        {-0.09375, -0.8125, 0.640625}, {0.046875, -0.8515625, 0.6328125},
        {-0.046875, -0.8515625, 0.6328125}, {0.0, -0.859375, 0.6328125},
        {0.171875, 0.21875, 0.78125}, {-0.171875, 0.21875, 0.78125},
        {0.1875, 0.15625, 0.7734375}, {-0.1875, 0.15625, 0.7734375},
        {0.3359375, 0.4296875, 0.7578125}, {-0.3359375, 0.4296875, 0.7578125},
        {0.2734375, 0.421875, 0.7734375}, {-0.2734375, 0.421875, 0.7734375},
        {0.421875, 0.3984375, 0.7734375}, {-0.421875, 0.3984375, 0.7734375},
        {0.5625, 0.3515625, 0.6953125}, {-0.5625, 0.3515625, 0.6953125},
        {0.5859375, 0.2890625, 0.6875}, {-0.5859375, 0.2890625, 0.6875},
        {0.578125, 0.1953125, 0.6796875}, {-0.578125, 0.1953125, 0.6796875},
        {0.4765625, 0.1015625, 0.71875}, {-0.4765625, 0.1015625, 0.71875},
        {0.375, 0.0625, 0.7421875}, {-0.375, 0.0625, 0.7421875},
        {0.2265625, 0.109375, 0.78125}, {-0.2265625, 0.109375, 0.78125},
        {0.1796875, 0.296875, 0.78125}, {-0.1796875, 0.296875, 0.78125},
        {0.2109375, 0.375, 0.78125}, {-0.2109375, 0.375, 0.78125},
        {0.234375, 0.359375, 0.7578125}, {-0.234375, 0.359375, 0.7578125},
        {0.1953125, 0.296875, 0.7578125}, {-0.1953125, 0.296875, 0.7578125},
        {0.2421875, 0.125, 0.7578125}, {-0.2421875, 0.125, 0.7578125},
        {0.375, 0.0859375, 0.7265625}, {-0.375, 0.0859375, 0.7265625},
        {0.4609375, 0.1171875, 0.703125}, {-0.4609375, 0.1171875, 0.703125},
        {0.546875, 0.2109375, 0.671875}, {-0.546875, 0.2109375, 0.671875},
        {0.5546875, 0.28125, 0.671875}, {-0.5546875, 0.28125, 0.671875},
        {0.53125, 0.3359375, 0.6796875}, {-0.53125, 0.3359375, 0.6796875},
        {0.4140625, 0.390625, 0.75}, {-0.4140625, 0.390625, 0.75},
        {0.28125, 0.3984375, 0.765625}, {-0.28125, 0.3984375, 0.765625},
        {0.3359375, 0.40625, 0.75}, {-0.3359375, 0.40625, 0.75},
        {0.203125, 0.171875, 0.75}, {-0.203125, 0.171875, 0.75},
        {0.1953125, 0.2265625, 0.75}, {-0.1953125, 0.2265625, 0.75},
        {0.109375, 0.4609375, 0.609375}, {-0.109375, 0.4609375, 0.609375},
        {0.1953125, 0.6640625, 0.6171875}, {-0.1953125, 0.6640625, 0.6171875},
        {0.3359375, 0.6875, 0.59375}, {-0.3359375, 0.6875, 0.59375},
        {0.484375, 0.5546875, 0.5546875}, {-0.484375, 0.5546875, 0.5546875},
        {0.6796875, 0.453125, 0.4921875}, {-0.6796875, 0.453125, 0.4921875},
        {0.796875, 0.40625, 0.4609375}, {-0.796875, 0.40625, 0.4609375},
        {0.7734375, 0.1640625, 0.375}, {-0.7734375, 0.1640625, 0.375},
        {0.6015625, 0.0, 0.4140625}, {-0.6015625, 0.0, 0.4140625},
        {0.4375, -0.09375, 0.46875}, {-0.4375, -0.09375, 0.46875},
        {0.0, 0.8984375, 0.2890625}, {0.0, 0.984375, -0.078125},
        {0.0, -0.1953125, -0.671875}, {0.0, -0.4609375, 0.1875},
        {0.0, -0.9765625, 0.4609375}, {0.0, -0.8046875, 0.34375},
        {0.0, -0.5703125, 0.3203125}, {0.0, -0.484375, 0.28125},
        {0.8515625, 0.234375, 0.0546875}, {-0.8515625, 0.234375, 0.0546875},
        {0.859375, 0.3203125, -0.046875}, {-0.859375, 0.3203125, -0.046875},
        {0.7734375, 0.265625, -0.4375}, {-0.7734375, 0.265625, -0.4375},
        {0.4609375, 0.4375, -0.703125}, {-0.4609375, 0.4375, -0.703125},
        {0.734375, -0.046875, 0.0703125}, {-0.734375, -0.046875, 0.0703125},
        {0.59375, -0.125, -0.1640625}, {-0.59375, -0.125, -0.1640625},
        {0.640625, -0.0078125, -0.4296875}, {-0.640625, -0.0078125, -0.4296875},
        {0.3359375, 0.0546875, -0.6640625}, {-0.3359375, 0.0546875, -0.6640625},
        {0.234375, -0.3515625, 0.40625}, {-0.234375, -0.3515625, 0.40625},
        {0.1796875, -0.4140625, 0.2578125}, {-0.1796875, -0.4140625, 0.2578125},
        {0.2890625, -0.7109375, 0.3828125}, {-0.2890625, -0.7109375, 0.3828125},
        {0.25, -0.5, 0.390625}, {-0.25, -0.5, 0.390625},
        {0.328125, -0.9140625, 0.3984375}, {-0.328125, -0.9140625, 0.3984375},
        {0.140625, -0.7578125, 0.3671875}, {-0.140625, -0.7578125, 0.3671875},
        {0.125, -0.5390625, 0.359375}, {-0.125, -0.5390625, 0.359375},
        {0.1640625, -0.9453125, 0.4375}, {-0.1640625, -0.9453125, 0.4375},
        {0.21875, -0.28125, 0.4296875}, {-0.21875, -0.28125, 0.4296875},
        {0.2109375, -0.2265625, 0.46875}, {-0.2109375, -0.2265625, 0.46875},
        {0.203125, -0.171875, 0.5}, {-0.203125, -0.171875, 0.5},
        {0.2109375, -0.390625, 0.1640625}, {-0.2109375, -0.390625, 0.1640625},
        {0.296875, -0.3125, -0.265625}, {-0.296875, -0.3125, -0.265625},
        {0.34375, -0.1484375, -0.5390625}, {-0.34375, -0.1484375, -0.5390625},
        {0.453125, 0.8671875, -0.3828125}, {-0.453125, 0.8671875, -0.3828125},
        {0.453125, 0.9296875, -0.0703125}, {-0.453125, 0.9296875, -0.0703125},
        {0.453125, 0.8515625, 0.234375}, {-0.453125, 0.8515625, 0.234375},
        {0.4609375, 0.5234375, 0.4296875}, {-0.4609375, 0.5234375, 0.4296875},
        {0.7265625, 0.40625, 0.3359375}, {-0.7265625, 0.40625, 0.3359375},
        {0.6328125, 0.453125, 0.28125}, {-0.6328125, 0.453125, 0.28125},
        {0.640625, 0.703125, 0.0546875}, {-0.640625, 0.703125, 0.0546875},
        {0.796875, 0.5625, 0.125}, {-0.796875, 0.5625, 0.125},
        {0.796875, 0.6171875, -0.1171875}, {-0.796875, 0.6171875, -0.1171875},
        {0.640625, 0.75, -0.1953125}, {-0.640625, 0.75, -0.1953125},
        {0.640625, 0.6796875, -0.4453125}, {-0.640625, 0.6796875, -0.4453125},
        {0.796875, 0.5390625, -0.359375}, {-0.796875, 0.5390625, -0.359375},
        {0.6171875, 0.328125, -0.5859375}, {-0.6171875, 0.328125, -0.5859375},
        {0.484375, 0.0234375, -0.546875}, {-0.484375, 0.0234375, -0.546875},
        {0.8203125, 0.328125, -0.203125}, {-0.8203125, 0.328125, -0.203125},
        {0.40625, -0.171875, 0.1484375}, {-0.40625, -0.171875, 0.1484375},
        {0.4296875, -0.1953125, -0.2109375},
        {-0.4296875, -0.1953125, -0.2109375}, {0.890625, 0.40625, -0.234375},
        {-0.890625, 0.40625, -0.234375}, {0.7734375, -0.140625, -0.125},
        {-0.7734375, -0.140625, -0.125}, {1.0390625, -0.1015625, -0.328125},
        {-1.0390625, -0.1015625, -0.328125}, {1.28125, 0.0546875, -0.4296875},
        {-1.28125, 0.0546875, -0.4296875}, {1.3515625, 0.3203125, -0.421875},
        {-1.3515625, 0.3203125, -0.421875}, {1.234375, 0.5078125, -0.421875},
        {-1.234375, 0.5078125, -0.421875}, {1.0234375, 0.4765625, -0.3125},
        {-1.0234375, 0.4765625, -0.3125}, {1.015625, 0.4140625, -0.2890625},
        {-1.015625, 0.4140625, -0.2890625}, {1.1875, 0.4375, -0.390625},
        {-1.1875, 0.4375, -0.390625}, {1.265625, 0.2890625, -0.40625},
        {-1.265625, 0.2890625, -0.40625}, {1.2109375, 0.078125, -0.40625},
        {-1.2109375, 0.078125, -0.40625}, {1.03125, -0.0390625, -0.3046875},
        {-1.03125, -0.0390625, -0.3046875}, {0.828125, -0.0703125, -0.1328125},
        {-0.828125, -0.0703125, -0.1328125}, {0.921875, 0.359375, -0.21875},
        {-0.921875, 0.359375, -0.21875}, {0.9453125, 0.3046875, -0.2890625},
        {-0.9453125, 0.3046875, -0.2890625}, {0.8828125, -0.0234375, -0.2109375},
        {-0.8828125, -0.0234375, -0.2109375}, {1.0390625, 0.0, -0.3671875},
        {-1.0390625, 0.0, -0.3671875}, {1.1875, 0.09375, -0.4453125},
        {-1.1875, 0.09375, -0.4453125}, {1.234375, 0.25, -0.4453125},
        {-1.234375, 0.25, -0.4453125}, {1.171875, 0.359375, -0.4375},
        {-1.171875, 0.359375, -0.4375}, {1.0234375, 0.34375, -0.359375},
        {-1.0234375, 0.34375, -0.359375}, {0.84375, 0.2890625, -0.2109375},
        {-0.84375, 0.2890625, -0.2109375}, {0.8359375, 0.171875, -0.2734375},
        {-0.8359375, 0.171875, -0.2734375}, {0.7578125, 0.09375, -0.2734375},
        {-0.7578125, 0.09375, -0.2734375}, {0.8203125, 0.0859375, -0.2734375},
        {-0.8203125, 0.0859375, -0.2734375}, {0.84375, 0.015625, -0.2734375},
        {-0.84375, 0.015625, -0.2734375}, {0.8125, -0.015625, -0.2734375},
        {-0.8125, -0.015625, -0.2734375}, {0.7265625, 0.0, -0.0703125},
        {-0.7265625, 0.0, -0.0703125}, {0.71875, -0.0234375, -0.171875},
        {-0.71875, -0.0234375, -0.171875}, {0.71875, 0.0390625, -0.1875},
        {-0.71875, 0.0390625, -0.1875}, {0.796875, 0.203125, -0.2109375},
        {-0.796875, 0.203125, -0.2109375}, {0.890625, 0.2421875, -0.265625},
        {-0.890625, 0.2421875, -0.265625}, {0.890625, 0.234375, -0.3203125},
        {-0.890625, 0.234375, -0.3203125}, {0.8125, -0.015625, -0.3203125},
        {-0.8125, -0.015625, -0.3203125}, {0.8515625, 0.015625, -0.3203125},
        {-0.8515625, 0.015625, -0.3203125}, {0.828125, 0.078125, -0.3203125},
        {-0.828125, 0.078125, -0.3203125}, {0.765625, 0.09375, -0.3203125},
        {-0.765625, 0.09375, -0.3203125}, {0.84375, 0.171875, -0.3203125},
        {-0.84375, 0.171875, -0.3203125}, {1.0390625, 0.328125, -0.4140625},
        {-1.0390625, 0.328125, -0.4140625}, {1.1875, 0.34375, -0.484375},
        {-1.1875, 0.34375, -0.484375}, {1.2578125, 0.2421875, -0.4921875},
        {-1.2578125, 0.2421875, -0.4921875}, {1.2109375, 0.0859375, -0.484375},
        {-1.2109375, 0.0859375, -0.484375}, {1.046875, 0.0, -0.421875},
        {-1.046875, 0.0, -0.421875}, {0.8828125, -0.015625, -0.265625},
        {-0.8828125, -0.015625, -0.265625}, {0.953125, 0.2890625, -0.34375},
        {-0.953125, 0.2890625, -0.34375}, {0.890625, 0.109375, -0.328125},
        {-0.890625, 0.109375, -0.328125}, {0.9375, 0.0625, -0.3359375},
        {-0.9375, 0.0625, -0.3359375}, {1.0, 0.125, -0.3671875},
        {-1.0, 0.125, -0.3671875}, {0.9609375, 0.171875, -0.3515625},
        {-0.9609375, 0.171875, -0.3515625}, {1.015625, 0.234375, -0.375},
        {-1.015625, 0.234375, -0.375}, {1.0546875, 0.1875, -0.3828125},
        {-1.0546875, 0.1875, -0.3828125}, {1.109375, 0.2109375, -0.390625},
        {-1.109375, 0.2109375, -0.390625}, {1.0859375, 0.2734375, -0.390625},
        {-1.0859375, 0.2734375, -0.390625}, {1.0234375, 0.4375, -0.484375},
        {-1.0234375, 0.4375, -0.484375}, {1.25, 0.46875, -0.546875},
        {-1.25, 0.46875, -0.546875}, {1.3671875, 0.296875, -0.5},
        {-1.3671875, 0.296875, -0.5}, {1.3125, 0.0546875, -0.53125},
        {-1.3125, 0.0546875, -0.53125}, {1.0390625, -0.0859375, -0.4921875},
        {-1.0390625, -0.0859375, -0.4921875}, {0.7890625, -0.125, -0.328125},
        {-0.7890625, -0.125, -0.328125}, {0.859375, 0.3828125, -0.3828125},
        {-0.859375, 0.3828125, -0.3828125}};
    static auto suzanne_triangles = vector<vec3i>{{60, 64, 48}, {49, 65, 61},
        {62, 64, 60}, {61, 65, 63}, {60, 58, 62}, {63, 59, 61}, {60, 56, 58},
        {59, 57, 61}, {60, 54, 56}, {57, 55, 61}, {60, 52, 54}, {55, 53, 61},
        {60, 50, 52}, {53, 51, 61}, {60, 48, 50}, {51, 49, 61}, {224, 228, 226},
        {227, 229, 225}, {72, 283, 73}, {73, 284, 72}, {341, 347, 383},
        {384, 348, 342}, {299, 345, 343}, {344, 346, 300}, {323, 379, 351},
        {352, 380, 324}, {441, 443, 445}, {446, 444, 442}, {463, 491, 465},
        {466, 492, 464}, {495, 497, 499}, {500, 498, 496}};
    static auto suzanne_quads = vector<vec4i>{{46, 0, 2, 44}, {3, 1, 47, 45},
        {44, 2, 4, 42}, {5, 3, 45, 43}, {2, 8, 6, 4}, {7, 9, 3, 5},
        {0, 10, 8, 2}, {9, 11, 1, 3}, {10, 12, 14, 8}, {15, 13, 11, 9},
        {8, 14, 16, 6}, {17, 15, 9, 7}, {14, 20, 18, 16}, {19, 21, 15, 17},
        {12, 22, 20, 14}, {21, 23, 13, 15}, {22, 24, 26, 20}, {27, 25, 23, 21},
        {20, 26, 28, 18}, {29, 27, 21, 19}, {26, 32, 30, 28}, {31, 33, 27, 29},
        {24, 34, 32, 26}, {33, 35, 25, 27}, {34, 36, 38, 32}, {39, 37, 35, 33},
        {32, 38, 40, 30}, {41, 39, 33, 31}, {38, 44, 42, 40}, {43, 45, 39, 41},
        {36, 46, 44, 38}, {45, 47, 37, 39}, {46, 36, 50, 48}, {51, 37, 47, 49},
        {36, 34, 52, 50}, {53, 35, 37, 51}, {34, 24, 54, 52}, {55, 25, 35, 53},
        {24, 22, 56, 54}, {57, 23, 25, 55}, {22, 12, 58, 56}, {59, 13, 23, 57},
        {12, 10, 62, 58}, {63, 11, 13, 59}, {10, 0, 64, 62}, {65, 1, 11, 63},
        {0, 46, 48, 64}, {49, 47, 1, 65}, {88, 173, 175, 90},
        {175, 174, 89, 90}, {86, 171, 173, 88}, {174, 172, 87, 89},
        {84, 169, 171, 86}, {172, 170, 85, 87}, {82, 167, 169, 84},
        {170, 168, 83, 85}, {80, 165, 167, 82}, {168, 166, 81, 83},
        {78, 91, 145, 163}, {146, 92, 79, 164}, {91, 93, 147, 145},
        {148, 94, 92, 146}, {93, 95, 149, 147}, {150, 96, 94, 148},
        {95, 97, 151, 149}, {152, 98, 96, 150}, {97, 99, 153, 151},
        {154, 100, 98, 152}, {99, 101, 155, 153}, {156, 102, 100, 154},
        {101, 103, 157, 155}, {158, 104, 102, 156}, {103, 105, 159, 157},
        {160, 106, 104, 158}, {105, 107, 161, 159}, {162, 108, 106, 160},
        {107, 66, 67, 161}, {67, 66, 108, 162}, {109, 127, 159, 161},
        {160, 128, 110, 162}, {127, 178, 157, 159}, {158, 179, 128, 160},
        {125, 155, 157, 178}, {158, 156, 126, 179}, {123, 153, 155, 125},
        {156, 154, 124, 126}, {121, 151, 153, 123}, {154, 152, 122, 124},
        {119, 149, 151, 121}, {152, 150, 120, 122}, {117, 147, 149, 119},
        {150, 148, 118, 120}, {115, 145, 147, 117}, {148, 146, 116, 118},
        {113, 163, 145, 115}, {146, 164, 114, 116}, {113, 180, 176, 163},
        {176, 181, 114, 164}, {109, 161, 67, 111}, {67, 162, 110, 112},
        {111, 67, 177, 182}, {177, 67, 112, 183}, {176, 180, 182, 177},
        {183, 181, 176, 177}, {134, 136, 175, 173}, {175, 136, 135, 174},
        {132, 134, 173, 171}, {174, 135, 133, 172}, {130, 132, 171, 169},
        {172, 133, 131, 170}, {165, 186, 184, 167}, {185, 187, 166, 168},
        {130, 169, 167, 184}, {168, 170, 131, 185}, {143, 189, 188, 186},
        {188, 189, 144, 187}, {184, 186, 188, 68}, {188, 187, 185, 68},
        {129, 130, 184, 68}, {185, 131, 129, 68}, {141, 192, 190, 143},
        {191, 193, 142, 144}, {139, 194, 192, 141}, {193, 195, 140, 142},
        {138, 196, 194, 139}, {195, 197, 138, 140}, {137, 70, 196, 138},
        {197, 70, 137, 138}, {189, 143, 190, 69}, {191, 144, 189, 69},
        {69, 190, 205, 207}, {206, 191, 69, 207}, {70, 198, 199, 196},
        {200, 198, 70, 197}, {196, 199, 201, 194}, {202, 200, 197, 195},
        {194, 201, 203, 192}, {204, 202, 195, 193}, {192, 203, 205, 190},
        {206, 204, 193, 191}, {198, 203, 201, 199}, {202, 204, 198, 200},
        {198, 207, 205, 203}, {206, 207, 198, 204}, {138, 139, 163, 176},
        {164, 140, 138, 176}, {139, 141, 210, 163}, {211, 142, 140, 164},
        {141, 143, 212, 210}, {213, 144, 142, 211}, {143, 186, 165, 212},
        {166, 187, 144, 213}, {80, 208, 212, 165}, {213, 209, 81, 166},
        {208, 214, 210, 212}, {211, 215, 209, 213}, {78, 163, 210, 214},
        {211, 164, 79, 215}, {130, 129, 71, 221}, {71, 129, 131, 222},
        {132, 130, 221, 219}, {222, 131, 133, 220}, {134, 132, 219, 217},
        {220, 133, 135, 218}, {136, 134, 217, 216}, {218, 135, 136, 216},
        {216, 217, 228, 230}, {229, 218, 216, 230}, {217, 219, 226, 228},
        {227, 220, 218, 229}, {219, 221, 224, 226}, {225, 222, 220, 227},
        {221, 71, 223, 224}, {223, 71, 222, 225}, {223, 230, 228, 224},
        {229, 230, 223, 225}, {182, 180, 233, 231}, {234, 181, 183, 232},
        {111, 182, 231, 253}, {232, 183, 112, 254}, {109, 111, 253, 255},
        {254, 112, 110, 256}, {180, 113, 251, 233}, {252, 114, 181, 234},
        {113, 115, 249, 251}, {250, 116, 114, 252}, {115, 117, 247, 249},
        {248, 118, 116, 250}, {117, 119, 245, 247}, {246, 120, 118, 248},
        {119, 121, 243, 245}, {244, 122, 120, 246}, {121, 123, 241, 243},
        {242, 124, 122, 244}, {123, 125, 239, 241}, {240, 126, 124, 242},
        {125, 178, 235, 239}, {236, 179, 126, 240}, {178, 127, 237, 235},
        {238, 128, 179, 236}, {127, 109, 255, 237}, {256, 110, 128, 238},
        {237, 255, 257, 275}, {258, 256, 238, 276}, {235, 237, 275, 277},
        {276, 238, 236, 278}, {239, 235, 277, 273}, {278, 236, 240, 274},
        {241, 239, 273, 271}, {274, 240, 242, 272}, {243, 241, 271, 269},
        {272, 242, 244, 270}, {245, 243, 269, 267}, {270, 244, 246, 268},
        {247, 245, 267, 265}, {268, 246, 248, 266}, {249, 247, 265, 263},
        {266, 248, 250, 264}, {251, 249, 263, 261}, {264, 250, 252, 262},
        {233, 251, 261, 279}, {262, 252, 234, 280}, {255, 253, 259, 257},
        {260, 254, 256, 258}, {253, 231, 281, 259}, {282, 232, 254, 260},
        {231, 233, 279, 281}, {280, 234, 232, 282}, {66, 107, 283, 72},
        {284, 108, 66, 72}, {107, 105, 285, 283}, {286, 106, 108, 284},
        {105, 103, 287, 285}, {288, 104, 106, 286}, {103, 101, 289, 287},
        {290, 102, 104, 288}, {101, 99, 291, 289}, {292, 100, 102, 290},
        {99, 97, 293, 291}, {294, 98, 100, 292}, {97, 95, 295, 293},
        {296, 96, 98, 294}, {95, 93, 297, 295}, {298, 94, 96, 296},
        {93, 91, 299, 297}, {300, 92, 94, 298}, {307, 308, 327, 337},
        {328, 308, 307, 338}, {306, 307, 337, 335}, {338, 307, 306, 336},
        {305, 306, 335, 339}, {336, 306, 305, 340}, {88, 90, 305, 339},
        {305, 90, 89, 340}, {86, 88, 339, 333}, {340, 89, 87, 334},
        {84, 86, 333, 329}, {334, 87, 85, 330}, {82, 84, 329, 331},
        {330, 85, 83, 332}, {329, 335, 337, 331}, {338, 336, 330, 332},
        {329, 333, 339, 335}, {340, 334, 330, 336}, {325, 331, 337, 327},
        {338, 332, 326, 328}, {80, 82, 331, 325}, {332, 83, 81, 326},
        {208, 341, 343, 214}, {344, 342, 209, 215}, {80, 325, 341, 208},
        {342, 326, 81, 209}, {78, 214, 343, 345}, {344, 215, 79, 346},
        {78, 345, 299, 91}, {300, 346, 79, 92}, {76, 323, 351, 303},
        {352, 324, 76, 303}, {303, 351, 349, 77}, {350, 352, 303, 77},
        {77, 349, 347, 304}, {348, 350, 77, 304}, {304, 347, 327, 308},
        {328, 348, 304, 308}, {325, 327, 347, 341}, {348, 328, 326, 342},
        {295, 297, 317, 309}, {318, 298, 296, 310}, {75, 315, 323, 76},
        {324, 316, 75, 76}, {301, 357, 355, 302}, {356, 358, 301, 302},
        {302, 355, 353, 74}, {354, 356, 302, 74}, {74, 353, 315, 75},
        {316, 354, 74, 75}, {291, 293, 361, 363}, {362, 294, 292, 364},
        {363, 361, 367, 365}, {368, 362, 364, 366}, {365, 367, 369, 371},
        {370, 368, 366, 372}, {371, 369, 375, 373}, {376, 370, 372, 374},
        {313, 377, 373, 375}, {374, 378, 314, 376}, {315, 353, 373, 377},
        {374, 354, 316, 378}, {353, 355, 371, 373}, {372, 356, 354, 374},
        {355, 357, 365, 371}, {366, 358, 356, 372}, {357, 359, 363, 365},
        {364, 360, 358, 366}, {289, 291, 363, 359}, {364, 292, 290, 360},
        {73, 359, 357, 301}, {358, 360, 73, 301}, {283, 285, 287, 289},
        {288, 286, 284, 290}, {283, 289, 359, 73}, {360, 290, 284, 73},
        {293, 295, 309, 361}, {310, 296, 294, 362}, {309, 311, 367, 361},
        {368, 312, 310, 362}, {311, 381, 369, 367}, {370, 382, 312, 368},
        {313, 375, 369, 381}, {370, 376, 314, 382}, {347, 349, 385, 383},
        {386, 350, 348, 384}, {317, 383, 385, 319}, {386, 384, 318, 320},
        {297, 299, 383, 317}, {384, 300, 298, 318}, {299, 343, 341, 383},
        {342, 344, 300, 384}, {313, 321, 379, 377}, {380, 322, 314, 378},
        {315, 377, 379, 323}, {380, 378, 316, 324}, {319, 385, 379, 321},
        {380, 386, 320, 322}, {349, 351, 379, 385}, {380, 352, 350, 386},
        {399, 387, 413, 401}, {414, 388, 400, 402}, {399, 401, 403, 397},
        {404, 402, 400, 398}, {397, 403, 405, 395}, {406, 404, 398, 396},
        {395, 405, 407, 393}, {408, 406, 396, 394}, {393, 407, 409, 391},
        {410, 408, 394, 392}, {391, 409, 411, 389}, {412, 410, 392, 390},
        {409, 419, 417, 411}, {418, 420, 410, 412}, {407, 421, 419, 409},
        {420, 422, 408, 410}, {405, 423, 421, 407}, {422, 424, 406, 408},
        {403, 425, 423, 405}, {424, 426, 404, 406}, {401, 427, 425, 403},
        {426, 428, 402, 404}, {401, 413, 415, 427}, {416, 414, 402, 428},
        {317, 319, 443, 441}, {444, 320, 318, 442}, {319, 389, 411, 443},
        {412, 390, 320, 444}, {309, 317, 441, 311}, {442, 318, 310, 312},
        {381, 429, 413, 387}, {414, 430, 382, 388}, {411, 417, 439, 443},
        {440, 418, 412, 444}, {437, 445, 443, 439}, {444, 446, 438, 440},
        {433, 445, 437, 435}, {438, 446, 434, 436}, {431, 447, 445, 433},
        {446, 448, 432, 434}, {429, 447, 431, 449}, {432, 448, 430, 450},
        {413, 429, 449, 415}, {450, 430, 414, 416}, {311, 447, 429, 381},
        {430, 448, 312, 382}, {311, 441, 445, 447}, {446, 442, 312, 448},
        {415, 449, 451, 475}, {452, 450, 416, 476}, {449, 431, 461, 451},
        {462, 432, 450, 452}, {431, 433, 459, 461}, {460, 434, 432, 462},
        {433, 435, 457, 459}, {458, 436, 434, 460}, {435, 437, 455, 457},
        {456, 438, 436, 458}, {437, 439, 453, 455}, {454, 440, 438, 456},
        {439, 417, 473, 453}, {474, 418, 440, 454}, {427, 415, 475, 463},
        {476, 416, 428, 464}, {425, 427, 463, 465}, {464, 428, 426, 466},
        {423, 425, 465, 467}, {466, 426, 424, 468}, {421, 423, 467, 469},
        {468, 424, 422, 470}, {419, 421, 469, 471}, {470, 422, 420, 472},
        {417, 419, 471, 473}, {472, 420, 418, 474}, {457, 455, 479, 477},
        {480, 456, 458, 478}, {477, 479, 481, 483}, {482, 480, 478, 484},
        {483, 481, 487, 485}, {488, 482, 484, 486}, {485, 487, 489, 491},
        {490, 488, 486, 492}, {463, 475, 485, 491}, {486, 476, 464, 492},
        {451, 483, 485, 475}, {486, 484, 452, 476}, {451, 461, 477, 483},
        {478, 462, 452, 484}, {457, 477, 461, 459}, {462, 478, 458, 460},
        {453, 473, 479, 455}, {480, 474, 454, 456}, {471, 481, 479, 473},
        {480, 482, 472, 474}, {469, 487, 481, 471}, {482, 488, 470, 472},
        {467, 489, 487, 469}, {488, 490, 468, 470}, {465, 491, 489, 467},
        {490, 492, 466, 468}, {391, 389, 503, 501}, {504, 390, 392, 502},
        {393, 391, 501, 499}, {502, 392, 394, 500}, {395, 393, 499, 497},
        {500, 394, 396, 498}, {397, 395, 497, 495}, {498, 396, 398, 496},
        {399, 397, 495, 493}, {496, 398, 400, 494}, {387, 399, 493, 505},
        {494, 400, 388, 506}, {493, 501, 503, 505}, {504, 502, 494, 506},
        {493, 495, 499, 501}, {500, 496, 494, 502}, {313, 381, 387, 505},
        {388, 382, 314, 506}, {313, 505, 503, 321}, {504, 506, 314, 322},
        {319, 321, 503, 389}, {504, 322, 320, 390}};

    positions = suzanne_pos;
    for (auto& p : positions) p *= size / 2;
    quads = suzanne_quads;
    for (auto& t : suzanne_triangles) {
        quads.push_back({t.x, t.y, t.z, t.z});
    }
}

// Watertight cube
void make_cube_shape(vector<vec4i>& quads,
vector<vec3f>& positions, const vec3f& size) {
    static auto cube_pos = vector<vec3f>{{-1, -1, -1}, {-1, +1, -1}, {+1, +1, -1},
        {+1, -1, -1}, {-1, -1, +1}, {-1, +1, +1}, {+1, +1, +1}, {+1, -1, +1}};
    static auto cube_quads   = vector<vec4i>{{0, 1, 2, 3}, {7, 6, 5, 4},
        {4, 5, 1, 0}, {6, 7, 3, 2}, {2, 1, 5, 6}, {0, 3, 7, 4}};
    static auto cube_quad_uv = vector<vec2f>{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    positions          = cube_pos;
    for (auto& p : positions) p *= size / 2;
    quads = cube_quads;
}

// Generate lines set along a quad.
void make_lines_shape(vector<vec2i>& lines, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords, vector<float>& radius, const vec2i& steps, const vec2f& size,
    const vec2f& uvsize, const vec2f& line_radius) {
    auto nverts = (steps.x + 1) * steps.y;
    auto nlines = steps.x * steps.y;
    auto vid    = [steps](int i, int j) { return j * (steps.x + 1) + i; };
    auto fid    = [steps](int i, int j) { return j * steps.x + i; };

    positions.resize(nverts);
    normals.resize(nverts);
    texturecoords.resize(nverts);
    radius.resize(nverts);
    if (steps.y > 1) {
        for (auto j = 0; j < steps.y; j++) {
            for (auto i = 0; i <= steps.x; i++) {
                auto uv                    = vec2f{i / (float)steps.x,
                    j / (float)(steps.y > 1 ? steps.y - 1 : 1)};
                positions[vid(i, j)] = {
                    (uv.x - 0.5f) * size.x, (uv.y - 0.5f) * size.y, 0};
                normals[vid(i, j)]       = {1, 0, 0};
                texturecoords[vid(i, j)] = uv * uvsize;
            }
        }
    } else {
        for (auto i = 0; i <= steps.x; i++) {
            auto uv                        = vec2f{i / (float)steps.x, 0};
            positions[vid(i, 0)]     = {(uv.x - 0.5f) * size.x, 0, 0};
            normals[vid(i, 0)]       = {1, 0, 0};
            texturecoords[vid(i, 0)] = uv * uvsize;
        }
    }

    lines.resize(nlines);
    for (int j = 0; j < steps.y; j++) {
        for (int i = 0; i < steps.x; i++) {
            lines[fid(i, j)] = {vid(i, j), vid(i + 1, j)};
        }
    }
}

// Generate a point set with points placed at the origin with texcoords
// varying along u.
void make_points_shape(vector<int>& points, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords, vector<float>& radius, int num, float uvsize, float point_radius) {
    points.resize(num);
    for (auto i = 0; i < num; i++) points[i] = i;
    positions.assign(num, {0, 0, 0});
    normals.assign(num, {0, 0, 1});
    texturecoords.assign(num, {0, 0});
    radius.assign(num, point_radius);
    for (auto i = 0; i < texturecoords.size(); i++)
        texturecoords[i] = {(float)i / (float)num, 0};
}

// Generate a point set.
void make_random_points_shape(vector<int>& points, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords, vector<float>& radius, int num, const vec3f& size,
    float uvsize, float point_radius, uint64_t seed) {
    make_points_shape(points, positions, normals, texturecoords, radius, num, uvsize, point_radius);
    auto rng   = make_rng(seed);
    for (auto i = 0; i < positions.size(); i++) {
        positions[i] = (get_random_vec3f(rng) - vec3f{0.5f, 0.5f, 0.5f}) *
                             size;
    }
}

// Make a point.
void make_point_shape(vector<int>& points, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords, vector<float>& radius, float point_radius) {
    points        = {0};
    positions     = {{0, 0, 0}};
    normals       = {{0, 0, 1}};
    texturecoords = {{0, 0}};
    radius        = {point_radius};
}

// Make a bezier circle. Returns bezier, pos.
void make_bezier_circle_shape(vector<vec4i>& beziers, vector<vec3f>& positions, float size) {
    // constant from http://spencermortensen.com/articles/bezier-circle/
    const auto  c              = 0.551915024494f;
    static auto circle_pos     = vector<vec3f>{{1, 0, 0}, {1, c, 0}, {c, 1, 0},
        {0, 1, 0}, {-c, 1, 0}, {-1, c, 0}, {-1, 0, 0}, {-1, -c, 0}, {-c, -1, 0},
        {0, -1, 0}, {c, -1, 0}, {1, -c, 0}};
    static auto circle_beziers = vector<vec4i>{
        {0, 1, 2, 3}, {3, 4, 5, 6}, {6, 7, 8, 9}, {9, 10, 11, 0}};
    positions = circle_pos;
    beziers   = circle_beziers;
    for(auto& p : positions) p *= size;
}

// Make a hair ball around a shape
void make_hair_shape(vector<vec2i>& lines, vector<vec3f>& positions, vector<vec3f>& normals,vector<vec2f>& texturecoords, vector<float>& radius, const vec2i& steps,
    const vector<vec3i>& striangles, const vector<vec4i>& squads,
    const vector<vec3f>& spos, const vector<vec3f>& snorm,
    const vector<vec2f>& stexcoord, const vec2f& len, const vec2f& rad,
    const vec2f& noise, const vec2f& clump, const vec2f& rotation, int seed) {
    vector<vec3f> bpos;
    vector<vec3f> bnorm;
    vector<vec2f> btexcoord;
    auto          alltriangles    = striangles;
    auto          quads_triangles = vector<vec3i>{};
    convert_quads_to_triangles(squads, quads_triangles);
    alltriangles.insert(
        alltriangles.end(), quads_triangles.begin(), quads_triangles.end());
    sample_triangles_points(alltriangles, spos, snorm, stexcoord, steps.y, bpos,
        bnorm, btexcoord, seed);

    auto rng  = make_rng(seed, 3);
    auto blen = vector<float>(bpos.size());
    for (auto& l : blen) l = lerp(len.x, len.y, get_random_float(rng));

    auto cidx = vector<int>();
    if (clump.x > 0) {
        for (auto bidx = 0; bidx < bpos.size(); bidx++) {
            cidx.push_back(0);
            auto cdist = maxf;
            for (auto c = 0; c < clump.y; c++) {
                auto d = length(bpos[bidx] - bpos[c]);
                if (d < cdist) {
                    cdist       = d;
                    cidx.back() = c;
                }
            }
        }
    }

    make_lines_shape(lines, positions, normals, texturecoords, radius, steps, {1, 1}, {1, 1});
    for (auto i = 0; i < positions.size(); i++) {
        auto u             = texturecoords[i].x;
        auto bidx          = i / (steps.x + 1);
        positions[i] = bpos[bidx] + bnorm[bidx] * u * blen[bidx];
        normals[i]   = bnorm[bidx];
        radius[i]    = lerp(rad.x, rad.y, u);
        if (clump.x > 0) {
            positions[i] = positions[i] +
                                 (positions[i + (cidx[bidx] - bidx) *
                                                          (steps.x + 1)] -
                                     positions[i]) *
                                     u * clump.x;
        }
        if (noise.x > 0) {
            auto nx = perlin_noise(positions[i] * noise.y + vec3f{0, 0, 0}) *
                      noise.x;
            auto ny = perlin_noise(
                          positions[i] * noise.y + vec3f{3, 7, 11}) *
                      noise.x;
            auto nz = perlin_noise(
                          positions[i] * noise.y + vec3f{13, 17, 19}) *
                      noise.x;
            positions[i] += {nx, ny, nz};
        }
    }

    if (clump.x > 0 || noise.x > 0 || rotation.x > 0)
        compute_vertex_tangents(lines, positions, normals);
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR COLOR UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Convert between CIE XYZ and xyY
vec3f xyz_to_xyY(const vec3f& xyz) {
    if (xyz == zero3f) return zero3f;
    return {xyz.x / (xyz.x + xyz.y + xyz.z), xyz.y / (xyz.x + xyz.y + xyz.z),
        xyz.y};
}
// Convert between CIE XYZ and xyY
vec3f xyY_to_xyz(const vec3f& xyY) {
    if (xyY.y == 0) return zero3f;
    return {xyY.x * xyY.z / xyY.y, xyY.z, (1 - xyY.x - xyY.y) * xyY.z / xyY.y};
}
// Convert between CIE XYZ and RGB
vec3f xyz_to_rgb(const vec3f& xyz) {
    // from http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    if (xyz == zero3f) return zero3f;
    return {+3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
        -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
        +0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z};
}
// Convert between CIE XYZ and RGB
vec3f rgb_to_xyz(const vec3f& rgb) {
    // from http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    if (rgb == zero3f) return zero3f;
    return {0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z,
        0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z,
        0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z};
}

// Convert HSV to RGB
vec3f hsv_to_rgb(const vec3f& hsv) {
    // from Imgui.cpp
    auto h = hsv.x, s = hsv.y, v = hsv.z;
    if (hsv.y == 0.0f) return {v, v, v};

    h       = fmodf(h, 1.0f) / (60.0f / 360.0f);
    int   i = (int)h;
    float f = h - (float)i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    switch (i) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        case 5: return {v, p, q};
        default: return {v, p, q};
    }
}
vec3f rgb_to_hsv(const vec3f& rgb) {
    // from Imgui.cpp
    auto  r = rgb.x, g = rgb.y, b = rgb.z;
    float K = 0.f;
    if (g < b) {
        swap(g, b);
        K = -1.f;
    }
    if (r < g) {
        swap(r, g);
        K = -2.f / 6.f - K;
    }

    float chroma = r - (g < b ? g : b);
    return {
        fabsf(K + (g - b) / (6.f * chroma + 1e-20f)), chroma / (r + 1e-20f), r};
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR IMAGE UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Gets an image size from a suggested size and an aspect ratio. The suggested
// size may have zeros in either components. In which case, we use the aspect
// ration to compute the other.
vec2i get_image_size(const vec2i& size, float aspect) {
    if (size == zero2i) {
        return {(int)round(720 * aspect), 720};
    } else if (size.y == 0) {
        return {size.x, (int)round(size.x / aspect)};
    } else if (size.x == 0) {
        return {(int)round(size.y * aspect), size.y};
    } else {
        return size;
    }
}

// Splits an image into an array of regions
void make_image_regions(vector<image_region>& regions, const vec2i& image_size, int region_size) {
    regions = vector<image_region>{};
    for (auto y = 0; y < image_size.y; y += region_size) {
        for (auto x = 0; x < image_size.x; x += region_size) {
            regions.push_back({{x, y}, {min(region_size, image_size.x - x),
                                           min(region_size, image_size.y - y)}});
        }
    }
}

// Conversion between linear and gamma-encoded images.
void gamma_to_linear(const image<vec4f>& srgb, image<vec4f>& lin, float gamma) {
    init_image(lin, srgb.size);
    for (auto idx = 0; idx < srgb.size.x * srgb.size.y; idx++) {
        lin.pixels[idx] = gamma_to_linear(srgb.pixels[idx], gamma);
    }
}
void linear_to_gamma(const image<vec4f>& lin, image<vec4f>& srgb, float gamma) {
    init_image(srgb, lin.size);
    for (auto idx = 0; idx < srgb.size.x * srgb.size.y; idx++) {
        srgb.pixels[idx] = linear_to_gamma(lin.pixels[idx], gamma);
    }
}

// Conversion between linear and gamma-encoded images.
void srgb_to_linear(const image<vec4f>& srgb,image<vec4f>& lin) {
    init_image(lin, srgb.size);
    for (auto idx = 0; idx < srgb.size.x * srgb.size.y; idx++) {
        lin.pixels[idx] = srgb_to_linear(srgb.pixels[idx]);
    }
}
void linear_to_srgb(const image<vec4f>& lin, image<vec4f>& srgb) {
    init_image(srgb, lin.size);
    for (auto idx = 0; idx < srgb.size.x * srgb.size.y; idx++) {
        srgb.pixels[idx] = linear_to_srgb(lin.pixels[idx]);
    }
}
void srgb_to_linear(const image<vec4b>& srgb,image<vec4f>& lin) {
    init_image(lin, srgb.size);
    for (auto idx = 0; idx < srgb.size.x * srgb.size.y; idx++) {
        lin.pixels[idx] = srgb_to_linear(byte_to_float(srgb.pixels[idx]));
    }
}
void linear_to_srgb(const image<vec4f>& lin, image<vec4b>& srgb) {
    init_image(srgb, lin.size);
    for (auto idx = 0; idx < srgb.size.x * srgb.size.y; idx++) {
        srgb.pixels[idx] = float_to_byte(linear_to_srgb(lin.pixels[idx]));
    }
}

// Conversion from/to floats.
void byte_to_float(const image<vec4b>& bt, image<vec4f>& fl) {
    init_image(fl, bt.size);
    for (auto idx = 0; idx < fl.size.x * fl.size.y; idx++) {
        fl.pixels[idx] = byte_to_float(bt.pixels[idx]);
    }
}
void float_to_byte(const image<vec4f>& fl, image<vec4b>& bt) {
    init_image(bt, fl.size);
    for (auto idx = 0; idx < fl.size.x * fl.size.y; idx++) {
        bt.pixels[idx] = float_to_byte(fl.pixels[idx]);
    }
}

// Tonemap image
void tonemap_image(
    const image<vec4f>& hdr, image<vec4f>& ldr, float exposure, bool filmic, bool srgb) {
    init_image(ldr, hdr.size);
    for (auto idx = 0; idx < hdr.size.x * hdr.size.y; idx++) {
        ldr.pixels[idx] = tonemap_filmic(hdr.pixels[idx], exposure, filmic, srgb);
    }
}
void tonemap_image(
    const image<vec4f>& hdr, image<vec4b>& ldr, float exposure, bool filmic, bool srgb) {
    init_image(ldr, hdr.size);
    for (auto idx = 0; idx < hdr.size.x * hdr.size.y; idx++) {
        ldr.pixels[idx] = float_to_byte(tonemap_filmic(hdr.pixels[idx], exposure, filmic, srgb));
    }
}

// Tonemap image
void tonemap_image_region(
    const image<vec4f>& hdr, image<vec4f>& ldr, const image_region& region, float exposure, bool filmic, bool srgb) {
    for (auto j = region.offset.y; j < region.offset.y + region.size.y; j++) {
        for (auto i = region.offset.x; i < region.offset.x + region.size.x; i++) {
            at(ldr, {i, j}) = tonemap_filmic(
                at(hdr, {i, j}), exposure, filmic, srgb);
        }
    }
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR IMAGE EXAMPLES
// -----------------------------------------------------------------------------
namespace ygl {

// Make a grid image
void make_grid_image(image<vec4f>& img, 
    const vec2i& size, int tiles, const vec4f& c0, const vec4f& c1) {
    init_image(img, size);
    auto tile = img.size.x / tiles;
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto c = i % tile == 0 || i % tile == tile - 1 || j % tile == 0 ||
                     j % tile == tile - 1;
            at(img, {i, j}) = (c) ? c0 : c1;
        }
    }
}

// Make a checkerboard image
void make_checker_image(image<vec4f>& img, 
    const vec2i& size, int tiles, const vec4f& c0, const vec4f& c1) {
    init_image(img, size);
    auto tile = img.size.x / tiles;
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto c          = (i / tile + j / tile) % 2 == 0;
            at(img, {i, j}) = (c) ? c0 : c1;
        }
    }
}

// Make an image with bumps and dimples.
void make_bumpdimple_image(image<vec4f>& img, const vec2i& size, int tiles) {
    init_image(img, size);
    auto tile = img.size.x / tiles;
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto c  = (i / tile + j / tile) % 2 == 0;
            auto ii = i % tile - tile / 2, jj = j % tile - tile / 2;
            auto r = sqrt(float(ii * ii + jj * jj)) /
                     sqrt(float(tile * tile) / 4);
            auto h = 0.5f;
            if (r < 0.5f) {
                h += (c) ? (0.5f - r) : -(0.5f - r);
            }
            at(img, {i, j}) = {h, h, h, 1};
        }
    }
}

// Make a uv colored grid
void make_ramp_image(image<vec4f>& img, const vec2i& size, const vec4f& c0, const vec4f& c1) {
    init_image(img, size);
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto u          = (float)i / (float)img.size.x;
            at(img, {i, j}) = c0 * (1 - u) + c1 * u;
        }
    }
}

// Make a gamma ramp image
void make_gammaramp_imagef(image<vec4f>& img, const vec2i& size) {
    init_image(img, size);
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto u = j / float(img.size.y - 1);
            if (i < img.size.x / 3) u = pow(u, 2.2f);
            if (i > (img.size.x * 2) / 3) u = pow(u, 1 / 2.2f);
            at(img, {i, j}) = {u, u, u, 1};
        }
    }
}

// Make an image color with red/green in the [0,1] range. Helpful to
// visualize uv texture coordinate application.
void make_uvramp_image(image<vec4f>& img, const vec2i& size) {
    init_image(img, size);
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            at(img, {i, j}) = {
                i / (float)(img.size.x - 1), j / (float)(img.size.y - 1), 0, 1};
        }
    }
}

// Make a uv colored grid
void make_uvgrid_image(image<vec4f>& img, const vec2i& size, int tiles, bool colored) {
    init_image(img, size);
    auto tile = img.size.x / tiles;
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto ii = i / tile, jj = j / tile;
            auto ww = img.size.x / tile, hh = img.size.y / tile;
            auto ph = (((256 / (ww * hh)) * (ii + jj * ww) - 64 + 256) % 256) /
                      360.f;
            auto pv = 0.5f;
            auto ps = 0.8f;
            if (i % (tile / 2) && j % (tile / 2)) {
                if ((i / tile + j / tile) % 2)
                    pv += 0.05f;
                else
                    pv -= 0.05f;
            } else {
                pv = 0.8f;
                ps = 0.2f;
            }
            auto rgb = (colored) ? hsv_to_rgb({ph, ps, pv}) : vec3f{pv, pv, pv};
            at(img, {i, img.size.y - j - 1}) = {rgb.x, rgb.y, rgb.z, 1};
        }
    }
}

// Comvert a bump map to a normal map.
void bump_to_normal_map(const image<vec4f>& img, image<vec4f>& norm, float scale) {
    init_image(norm, img.size);
    auto dx = 1.0f / img.size.x, dy = 1.0f / img.size.y;
    for (int j = 0; j < img.size.y; j++) {
        for (int i = 0; i < img.size.x; i++) {
            auto i1 = (i + 1) % img.size.x, j1 = (j + 1) % img.size.y;
            auto p00 = at(img, {i, j}), p10 = at(img, {i1, j}),
                 p01    = at(img, {i, j1});
            auto g00    = (p00.x + p00.y + p00.z) / 3;
            auto g01    = (p01.x + p01.y + p01.z) / 3;
            auto g10    = (p10.x + p10.y + p10.z) / 3;
            auto normal = vec3f{
                scale * (g00 - g10) / dx, scale * (g00 - g01) / dy, 1.0f};
            normal.y = -normal.y;  // make green pointing up, even if y axis
                                   // points down
            normal = normalize(normal) * 0.5f + vec3f{0.5f, 0.5f, 0.5f};
            at(norm, {i, j}) = {normal.x, normal.y, normal.z, 1};
        }
    }
}

// Implementation of sunsky modified heavily from pbrt
void make_sunsky_image(image<vec4f>& img, const vec2i& size, float thetaSun,
    float turbidity, bool has_sun, const vec3f& ground_albedo) {
    auto wSun = vec3f{0, cos(thetaSun), sin(thetaSun)};

    // sunSpectralRad =  ComputeAttenuatedSunlight(thetaS, turbidity);
    auto sunAngularRadius = 9.35e-03f / 2;  // Wikipedia
    auto thetaS           = thetaSun;

    auto t1 = thetaSun, t2 = thetaSun * thetaSun,
         t3 = thetaSun * thetaSun * thetaSun;
    auto T  = turbidity;
    auto T2 = turbidity * turbidity;

    auto zenith_xyY = vec3f{
        (+0.00165f * t3 - 0.00374f * t2 + 0.00208f * t1 + 0) * T2 +
            (-0.02902f * t3 + 0.06377f * t2 - 0.03202f * t1 + 0.00394f) * T +
            (+0.11693f * t3 - 0.21196f * t2 + 0.06052f * t1 + 0.25885f),
        (+0.00275f * t3 - 0.00610f * t2 + 0.00316f * t1 + 0) * T2 +
            (-0.04214f * t3 + 0.08970f * t2 - 0.04153f * t1 + 0.00515f) * T +
            (+0.15346f * t3 - 0.26756f * t2 + 0.06669f * t1 + 0.26688f),
        1000 * (4.0453f * T - 4.9710f) *
                tan((4.0f / 9.0f - T / 120.0f) * (pif - 2 * t1)) -
            .2155f * T + 2.4192f};

    auto perez_A_xyY = vec3f{-0.01925f * T - 0.25922f, -0.01669f * T - 0.26078f,
        +0.17872f * T - 1.46303f};
    auto perez_B_xyY = vec3f{-0.06651f * T + 0.00081f, -0.09495f * T + 0.00921f,
        -0.35540f * T + 0.42749f};
    auto perez_C_xyY = vec3f{-0.00041f * T + 0.21247f, -0.00792f * T + 0.21023f,
        -0.02266f * T + 5.32505f};
    auto perez_D_xyY = vec3f{-0.06409f * T - 0.89887f, -0.04405f * T - 1.65369f,
        +0.12064f * T - 2.57705f};
    auto perez_E_xyY = vec3f{-0.00325f * T + 0.04517f, -0.01092f * T + 0.05291f,
        -0.06696f * T + 0.37027f};

    auto perez_f = [thetaS](float A, float B, float C, float D, float E,
                       float theta, float gamma, float zenith) -> float {
        auto den = ((1 + A * exp(B)) *
                    (1 + C * exp(D * thetaS) + E * cos(thetaS) * cos(thetaS)));
        auto num = ((1 + A * exp(B / cos(theta))) *
                    (1 + C * exp(D * gamma) + E * cos(gamma) * cos(gamma)));
        return zenith * num / den;
    };

    auto sky = [&perez_f, perez_A_xyY, perez_B_xyY, perez_C_xyY, perez_D_xyY,
                   perez_E_xyY, zenith_xyY](auto theta, auto gamma) -> vec3f {
        auto x = perez_f(perez_A_xyY.x, perez_B_xyY.x, perez_C_xyY.x,
            perez_D_xyY.x, perez_E_xyY.x, theta, gamma, zenith_xyY.x);
        auto y = perez_f(perez_A_xyY.y, perez_B_xyY.y, perez_C_xyY.y,
            perez_D_xyY.y, perez_E_xyY.y, theta, gamma, zenith_xyY.y);
        auto Y = perez_f(perez_A_xyY.z, perez_B_xyY.z, perez_C_xyY.z,
            perez_D_xyY.z, perez_E_xyY.z, theta, gamma, zenith_xyY.z);
        return xyz_to_rgb(xyY_to_xyz({x, y, Y})) / 10000.0f;
    };

    // compute sun luminance
    // TODO: how this relates to zenith intensity?
    auto sun_ko     = vec3f{0.48f, 0.75f, 0.14f};
    auto sun_kg     = vec3f{0.1f, 0.0f, 0.0f};
    auto sun_kwa    = vec3f{0.02f, 0.0f, 0.0f};
    auto sun_sol    = vec3f{20000.0f, 27000.0f, 30000.0f};
    auto sun_lambda = vec3f{680, 530, 480};
    auto sun_beta   = 0.04608365822050f * turbidity - 0.04586025928522f;
    auto sun_m      = 1.0f /
                 (cos(thetaSun) + 0.000940f * pow(1.6386f - thetaSun, -1.253f));

    auto sun_le = zero3f;
    for (auto i = 0; i < 3; i++) {
        auto tauR = exp(
            -sun_m * 0.008735f * pow((&sun_lambda.x)[i] / 1000, -4.08f));
        auto tauA = exp(
            -sun_m * sun_beta * pow((&sun_lambda.x)[i] / 1000, -1.3f));
        auto tauO      = exp(-sun_m * (&sun_ko.x)[i] * .35f);
        auto tauG      = exp(-1.41f * (&sun_kg.x)[i] * sun_m /
                        pow(1 + 118.93f * (&sun_kg.x)[i] * sun_m, 0.45f));
        auto tauWA     = exp(-0.2385f * (&sun_kwa.x)[i] * 2.0f * sun_m /
                         pow(1 + 20.07f * (&sun_kwa.x)[i] * 2.0f * sun_m, 0.45f));
        (&sun_le.x)[i] = (&sun_sol.x)[i] * tauR * tauA * tauO * tauG * tauWA;
    }

    auto sun = [has_sun, sunAngularRadius, sun_le](auto theta, auto gamma) {
        return (has_sun && gamma < sunAngularRadius) ? sun_le / 10000.0f : zero3f;
    };

    init_image(img, size, {0, 0, 0, 1});
    for (auto j = 0; j < img.size.y / 2; j++) {
        auto theta = pif * ((j + 0.5f) / img.size.y);
        theta      = clamp(theta, 0.0f, pif / 2 - epsf);
        for (int i = 0; i < img.size.x; i++) {
            auto phi = 2 * pif * (float(i + 0.5f) / img.size.x);
            auto w   = vec3f{
                cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta)};
            auto gamma      = acos(clamp(dot(w, wSun), -1.0f, 1.0f));
            auto col        = sky(theta, gamma) + sun(theta, gamma);
            at(img, {i, j}) = {col.x, col.y, col.z, 1};
        }
    }

    if (ground_albedo != zero3f) {
        auto ground = zero3f;
        for (auto j = 0; j < img.size.y / 2; j++) {
            auto theta = pif * ((j + 0.5f) / img.size.y);
            for (int i = 0; i < img.size.x; i++) {
                auto pxl   = at(img, {i, j});
                auto le    = vec3f{pxl.x, pxl.y, pxl.z};
                auto angle = sin(theta) * 4 * pif / (img.size.x * img.size.y);
                ground += le * (ground_albedo / pif) * cos(theta) * angle;
            }
        }
        for (auto j = img.size.y / 2; j < img.size.y; j++) {
            for (int i = 0; i < img.size.x; i++) {
                at(img, {i, j}) = {ground.x, ground.y, ground.z, 1};
            }
        }
    }
}

// Make an image of multiple lights.
void make_lights_image(image<vec4f>& img, const vec2i& size, const vec3f& le, int nlights,
    float langle, float lwidth, float lheight) {
    init_image(img, size, {0, 0, 0, 1});
    for (auto j = 0; j < img.size.y / 2; j++) {
        auto theta = pif * ((j + 0.5f) / img.size.y);
        theta      = clamp(theta, 0.0f, pif / 2 - epsf);
        if (fabs(theta - langle) > lheight / 2) continue;
        for (int i = 0; i < img.size.x; i++) {
            auto phi     = 2 * pif * (float(i + 0.5f) / img.size.x);
            auto inlight = false;
            for (auto l = 0; l < nlights; l++) {
                auto lphi = 2 * pif * (l + 0.5f) / nlights;
                inlight   = inlight || fabs(phi - lphi) < lwidth / 2;
            }
            at(img, {i, j}) = {le.x, le.y, le.z, 1};
        }
    }
}

// Make a noise image. Wrap works only if size is a power of two.
void make_noise_image(image<vec4f>& img, const vec2i& size, float scale, bool wrap) {
    init_image(img, size);
    auto wrap3i = (wrap) ? vec3i{img.size.x, img.size.y, 2} : zero3i;
    for (auto j = 0; j < img.size.y; j++) {
        for (auto i = 0; i < img.size.x; i++) {
            auto p = vec3f{i / (float)img.size.x, j / (float)img.size.y, 0.5f} *
                     scale;
            auto g          = perlin_noise(p, wrap3i);
            g               = clamp(0.5f + 0.5f * g, 0.0f, 1.0f);
            at(img, {i, j}) = {g, g, g, 1};
        }
    }
}

// Make a noise image. Wrap works only if size is a power of two.
void make_fbm_image(image<vec4f>& img, const vec2i& size, float scale, float lacunarity,
    float gain, int octaves, bool wrap) {
    init_image(img, size);
    auto wrap3i = (wrap) ? vec3i{img.size.x, img.size.y, 2} : zero3i;
    for (auto j = 0; j < img.size.y; j++) {
        for (auto i = 0; i < img.size.x; i++) {
            auto p = vec3f{i / (float)img.size.x, j / (float)img.size.y, 0.5f} *
                     scale;
            auto g = perlin_fbm_noise(p, lacunarity, gain, octaves, wrap3i);
            g      = clamp(0.5f + 0.5f * g, 0.0f, 1.0f);
            at(img, {i, j}) = {g, g, g, 1};
        }
    }
}

// Make a noise image. Wrap works only if size is a power of two.
void make_ridge_image(image<vec4f>& img, const vec2i& size, float scale, float lacunarity,
    float gain, float offset, int octaves, bool wrap) {
    init_image(img, size);
    auto wrap3i = (wrap) ? vec3i{img.size.x, img.size.y, 2} : zero3i;
    for (auto j = 0; j < img.size.y; j++) {
        for (auto i = 0; i < img.size.x; i++) {
            auto p = vec3f{i / (float)img.size.x, j / (float)img.size.y, 0.5f} *
                     scale;
            auto g = perlin_ridge_noise(
                p, lacunarity, gain, offset, octaves, wrap3i);
            g               = clamp(g, 0.0f, 1.0f);
            at(img, {i, j}) = {g, g, g, 1};
        }
    }
}

// Make a noise image. Wrap works only if size is a power of two.
void make_turbulence_image(image<vec4f>& img, const vec2i& size, float scale,
    float lacunarity, float gain, int octaves, bool wrap) {
    init_image(img, size);
    auto wrap3i = (wrap) ? vec3i{img.size.x, img.size.y, 2} : zero3i;
    for (auto j = 0; j < img.size.y; j++) {
        for (auto i = 0; i < img.size.x; i++) {
            auto p = vec3f{i / (float)img.size.x, j / (float)img.size.y, 0.5f} *
                     scale;
            auto g = perlin_turbulence_noise(
                p, lacunarity, gain, octaves, wrap3i);
            g               = clamp(g, 0.0f, 1.0f);
            at(img, {i, j}) = {g, g, g, 1};
        }
    }
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR VOLUME EXAMPLES
// -----------------------------------------------------------------------------
namespace ygl {

// make a simple example volume
void make_test_volume1f(volume<float>& vol, const vec3i& size, float scale, float exponent) {
    init_volume(vol, size);
    for (auto k = 0; k < vol.size.z; k++) {
        for (auto j = 0; j < vol.size.y; j++) {
            for (auto i = 0; i < vol.size.x; i++) {
                auto p = vec3f{
                    i / (float)size.x, j / (float)size.y, k / (float)size.z};
                float value = pow(
                    max(max(cos(scale * p.x), cos(scale * p.y)), 0.0f), exponent);
                at(vol, {i, j, k}) = clamp(value, 0.0f, 1.0f);
            }
        }
    }
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SCENE UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Computes a shape bounding box.
bbox3f compute_shape_bounds(const yocto_shape& shape) {
    auto bbox = invalid_bbox3f;
    for (auto p : shape.positions) bbox += p;
    return bbox;
}

// Computes a surface bounding box.
bbox3f compute_surface_bounds(const yocto_surface& surface) {
    auto bbox = invalid_bbox3f;
    for (auto p : surface.positions) bbox += p;
    return bbox;
}

// Updates the scene and scene's instances bounding boxes
bbox3f compute_scene_bounds(const yocto_scene& scene) {
    auto shape_bbox = vector<bbox3f>(scene.shapes.size());
    for (auto shape_id = 0; shape_id < scene.shapes.size(); shape_id++)
        shape_bbox[shape_id] = compute_shape_bounds(scene.shapes[shape_id]);
    auto surface_bbox = vector<bbox3f>(scene.shapes.size());
    for (auto surface_id = 0; surface_id < scene.surfaces.size(); surface_id++)
        surface_bbox[surface_id] = compute_surface_bounds(
            scene.surfaces[surface_id]);
    auto bbox = invalid_bbox3f;
    for (auto& instance : scene.instances) {
        if (instance.shape >= 0) {
            bbox += transform_bbox(instance.frame, shape_bbox[instance.shape]);
        } else if (instance.surface >= 0) {
            bbox += transform_bbox(
                instance.frame, surface_bbox[instance.surface]);
        }
    }
    return bbox;
}

// Compute vertex normals
void compute_shape_normals(const yocto_shape& shape, vector<vec3f>& normals) {
    if (!shape.points.empty()) {
        normals = vector<vec3f>(shape.positions.size(), {0, 0, 1});
    } else if (!shape.lines.empty()) {
        return compute_vertex_tangents(shape.lines, shape.positions, normals);
    } else if (!shape.triangles.empty()) {
        return compute_vertex_normals(shape.triangles, shape.positions, normals);
    } else {
        normals = {};
        return;
    }
}

// Compute vertex normals
void compute_surface_normals(const yocto_surface& shape, vector<vec3f>& normals) {
    if (!shape.quads_positions.empty()) {
        return compute_vertex_normals(
            shape.quads_positions, shape.positions, normals);
    } else {
        normals = {};
        return;
    }
}

// Apply subdivision and displacement rules.
yocto_shape subdivide_shape(const yocto_shape& shape, int subdivision_level,
    bool catmull_clark, bool compute_normals) {
    if (!subdivision_level) return shape;
    auto subdivided = shape;
    if (!subdivided.points.empty()) {
        log_error("point subdivision not supported");
    } else if (!subdivided.lines.empty()) {
        auto lines_positions = subdivided.lines;
        subdivide_lines(lines_positions, subdivided.positions, subdivision_level);
        auto lines_normals = subdivided.lines;
        subdivide_lines(lines_normals, subdivided.normals, subdivision_level);
        auto lines_texturecoords = subdivided.lines;
        subdivide_lines(
            lines_texturecoords, subdivided.texturecoords, subdivision_level);
        auto lines_colors = subdivided.lines;
        subdivide_lines(lines_colors, subdivided.colors, subdivision_level);
        auto lines_radius = subdivided.lines;
        subdivide_lines(lines_radius, subdivided.colors, subdivision_level);
        subdivided.lines = lines_positions;
    } else if (!subdivided.triangles.empty()) {
        auto triangles_positions = subdivided.triangles;
        subdivide_triangles(
            triangles_positions, subdivided.positions, subdivision_level);
        auto triangles_normals = subdivided.triangles;
        subdivide_triangles(
            triangles_normals, subdivided.normals, subdivision_level);
        auto triangles_texturecoords = subdivided.triangles;
        subdivide_triangles(triangles_texturecoords, subdivided.texturecoords,
            subdivision_level);
        auto triangles_colors = subdivided.triangles;
        subdivide_triangles(
            triangles_colors, subdivided.colors, subdivision_level);
        subdivided.triangles = triangles_positions;
    } else if (!subdivided.quads.empty() && !catmull_clark) {
        auto quads_positions = subdivided.quads;
        subdivide_quads(quads_positions, subdivided.positions, subdivision_level);
        auto quads_normals = subdivided.quads;
        subdivide_quads(quads_normals, subdivided.normals, subdivision_level);
        auto quads_texturecoords = subdivided.quads;
        subdivide_quads(
            quads_texturecoords, subdivided.texturecoords, subdivision_level);
        auto quads_colors = subdivided.quads;
        subdivide_quads(quads_colors, subdivided.colors, subdivision_level);
        subdivided.quads = quads_positions;
    } else if (!subdivided.quads.empty() && catmull_clark) {
        auto quads_positions = subdivided.quads;
        subdivide_catmullclark(
            quads_positions, subdivided.positions, subdivision_level);
        auto quads_normals = subdivided.quads;
        subdivide_catmullclark(
            quads_normals, subdivided.normals, subdivision_level);
        auto quads_texturecoords = subdivided.quads;
        subdivide_catmullclark(
            quads_texturecoords, subdivided.texturecoords, subdivision_level);
        auto quads_colors = subdivided.quads;
        subdivide_catmullclark(
            quads_colors, subdivided.colors, subdivision_level);
        subdivided.quads = quads_positions;
    }

    if (compute_normals) {
        compute_shape_normals(subdivided, subdivided.normals);
    }

    return subdivided;
}
// Apply subdivision and displacement rules.
yocto_surface subdivide_surface(const yocto_surface& surface,
    int subdivision_level, bool catmull_clark, bool compute_normals) {
    if (!subdivision_level) return surface;
    auto subdivided    = surface;
    auto subdivide_ids = [](const vector<vec4i>& quads, vector<int>& ids,
                             int level) {
        if (ids.empty() || level <= 0) return;
        auto new_ids = vector<int>();
        for (int quad_id = 0; quad_id < quads.size(); quad_id++) {
            auto quad = quads[quad_id];
            for (auto i = 0; i < (quad.z == quad.w ? 3 : 4); i++)
                new_ids.push_back(ids[quad_id]);
        }
        swap(new_ids, ids);
        for (auto l = 1; l < level; l++) {
            new_ids = vector<int>();
            for (auto id : ids) {
                for (auto i = 0; i < 4; i++) new_ids.push_back(id);
            }
            swap(new_ids, ids);
        }
    };
    if (!subdivided.quads_positions.empty() && !catmull_clark) {
        subdivide_ids(subdivided.quads_positions, subdivided.quads_materials,
            subdivision_level);
        subdivide_quads(subdivided.quads_positions, subdivided.positions,
            subdivision_level);
        subdivide_quads(
            subdivided.quads_normals, subdivided.normals, subdivision_level);
        subdivide_quads(subdivided.quads_texturecoords,
            subdivided.texturecoords, subdivision_level);
    } else if (!subdivided.quads_positions.empty() && catmull_clark) {
        subdivide_ids(subdivided.quads_positions, subdivided.quads_materials,
            subdivision_level);
        subdivide_catmullclark(subdivided.quads_positions, subdivided.positions,
            subdivision_level);
        subdivide_catmullclark(subdivided.quads_texturecoords,
            subdivided.texturecoords, subdivision_level, true);
    }

    if (compute_normals) {
        if (!subdivided.quads_positions.empty())
            subdivided.quads_normals = subdivided.quads_positions;
        compute_surface_normals(subdivided, subdivided.normals);
    }

    return subdivided;
}
yocto_shape displace_shape(const yocto_shape& shape,
    const yocto_texture& displacement, bool compute_normals) {
    if (shape.texturecoords.empty()) {
        log_error("missing texture coordinates");
        return shape;
    }
    auto displaced_shape = shape;
    auto normals         = vector<vec3f>();
    if (shape.normals.empty())
        compute_shape_normals(shape, normals);
    else
        normals = shape.normals;
    for (auto vid = 0; vid < shape.positions.size(); vid++) {
        displaced_shape.positions[vid] += normals[vid] *
                                          displacement.height_scale *
                                          mean(xyz(evaluate_texture(displacement,
                                              shape.texturecoords[vid])));
    }

    if (compute_normals)
        compute_shape_normals(displaced_shape, displaced_shape.normals);

    return displaced_shape;
}
yocto_surface displace_surface(const yocto_surface& surface,
    const yocto_texture& displacement, bool compute_normals) {
    if (surface.texturecoords.empty()) {
        log_error("missing texture coordinates");
        return surface;
    }
    auto displaced_surface = surface;
    auto offset            = vector<float>(surface.positions.size(), 0);
    auto count             = vector<int>(surface.positions.size(), 0);
    for (auto fid = 0; fid < surface.quads_positions.size(); fid++) {
        auto qpos = surface.quads_positions[fid];
        auto qtxt = surface.quads_texturecoords[fid];
        for (auto i = 0; i < 4; i++) {
            offset[(&qpos.x)[i]] += displacement.height_scale *
                                    mean(xyz(evaluate_texture(displacement,
                                        surface.texturecoords[(&qtxt.x)[i]])));
            count[(&qpos.x)[i]] += 1;
        }
    }
    auto normals = vector<vec3f>();
    compute_vertex_normals(surface.quads_positions, surface.positions, normals);
    for (auto vid = 0; vid < surface.positions.size(); vid++) {
        displaced_surface.positions[vid] += normals[vid] * offset[vid] /
                                            count[vid];
    }

    if (compute_normals) {
        displaced_surface.quads_normals = displaced_surface.quads_positions;
        compute_surface_normals(displaced_surface, displaced_surface.normals);
    }

    return displaced_surface;
}

// Updates tesselation.
void tesselate_shapes_and_surfaces(yocto_scene& scene) {
    auto scope = log_trace_scoped("tesselating surfaces");
    for (auto& shape : scene.shapes) {
        auto& material = scene.materials[shape.material];
        if (!shape.subdivision_level && material.displacement_texture < 0)
            continue;
        auto tesselated_shape = shape;
        if (shape.subdivision_level) {
            tesselated_shape = subdivide_shape(tesselated_shape,
                tesselated_shape.subdivision_level,
                tesselated_shape.catmull_clark,
                tesselated_shape.compute_vertex_normals);
            tesselated_shape.subdivision_level = 0;
        }
        if (material.displacement_texture >= 0) {
            tesselated_shape = displace_shape(tesselated_shape,
                scene.textures[material.displacement_texture],
                shape.compute_vertex_normals);
        }
        shape = tesselated_shape;
    }
    for (auto& surface : scene.surfaces) {
        auto& material = scene.materials[surface.materials.front()];
        if (!surface.subdivision_level && material.displacement_texture < 0)
            continue;
        auto tesselated_surface = surface;
        if (surface.subdivision_level) {
            tesselated_surface = subdivide_surface(tesselated_surface,
                tesselated_surface.subdivision_level,
                tesselated_surface.catmull_clark,
                tesselated_surface.compute_vertex_normals);
            tesselated_surface.subdivision_level = 0;
        }
        if (material.displacement_texture >= 0) {
            tesselated_surface = displace_surface(tesselated_surface,
                scene.textures[material.displacement_texture],
                surface.compute_vertex_normals);
        }
        surface = tesselated_surface;
    }
}

// Update animation transforms
void update_transforms(yocto_scene& scene, yocto_animation& animation,
    float time, const string& anim_group) {
    if (anim_group != "" && anim_group != animation.animation_group) return;

    if (!animation.translation_keyframes.empty()) {
        auto value = vec3f{0, 0, 0};
        switch (animation.interpolation_type) {
            case yocto_interpolation_type::step:
                value = evaluate_keyframed_step(animation.keyframes_times,
                    animation.translation_keyframes, time);
                break;
            case yocto_interpolation_type::linear:
                value = evaluate_keyframed_linear(animation.keyframes_times,
                    animation.translation_keyframes, time);
                break;
            case yocto_interpolation_type::bezier:
                value = evaluate_keyframed_bezier(animation.keyframes_times,
                    animation.translation_keyframes, time);
                break;
            default: log_error("should not have been here");
        }
        for (auto target : animation.node_targets)
            scene.nodes[target].translation = value;
    }
    if (!animation.rotation_keyframes.empty()) {
        auto value = vec4f{0, 0, 0, 1};
        switch (animation.interpolation_type) {
            case yocto_interpolation_type::step:
                value = evaluate_keyframed_step(animation.keyframes_times,
                    animation.rotation_keyframes, time);
                break;
            case yocto_interpolation_type::linear:
                value = evaluate_keyframed_linear(animation.keyframes_times,
                    animation.rotation_keyframes, time);
                break;
            case yocto_interpolation_type::bezier:
                value = evaluate_keyframed_bezier(animation.keyframes_times,
                    animation.rotation_keyframes, time);
                break;
        }
        for (auto target : animation.node_targets)
            scene.nodes[target].rotation = value;
    }
    if (!animation.scale_keyframes.empty()) {
        auto value = vec3f{1, 1, 1};
        switch (animation.interpolation_type) {
            case yocto_interpolation_type::step:
                value = evaluate_keyframed_step(
                    animation.keyframes_times, animation.scale_keyframes, time);
                break;
            case yocto_interpolation_type::linear:
                value = evaluate_keyframed_linear(
                    animation.keyframes_times, animation.scale_keyframes, time);
                break;
            case yocto_interpolation_type::bezier:
                value = evaluate_keyframed_bezier(
                    animation.keyframes_times, animation.scale_keyframes, time);
                break;
        }
        for (auto target : animation.node_targets)
            scene.nodes[target].scale = value;
    }
}

// Update node transforms
void update_transforms(yocto_scene& scene, yocto_scene_node& node,
    const frame3f& parent = identity_frame3f) {
    auto frame = parent * node.local * translation_frame(node.translation) *
                 rotation_frame(node.rotation) * scaling_frame(node.scale);
    if (node.instance >= 0) scene.instances[node.instance].frame = frame;
    if (node.camera >= 0) scene.cameras[node.camera].frame = frame;
    if (node.environment >= 0)
        scene.environments[node.environment].frame = frame;
    for (auto child : node.children)
        update_transforms(scene, scene.nodes[child], frame);
}

// Update node transforms
void update_transforms(yocto_scene& scene, float time, const string& anim_group) {
    for (auto& agr : scene.animations)
        update_transforms(scene, agr, time, anim_group);
    for (auto& node : scene.nodes) node.children.clear();
    for (auto node_id = 0; node_id < scene.nodes.size(); node_id++) {
        auto& node = scene.nodes[node_id];
        if (node.parent >= 0)
            scene.nodes[node.parent].children.push_back(node_id);
    }
    for (auto& node : scene.nodes)
        if (node.parent >= 0) update_transforms(scene, node);
}

// Compute animation range
vec2f compute_animation_range(const yocto_scene& scene, const string& anim_group) {
    if (scene.animations.empty()) return zero2f;
    auto range = vec2f{+maxf, -maxf};
    for (auto& animation : scene.animations) {
        if (anim_group != "" && animation.animation_group != anim_group)
            continue;
        range.x = min(range.x, animation.keyframes_times.front());
        range.y = max(range.y, animation.keyframes_times.back());
    }
    if (range.y < range.x) return zero2f;
    return range;
}

// Generate a distribution for sampling a shape uniformly based on area/length.
void compute_shape_elements_cdf(const yocto_shape& shape, vector<float>& cdf) {
    if (!shape.triangles.empty()) {
        return sample_triangles_element_cdf(
            shape.triangles, shape.positions, cdf);
    } else if (!shape.quads.empty()) {
        return sample_quads_element_cdf(shape.quads, shape.positions, cdf);
    } else if (!shape.lines.empty()) {
        return sample_lines_element_cdf(shape.lines, shape.positions, cdf);
    } else if (!shape.points.empty()) {
        return sample_points_element_cdf(shape.points.size(), cdf);
    } else {
        log_error("empty shape not supported");
        cdf = {};
        return;
    }
}

// Sample a shape based on a distribution.
pair<int, vec2f> sample_shape_element(const yocto_shape& shape,
    const vector<float>& elements_cdf, float re, const vec2f& ruv) {
    // TODO: implement sampling without cdf
    if (elements_cdf.empty()) return {};
    if (!shape.triangles.empty()) {
        return sample_triangles_element(elements_cdf, re, ruv);
    } else if (!shape.quads.empty()) {
        return sample_quads_element(elements_cdf, re, ruv);
    } else if (!shape.lines.empty()) {
        return {get<0>(sample_lines_element(elements_cdf, re, ruv.x)), ruv};
    } else if (!shape.points.empty()) {
        return {sample_points_element(elements_cdf, re), ruv};
    } else {
        return {0, zero2f};
    }
}

float sample_shape_element_pdf(const yocto_shape& shape,
    const vector<float>& elements_cdf, int element_id, const vec2f& element_uv) {
    // prob triangle * area triangle = area triangle mesh
    return 1 / elements_cdf.back();
}

// Generate a distribution for sampling a shape uniformly based on area/length.
void compute_surface_elements_cdf(
    const yocto_surface& surface, vector<float>& cdf) {
    if (!surface.quads_positions.empty()) {
        return sample_quads_element_cdf(
            surface.quads_positions, surface.positions, cdf);
    } else {
        log_error("empty shape not supported");
        cdf = {};
        return;
    }
}

// Sample a shape based on a distribution.
pair<int, vec2f> sample_surface_element(const yocto_surface& surface,
    const vector<float>& elements_cdf, float re, const vec2f& ruv) {
    // TODO: implement sampling without cdf
    if (elements_cdf.empty()) return {};
    if (!surface.quads_positions.empty()) {
        return sample_quads_element(elements_cdf, re, ruv);
    } else {
        return {0, zero2f};
    }
}

float sample_surface_element_pdf(const yocto_surface& surface,
    const vector<float>& elements_cdf, int element_id, const vec2f& element_uv) {
    // prob triangle * area triangle = area triangle mesh
    return 1 / elements_cdf.back();
}

// Update environment CDF for sampling.
vector<float> compute_environment_texels_cdf(
    const yocto_scene& scene, const yocto_environment& environment) {
    if (environment.emission_texture < 0) return {};
    auto& texture  = scene.textures[environment.emission_texture];
    auto  size     = evaluate_texture_size(texture);
    auto  elem_cdf = vector<float>(size.x * size.y);
    if (size != zero2i) {
        for (auto i = 0; i < elem_cdf.size(); i++) {
            auto ij     = vec2i{i % size.x, i / size.x};
            auto th     = (ij.y + 0.5f) * pif / size.y;
            auto value  = lookup_texture(texture, ij);
            elem_cdf[i] = max(xyz(value)) * sin(th);
            if (i) elem_cdf[i] += elem_cdf[i - 1];
        }
    } else {
        log_error("empty texture");
    }
    return elem_cdf;
}

// Sample an environment based on texels
vec3f sample_environment_direction(const yocto_scene& scene,
    const yocto_environment& environment, const vector<float>& texels_cdf,
    float re, const vec2f& ruv) {
    if (!texels_cdf.empty() && environment.emission_texture >= 0) {
        auto& texture = scene.textures[environment.emission_texture];
        auto  idx     = sample_discrete_distribution(texels_cdf, re);
        auto  size    = evaluate_texture_size(texture);
        auto  u       = (idx % size.x + 0.5f) / size.x;
        auto  v       = (idx / size.x + 0.5f) / size.y;
        return evaluate_environment_direction(environment, {u, v});
    } else {
        return sample_sphere_direction(ruv);
    }
}

// Sample an environment based on texels
float sample_environment_direction_pdf(const yocto_scene& scene,
    const yocto_environment& environment, const vector<float>& texels_cdf,
    const vec3f& direction) {
    if (!texels_cdf.empty() && environment.emission_texture >= 0) {
        auto& texture = scene.textures[environment.emission_texture];
        auto  size    = evaluate_texture_size(texture);
        auto texcoord = evaluate_environment_texturecoord(environment, direction);
        auto i        = (int)(texcoord.x * size.x);
        auto j        = (int)(texcoord.y * size.y);
        auto idx      = j * size.x + i;
        auto prob     = sample_discrete_distribution_pdf(texels_cdf, idx) /
                    texels_cdf.back();
        auto angle = (2 * pif / size.x) * (pif / size.y) *
                     sin(pif * (j + 0.5f) / size.y);
        return prob / angle;
    } else {
        return sample_sphere_direction_pdf(direction);
    }
}

// Build a shape BVH
void build_shape_bvh(
    const yocto_shape& shape, bvh_shape& bvh, bool high_quality, bool embree) {
    // create bvh
    bvh = bvh_shape{};

    // set data
    bvh.positions = shape.positions;
    bvh.radius    = shape.radius;
    bvh.points    = shape.points;
    bvh.lines     = shape.lines;
    bvh.triangles = shape.triangles;
    bvh.quads     = shape.quads;

    // build bvh
    build_shape_bvh_embree(bvh, high_quality, embree);
}

// Build a shape BVH
void build_surface_bvh(
    const yocto_surface& surface, bvh_shape& bvh, bool high_quality, bool embree) {
    // create bvh
    bvh = bvh_shape{};

    // set data
    bvh.positions = surface.positions;
    bvh.quads     = surface.quads_positions;

    // build bvh
    build_shape_bvh_embree(bvh, high_quality, embree);
}

// Build a scene BVH
void build_scene_bvh(
    const yocto_scene& scene, bvh_scene& bvh, bool high_quality, bool embree) {
    auto scope = log_trace_scoped("building scene bvh");
    // create bvh
    bvh = bvh_scene{};

    // shapes
    bvh.shape_bvhs.resize(scene.shapes.size());
    for (auto shape_id = 0; shape_id < scene.shapes.size(); shape_id++) {
        build_shape_bvh(scene.shapes[shape_id], bvh.shape_bvhs[shape_id], high_quality, embree);
    }

    // surfaces
    bvh.surface_bvhs.resize(scene.surfaces.size());
    for (auto surface_id = 0; surface_id < scene.surfaces.size(); surface_id++) {
        build_surface_bvh(scene.surfaces[surface_id], bvh.surface_bvhs[surface_id], high_quality, embree);
    }

    // instances
    bvh.instances.resize(scene.instances.size());
    for (auto instance_id = 0; instance_id < scene.instances.size(); instance_id++) {
        auto& instance = scene.instances[instance_id];
        bvh.instances[instance_id] = {instance.frame, inverse(instance.frame, false),
            instance.shape, instance.surface};
    }

    // build bvh
    build_scene_bvh_embree(bvh, high_quality, embree);
}

// Refits a shape BVH
void refit_shape_bvh(const yocto_shape& shape, bvh_shape& bvh) {
    bvh.positions = shape.positions;
    bvh.radius    = shape.radius;
    refit_shape_bvh(bvh);
}

// Refits a surface BVH
void refit_surface_bvh(const yocto_surface& surface, bvh_shape& bvh) {
    bvh.positions = surface.positions;
    refit_shape_bvh(bvh);
}

// Refits a scene BVH
void refit_scene_bvh(const yocto_scene& scene, bvh_scene& bvh) {
    for (auto instance_id = 0; instance_id < scene.instances.size();
         instance_id++) {
        auto& instance             = scene.instances[instance_id];
        bvh.instances[instance_id] = {instance.frame, inverse(instance.frame),
            instance.shape, instance.surface};
    }
    refit_scene_bvh(bvh);
}

// Add missing names and resolve duplicated names.
void add_missing_names(yocto_scene& scene) {
    auto fix_names = [](auto& vals, const string& base) {
        auto nmap = unordered_map<string, int>();
        for (auto& value : vals) {
            if (value.name == "") value.name = base;
            if (nmap.find(value.name) == nmap.end()) {
                nmap[value.name] = 0;
            } else {
                nmap[value.name] += 1;
                value.name = value.name + "_" + std::to_string(nmap[value.name]);
            }
        }
    };
    fix_names(scene.cameras, "camera");
    fix_names(scene.shapes, "shape");
    fix_names(scene.surfaces, "surface");
    fix_names(scene.textures, "texture");
    fix_names(scene.voltextures, "voltexture");
    fix_names(scene.materials, "material");
    fix_names(scene.instances, "instance");
    fix_names(scene.environments, "environment");
    fix_names(scene.nodes, "node");
    fix_names(scene.animations, "animation");
}

// Add missing tangent space if needed.
void add_missing_tangent_space(yocto_scene& scene) {
    for (auto& shape : scene.shapes) {
        auto& material = scene.materials[shape.material];
        if (!shape.tangentspaces.empty() || shape.texturecoords.empty())
            continue;
        if (material.normal_texture < 0 && material.bump_texture < 0) continue;
        if (!shape.triangles.empty()) {
            if (shape.normals.empty())
                compute_vertex_normals(
                    shape.triangles, shape.positions, shape.normals);
            compute_tangent_spaces(shape.triangles, shape.positions,
                shape.normals, shape.texturecoords, shape.tangentspaces);
        } else {
            log_error("type not supported");
        }
    }
}

// Add missing materials.
void add_missing_materials(yocto_scene& scene) {
    auto material_id = -1;
    for (auto& shape : scene.shapes) {
        if (shape.material >= 0) continue;
        if (material_id < 0) {
            auto material    = yocto_material{};
            material.name    = "<default>";
            material.diffuse = {0.2f, 0.2f, 0.2f};
            scene.materials.push_back(material);
            material_id = (int)scene.materials.size() - 1;
        }
        shape.material = material_id;
    }
    for (auto& surface : scene.surfaces) {
        if (!surface.materials.empty()) continue;
        if (material_id < 0) {
            auto material    = yocto_material{};
            material.name    = "<default>";
            material.diffuse = {0.2f, 0.2f, 0.2f};
            scene.materials.push_back(material);
            material_id = (int)scene.materials.size() - 1;
        }
        surface.materials.push_back(material_id);
    }
}

// Add missing cameras.
void add_missing_cameras(yocto_scene& scene) {
    if (scene.cameras.empty()) {
        auto camera = yocto_camera{};
        camera.name = "<view>";
        set_camera_view(camera, compute_scene_bounds(scene), {0, 0, 1});
        scene.cameras.push_back(camera);
    }
}

// Add a sky environment
void add_sky_environment(yocto_scene& scene, float sun_angle) {
    auto texture      = yocto_texture{};
    texture.name      = "<sky>";
    texture.filename  = "textures/sky.hdr";
    make_sunsky_image(texture.hdr_image, {1024, 512}, sun_angle);
    scene.textures.push_back(texture);
    auto environment             = yocto_environment{};
    environment.name             = "<sky>";
    environment.emission         = {1, 1, 1};
    environment.emission_texture = (int)scene.textures.size() - 1;
    scene.environments.push_back(environment);
}

// Checks for validity of the scene.
vector<string> validate_scene(const yocto_scene& scene, bool skip_textures) {
    auto errs        = vector<string>();
    auto check_names = [&errs](const auto& vals, const string& base) {
        auto used = unordered_map<string, int>();
        for (auto& value : vals) used[value.name] += 1;
        for (auto& kv : used) {
            if (kv.first == "") {
                errs.push_back("empty " + base + " name");
            } else if (kv.second > 1) {
                errs.push_back("duplicated " + base + " name " + kv.first);
            }
        }
    };
    auto check_empty_textures = [&errs](const vector<yocto_texture>& vals) {
        for (auto& value : vals) {
            if (value.hdr_image.pixels.empty() && value.ldr_image.pixels.empty()) {
                errs.push_back("empty texture " + value.name);
            }
        }
    };

    check_names(scene.cameras, "camera");
    check_names(scene.shapes, "shape");
    check_names(scene.surfaces, "surface");
    check_names(scene.textures, "texture");
    check_names(scene.voltextures, "voltexture");
    check_names(scene.materials, "material");
    check_names(scene.instances, "instance");
    check_names(scene.environments, "environment");
    check_names(scene.nodes, "node");
    check_names(scene.animations, "animation");
    if (!skip_textures) check_empty_textures(scene.textures);

    return errs;
}

// Logs validations errors
void log_validation_errors(const yocto_scene& scene, bool skip_textures) {
    for (auto err : validate_scene(scene, skip_textures))
        log_error(err + " [validation]");
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR EVAL AND SAMPLING FUNCTIONS
// -----------------------------------------------------------------------------
namespace ygl {

// Scene intersection.
scene_intersection intersect_scene(const yocto_scene& scene,
    const bvh_scene& bvh, const ray3f& ray, bool find_any) {
    auto isec = scene_intersection();
    if (!intersect_scene_bvh(bvh, ray, find_any, isec.distance,
            isec.instance_id, isec.element_id, isec.element_uv))
        return {};
    return isec;
}

// Instance intersection.
scene_intersection intersect_scene(const yocto_scene& scene, int instance_id,
    const bvh_scene& bvh, const ray3f& ray, bool find_any) {
    auto& instance = scene.instances[instance_id];
    auto  isec     = scene_intersection();
    auto  tray     = transform_ray_inverse(instance.frame, ray);
    if (instance.shape >= 0) {
        if (!intersect_shape_bvh(bvh.shape_bvhs[instance.shape], tray, find_any,
                isec.distance, isec.element_id, isec.element_uv))
            return {};
    } else if (instance.surface) {
        if (!intersect_shape_bvh(bvh.shape_bvhs[instance.surface], tray,
                find_any, isec.distance, isec.element_id, isec.element_uv))
            return {};
    } else {
        return {};
    }
    isec.instance_id = instance_id;
    return isec;
}

// Shape element normal.
vec3f evaluate_shape_element_normal(const yocto_shape& shape, int element_id) {
    auto norm = zero3f;
    if (!shape.triangles.empty()) {
        auto t = shape.triangles[element_id];
        norm   = triangle_normal(
            shape.positions[t.x], shape.positions[t.y], shape.positions[t.z]);
    } else if (!shape.quads.empty()) {
        auto q = shape.quads[element_id];
        norm   = quad_normal(shape.positions[q.x], shape.positions[q.y],
            shape.positions[q.z], shape.positions[q.w]);
    } else if (!shape.lines.empty()) {
        auto l = shape.lines[element_id];
        norm   = line_tangent(shape.positions[l.x], shape.positions[l.y]);
    } else {
        norm = {0, 0, 1};
    }
    return norm;
}

// Shape element normal.
vec4f evaluate_shape_element_tangentspace(
    const yocto_shape& shape, int element_id) {
    auto tangsp = zero4f;
    if (!shape.triangles.empty()) {
        auto t    = shape.triangles[element_id];
        auto norm = triangle_normal(
            shape.positions[t.x], shape.positions[t.y], shape.positions[t.z]);
        auto txty = pair<vec3f, vec3f>();
        if (shape.texturecoords.empty()) {
            txty = triangle_tangents_fromuv(shape.positions[t.x],
                shape.positions[t.y], shape.positions[t.z], {0, 0}, {1, 0},
                {0, 1});
        } else {
            txty = triangle_tangents_fromuv(shape.positions[t.x],
                shape.positions[t.y], shape.positions[t.z],
                shape.texturecoords[t.x], shape.texturecoords[t.y],
                shape.texturecoords[t.z]);
        }
        auto tx = get<0>(txty), ty = get<1>(txty);
        tx     = orthonormalize(tx, norm);
        auto s = (dot(cross(norm, tx), ty) < 0) ? -1.0f : 1.0f;
        tangsp = {tx.x, tx.y, tx.z, s};
    }
    return tangsp;
}

// Shape value interpolated using barycentric coordinates
template <typename T>
T evaluate_shape_elem(const yocto_shape& shape, const vector<T>& vals,
    int element_id, const vec2f& element_uv) {
    if (vals.empty()) return {};
    if (!shape.triangles.empty()) {
        auto t = shape.triangles[element_id];
        return interpolate_triangle(vals[t.x], vals[t.y], vals[t.z], element_uv);
    } else if (!shape.quads.empty()) {
        auto q = shape.quads[element_id];
        if (q.w == q.z)
            return interpolate_triangle(
                vals[q.x], vals[q.y], vals[q.z], element_uv);
        return interpolate_quad(
            vals[q.x], vals[q.y], vals[q.z], vals[q.w], element_uv);
    } else if (!shape.lines.empty()) {
        auto l = shape.lines[element_id];
        return interpolate_line(vals[l.x], vals[l.y], element_uv.x);
    } else if (!shape.points.empty()) {
        return vals[shape.points[element_id]];
    } else {
        return {};
    }
}

// Shape values interpolated using barycentric coordinates
vec3f evaluate_shape_position(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    return evaluate_shape_elem(shape, shape.positions, element_id, element_uv);
}
vec3f evaluate_shape_normal(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    if (shape.normals.empty())
        return evaluate_shape_element_normal(shape, element_id);
    return normalize(
        evaluate_shape_elem(shape, shape.normals, element_id, element_uv));
}
vec2f evaluate_shape_texturecoord(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    if (shape.texturecoords.empty()) return element_uv;
    return evaluate_shape_elem(
        shape, shape.texturecoords, element_id, element_uv);
}
vec4f evaluate_shape_color(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    if (shape.colors.empty()) return {1, 1, 1, 1};
    return evaluate_shape_elem(shape, shape.colors, element_id, element_uv);
}
float evaluate_shape_radius(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    if (shape.radius.empty()) return 0.001f;
    return evaluate_shape_elem(shape, shape.radius, element_id, element_uv);
}
vec4f evaluate_shape_tangentspace(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    if (shape.tangentspaces.empty())
        return evaluate_shape_element_tangentspace(shape, element_id);
    return evaluate_shape_elem(
        shape, shape.tangentspaces, element_id, element_uv);
}
vec3f evaluate_shape_tangentspace(const yocto_shape& shape, int element_id,
    const vec2f& element_uv, bool& left_handed) {
    auto tangsp = (shape.tangentspaces.empty()) ?
                      evaluate_shape_element_tangentspace(shape, element_id) :
                      evaluate_shape_elem(
                          shape, shape.tangentspaces, element_id, element_uv);
    left_handed = tangsp.w < 0;
    return {tangsp.x, tangsp.y, tangsp.z};
}
// Shading normals including material perturbations.
vec3f evaluate_shape_shading_normal(const yocto_scene& scene,
    const yocto_shape& shape, int element_id, const vec2f& element_uv,
    const vec3f& outgoing) {
    if (!shape.triangles.empty()) {
        auto  normal   = evaluate_shape_normal(shape, element_id, element_uv);
        auto& material = scene.materials[shape.material];
        if (material.normal_texture >= 0) {
            auto texcoord = evaluate_shape_texturecoord(
                shape, element_id, element_uv);
            auto  left_handed    = false;
            auto& normal_texture = scene.textures[material.normal_texture];
            auto  texture = xyz(evaluate_texture(normal_texture, texcoord));
            texture       = texture * 2 - vec3f{1, 1, 1};
            texture.y = -texture.y;  // flip vertical axis to align green with
                                     // image up
            auto tu = orthonormalize(evaluate_shape_tangentspace(shape,
                                         element_id, element_uv, left_handed),
                normal);
            auto tv = normalize(cross(normal, tu) * (left_handed ? -1.0f : 1.0f));
            normal = normalize(
                texture.x * tu + texture.y * tv + texture.z * normal);
        }
        if (material.double_sided && dot(normal, outgoing) < 0)
            normal = -normal;
        return normal;
    } else if (!shape.quads.empty()) {
        auto  normal   = evaluate_shape_normal(shape, element_id, element_uv);
        auto& material = scene.materials[shape.material];
        if (material.double_sided && dot(normal, outgoing) < 0)
            normal = -normal;
        return normal;
    } else if (!shape.lines.empty()) {
        return orthonormalize(
            outgoing, evaluate_shape_normal(shape, element_id, element_uv));
    } else {
        return outgoing;
    }
}

// Shape element normal.
vec3f evaluate_surface_element_normal(
    const yocto_surface& surface, int element_id) {
    auto norm = zero3f;
    if (!surface.quads_positions.empty()) {
        auto q = surface.quads_positions[element_id];
        norm   = quad_normal(surface.positions[q.x], surface.positions[q.y],
            surface.positions[q.z], surface.positions[q.w]);
    } else {
        norm = {0, 0, 1};
    }
    return norm;
}

// Shape element normal.
vec4f evaluate_surface_element_tangentspace(
    const yocto_surface& surface, int element_id) {
    return zero4f;
}

// override for face-varying data
template <typename T>
T evaluate_surface_elem(const yocto_surface& surface, const vector<T>& vals,
    const vector<vec4i>& quads, int element_id, const vec2f& element_uv) {
    if (vals.empty()) return {};
    auto q = quads[element_id];
    if (q.w == q.z)
        return interpolate_triangle(vals[q.x], vals[q.y], vals[q.z], element_uv);
    return interpolate_quad(
        vals[q.x], vals[q.y], vals[q.z], vals[q.w], element_uv);
}

// Shape values interpolated using barycentric coordinates
vec3f evaluate_surface_position(
    const yocto_surface& surface, int element_id, const vec2f& element_uv) {
    return evaluate_surface_elem(surface, surface.positions,
        surface.quads_positions, element_id, element_uv);
}
vec3f evaluate_surface_normal(
    const yocto_surface& surface, int element_id, const vec2f& element_uv) {
    if (surface.normals.empty())
        return evaluate_surface_element_normal(surface, element_id);
    return normalize(evaluate_surface_elem(surface, surface.normals,
        surface.quads_normals, element_id, element_uv));
}
vec2f evaluate_surface_texturecoord(
    const yocto_surface& surface, int element_id, const vec2f& element_uv) {
    if (surface.texturecoords.empty()) return element_uv;
    return evaluate_surface_elem(surface, surface.texturecoords,
        surface.quads_texturecoords, element_id, element_uv);
}
vec4f evaluate_surface_tangentspace(
    const yocto_shape& shape, int element_id, const vec2f& element_uv) {
    return evaluate_shape_element_tangentspace(shape, element_id);
}
vec3f evaluate_surface_tangentspace(const yocto_surface& surface,
    int element_id, const vec2f& element_uv, bool& left_handed) {
    auto tangsp = evaluate_surface_element_tangentspace(surface, element_id);
    left_handed = tangsp.w < 0;
    return {tangsp.x, tangsp.y, tangsp.z};
}
// Shading normals including material perturbations.
vec3f evaluate_surface_shading_normal(const yocto_scene& scene,
    const yocto_surface& surface, int element_id, const vec2f& element_uv,
    const vec3f& outgoing) {
    if (!surface.quads_positions.empty()) {
        auto  normal = evaluate_surface_normal(surface, element_id, element_uv);
        auto& material = scene.materials[get_surface_element_material(
            surface, element_id)];
        if (material.double_sided && dot(normal, outgoing) < 0)
            normal = -normal;
        return normal;
    } else {
        return outgoing;
    }
}
// Per-element material.
int get_surface_element_material(const yocto_surface& surface, int element_id) {
    if (surface.materials.empty()) return -1;
    if (surface.quads_materials.empty()) return surface.materials.front();
    return surface.materials[surface.quads_materials[element_id]];
}

// Instance values interpolated using barycentric coordinates.
vec3f evaluate_instance_position(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv) {
    if (instance.shape >= 0) {
        return transform_point(instance.frame,
            evaluate_shape_position(
                scene.shapes[instance.shape], element_id, element_uv));
    } else if (instance.surface >= 0) {
        return transform_point(instance.frame,
            evaluate_surface_position(
                scene.surfaces[instance.surface], element_id, element_uv));
    } else {
        log_error("empty instance");
        return zero3f;
    }
}
vec3f evaluate_instance_normal(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv) {
    if (instance.shape >= 0) {
        return transform_direction(
            instance.frame, evaluate_shape_normal(scene.shapes[instance.shape],
                                element_id, element_uv));
    } else if (instance.surface >= 0) {
        return transform_direction(instance.frame,
            evaluate_surface_normal(
                scene.surfaces[instance.surface], element_id, element_uv));
    } else {
        log_error("empty instance");
        return zero3f;
    }
}
vec2f evaluate_instance_texturecoord(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv) {
    if (instance.shape >= 0) {
        return evaluate_shape_texturecoord(
            scene.shapes[instance.shape], element_id, element_uv);
    } else if (instance.surface >= 0) {
        return evaluate_surface_texturecoord(
            scene.surfaces[instance.surface], element_id, element_uv);
    } else {
        log_error("empty instance");
        return zero2f;
    }
}
vec3f evaluate_instance_tangentspace(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv,
    bool& left_handed) {
    if (instance.shape >= 0) {
        return transform_direction(instance.frame,
            evaluate_shape_tangentspace(scene.shapes[instance.shape],
                element_id, element_uv, left_handed));
    } else if (instance.surface >= 0) {
        return transform_direction(instance.frame,
            evaluate_surface_tangentspace(scene.surfaces[instance.surface],
                element_id, element_uv, left_handed));
    } else {
        log_error("empty instance");
        return zero3f;
    }
}
// Instance element values.
vec3f evaluate_instance_element_normal(
    const yocto_scene& scene, const yocto_instance& instance, int element_id) {
    if (instance.shape >= 0) {
        return transform_direction(
            instance.frame, evaluate_shape_element_normal(
                                scene.shapes[instance.shape], element_id));
    } else if (instance.surface >= 0) {
        return transform_direction(
            instance.frame, evaluate_surface_element_normal(
                                scene.surfaces[instance.surface], element_id));
    } else {
        log_error("empty instance");
        return zero3f;
    }
}
// Shading normals including material perturbations.
vec3f evaluate_instance_shading_normal(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv,
    const vec3f& outgoing) {
    if (instance.shape >= 0) {
        return transform_direction(instance.frame,
            evaluate_shape_shading_normal(scene, scene.shapes[instance.shape],
                element_id, element_uv,
                transform_direction_inverse(instance.frame, outgoing)));
    } else if (instance.surface >= 0) {
        return transform_direction(instance.frame,
            evaluate_surface_shading_normal(scene,
                scene.surfaces[instance.surface], element_id, element_uv,
                transform_direction_inverse(instance.frame, outgoing)));
    } else {
        log_error("empty instance");
        return zero3f;
    }
}

// Material values
vec3f evaluate_instance_emission(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv) {
    if (instance.shape >= 0) {
        auto& shape = scene.shapes[instance.shape];
        return evaluate_material_emission(scene, scene.materials[shape.material],
            evaluate_shape_texturecoord(shape, element_id, element_uv),
            evaluate_shape_color(shape, element_id, element_uv));
    } else if (instance.surface >= 0) {
        auto& surface = scene.surfaces[instance.surface];
        return evaluate_material_emission(scene,
            scene.materials[get_surface_element_material(surface, element_id)],
            evaluate_surface_texturecoord(surface, element_id, element_uv),
            {1, 1, 1, 1});
    } else {
        return zero3f;
    }
}
microfacet_brdf evaluate_instance_brdf(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv) {
    if (instance.shape >= 0) {
        auto& shape = scene.shapes[instance.shape];
        return evaluate_material_brdf(scene, scene.materials[shape.material],
            evaluate_shape_texturecoord(shape, element_id, element_uv),
            evaluate_shape_color(shape, element_id, element_uv));
    } else if (instance.surface >= 0) {
        auto& surface = scene.surfaces[instance.surface];
        return evaluate_material_brdf(scene,
            scene.materials[get_surface_element_material(surface, element_id)],
            evaluate_surface_texturecoord(surface, element_id, element_uv),
            {1, 1, 1, 1});
    } else {
        return {};
    }
}
float evaluate_instance_opacity(const yocto_scene& scene,
    const yocto_instance& instance, int element_id, const vec2f& element_uv) {
    if (instance.shape >= 0) {
        auto& shape = scene.shapes[instance.shape];
        return evaluate_material_opacity(scene, scene.materials[shape.material],
            evaluate_shape_texturecoord(shape, element_id, element_uv),
            evaluate_shape_color(shape, element_id, element_uv));
    } else if (instance.surface >= 0) {
        auto& surface = scene.surfaces[instance.surface];
        return evaluate_material_opacity(scene,
            scene.materials[get_surface_element_material(surface, element_id)],
            evaluate_surface_texturecoord(surface, element_id, element_uv),
            {1, 1, 1, 1});
    } else {
        return 0;
    }
}
bool is_instance_emissive(
    const yocto_scene& scene, const yocto_instance& instance) {
    if (instance.shape >= 0) {
        auto& shape = scene.shapes[instance.shape];
        return scene.materials[shape.material].emission != zero3f;
    } else if (instance.surface >= 0) {
        auto& surface = scene.surfaces[instance.surface];
        for (auto material_id : surface.materials)
            if (scene.materials[material_id].emission != zero3f) return true;
        return false;
    } else {
        return false;
    }
}

// Environment texture coordinates from the direction.
vec2f evaluate_environment_texturecoord(
    const yocto_environment& environment, const vec3f& direction) {
    auto wl = transform_direction_inverse(environment.frame, direction);
    auto environment_uv = vec2f{
        atan2(wl.z, wl.x) / (2 * pif), acos(clamp(wl.y, -1.0f, 1.0f)) / pif};
    if (environment_uv.x < 0) environment_uv.x += 1;
    return environment_uv;
}
// Evaluate the environment direction.
vec3f evaluate_environment_direction(
    const yocto_environment& environment, const vec2f& environment_uv) {
    return transform_direction(environment.frame,
        {cos(environment_uv.x * 2 * pif) * sin(environment_uv.y * pif),
            cos(environment_uv.y * pif),
            sin(environment_uv.x * 2 * pif) * sin(environment_uv.y * pif)});
}
// Evaluate the environment color.
vec3f evaluate_environment_emission(const yocto_scene& scene,
    const yocto_environment& environment, const vec3f& direction) {
    auto ke = environment.emission;
    if (environment.emission_texture >= 0) {
        auto& emission_texture = scene.textures[environment.emission_texture];
        ke *= xyz(evaluate_texture(emission_texture,
            evaluate_environment_texturecoord(environment, direction)));
    }
    return ke;
}
// Evaluate all environment color.
vec3f evaluate_environment_emission(
    const yocto_scene& scene, const vec3f& direction) {
    auto ke = zero3f;
    for (auto& environment : scene.environments)
        ke += evaluate_environment_emission(scene, environment, direction);
    return ke;
}

// Check texture size
vec2i evaluate_texture_size(const yocto_texture& texture) {
    if (!texture.hdr_image.pixels.empty()) {
        return texture.hdr_image.size;
    } else if (!texture.ldr_image.pixels.empty()) {
        return texture.ldr_image.size;
    } else {
        return zero2i;
    }
}

// Lookup a texture value
vec4f lookup_texture(const yocto_texture& texture, const vec2i& ij) {
    if (!texture.hdr_image.pixels.empty()) {
        return at(texture.hdr_image, ij);
    } else if (!texture.ldr_image.pixels.empty() && !texture.ldr_as_linear) {
        return srgb_to_linear(byte_to_float(at(texture.ldr_image, ij)));
    } else if (!texture.ldr_image.pixels.empty() && texture.ldr_as_linear) {
        return byte_to_float(at(texture.ldr_image, ij));
    } else {
        return zero4f;
    }
}

// Evaluate a texture
vec4f evaluate_texture(const yocto_texture& texture, const vec2f& texcoord) {
    if (texture.hdr_image.pixels.empty() && texture.ldr_image.pixels.empty())
        return {1, 1, 1, 1};

    // get image width/height
    auto size  = evaluate_texture_size(texture);
    auto width = size.x, height = size.y;

    // get coordinates normalized for tiling
    auto s = 0.0f, t = 0.0f;
    if (texture.clamp_to_edge) {
        s = clamp(texcoord.x, 0.0f, 1.0f) * width;
        t = clamp(texcoord.y, 0.0f, 1.0f) * height;
    } else {
        s = fmod(texcoord.x, 1.0f) * width;
        if (s < 0) s += width;
        t = fmod(texcoord.y, 1.0f) * height;
        if (t < 0) t += height;
    }

    // get image coordinates and residuals
    auto i = clamp((int)s, 0, width - 1), j = clamp((int)t, 0, height - 1);
    auto ii = (i + 1) % width, jj = (j + 1) % height;
    auto u = s - i, v = t - j;

    // nearest-neighbor interpolation
    if (texture.no_interpolation) {
        i = u < 0.5 ? i : min(i + 1, width - 1);
        j = v < 0.5 ? j : min(j + 1, height - 1);
        return lookup_texture(texture, {i, j});
    }

    // handle interpolation
    return lookup_texture(texture, {i, j}) * (1 - u) * (1 - v) +
           lookup_texture(texture, {i, jj}) * (1 - u) * v +
           lookup_texture(texture, {ii, j}) * u * (1 - v) +
           lookup_texture(texture, {ii, jj}) * u * v;
}

// Lookup a texture value
float lookup_voltexture(const yocto_voltexture& texture, const vec3i& ijk) {
    if (texture.volume_data.voxels.empty()) {
        return at(texture.volume_data, ijk);
    } else {
        return 0;
    }
}

// Evaluate a volume texture
float evaluate_voltexture(const yocto_voltexture& texture, const vec3f& texcoord) {
    if (texture.volume_data.voxels.empty()) return 1;

    // get image width/height
    auto width  = texture.volume_data.size.x;
    auto height = texture.volume_data.size.y;
    auto depth  = texture.volume_data.size.z;

    // get coordinates normalized for tiling
    auto s = clamp((texcoord.x + 1.0f) * 0.5f, 0.0f, 1.0f) * width;
    auto t = clamp((texcoord.y + 1.0f) * 0.5f, 0.0f, 1.0f) * height;
    auto r = clamp((texcoord.z + 1.0f) * 0.5f, 0.0f, 1.0f) * depth;

    // get image coordinates and residuals
    auto i  = clamp((int)s, 0, width - 1);
    auto j  = clamp((int)t, 0, height - 1);
    auto k  = clamp((int)r, 0, depth - 1);
    auto ii = (i + 1) % width, jj = (j + 1) % height, kk = (k + 1) % depth;
    auto u = s - i, v = t - j, w = r - k;

    // nearest-neighbor interpolation
    if (texture.no_interpolation) {
        i = u < 0.5 ? i : min(i + 1, width - 1);
        j = v < 0.5 ? j : min(j + 1, height - 1);
        k = w < 0.5 ? k : min(k + 1, depth - 1);
        return lookup_voltexture(texture, {i, j, k});
    }

    // trilinear interpolation
    return lookup_voltexture(texture, {i, j, k}) * (1 - u) * (1 - v) * (1 - w) +
           lookup_voltexture(texture, {ii, j, k}) * u * (1 - v) * (1 - w) +
           lookup_voltexture(texture, {i, jj, k}) * (1 - u) * v * (1 - w) +
           lookup_voltexture(texture, {i, j, kk}) * (1 - u) * (1 - v) * w +
           lookup_voltexture(texture, {i, jj, kk}) * (1 - u) * v * w +
           lookup_voltexture(texture, {ii, j, kk}) * u * (1 - v) * w +
           lookup_voltexture(texture, {ii, jj, k}) * u * v * (1 - w) +
           lookup_voltexture(texture, {ii, jj, kk}) * u * v * w;
}

// Set and evaluate camera parameters. Setters take zeros as default values.
float get_camera_fovx(const yocto_camera& camera) {
    return 2 * atan(camera.film_size.x / (2 * camera.focal_length));
}
float get_camera_fovy(const yocto_camera& camera) {
    return 2 * atan(camera.film_size.y / (2 * camera.focal_length));
}
float get_camera_aspect(const yocto_camera& camera) {
    return camera.film_size.x / camera.film_size.y;
}
vec2i get_camera_image_size(const yocto_camera& camera, const vec2i& size) {
    return get_image_size(size, camera.film_size.x / camera.film_size.y);
}
void set_camera_fovy(yocto_camera& camera, float fovy, float aspect, float width) {
    camera.film_size    = {width, width / aspect};
    camera.focal_length = camera.film_size.y / (2 * tan(fovy / 2));
}

// add missing camera
void set_camera_view(yocto_camera& camera, const bbox3f& bbox,
    const vec3f& view_direction, const vec2f& film, float focal) {
    camera.orthographic = false;
    if (film != zero2f) camera.film_size = film;
    if (focal != 0) camera.focal_length = focal;
    auto bbox_center = (bbox.max + bbox.min) / 2.0f;
    auto bbox_radius = length(bbox.max - bbox.min) / 2;
    auto camera_dir = (view_direction == zero3f) ? camera.frame.o - bbox_center :
                                                   view_direction;
    if (camera_dir == zero3f) camera_dir = {0, 0, 1};
    auto camera_fov = min(get_camera_fovx(camera), get_camera_fovy(camera));
    if (camera_fov == 0) camera_fov = 45 * pif / 180;
    auto camera_dist      = bbox_radius / sin(camera_fov / 2);
    auto from             = camera_dir * (camera_dist * 1) + bbox_center;
    auto to               = bbox_center;
    auto up               = vec3f{0, 1, 0};
    camera.frame          = lookat_frame(from, to, up);
    camera.focus_distance = length(from - to);
    camera.lens_aperture  = 0;
}

// Generates a ray from a camera for image plane coordinate uv and
// the lens coordinates luv.
ray3f evaluate_camera_ray(
    const yocto_camera& camera, const vec2f& image_uv, const vec2f& lens_uv) {
    auto distance = camera.focal_length;
    if (camera.focus_distance < maxf) {
        distance = camera.focal_length * camera.focus_distance /
                   (camera.focus_distance - camera.focal_length);
    }
    auto e = vec3f{
        lens_uv.x * camera.lens_aperture, lens_uv.y * camera.lens_aperture, 0};
    auto q   = vec3f{camera.film_size.x * (0.5f - image_uv.x),
        camera.film_size.y * (image_uv.y - 0.5f), distance};
    auto ray = make_ray(transform_point(camera.frame, e),
        transform_direction(camera.frame, normalize(e - q)));
    return ray;
}

// Generates a ray from a camera.
ray3f evaluate_camera_ray(const yocto_camera& camera, const vec2i& image_ij,
    const vec2i& image_size, const vec2f& pixel_uv, const vec2f& lens_uv) {
    auto image_uv = vec2f{(image_ij.x + pixel_uv.x) / image_size.x,
        (image_ij.y + pixel_uv.y) / image_size.y};
    return evaluate_camera_ray(camera, image_uv, lens_uv);
}

// Generates a ray from a camera.
ray3f evaluate_camera_ray(const yocto_camera& camera, int idx,
    const vec2i& image_size, const vec2f& pixel_uv, const vec2f& lens_uv) {
    auto image_ij = vec2i{idx % image_size.x, idx / image_size.x};
    auto image_uv = vec2f{(image_ij.x + pixel_uv.x) / image_size.x,
        (image_ij.y + pixel_uv.y) / image_size.y};
    return evaluate_camera_ray(camera, image_uv, lens_uv);
}

// Evaluates material parameters.
vec3f evaluate_material_emission(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    auto emission = material.emission * xyz(shape_color);
    if (material.emission_texture >= 0) {
        auto& emission_texture = scene.textures[material.emission_texture];
        emission *= xyz(evaluate_texture(emission_texture, texturecoord));
    }
    return emission;
}
vec3f evaluate_material_diffuse(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    if (!material.base_metallic) {
        auto diffuse = material.diffuse * xyz(shape_color);
        if (material.diffuse_texture >= 0) {
            auto& diffuse_texture = scene.textures[material.diffuse_texture];
            diffuse *= xyz(evaluate_texture(diffuse_texture, texturecoord));
        }
        return diffuse;
    } else {
        auto base = material.diffuse * xyz(shape_color);
        if (material.diffuse_texture >= 0) {
            auto& diffuse_texture = scene.textures[material.diffuse_texture];
            base *= xyz(evaluate_texture(diffuse_texture, texturecoord));
        }
        auto metallic = material.specular;
        if (material.specular_texture >= 0) {
            auto& specular_texture = scene.textures[material.specular_texture];
            metallic *= evaluate_texture(specular_texture, texturecoord).z;
        }
        return base * (1 - metallic);
    }
}
vec3f evaluate_material_specular(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    if (!material.base_metallic) {
        auto specular = material.specular * xyz(shape_color);
        if (material.specular_texture >= 0) {
            auto& specular_texture = scene.textures[material.specular_texture];
            specular *= xyz(evaluate_texture(specular_texture, texturecoord));
        }
        return specular;
    } else {
        auto base = material.diffuse * xyz(shape_color);
        if (material.diffuse_texture >= 0) {
            auto& diffuse_texture = scene.textures[material.diffuse_texture];
            base *= xyz(evaluate_texture(diffuse_texture, texturecoord));
        }
        auto metallic = material.specular.x;
        if (material.specular_texture >= 0) {
            auto& specular_texture = scene.textures[material.specular_texture];
            metallic *= evaluate_texture(specular_texture, texturecoord).z;
        }
        return base * metallic + 0.04f * (1 - metallic);
    }
}
float evaluate_material_roughness(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    if (!material.base_metallic) {
        if (!material.gltf_textures) {
            auto roughness = material.roughness;
            if (material.roughness_texture >= 0) {
                auto& roughness_texture = scene.textures[material.roughness_texture];
                roughness *= evaluate_texture(roughness_texture, texturecoord).x;
            }
            return roughness * roughness;
        } else {
            auto glossiness = 1 - material.roughness;
            if (material.roughness_texture >= 0) {
                auto& roughness_texture = scene.textures[material.roughness_texture];
                glossiness *= evaluate_texture(roughness_texture, texturecoord).w;
            }
            auto roughness = 1 - glossiness;
            return roughness * roughness;
        }
    } else {
        auto roughness = material.roughness;
        if (material.roughness_texture >= 0) {
            auto& roughness_texture = scene.textures[material.roughness_texture];
            roughness *= evaluate_texture(roughness_texture, texturecoord).y;
        }
        return roughness * roughness;
    }
}
vec3f evaluate_material_transmission(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    auto transmission = material.transmission * xyz(shape_color);
    if (material.transmission_texture >= 0) {
        auto& transmission_texture = scene.textures[material.transmission_texture];
        transmission *= xyz(evaluate_texture(transmission_texture, texturecoord));
    }
    return transmission;
}
float evaluate_material_opacity(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    auto opacity = material.opacity * shape_color.w;
    if (material.opacity_texture >= 0) {
        auto& opacity_texture = scene.textures[material.opacity_texture];
        opacity *= evaluate_texture(opacity_texture, texturecoord).w;
    }
    return opacity;
}

// Evaluates the microfacet_brdf at a location.
microfacet_brdf evaluate_material_brdf(const yocto_scene& scene,
    const yocto_material& material, const vec2f& texturecoord,
    const vec4f& shape_color) {
    auto brdf    = microfacet_brdf();
    brdf.diffuse = evaluate_material_diffuse(
        scene, material, texturecoord, shape_color);
    brdf.specular = evaluate_material_specular(
        scene, material, texturecoord, shape_color);
    brdf.transmission = evaluate_material_transmission(
        scene, material, texturecoord, shape_color);
    brdf.roughness = evaluate_material_roughness(
        scene, material, texturecoord, shape_color);
    brdf.refract = material.refract;
    if (brdf.diffuse != zero3f) {
        brdf.roughness = clamp(brdf.roughness, 0.03f * 0.03f, 1.0f);
    } else if (brdf.roughness <= 0.03f * 0.03f)
        brdf.roughness = 0;
    return brdf;
}
bool is_brdf_delta(const microfacet_brdf& brdf) {
    return brdf.roughness == 0 && brdf.diffuse == zero3f &&
           (brdf.specular != zero3f || brdf.transmission != zero3f);
}
bool is_brdf_zero(const microfacet_brdf& brdf) {
    return brdf.diffuse == zero3f && brdf.specular == zero3f &&
           brdf.transmission == zero3f;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// VOLUME, EVAL AND SAMPLING FUNCTIONS
// -----------------------------------------------------------------------------
namespace ygl {

bool is_material_volume_homogeneus(const yocto_material& material) {
    return material.volume_density_texture < 0;
}

bool is_material_volume_colored(const yocto_material& material) {
    return !(material.volume_density.x == material.volume_density.y &&
             material.volume_density.y == material.volume_density.z);
}

vec3f evaluate_transmission(const yocto_scene& scene,
    const yocto_material& material, const vec3f& from, const vec3f& dir,
    float distance, int channel, rng_state& rng) {
    auto& vd = material.volume_density;
    if (is_material_volume_homogeneus(material))
        return vec3f{exp(-distance * vd.x), exp(-distance * vd.y),
            exp(-distance * vd.z)};

    // ratio tracking
    auto tr = 1.0f, t = 0.0f;
    auto pos = from;
    while (true) {
        auto step = -log(1 - get_random_float(rng)) / at(vd, channel);
        t += step;
        if (t >= distance) break;
        pos += dir * step;
        auto density = material.volume_density;
        if (material.volume_density_texture >= 0) {
            auto& volume_density_texture = scene.voltextures[material.volume_density_texture];
            density *= evaluate_voltexture(volume_density_texture, pos);
        }
        tr *= 1.0f - max(0.0f, at(density, channel) / at(vd, channel));
    }
    return {tr, tr, tr};
}

float sample_distance(const yocto_scene& scene, const yocto_material& material,
    const vec3f& from, const vec3f& dir, int channel, rng_state& rng) {
    auto pos      = from;
    auto majorant = at(material.volume_density, channel);
    if (majorant == 0) return maxf;

    // delta tracking
    auto distance = 0.0f;
    while (true) {
        auto r = get_random_float(rng);
        if (r == 0) return maxf;
        auto step = -log(r) / majorant;
        if (is_material_volume_homogeneus(material)) return step;

        pos += dir * step;
        distance += step;
        auto density = material.volume_density;
        if (material.volume_density_texture >= 0) {
            auto& volume_density_texture = scene.voltextures[material.volume_density_texture];
            density *= evaluate_voltexture(volume_density_texture, pos);
        }
        if (at(density, channel) / majorant >= get_random_float(rng))
            return distance;

        // Escape from volume.
        if (pos.x > 1 || pos.y > 1 || pos.z > 1) return maxf;
        if (pos.x < -1 || pos.y < -1 || pos.z < -1) return maxf;
    }
}

float sample_distance(const yocto_scene& scene, const yocto_instance& instance,
    const bbox3f& bbox, const vec3f& from, const vec3f& dir, int channel,
    rng_state& rng) {
    auto& shape    = scene.shapes[instance.shape];
    auto& material = scene.materials[shape.material];
    if (material.volume_density == zero3f) return maxf;

    // Transform coordinates so that every position in the bounding box of the
    // instance is mapped to the cube [-1,1]^3 (the same space of volume texture
    // sampling).
    auto scale    = bbox.max - bbox.min;
    auto frame    = instance.frame;
    auto froml    = transform_point_inverse(frame, from) / scale;
    auto dirl     = transform_direction_inverse(frame, dir) / scale;
    auto ll       = length(dirl);
    auto distance = sample_distance(
        scene, material, froml, dirl / ll, channel, rng);
    return distance * ll;
}

vec3f sample_phase_function(float g, const vec2f& u) {
    auto cos_theta = 0.0f;
    if (abs(g) < 1e-3) {
        cos_theta = 1 - 2 * u.x;
    } else {
        float square = (1 - g * g) / (1 - g + 2 * g * u.x);
        cos_theta    = (1 + g * g - square * square) / (2 * g);
    }

    auto sin_theta = sqrt(max(0.0f, 1 - cos_theta * cos_theta));
    auto phi       = 2 * pif * u.y;
    return {sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
}

float evaluate_phase_function(float cos_theta, float g) {
    auto denom = 1 + g * g + 2 * g * cos_theta;
    return (1 - g * g) / (4 * pif * denom * sqrt(denom));
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SCENE UTILITIES
// -----------------------------------------------------------------------------
namespace ygl {

// Merge scene into one another
void merge_scene(yocto_scene& merge_scene, const yocto_scene& merge_from) {
    log_error("this is  broken since we did not fix references");
    auto merge = [](auto& v1, auto& v2) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    };
    auto merge_ = [](auto& v1, auto& v2) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    };
    merge_(merge_scene.cameras, merge_from.cameras);
    merge(merge_scene.textures, merge_from.textures);
    merge(merge_scene.voltextures, merge_from.voltextures);
    merge(merge_scene.materials, merge_from.materials);
    merge(merge_scene.shapes, merge_from.shapes);
    merge(merge_scene.surfaces, merge_from.surfaces);
    merge(merge_scene.instances, merge_from.instances);
    merge(merge_scene.environments, merge_from.environments);
    merge_(merge_scene.nodes, merge_from.nodes);
    merge_(merge_scene.animations, merge_from.animations);
}

void print_stats(const yocto_scene& scene) {
    // using long long instead of uint64_t to avoid printf macros
    auto num_cameras      = (long long)0;
    auto num_shapes       = (long long)0;
    auto num_surfaces     = (long long)0;
    auto num_instances    = (long long)0;
    auto num_materials    = (long long)0;
    auto num_textures     = (long long)0;
    auto num_voltextures  = (long long)0;
    auto num_environments = (long long)0;
    auto num_nodes        = (long long)0;
    auto num_animations   = (long long)0;

    auto elem_points    = (long long)0;
    auto elem_lines     = (long long)0;
    auto elem_triangles = (long long)0;
    auto elem_quads     = (long long)0;
    auto vert_pos       = (long long)0;
    auto vert_norm      = (long long)0;
    auto vert_texcoord  = (long long)0;
    auto vert_color     = (long long)0;
    auto vert_radius    = (long long)0;
    auto vert_tangsp    = (long long)0;

    auto elem_quads_pos      = (long long)0;
    auto elem_quads_norm     = (long long)0;
    auto elem_quads_texcoord = (long long)0;
    auto vert_quads_pos      = (long long)0;
    auto vert_quads_norm     = (long long)0;
    auto vert_quads_texcoord = (long long)0;

    auto texel_hdr = (long long)0;
    auto texel_ldr = (long long)0;
    auto voxel_hdr = (long long)0;

    auto memory_imgs    = (long long)0;
    auto memory_vols    = (long long)0;
    auto memory_elems   = (long long)0;
    auto memory_verts   = (long long)0;
    auto memory_fvelems = (long long)0;
    auto memory_fvverts = (long long)0;

    auto bbox = compute_scene_bounds(scene);

    num_cameras      = scene.cameras.size();
    num_shapes       = scene.shapes.size();
    num_surfaces     = scene.surfaces.size();
    num_materials    = scene.materials.size();
    num_textures     = scene.textures.size();
    num_voltextures  = scene.voltextures.size();
    num_environments = scene.environments.size();
    num_instances    = scene.instances.size();
    num_nodes        = scene.nodes.size();
    num_animations   = scene.animations.size();

    for (auto& shape : scene.shapes) {
        elem_points += shape.points.size();
        elem_lines += shape.lines.size();
        elem_triangles += shape.triangles.size();
        elem_quads += shape.quads.size();
        vert_pos += shape.positions.size();
        vert_norm += shape.normals.size();
        vert_texcoord += shape.texturecoords.size();
        vert_color += shape.colors.size();
        vert_radius += shape.radius.size();
        vert_tangsp += shape.tangentspaces.size();
    }

    memory_elems = elem_points * sizeof(int) + elem_lines * sizeof(vec2i) +
                   elem_triangles * sizeof(vec3i) + elem_quads * sizeof(vec4i);
    memory_verts = vert_pos * sizeof(vec3f) + vert_norm * sizeof(vec3f) +
                   vert_texcoord * sizeof(vec2f) + vert_color * sizeof(vec4f) +
                   vert_tangsp * sizeof(vec4f) + vert_radius * sizeof(float);

    for (auto& surface : scene.surfaces) {
        elem_quads_pos += surface.quads_positions.size();
        elem_quads_norm += surface.quads_normals.size();
        elem_quads_texcoord += surface.quads_texturecoords.size();
        vert_quads_pos += surface.positions.size();
        vert_quads_norm += surface.normals.size();
        vert_quads_texcoord += surface.texturecoords.size();
    }

    memory_fvelems = elem_quads_pos * sizeof(vec4i) +
                     elem_quads_norm * sizeof(vec4i) +
                     elem_quads_texcoord * sizeof(vec4i);
    memory_fvverts = vert_quads_pos * sizeof(vec3f) +
                     vert_quads_norm * sizeof(vec3f) +
                     vert_quads_texcoord * sizeof(vec2f);

    for (auto& texture : scene.textures) {
        texel_hdr += texture.hdr_image.size.x * texture.hdr_image.size.y;
        texel_ldr += texture.ldr_image.size.x * texture.ldr_image.size.y;
    }
    memory_imgs = texel_hdr * sizeof(vec4f) + texel_ldr * sizeof(vec4b);

    for (auto& voltexture : scene.voltextures) {
        voxel_hdr += voltexture.volume_data.size.x *
                     voltexture.volume_data.size.y *
                     voltexture.volume_data.size.z;
    }
    memory_vols = voxel_hdr * sizeof(float);

    printf("num_cameras: %lld\n", num_cameras);
    printf("num_shapes: %lld\n", num_shapes);
    printf("num_surface: %lld\n", num_surfaces);
    printf("num_instances: %lld\n", num_instances);
    printf("num_materials: %lld\n", num_materials);
    printf("num_textures: %lld\n", num_textures);
    printf("num_voltextures: %lld\n", num_voltextures);
    printf("num_environments: %lld\n", num_environments);
    printf("num_nodes: %lld\n", num_nodes);
    printf("num_animations: %lld\n", num_animations);

    printf("elem_points: %lld\n", elem_points);
    printf("elem_lines: %lld\n", elem_lines);
    printf("elem_triangles: %lld\n", elem_triangles);
    printf("elem_quads: %lld\n", elem_quads);
    printf("vert_pos: %lld\n", vert_pos);
    printf("vert_norm: %lld\n", vert_norm);
    printf("vert_texcoord: %lld\n", vert_texcoord);
    printf("vert_color: %lld\n", vert_color);
    printf("vert_radius: %lld\n", vert_radius);
    printf("vert_tangsp: %lld\n", vert_tangsp);

    printf("elem_points: %lld\n", elem_points);
    printf("elem_lines: %lld\n", elem_lines);
    printf("elem_triangles: %lld\n", elem_triangles);
    printf("elem_quads: %lld\n", elem_quads);
    printf("vert_pos: %lld\n", vert_pos);
    printf("vert_norm: %lld\n", vert_norm);
    printf("vert_texcoord: %lld\n", vert_texcoord);

    printf("texel_hdr: %lld\n", texel_hdr);
    printf("texel_ldr: %lld\n", texel_ldr);

    printf("memory_imgs: %lld\n", memory_imgs);
    printf("memory_vols: %lld\n", memory_vols);
    printf("memory_elems: %lld\n", memory_elems);
    printf("memory_verts: %lld\n", memory_verts);
    printf("memory_fvelems: %lld\n", memory_fvelems);
    printf("memory_fvverts: %lld\n", memory_fvverts);

    printf("bbox min: %g %g %g\n", bbox.min.x, bbox.min.y, bbox.min.z);
    printf("bbox max: %g %g %g\n", bbox.max.x, bbox.max.y, bbox.max.z);
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace ygl {

// Trace stats.
atomic<uint64_t> _trace_npaths{0};
atomic<uint64_t> _trace_nrays{0};

// Trace point
struct trace_point {
    int             instance_id  = -1;
    int             element_id   = -1;
    vec2f           element_uv   = zero2f;
    vec3f           position     = zero3f;
    vec3f           normal       = zero3f;
    vec2f           texturecoord = zero2f;
    vec3f           emission     = zero3f;
    microfacet_brdf brdf         = {};
    float           opacity      = 1;
};

// Make a trace point
trace_point make_trace_point(const yocto_scene& scene, int instance_id,
    int element_id, const vec2f& element_uv,
    const vec3f& shading_direction = zero3f) {
    auto& instance    = scene.instances[instance_id];
    auto  point       = trace_point();
    point.instance_id = instance_id;
    point.element_id  = element_id;
    point.element_uv  = element_uv;
    point.position    = evaluate_instance_position(
        scene, instance, element_id, element_uv);
    point.normal = evaluate_instance_shading_normal(
        scene, instance, element_id, element_uv, -shading_direction);
    point.texturecoord = evaluate_instance_texturecoord(
        scene, instance, element_id, element_uv);
    point.emission = evaluate_instance_emission(
        scene, instance, element_id, element_uv);
    point.brdf = evaluate_instance_brdf(scene, instance, element_id, element_uv);
    point.opacity = evaluate_instance_opacity(
        scene, instance, element_id, element_uv);
    return point;
}

// Intersects a ray and returns a point
trace_point trace_ray(const yocto_scene& scene, const bvh_scene& bvh,
    const vec3f& position, const vec3f& direction) {
    auto isec = intersect_scene(scene, bvh, make_ray(position, direction));
    _trace_nrays += 1;
    if (isec.instance_id >= 0) {
        return make_trace_point(scene, isec.instance_id, isec.element_id,
            isec.element_uv, direction);
    } else {
        auto point     = trace_point();
        point.emission = evaluate_environment_emission(scene, direction);
        return point;
    }
}

// Intersects a ray and returns a point accounting for opacity treated as
// coveregae
trace_point trace_ray_with_opacity(const yocto_scene& scene,
    const bvh_scene& bvh, const vec3f& position_, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    auto position = position_;
    for (auto b = 0; b < max_bounces; b++) {
        auto point = trace_ray(scene, bvh, position, direction);
        if (point.instance_id < 0) return point;
        if (point.opacity > 0.999f) return point;
        if (get_random_float(rng) < point.opacity) return point;
        position = point.position;
    }
    return {};
}

// Intersect a scene handling opacity.
scene_intersection intersect_scene_with_opacity(const yocto_scene& scene,
    const bvh_scene& bvh, const ray3f& ray_, rng_state& rng, int max_bounces) {
    auto ray = ray_;
    for (auto b = 0; b < max_bounces; b++) {
        _trace_nrays += 1;
        auto isec = intersect_scene(scene, bvh, ray);
        if (isec.instance_id < 0) return isec;
        auto& instance = scene.instances[isec.instance_id];
        auto& shape    = scene.shapes[instance.shape];
        auto& material = scene.materials[shape.material];
        auto  op       = evaluate_material_opacity(scene, material,
            evaluate_shape_texturecoord(shape, isec.element_id, isec.element_uv),
            evaluate_shape_color(shape, isec.element_id, isec.element_uv));
        if (op > 0.999f) return isec;
        if (get_random_float(rng) < op) return isec;
        ray = make_ray(evaluate_instance_position(
                           scene, instance, isec.element_id, isec.element_uv),
            ray.d);
    }
    return {};
}

// Sample camera
ray3f sample_camera_ray(const yocto_camera& camera, const vec2i& ij,
    const vec2i& image_size, rng_state& rng) {
    auto puv = get_random_vec2f(rng);  // force order of evaluation with
                                       // assignments
    auto luv = get_random_vec2f(rng);  // force order of evaluation with
                                       // assignments
    return evaluate_camera_ray(camera, ij, image_size, puv, luv);
}

// Check if we are near the mirror direction.
inline bool check_near_mirror(vec3f normal, vec3f outgoing, vec3f incoming) {
    return fabs(dot(incoming,
                    normalize(normal * 2.0f * dot(outgoing, normal) - outgoing)) -
                1) < 0.001f;
}

// Schlick approximation of the Fresnel term
vec3f fresnel_schlick(const vec3f& ks, const vec3f& h, const vec3f& incoming) {
    if (ks == zero3f) return zero3f;
    return ks + (vec3f{1, 1, 1} - ks) *
                    pow(clamp(1.0f - fabs(dot(h, incoming)), 0.0f, 1.0f), 5.0f);
}
vec3f fresnel_schlick(
    const vec3f& ks, const vec3f& h, const vec3f& incoming, float rs) {
    if (ks == zero3f) return zero3f;
    auto fks = fresnel_schlick(ks, fabs(dot(h, incoming)));
    return ks + (fks - ks) * (1 - sqrt(clamp(rs, 0.0f, 1.0f)));
}

// Evaluates the GGX distribution and geometric term
float evaluate_ggx_dist(float rs, const vec3f& normal, const vec3f& h) {
    auto di = (dot(normal, h) * dot(normal, h)) * (rs * rs - 1) + 1;
    return rs * rs / (pif * di * di);
}
float evaluate_ggx_sm(float rs, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
#if 0
    // evaluate G from Heitz
    auto lambda_o = (-1 + sqrt(1 + alpha2 * (1 - ndo * ndo) / (ndo * ndo))) / 2;
    auto lambda_i = (-1 + sqrt(1 + alpha2 * (1 - ndi * ndi) / (ndi * ndi))) / 2;
    auto g = 1 / (1 + lambda_o + lambda_i);
#else
    auto Go = (2 * fabs(dot(normal, outgoing))) /
              (fabs(dot(normal, outgoing)) +
                  sqrt(rs * rs + (1 - rs * rs) * dot(normal, outgoing) *
                                     dot(normal, outgoing)));
    auto Gi = (2 * fabs(dot(normal, incoming))) /
              (fabs(dot(normal, incoming)) +
                  sqrt(rs * rs + (1 - rs * rs) * dot(normal, incoming) *
                                     dot(normal, incoming)));
    return Go * Gi;
#endif
}

// Evaluates the BRDF scaled by the cosine of the incoming direction.
// - ggx from [Heitz 2014] and [Walter 2007] and [Lagarde 2014]
// "Understanding the Masking-Shadowing Function in Microfacet-Based
// BRDFs" http://jcgt.org/published/0003/02/03/
// - "Microfacet Models for Refraction through Rough Surfaces" EGSR 07
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
vec3f evaluate_smooth_brdf_cosine(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (is_brdf_delta(brdf)) return zero3f;
    auto brdf_cosine = zero3f;

    // diffuse
    if (brdf.diffuse != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto h = normalize(incoming + outgoing);
        auto F = fresnel_schlick(brdf.specular, h, outgoing);
        brdf_cosine += brdf.diffuse * (vec3f{1, 1, 1} - F) / pif;
    }

    // specular
    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto h = normalize(incoming + outgoing);
        auto F = fresnel_schlick(brdf.specular, h, outgoing);
        auto D = evaluate_ggx_dist(brdf.roughness, normal, h);
        auto G = evaluate_ggx_sm(brdf.roughness, normal, outgoing, incoming);
        brdf_cosine += F * D * G /
                       (4 * fabs(dot(normal, outgoing)) *
                           fabs(dot(normal, incoming)));
    }

    // transmission (thin sheet)
    if (brdf.transmission != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto ir = (dot(normal, outgoing) >= 0) ? reflect(-incoming, normal) :
                                                 reflect(-incoming, -normal);
        auto h = normalize(ir + outgoing);
        auto F = fresnel_schlick(brdf.specular, h, outgoing);
        auto D = evaluate_ggx_dist(brdf.roughness, normal, h);
        auto G = evaluate_ggx_sm(brdf.roughness, normal, outgoing, ir);
        brdf_cosine += brdf.transmission * (vec3f{1, 1, 1} - F) * D * G /
                       (4 * fabs(dot(normal, outgoing)) * fabs(dot(normal, ir)));
    }

    return brdf_cosine * abs(dot(normal, incoming));
}

// Evaluates the BRDF assuming that it is called only from the directions
// generated by sample_smooth_brdf_direction.
vec3f evaluate_delta_brdf_cosine(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (!is_brdf_delta(brdf)) return zero3f;
    auto microfacet_brdf = zero3f;

    // specular
    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto F = fresnel_schlick(brdf.specular, normal, outgoing);
        microfacet_brdf += F;
    }

    // transmission (thin sheet)
    if (brdf.transmission != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto F = fresnel_schlick(brdf.specular, normal, outgoing);
        microfacet_brdf += brdf.transmission * (1 - F);
    }

    return microfacet_brdf;
}

// Evalates the BRDF value. For delta BRDFs this assumes that it is called only
// from the directions generated by sample_delta_brdf_direction.
vec3f evaluate_brdf_cosine(const microfacet_brdf& brdf, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
    if (is_brdf_delta(brdf)) {
        return evaluate_delta_brdf_cosine(brdf, normal, outgoing, incoming);
    } else {
        return evaluate_smooth_brdf_cosine(brdf, normal, outgoing, incoming);
    }
}

// Picks a direction based on the BRDF
vec3f sample_smooth_brdf_direction(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
    if (is_brdf_delta(brdf)) return zero3f;
    auto F    = fresnel_schlick(brdf.specular, normal, outgoing);
    auto prob = vec3f{max(brdf.diffuse * (vec3f{1, 1, 1} - F)), max(F),
        max(brdf.transmission * (vec3f{1, 1, 1} - F))};
    if (prob == zero3f) return zero3f;
    prob /= prob.x + prob.y + prob.z;

    // sample according to diffuse
    if (brdf.diffuse != zero3f && rnl < prob.x) {
        auto rz = sqrtf(rn.y), rr = sqrtf(1 - rz * rz), rphi = 2 * pif * rn.x;
        auto il = vec3f{rr * cosf(rphi), rr * sinf(rphi), rz};
        auto fp = dot(normal, outgoing) >= 0 ? make_frame_fromz(zero3f, normal) :
                                               make_frame_fromz(zero3f, -normal);
        return transform_direction(fp, il);
    }
    // sample according to specular GGX
    else if (brdf.specular != zero3f && rnl < prob.x + prob.y) {
        auto hl = sample_ggx(brdf.roughness, rn);
        auto fp = dot(normal, outgoing) >= 0 ? make_frame_fromz(zero3f, normal) :
                                               make_frame_fromz(zero3f, -normal);
        auto h = transform_direction(fp, hl);
        return reflect(outgoing, h);
    }
    // transmission hack
    else if (brdf.transmission != zero3f && rnl < prob.x + prob.y + prob.z) {
        auto hl = sample_ggx(brdf.roughness, rn);
        auto fp = dot(normal, outgoing) >= 0 ? make_frame_fromz(zero3f, normal) :
                                               make_frame_fromz(zero3f, -normal);
        auto h  = transform_direction(fp, hl);
        auto ir = reflect(outgoing, h);
        return dot(normal, outgoing) >= 0 ? reflect(-ir, -normal) :
                                            reflect(-ir, normal);
    } else {
        return zero3f;
    }
}

// Picks a direction based on the BRDF
vec3f sample_smooth_brdf_direction(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, rng_state& rng) {
    auto rnl = get_random_float(rng);  // force order of evaluation with
                                       // assignments
    auto rni = get_random_vec2f(rng);  // force order of evaluation with
                                       // assignments
    return sample_smooth_brdf_direction(brdf, normal, outgoing, rnl, rni);
}

// Picks a direction based on the BRDF
vec3f sample_delta_brdf_direction(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
    if (!is_brdf_delta(brdf)) return zero3f;
    auto F    = fresnel_schlick(brdf.specular, normal, outgoing);
    auto prob = vec3f{0, max(F), max(brdf.transmission * (vec3f{1, 1, 1} - F))};
    if (prob == zero3f) return zero3f;
    prob /= prob.x + prob.y + prob.z;

    // sample according to specular mirror
    if (brdf.specular != zero3f && rnl < prob.x + prob.y) {
        return reflect(outgoing, dot(normal, outgoing) >= 0 ? normal : -normal);
    }
    // sample according to transmission
    else if (brdf.transmission != zero3f && !brdf.refract &&
             rnl < prob.x + prob.y + prob.z) {
        return -outgoing;
    }
    // sample according to transmission
    else if (brdf.transmission != zero3f && brdf.refract &&
             rnl < prob.x + prob.y + prob.z) {
        if (dot(normal, outgoing) >= 0) {
            return refract(outgoing, normal, 1 / specular_to_eta(brdf.specular));
        } else {
            return refract(outgoing, -normal, specular_to_eta(brdf.specular));
        }
    }
    // no sampling
    else {
        return zero3f;
    }
}

// Picks a direction based on the BRDF
vec3f sample_delta_brdf_direction(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, rng_state& rng) {
    auto rnl = get_random_float(rng);  // force order of evaluation with
                                       // assignments
    auto rni = get_random_vec2f(rng);  // force order of evaluation with
                                       // assignments
    return sample_delta_brdf_direction(brdf, normal, outgoing, rnl, rni);
}

// Picks a direction based on the BRDF
vec3f sample_brdf_direction(const microfacet_brdf& brdf, const vec3f& normal,
    const vec3f& outgoing, rng_state& rng) {
    if (is_brdf_delta(brdf)) {
        return sample_delta_brdf_direction(brdf, normal, outgoing, rng);
    } else {
        return sample_smooth_brdf_direction(brdf, normal, outgoing, rng);
    }
}

// Compute the weight for sampling the BRDF
float sample_smooth_brdf_direction_pdf(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (is_brdf_delta(brdf)) return 0;
    auto F    = fresnel_schlick(brdf.specular, normal, outgoing);
    auto prob = vec3f{max(brdf.diffuse * (vec3f{1, 1, 1} - F)), max(F),
        max(brdf.transmission * (vec3f{1, 1, 1} - F))};
    if (prob == zero3f) return 0;
    prob /= prob.x + prob.y + prob.z;

    auto pdf = 0.0f;

    if (brdf.diffuse != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        pdf += prob.x * fabs(dot(normal, incoming)) / pif;
    }
    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto h = normalize(incoming + outgoing);
        auto d = sample_ggx_pdf(brdf.roughness, fabs(dot(normal, h)));
        pdf += prob.y * d / (4 * fabs(dot(outgoing, h)));
    }
    if (brdf.transmission != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto ir = (dot(normal, outgoing) >= 0) ? reflect(-incoming, normal) :
                                                 reflect(-incoming, -normal);
        auto h = normalize(ir + outgoing);
        auto d = sample_ggx_pdf(brdf.roughness, fabs(dot(normal, h)));
        pdf += prob.z * d / (4 * fabs(dot(outgoing, h)));
    }

    return pdf;
}

// Compute the weight for sampling the BRDF
float sample_delta_brdf_direction_pdf(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (!is_brdf_delta(brdf)) return 0;
    auto F    = fresnel_schlick(brdf.specular, normal, outgoing);
    auto prob = vec3f{0, max(F), max(brdf.transmission * (vec3f{1, 1, 1} - F))};
    if (prob == zero3f) return 0;
    prob /= prob.x + prob.y + prob.z;

    auto pdf = 0.0f;

    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        return prob.y;
    }
    if (brdf.transmission != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        return prob.z;
    }

    return pdf;
}

// Compute the weight for BRDF smapling
float sample_brdf_direction_pdf(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (is_brdf_delta(brdf)) {
        return sample_delta_brdf_direction_pdf(brdf, normal, outgoing, incoming);
    } else {
        return sample_smooth_brdf_direction_pdf(brdf, normal, outgoing, incoming);
    }
}

// Picks a point on a light.
trace_point sample_instance_point(const yocto_scene& scene,
    const trace_lights& lights, int instance_id, float rel, const vec2f& ruv) {
    auto& instance = scene.instances[instance_id];
    auto  sample   = pair<int, vec2f>();
    if (instance.shape >= 0) {
        auto& shape        = scene.shapes[instance.shape];
        auto& elements_cdf = lights.shape_elements_cdf[instance.shape];
        sample = sample_shape_element(shape, elements_cdf, rel, ruv);
    } else if (instance.surface >= 0) {
        auto& surface      = scene.surfaces[instance.surface];
        auto& elements_cdf = lights.surface_elements_cdf[instance.shape];
        sample = sample_surface_element(surface, elements_cdf, rel, ruv);
    } else {
        log_error("empty instance");
    }
    return make_trace_point(scene, instance_id, sample.first, sample.second);
}

// Sample pdf for a light point.
float sample_instance_point_pdf(const yocto_scene& scene,
    const trace_lights& lights, int instance_id, int element_id,
    const vec2f& element_uv) {
    auto& instance = scene.instances[instance_id];
    if (!is_instance_emissive(scene, instance)) return 0;
    if (instance.shape >= 0) {
        auto& shape        = scene.shapes[instance.shape];
        auto& elements_cdf = lights.shape_elements_cdf[instance.shape];
        return sample_shape_element_pdf(
            shape, elements_cdf, element_id, element_uv);
    } else if (instance.surface >= 0) {
        auto& surface      = scene.shapes[instance.surface];
        auto& elements_cdf = lights.surface_elements_cdf[instance.surface];
        return sample_shape_element_pdf(
            surface, elements_cdf, element_id, element_uv);
    } else {
        return 0;
    }
}

// Sample a point from all shape lights.
trace_point sample_lights_point(const yocto_scene& scene,
    const trace_lights& lights, const vec3f& position, rng_state& rng) {
    if (lights.instances.empty()) return {};
    auto light_id = sample_uniform_index(
        lights.instances.size(), get_random_float(rng));
    auto instance_id = lights.instances[light_id];
    auto rel         = get_random_float(rng);  // force order of evaluation
    auto ruv         = get_random_vec2f(rng);  // force order of evaluation
    auto point = sample_instance_point(scene, lights, instance_id, rel, ruv);
    auto direction = normalize(position - point.position);
    point.normal   = evaluate_instance_shading_normal(scene,
        scene.instances[instance_id], point.element_id, point.element_uv,
        direction);
    return point;
}

// Sample pdf for a light point.
float sample_lights_point_pdf(const yocto_scene& scene,
    const trace_lights& lights, const vec3f& position,
    const trace_point& light_point) {
    auto& instance = scene.instances[light_point.instance_id];
    auto& shape    = scene.shapes[instance.shape];
    auto& material = scene.materials[shape.material];
    if (lights.instances.empty()) return 0;
    if (material.emission == zero3f) return 0;
    return sample_instance_point_pdf(scene, lights, light_point.instance_id,
               light_point.element_id, light_point.element_uv) *
           sample_uniform_index_pdf(lights.instances.size());
}

// Sample pdf for an environment.
float sample_environment_direction_pdf(const yocto_scene& scene,
    const trace_lights& lights, int environment_id, const vec3f& incoming) {
    auto& environment = scene.environments[environment_id];
    if (environment.emission_texture >= 0) {
        auto& elements_cdf = lights.environment_texture_cdf[environment.emission_texture];
        auto& emission_texture = scene.textures[environment.emission_texture];
        auto  size             = evaluate_texture_size(emission_texture);
        auto texcoord = evaluate_environment_texturecoord(environment, incoming);
        auto i        = (int)(texcoord.x * size.x);
        auto j        = (int)(texcoord.y * size.y);
        auto idx      = j * size.x + i;
        auto prob     = sample_discrete_distribution_pdf(elements_cdf, idx) /
                    elements_cdf.back();
        auto angle = (2 * pif / size.x) * (pif / size.y) *
                     sin(pif * (j + 0.5f) / size.y);
        return prob / angle;
    } else {
        return 1 / (4 * pif);
    }
}

// Picks a point on an environment.
vec3f sample_environment_direction(const yocto_scene& scene,
    const trace_lights& lights, int environment_id, float rel, const vec2f& ruv) {
    auto& environment = scene.environments[environment_id];
    if (environment.emission_texture >= 0) {
        auto& elements_cdf = lights.environment_texture_cdf[environment.emission_texture];
        auto& emission_texture = scene.textures[environment.emission_texture];
        auto  idx  = sample_discrete_distribution(elements_cdf, rel);
        auto  size = evaluate_texture_size(emission_texture);
        auto  u    = (idx % size.x + 0.5f) / size.x;
        auto  v    = (idx / size.x + 0.5f) / size.y;
        return evaluate_environment_direction(environment, {u, v});
    } else {
        return sample_sphere_direction(ruv);
    }
}

vec3f sample_environment_direction(const yocto_scene& scene,
    const trace_lights& lights, int environment_id, rng_state& rng) {
    auto rel = get_random_float(rng);  // force order of evaluation with
                                       // assignments
    auto ruv = get_random_vec2f(rng);  // force order of evaluation with
                                       // assignments
    return sample_environment_direction(scene, lights, environment_id, rel, ruv);
}

// Picks a point on a light.
vec3f sample_instance_direction(const yocto_scene& scene,
    const trace_lights& lights, int instance_id, const vec3f& p, float rel,
    const vec2f& ruv) {
    auto& instance = scene.instances[instance_id];
    auto  sample   = pair<int, vec2f>();
    if (instance.shape >= 0) {
        auto& shape        = scene.shapes[instance.shape];
        auto& elements_cdf = lights.shape_elements_cdf[instance.shape];
        sample = sample_shape_element(shape, elements_cdf, rel, ruv);
    } else if (instance.surface >= 0) {
        auto& surface      = scene.surfaces[instance.surface];
        auto& elements_cdf = lights.surface_elements_cdf[instance.surface];
        sample = sample_surface_element(surface, elements_cdf, rel, ruv);
    } else {
    }
    return normalize(evaluate_instance_position(
                         scene, instance, get<0>(sample), get<1>(sample)) -
                     p);
}

// Sample pdf for a light point.
float sample_instance_direction_pdf(const yocto_scene& scene,
    const trace_lights& lights, int instance_id, const bvh_scene& bvh,
    const vec3f& position, const vec3f& direction) {
    auto& instance = scene.instances[instance_id];
    if (!is_instance_emissive(scene, instance)) return 0;
    auto& elements_cdf = instance.shape >= 0 ?
                             lights.shape_elements_cdf[instance.shape] :
                             lights.surface_elements_cdf[instance.surface];
    // check all intersection
    auto pdf  = 0.0f;
    auto ray  = make_ray(position, direction);
    auto isec = intersect_scene(scene, instance_id, bvh, ray);
    while (isec.instance_id >= 0) {
        // accumulate pdf
        auto& instance       = scene.instances[isec.instance_id];
        auto  light_position = evaluate_instance_position(
            scene, instance, isec.element_id, isec.element_uv);
        auto light_normal = evaluate_instance_shading_normal(
            scene, instance, isec.element_id, isec.element_uv, direction);
        // prob triangle * area triangle = area triangle mesh
        auto area = elements_cdf.back();
        pdf += distance_squared(light_position, position) /
               (abs(dot(light_normal, direction)) * area);
        // continue
        ray  = make_ray(light_position + direction * 1e-3f, direction);
        isec = intersect_scene(scene, instance_id, bvh, ray);
    }
    return pdf;
}

// Sample lights wrt solid angle
vec3f sample_lights_direction(const yocto_scene& scene,
    const trace_lights& lights, const bvh_scene& bvh, const vec3f& position,
    rng_state& rng) {
    auto light_id = sample_uniform_index(
        lights.instances.size() + lights.environments.size(),
        get_random_float(rng));
    if (light_id < lights.instances.size()) {
        auto instance = lights.instances[light_id];
        auto rel      = get_random_float(rng);  // force order of evaluation
        auto ruv      = get_random_vec2f(rng);  // force order of evaluation
        return sample_instance_direction(
            scene, lights, instance, position, rel, ruv);
    } else {
        auto environment = lights.environments[light_id -
                                               (int)lights.instances.size()];
        auto rel         = get_random_float(rng);  // force order of evaluation
        auto ruv         = get_random_vec2f(rng);  // force order of evaluation
        return sample_environment_direction(scene, lights, environment, rel, ruv);
    }
}

// Sample lights pdf
float sample_lights_direction_pdf(const yocto_scene& scene,
    const trace_lights& lights, const bvh_scene& bvh, const vec3f& position,
    const vec3f& direction) {
    auto pdf = 0.0f;
    for (auto instance : lights.instances) {
        pdf += sample_instance_direction_pdf(
            scene, lights, instance, bvh, position, direction);
    }
    for (auto environment : lights.environments) {
        pdf += sample_environment_direction_pdf(
            scene, lights, environment, direction);
    }
    pdf *= sample_uniform_index_pdf(
        lights.instances.size() + lights.environments.size());
    return pdf;
}

// Sample a direction accoding to either ligts or brdf
vec3f sample_lights_or_brdf_direction(const yocto_scene& scene,
    const trace_lights& lights, const bvh_scene& bvh,
    const microfacet_brdf& brdf, const vec3f& position, const vec3f& normal,
    const vec3f& outgoing, rng_state& rng) {
    auto rmode = get_random_float(rng);
    if (rmode < 0.5f) {
        return sample_lights_direction(scene, lights, bvh, position, rng);
    } else {
        return sample_smooth_brdf_direction(brdf, normal, outgoing, rng);
    }
}

// Pdf for direction sampling
float sample_lights_or_brdf_direction_pdf(const yocto_scene& scene,
    const trace_lights& lights, const bvh_scene& bvh,
    const microfacet_brdf& brdf, const vec3f& position, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
    return 0.5f * sample_lights_direction_pdf(
                      scene, lights, bvh, position, incoming) +
           0.5f * sample_smooth_brdf_direction_pdf(
                      brdf, normal, outgoing, incoming);
}

// Russian roulette
bool sample_russian_roulette(
    const vec3f& weight, int bounce, rng_state& rng, int min_bounce = 2) {
    if (bounce <= min_bounce) return false;
    auto rrprob = 1.0f - min(max(weight), 0.95f);
    return get_random_float(rng) < rrprob;
}
float sample_russian_roulette_pdf(
    const vec3f& weight, int bounce, int min_bounce = 2) {
    if (bounce <= min_bounce) return 1;
    auto rrprob = 1.0f - min(max(weight), 0.95f);
    return 1 - rrprob;
}

#if 0
// Test occlusion.
vec3f evaluate_transmission(const yocto_scene& scene, const bvh_scene& bvh,
    const vec3f& from, const vec3f& to, int max_bounces) {
    auto weight = vec3f{1, 1, 1};
    auto p      = from;
    for (auto bounce = 0; bounce < max_bounces; bounce++) {
        auto ray  = make_segment(p, to);
        auto isec = intersect_scene(scene, bvh, ray);
        if (!isec.instance) break;
        auto f = evaluate_brdf(isec.instance.shape, isec.element_id, isec.element_uv);
        auto op = evaluate_opacity(
            isec.instance, isec.element_id, isec.element_uv);
        weight *= brdf.transmission + vec3f{1 - op, 1 - op, 1 - op};
        if (weight == zero3f) break;
        p = evaluate_position(isec.instance, isec.element_id, isec.element_uv);
    }
    return weight;
}

#endif

// Probability of computing direct illumination.
float prob_direct(const microfacet_brdf& brdf) {
    // This is just heuristic. Any other choice is equally correct.
    if (brdf.diffuse + brdf.specular == zero3f) return 0;
    auto kd = max(brdf.diffuse);
    auto ks = max(brdf.specular);
    return (kd + brdf.roughness * ks) / (kd + ks);
}

// Sample a direction of direct illumination from the point p, which is inside
// mediums.back(). pdf and incoming radiance le are returned in reference. It
// works for both surface rendering and volume rendering.
vec3f direct_illumination(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& p, int channel,
    const vector<int>& mediums_, rng_state& rng, float& pdf, vec3f& le) {
    auto  incoming = zero3f;
    vec3f weight   = vec3f{1, 1, 1};
    auto  mediums  = mediums_;

    auto idx = sample_uniform_index(
        lights.instances.size() + lights.environments.size(),
        get_random_float(rng));
    pdf = 1.0f / (lights.instances.size() + lights.environments.size());
    if (idx < lights.instances.size()) {
        auto instance_id = lights.instances[idx];
        incoming = sample_instance_direction(scene, lights, instance_id, p,
            get_random_float(rng), get_random_vec2f(rng));
        auto& instance = scene.instances[instance_id];
        pdf *= 1.0 / lights.shape_elements_cdf[instance.shape].back();
    } else {
        auto environment_id = lights.environments[idx - lights.instances.size()];
        incoming = sample_environment_direction(scene, lights, environment_id,
            get_random_float(rng), get_random_vec2f(rng));
        pdf *= sample_environment_direction_pdf(
            scene, lights, environment_id, incoming);
        auto isec = intersect_scene_with_opacity(
            scene, bvh, make_ray(p, incoming), rng, 10);
        if (isec.instance_id < 0) {
            auto& environment = scene.environments[environment_id];
            le = evaluate_environment_emission(scene, environment, incoming);
            return incoming;
        }
    }

    auto isec = intersect_scene(scene, bvh, make_ray(p, incoming));

    while (isec.instance_id >= 0) {
        auto& isec_instance = scene.instances[isec.instance_id];
        auto& isec_shape    = scene.shapes[isec_instance.shape];
        auto  lp            = evaluate_instance_position(
            scene, isec_instance, isec.element_id, isec.element_uv);
        auto ln = evaluate_instance_shading_normal(
            scene, isec_instance, isec.element_id, isec.element_uv, -incoming);
        auto& isec_material = scene.materials[isec_shape.material];
        auto  emission      = evaluate_material_emission(scene, isec_material,
            evaluate_shape_texturecoord(
                isec_shape, isec.element_id, isec.element_uv),
            evaluate_shape_color(isec_shape, isec.element_id, isec.element_uv));

        auto& medium_instance = scene.instances[mediums.back()];
        auto& medium_shape    = scene.shapes[medium_instance.shape];
        auto& medium_material = scene.materials[medium_shape.material];
        if (medium_material.volume_density != zero3f)
            weight *= evaluate_transmission(scene, medium_material, lp,
                incoming, isec.distance, channel, rng);

        // Hack: Uncomment this or the result will be biased
        // If mediums refracts, the transmission ray won't reach the sampled
        // light point if(isec.instance.mat->refract) break;

        if (emission != zero3f) {
            // Geometric term.
            weight *= fabs(dot(ln, incoming)) / dot(lp - p, lp - p);
            le += weight * emission;
            break;
        }

        auto brdf = evaluate_material_brdf(scene, isec_material,
            evaluate_shape_texturecoord(
                isec_shape, isec.element_id, isec.element_uv),
            evaluate_shape_color(isec_shape, isec.element_id, isec.element_uv));
        if (brdf.transmission == zero3f) {
            le = zero3f;
            break;
        }

        auto  ndi       = dot(incoming, ln);
        float threshold = 0.05;

        if (ndi > threshold) {
            // Exiting from medium.
            if (isec.instance_id != mediums.back()) {  // exiting a different
                                                       // medium??
                pdf = 0;
                return zero3f;
            }
            if (mediums.size() <= 1) {
                pdf = 0;
                return zero3f;
            }
            mediums.pop_back();
        } else if (ndi < -threshold) {
            // Entering new medium.
            if (isec.instance_id == mediums.back()) {  // entering the same
                                                       // medium??
                pdf = 0;
                return zero3f;
            }
            mediums.push_back(isec.instance_id);
        } else {
            pdf = 0;
            return zero3f;
        }

        isec = intersect_scene(scene, bvh, make_ray(lp, incoming));  //@Hack:
                                                                     // 10? Don't
                                                                     // know...
    }

    return incoming;
}

// Recursive path tracing.
vec4f trace_path(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto weight   = vec3f{1, 1, 1};
    auto outgoing = -direction;

    // trace  path
    for (auto bounce = 0; bounce < max_bounces; bounce++) {
        // direct
        if (!is_brdf_delta(point.brdf) && !empty(lights)) {
            auto light_direction = sample_lights_or_brdf_direction(scene, lights,
                bvh, point.brdf, point.position, point.normal, outgoing, rng);
            auto light_pdf = sample_lights_or_brdf_direction_pdf(scene, lights,
                bvh, point.brdf, point.position, point.normal, outgoing,
                light_direction);
            auto light_point = trace_ray_with_opacity(
                scene, bvh, point.position, light_direction, rng, max_bounces);
            auto brdf_cosine = evaluate_smooth_brdf_cosine(
                point.brdf, point.normal, outgoing, light_direction);
            if (light_pdf)
                radiance += weight * light_point.emission * brdf_cosine /
                            light_pdf;
        }

        // continue path
        auto next_direction = sample_brdf_direction(
            point.brdf, point.normal, outgoing, rng);
        auto brdf_cosine = evaluate_brdf_cosine(
            point.brdf, point.normal, outgoing, next_direction);
        auto next_pdf = sample_brdf_direction_pdf(
            point.brdf, point.normal, outgoing, next_direction);

        // accumulate weight
        if (next_pdf == 0) break;
        weight *= brdf_cosine / next_pdf;
        if (weight == zero3f) break;

        // russian roulette
        if (sample_russian_roulette(weight, bounce, rng)) break;
        weight /= sample_russian_roulette_pdf(weight, bounce);

        // intersect next point
        auto next_point = trace_ray_with_opacity(
            scene, bvh, point.position, next_direction, rng, max_bounces);
        if (is_brdf_delta(point.brdf)) radiance += weight * next_point.emission;
        if (next_point.instance_id < 0 || is_brdf_zero(next_point.brdf)) break;

        // setup next iteration
        point    = next_point;
        outgoing = -next_direction;
    }

    return {radiance.x, radiance.y, radiance.z, 1};
}

// Evaluates the weight after sampling distance in a medium.
vec3f evaluate_transmission_div_pdf(const vec3f& vd, float distance, int ch) {
    auto weight = zero3f;

    // For the sampled channel, transmission / pdf == 1.0
    at(weight, ch) = 1.0;

    // Compute weight for the remaining channels i.
    // In order to avoid numerical nasties (NaNs) transmission / pdf is
    // evaluated. transmission[i] = exp(-distance * vd[i]) pdf             =
    // exp(-distance * vd[channel])
    int i = (ch + 1) % 3, j = (ch + 2) % 3;
    at(weight, i) = exp(-distance * (at(vd, i) - at(vd, ch)));
    at(weight, j) = exp(-distance * (at(vd, j) - at(vd, ch)));
    return weight;
}

// Iterative volume path tracing.
vec4f trace_volpath(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    if (empty(lights)) return zero4f;

    // initialize
    auto radiance = zero3f;
    auto weight   = vec3f{1, 1, 1};
    auto emission = true;
    auto ray      = make_ray(position, direction);

#if 0
    // @Hack: air volume properties should be set in the scene struct.
    if (air == nullptr) {
        air = new instance();
        air->name = "air";
        air->mat = new material();
        air->mat->vd = vec3f{0.0, 0.0, 0.0};
        air->mat->va = vec3f{0.0, 0.0, 0.0};
        air->mat->vg = 0.0;
    }
#endif

    // List of mediums that contains the path. The path starts in air.
    auto mediums = vector<int>{-1};

    // Sample color channel. This won't matter if there are no heterogeneus
    // materials.
    auto ch             = sample_uniform_index(3, get_random_float(rng));
    auto single_channel = false;

    int bounce = 0;
    while (bounce < max_bounces) {
        auto  medium   = mediums.back();
        auto& instance = scene.instances[medium];
        auto& shape    = scene.shapes[instance.shape];
        auto& material = scene.materials[shape.material];
        auto  ve       = material.volume_emission;
        auto  va       = material.volume_albedo;
        auto  vd       = material.volume_density;
        auto  vg       = material.volume_phaseg;

        // If medium has color but must use delta tracking, integrate only the
        // sampled spectrum.
        if (!single_channel && is_material_volume_colored(material) &&
            !is_material_volume_homogeneus(material)) {
            at(weight, ch) *= 3;
            at(weight, (ch + 1) % 3) = 0;
            at(weight, (ch + 2) % 3) = 0;
            single_channel           = true;
        }

        // TODO: FIXME REMOVING BBOX
        // Sample distance of next absorption/scattering event in the medium.
        // dist_pdf is unknown due to delta tracking.
        auto bbox = transform_bbox(
            instance.frame, bbox3f{{-1, -1, -1}, {1, 1, 1}});
        auto distance = sample_distance(
            scene, instance, bbox, ray.o, ray.d, ch, rng);

        // Create ray and clamp it to make the intersection faster.
        ray       = make_ray(ray.o, ray.d);
        ray.tmax  = distance;
        auto isec = intersect_scene_with_opacity(
            scene, bvh, ray, rng, max_bounces);

        // @Hack: When isec.instance == nullptr, we must discern if the ray hit
        // nothing (the environment)
        //        or a medium interaction was sampled. Doing isec.distance ==
        //        maxf doesn't work, why??
        auto scene_size = max(bvh.nodes[0].bbox.max - bvh.nodes[0].bbox.min);

        // environment
        if (isec.instance_id < 0 && distance > scene_size) {
            if (emission) {
                for (auto& environment : scene.environments)
                    radiance += weight * evaluate_environment_emission(
                                             scene, environment, ray.d);
            }
            return {radiance.x, radiance.y, radiance.z, 0};
        }

        // surface intersection
        if (isec.instance_id >= 0) {
            auto& isec_instance = scene.instances[isec.instance_id];
            auto& isec_shape    = scene.shapes[isec_instance.shape];
            auto  outgoing      = -ray.d;
            auto& isec_material = scene.materials[isec_shape.material];
            auto  p             = evaluate_instance_position(
                scene, isec_instance, isec.element_id, isec.element_uv);
            auto normal = evaluate_instance_shading_normal(scene, isec_instance,
                isec.element_id, isec.element_uv, outgoing);
            auto brdf   = evaluate_material_brdf(scene, isec_material,
                evaluate_shape_texturecoord(
                    isec_shape, isec.element_id, isec.element_uv),
                evaluate_shape_color(
                    isec_shape, isec.element_id, isec.element_uv));

            // distance sampling pdf is unknown due to delta tracking, but we do
            // know the value of transmission / pdf_dist.
            weight *= evaluate_transmission_div_pdf(vd, isec.distance, ch);

            // emission
            if (emission)
                radiance += weight * evaluate_material_emission(scene,
                                         isec_material,
                                         evaluate_shape_texturecoord(isec_shape,
                                             isec.element_id, isec.element_uv),
                                         evaluate_shape_color(isec_shape,
                                             isec.element_id, isec.element_uv));

            // early exit
            if (brdf.diffuse + brdf.specular + brdf.transmission == zero3f ||
                bounce >= max_bounces - 1)
                break;

            // direct lighting
            if (get_random_float(rng) < prob_direct(brdf)) {
                // With some probabilty, this is a naive path tracer (works
                // great with delta-like brdfs)
                vec3f direct;
                float pdf;
                vec3f incoming = direct_illumination(
                    scene, bvh, lights, p, ch, mediums, rng, pdf, direct);
                if (pdf != 0) {
                    auto brdf_cosine = evaluate_smooth_brdf_cosine(
                        brdf, normal, outgoing, incoming);
                    radiance += weight * direct * brdf_cosine / pdf;
                    emission = false;
                }
            } else
                emission = true;

            // continue path
            vec3f incoming, brdf_cosine;
            float pdf = 0;
            if (!is_brdf_delta(brdf)) {
                incoming = sample_smooth_brdf_direction(
                    brdf, normal, outgoing, rng);
                brdf_cosine = evaluate_smooth_brdf_cosine(
                    brdf, normal, outgoing, incoming);
                pdf = sample_smooth_brdf_direction_pdf(
                    brdf, normal, outgoing, incoming);
            } else {
                incoming = sample_delta_brdf_direction(
                    brdf, normal, outgoing, rng);
                brdf_cosine = evaluate_delta_brdf_cosine(
                    brdf, normal, outgoing, incoming);
                pdf = sample_delta_brdf_direction_pdf(
                    brdf, normal, outgoing, incoming);
            }
            auto ndi = dot(normal, incoming);
            auto ndo = dot(normal, outgoing);

            // accumulate weight
            if (pdf == 0) break;
            weight *= brdf_cosine / pdf;
            if (weight == zero3f) break;
            ray.o            = p;
            ray.d            = incoming;
            bool transmitted = (ndi > 0) != (ndo > 0);

            // transmission in medium
            if (transmitted) {
                float tr = 0.05;  // avoid numerical errors
                if (ndo < -tr) {
                    // Exiting from medium.
                    if (isec.instance_id != medium) break;
                    if (mediums.size() <= 1) break;
                    mediums.pop_back();
                } else if (ndo > tr) {
                    // Entering new medium.
                    if (isec.instance_id == medium) break;
                    mediums.push_back(isec.instance_id);
                } else
                    break;
            }
            bounce += 1;
        }
        // medium interaction
        else {
            ray.o += ray.d * distance;
            float scattering_prob = at(va, ch);

            // absorption and emission
            if (get_random_float(rng) >= scattering_prob) {
                weight /= 1 - scattering_prob;
                radiance += weight * ve;
                break;
            }

            // scattering event
            weight /= scattering_prob;
            weight *= evaluate_transmission_div_pdf(vd, distance, ch);

            // direct lighting
            vec3f direct;
            float pdf_direct;
            vec3f l = direct_illumination(scene, bvh, lights, ray.o, ch,
                mediums, rng, pdf_direct, direct);
            if (pdf_direct != 0) {
                auto f = va * evaluate_phase_function(dot(l, -ray.d), vg);
                radiance += weight * direct * f / pdf_direct;
                emission = false;
            }

            // indirect
            vec3f incoming = sample_phase_function(vg, get_random_vec2f(rng));
            weight *= va;
            ray.d = transform_direction(
                make_frame_fromz(zero3f, ray.d), incoming);
        }

        // russian roulette
        if (bounce > 2) {
            auto rrprob = 1.0f - min(max(weight), 0.95f);
            if (get_random_float(rng) < rrprob) break;
            weight *= 1 / (1 - rrprob);
        }
    }

    return {radiance.x, radiance.y, radiance.z, 1};
}

// Recursive path tracing.
vec4f trace_path_naive(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto weight   = vec3f{1, 1, 1};
    auto outgoing = -direction;

    // trace  path
    for (auto bounce = 0; bounce < max_bounces; bounce++) {
        // continue path
        auto next_direction = sample_brdf_direction(
            point.brdf, point.normal, outgoing, rng);
        auto brdf_cosine = evaluate_brdf_cosine(
            point.brdf, point.normal, outgoing, next_direction);
        auto next_pdf = sample_brdf_direction_pdf(
            point.brdf, point.normal, outgoing, next_direction);

        // accumulate weight
        if (next_pdf == 0) break;
        weight *= brdf_cosine / next_pdf;
        if (weight == zero3f) break;

        // russian roulette
        if (sample_russian_roulette(weight, bounce, rng)) break;
        weight /= sample_russian_roulette_pdf(weight, bounce);

        // intersect next point
        auto next_point = trace_ray_with_opacity(
            scene, bvh, point.position, next_direction, rng, max_bounces);
        radiance += weight * next_point.emission;
        if (next_point.instance_id < 0 || is_brdf_zero(next_point.brdf)) break;

        // setup next iteration
        point    = next_point;
        outgoing = -next_direction;
    }

    return {radiance.x, radiance.y, radiance.z, 1};
}

// Recursive path tracing.
vec4f trace_path_nomis(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto weight   = vec3f{1, 1, 1};
    auto outgoing = -direction;

    // trace  path
    for (auto bounce = 0; bounce < max_bounces; bounce++) {
        // direct
        if (!is_brdf_delta(point.brdf) && !empty(lights)) {
            auto light_point = sample_lights_point(
                scene, lights, point.position, rng);
            auto light_pdf = sample_lights_point_pdf(
                scene, lights, point.position, light_point);
            auto light_direction = normalize(
                light_point.position - point.position);
            auto intersection_point = trace_ray_with_opacity(
                scene, bvh, point.position, light_direction, rng, max_bounces);
            if (light_pdf &&
                light_point.instance_id == intersection_point.instance_id) {
                auto brdf_cosine = evaluate_smooth_brdf_cosine(
                    point.brdf, point.normal, outgoing, light_direction);
                auto geometric_term = abs(dot(light_point.normal,
                                          light_direction)) /
                                      distance_squared(
                                          light_point.position, point.position);
                radiance += weight * light_point.emission * brdf_cosine *
                            geometric_term / light_pdf;
            }
        }

        // continue path
        auto next_direction = sample_brdf_direction(
            point.brdf, point.normal, outgoing, rng);
        auto brdf_cosine = evaluate_brdf_cosine(
            point.brdf, point.normal, outgoing, next_direction);
        auto next_pdf = sample_brdf_direction_pdf(
            point.brdf, point.normal, outgoing, next_direction);

        // accumulate weight
        if (next_pdf == 0) break;
        weight *= brdf_cosine / next_pdf;
        if (weight == zero3f) break;

        // russian roulette
        if (sample_russian_roulette(weight, bounce, rng)) break;
        weight /= sample_russian_roulette_pdf(weight, bounce);

        // intersect next point
        auto next_point = trace_ray_with_opacity(
            scene, bvh, point.position, next_direction, rng, max_bounces);
        if (next_point.instance_id < 0 || is_brdf_delta(point.brdf))
            radiance += weight * next_point.emission;
        if (next_point.instance_id < 0 || is_brdf_zero(next_point.brdf)) break;

        // setup next iteration
        point    = next_point;
        outgoing = -next_direction;
    }

    return {radiance.x, radiance.y, radiance.z, 1};
}

// Direct illumination.
vec4f trace_direct(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto outgoing = -direction;

    // direct
    if (!is_brdf_delta(point.brdf) && !empty(lights)) {
        auto light_direction = sample_lights_or_brdf_direction(scene, lights,
            bvh, point.brdf, point.position, point.normal, outgoing, rng);
        auto light_pdf = sample_lights_or_brdf_direction_pdf(scene, lights, bvh,
            point.brdf, point.position, point.normal, outgoing, light_direction);
        auto light_point = trace_ray_with_opacity(
            scene, bvh, point.position, light_direction, rng, max_bounces);
        auto brdf_cosine = evaluate_smooth_brdf_cosine(
            point.brdf, point.normal, outgoing, light_direction);
        if (light_pdf)
            radiance += light_point.emission * brdf_cosine / light_pdf;
    }

    // deltas
    if (is_brdf_delta(point.brdf) && max_bounces) {
        auto next_direction = sample_delta_brdf_direction(
            point.brdf, point.normal, outgoing, rng);
        auto brdf_cosine = evaluate_delta_brdf_cosine(
            point.brdf, point.normal, outgoing, next_direction);
        auto next_pdf = sample_delta_brdf_direction_pdf(
            point.brdf, point.normal, outgoing, next_direction);
        auto incoming_radiance = xyz(trace_direct(scene, bvh, lights,
            point.position, next_direction, rng, max_bounces - 1));
        radiance += brdf_cosine * incoming_radiance / next_pdf;
    }

    // done
    return {radiance.x, radiance.y, radiance.z, 1};
}

// Direct illumination.
vec4f trace_direct_nomis(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto outgoing = -direction;

    // direct
    if (!is_brdf_delta(point.brdf) && !lights.instances.empty()) {
        auto light_point = sample_lights_point(
            scene, lights, point.position, rng);
        auto light_pdf = sample_lights_point_pdf(
            scene, lights, point.position, light_point);
        auto light_direction = normalize(light_point.position - point.position);
        auto intersection_point = trace_ray_with_opacity(
            scene, bvh, point.position, light_direction, rng, max_bounces);
        if (light_pdf &&
            light_point.instance_id == intersection_point.instance_id) {
            auto brdf_cosine = evaluate_smooth_brdf_cosine(
                point.brdf, point.normal, outgoing, light_direction);
            auto geometric_term = abs(dot(light_point.normal, light_direction)) /
                                  distance_squared(
                                      light_point.position, point.position);
            radiance += light_point.emission * brdf_cosine * geometric_term /
                        light_pdf;
        }
    }

    // environments
    if (!is_brdf_delta(point.brdf) && !lights.environments.empty()) {
        auto next_direction = sample_brdf_direction(
            point.brdf, point.normal, outgoing, rng);
        auto brdf_cosine = evaluate_brdf_cosine(
            point.brdf, point.normal, outgoing, next_direction);
        auto next_pdf = sample_brdf_direction_pdf(
            point.brdf, point.normal, outgoing, next_direction);
        auto emission = evaluate_environment_emission(scene, next_direction);
        if (next_pdf &&
            intersect_scene_with_opacity(scene, bvh,
                make_ray(point.position, next_direction), rng, max_bounces)
                    .instance_id < 0)
            radiance += emission * brdf_cosine / next_pdf;
    }

    // deltas
    if (is_brdf_delta(point.brdf)) {
        auto next_direction = sample_delta_brdf_direction(
            point.brdf, point.normal, outgoing, rng);
        auto brdf_cosine = evaluate_delta_brdf_cosine(
            point.brdf, point.normal, outgoing, next_direction);
        auto next_pdf = sample_delta_brdf_direction_pdf(
            point.brdf, point.normal, outgoing, next_direction);
        auto incoming_radiance = xyz(trace_direct_nomis(scene, bvh, lights,
            point.position, next_direction, rng, max_bounces - 1));
        radiance += brdf_cosine * incoming_radiance * next_pdf;
    }

    // done
    return {radiance.x, radiance.y, radiance.z, 1};
}

// Environment illumination only with no shadows.
vec4f trace_environment(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto outgoing = -direction;

    // continue path
    auto next_direction = sample_brdf_direction(
        point.brdf, point.normal, outgoing, rng);
    auto brdf_cosine = evaluate_brdf_cosine(
        point.brdf, point.normal, outgoing, next_direction);
    auto next_pdf = sample_brdf_direction_pdf(
        point.brdf, point.normal, outgoing, next_direction);

    // accumulate environment illumination
    if (next_pdf)
        radiance += brdf_cosine *
                    evaluate_environment_emission(scene, next_direction) /
                    next_pdf;

    // done
    return {radiance.x, radiance.y, radiance.z, 1};
}

// Eyelight for quick previewing.
vec4f trace_eyelight(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0)
        return {point.emission.x, point.emission.y, point.emission.z, 0};

    // initialize
    auto radiance = point.emission;
    auto outgoing = -direction;

    // microfacet_brdf * light
    radiance += evaluate_smooth_brdf_cosine(
                    point.brdf, point.normal, outgoing, outgoing) *
                pif;

    // done
    return {radiance.x, radiance.y, radiance.z, 1};
}

// Debug previewing.
vec4f trace_debug_normal(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    auto normal = point.normal * 0.5f + 0.5f;
    return {normal.x, normal.y, normal.z, 1};
}

// Debug frontfacing.
vec4f trace_debug_frontfacing(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    auto outgoing    = -direction;
    auto frontfacing = dot(point.normal, outgoing) > 0 ? vec3f{0, 1, 0} :
                                                         vec3f{1, 0, 0};
    return {frontfacing.x, frontfacing.y, frontfacing.z, 1};
}

// Debug previewing.
vec4f trace_debug_albedo(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    auto albedo = point.brdf.diffuse + point.brdf.specular +
                  point.brdf.transmission;
    return {albedo.x, albedo.y, albedo.z, 1};
}

// Debug previewing.
vec4f trace_debug_diffuse(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    return {point.brdf.diffuse.x, point.brdf.diffuse.y, point.brdf.diffuse.z, 1};
}

// Debug previewing.
vec4f trace_debug_specular(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    return {
        point.brdf.specular.x, point.brdf.specular.y, point.brdf.specular.z, 1};
}

// Debug previewing.
vec4f trace_debug_roughness(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    return {point.brdf.roughness, point.brdf.roughness, point.brdf.roughness, 1};
}

// Debug previewing.
vec4f trace_debug_texcoord(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, int max_bounces) {
    // intersect ray
    auto point = trace_ray_with_opacity(
        scene, bvh, position, direction, rng, max_bounces);
    if (point.instance_id < 0) return zero4f;

    // shade
    return {point.texturecoord.x, point.texturecoord.y, 0, 1};
}

// Trace a single ray from the camera using the given algorithm.
trace_sampler_func get_trace_sampler_func(trace_sampler_type type) {
    switch (type) {
        case trace_sampler_type::path: return trace_path;
        // case trace_type::volpath:
        //     return trace_volpath(
        //         scene, bvh, lights, position, direction, rng, max_bounces);
        case trace_sampler_type::direct: return trace_direct;
        case trace_sampler_type::environment: return trace_environment;
        case trace_sampler_type::eyelight: return trace_eyelight;
        case trace_sampler_type::path_nomis: return trace_path_nomis;
        case trace_sampler_type::path_naive: return trace_path_naive;
        case trace_sampler_type::direct_nomis: return trace_direct_nomis;
        case trace_sampler_type::debug_normal: return trace_debug_normal;
        case trace_sampler_type::debug_albedo: return trace_debug_albedo;
        case trace_sampler_type::debug_texcoord: return trace_debug_texcoord;
        case trace_sampler_type::debug_frontfacing:
            return trace_debug_frontfacing;
        case trace_sampler_type::debug_diffuse: return trace_debug_diffuse;
        case trace_sampler_type::debug_specular: return trace_debug_specular;
        case trace_sampler_type::debug_roughness: return trace_debug_roughness;
    }
    return {};
}

// Trace a block of samples
void trace_image_region(image<vec4f>& rendered_image, const yocto_scene& scene,
    const yocto_camera& camera, const bvh_scene& bvh,
    const trace_lights& lights, const trace_sampler_func& sampler,
    const image_region& region, int current_sample, int num_samples,
    int max_bounces, image<rng_state>& rngs, float pixel_clamp) {
    for (auto j = region.offset.y; j < region.offset.y + region.size.y; j++) {
        for (auto i = region.offset.x; i < region.offset.x + region.size.x; i++) {
            at(rendered_image, {i, j}) *= current_sample;
            for (auto s = 0; s < num_samples; s++) {
                _trace_npaths += 1;
                auto& rng = at(rngs, {i, j});
                auto  ray = sample_camera_ray(
                    camera, {i, j}, rendered_image.size, rng);
                auto radiance_hit = sampler(
                    scene, bvh, lights, ray.o, ray.d, rng, max_bounces);
                if (!isfinite(radiance_hit.x) || !isfinite(radiance_hit.y) ||
                    !isfinite(radiance_hit.z)) {
                    log_error("NaN detected");
                    radiance_hit = zero4f;
                }
                if (max(xyz(radiance_hit)) > pixel_clamp)
                    xyz(radiance_hit) = xyz(radiance_hit) *
                                        (pixel_clamp / max(xyz(radiance_hit)));
                at(rendered_image, {i, j}) += {radiance_hit.x, radiance_hit.y,
                    radiance_hit.z,
                    (radiance_hit.w > 0 || !scene.environments.empty()) ? 1.0f :
                                                                          0.0f};
            }
            at(rendered_image, {i, j}) /= current_sample + num_samples;
        }
    }
}

// Init a sequence of random number generators.
void make_trace_rngs(image<rng_state>& rngs, const vec2i& image_size, uint64_t seed) {
    init_image(rngs, image_size);
    auto rng  = make_rng(1301081);
    for (auto j = 0; j < rngs.size.y; j++) {
        for (auto i = 0; i < rngs.size.x; i++) {
            at(rngs, {i, j}) = make_rng(
                seed, get_random_int(rng, 1 << 31) / 2 + 1);
        }
    }
}

// Init trace lights
void make_trace_lights(trace_lights& lights, const yocto_scene& scene) {
    auto scope  = log_trace_scoped("making trace lights");
    lights = trace_lights{};

    lights.shape_elements_cdf.resize(scene.shapes.size());
    lights.surface_elements_cdf.resize(scene.surfaces.size());
    lights.environment_texture_cdf.resize(scene.textures.size());

    for (auto instance_id = 0; instance_id < scene.instances.size();
         instance_id++) {
        auto& instance = scene.instances[instance_id];
        if (!is_instance_emissive(scene, instance)) continue;
        if (instance.shape >= 0) {
            auto& shape = scene.shapes[instance.shape];
            if (shape.triangles.empty() && shape.quads.empty()) continue;
            lights.instances.push_back(instance_id);
            compute_shape_elements_cdf(
                shape, lights.shape_elements_cdf[instance.shape]);
        } else if (instance.surface >= 0) {
            auto& surface = scene.surfaces[instance.surface];
            if (surface.quads_positions.empty()) continue;
            lights.instances.push_back(instance_id);
            compute_surface_elements_cdf(
                surface, lights.surface_elements_cdf[instance.surface]);
        } else {
            continue;
        }
    }

    for (auto environment_id = 0; environment_id < scene.environments.size();
         environment_id++) {
        auto& environment = scene.environments[environment_id];
        if (environment.emission == zero3f) continue;
        lights.environments.push_back(environment_id);
        if (environment.emission_texture >= 0) {
            lights.environment_texture_cdf[environment.emission_texture] = compute_environment_texels_cdf(
                scene, environment);
        }
    }

    if (lights.instances.empty() && lights.environments.empty()) lights = {};
}

// Progressively compute an image by calling trace_samples multiple times.
void trace_image(image<vec4f>& rendered_image, const yocto_scene& scene,
    const yocto_camera& camera, const bvh_scene& bvh,
    const trace_lights& lights, const trace_sampler_func& sampler,
    int num_samples, int max_bounces, float pixel_clamp, bool no_parallel) {
    auto scope = log_trace_scoped("tracing image");
    auto rngs  = image<rng_state>{};
    make_trace_rngs(rngs, rendered_image.size);
    auto regions = vector<image_region>{};
    make_image_regions(regions, rendered_image.size);

    if (no_parallel) {
        for (auto& region : regions) {
            trace_image_region(rendered_image, scene, camera, bvh, lights,
                sampler, region, 0, num_samples, max_bounces, rngs, pixel_clamp);
        }
    } else {
        auto nthreads = thread::hardware_concurrency();
        auto threads  = vector<thread>();
        for (auto tid = 0; tid < nthreads; tid++) {
            threads.push_back(thread([&, tid]() {
                for (auto region_id = tid; region_id < regions.size();
                     region_id += nthreads) {
                    auto& region = regions[region_id];
                    trace_image_region(rendered_image, scene, camera, bvh,
                        lights, sampler, region, 0, num_samples, max_bounces,
                        rngs, pixel_clamp);
                }
            }));
        }
        for (auto& t : threads) t.join();
    }
}

// Progressively compute an image by calling trace_samples multiple times.
void trace_samples(image<vec4f>& rendered_image, const yocto_scene& scene,
    const yocto_camera& camera, const bvh_scene& bvh,
    const trace_lights& lights, const trace_sampler_func& sampler,
    int current_sample, int num_samples, int max_bounces,
    image<rng_state>& rngs, float pixel_clamp, bool no_parallel) {
    auto regions = vector<image_region>{};
    make_image_regions(regions, rendered_image.size);
    auto scope = log_trace_scoped(
        "tracing samples {}-{}", current_sample, current_sample + num_samples);
    if (no_parallel) {
        for (auto& region : regions) {
            trace_image_region(rendered_image, scene, camera, bvh, lights,
                sampler, region, current_sample, num_samples, max_bounces, rngs,
                pixel_clamp);
        }
    } else {
        auto nthreads = thread::hardware_concurrency();
        auto threads  = vector<thread>();
        for (auto tid = 0; tid < nthreads; tid++) {
            threads.push_back(thread([&, tid]() {
                for (auto region_id = tid; region_id < regions.size();
                     region_id += nthreads) {
                    auto& region = regions[region_id];
                    trace_image_region(rendered_image, scene, camera, bvh,
                        lights, sampler, region, current_sample, num_samples,
                        max_bounces, rngs, pixel_clamp);
                }
            }));
        }
        for (auto& t : threads) t.join();
    }
}

// Starts an anyncrhounous renderer.
void trace_async_start(image<vec4f>& rendered_image, const yocto_scene& scene,
    const yocto_camera& camera, const bvh_scene& bvh,
    const trace_lights& lights, const trace_sampler_func& sampler,
    int num_samples, int max_bounces, image<rng_state>& rngs,
    vector<thread>& threads, bool& stop_flag, int& current_sample,
    concurrent_queue<image_region>& queue, float pixel_clamp) {
    log_trace("start tracing async");
    auto nthreads = thread::hardware_concurrency();
    threads.clear();
    stop_flag = false;
    auto regions = vector<image_region>{};
    make_image_regions(regions, rendered_image.size);
    for (auto tid = 0; tid < nthreads; tid++) {
        threads.push_back(thread([&, tid, nthreads, num_samples, pixel_clamp,
                                     max_bounces, sampler, regions]() {
            for (auto s = 0; s < num_samples; s++) {
                if (!tid) current_sample = s;
                for (auto region_id = tid; region_id < regions.size();
                     region_id += nthreads) {
                    if (stop_flag) return;
                    auto& region = regions[region_id];
                    trace_image_region(rendered_image, scene, camera, bvh, lights,
                        sampler, region, s, 1, max_bounces, rngs, pixel_clamp);
                    queue.push(region);
                }
            }
            if (!tid) current_sample = num_samples;
        }));
    }
}

// Stop the asynchronous renderer.
void trace_async_stop(vector<thread>& threads, bool& stop_flag,
    concurrent_queue<image_region>& queue) {
    stop_flag = true;
    for (auto& t : threads) t.join();
    threads.clear();
    queue.clear();
}

// Trace statistics for last run used for fine tuning implementation.
// For now returns number of paths and number of rays.
pair<uint64_t, uint64_t> get_trace_stats() {
    return {_trace_nrays, _trace_npaths};
}
void reset_trace_stats() {
    _trace_nrays  = 0;
    _trace_npaths = 0;
}

}  // namespace ygl
// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING SUPPORT FUNCTION
// -----------------------------------------------------------------------------
namespace ygl {

// Phong exponent to roughness.
float specular_exponent_to_roughness(float exponent) {
    return sqrtf(2 / (exponent + 2));
}

// Specular to fresnel eta.
void specular_fresnel_from_ks(const vec3f& ks, vec3f& es, vec3f& esk) {
    es  = {(1 + sqrt(ks.x)) / (1 - sqrt(ks.x)),
        (1 + sqrt(ks.y)) / (1 - sqrt(ks.y)), (1 + sqrt(ks.z)) / (1 - sqrt(ks.z))};
    esk = {0, 0, 0};
}

// Specular to  eta.
float specular_to_eta(const vec3f& ks) {
    auto f0 = (ks.x + ks.y + ks.z) / 3;
    return (1 + sqrt(f0)) / (1 - sqrt(f0));
}

// Compute the fresnel term for dielectrics. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3f fresnel_dielectric(float cosw, const vec3f& eta_) {
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
vec3f fresnel_metal(float cosw, const vec3f& eta, const vec3f& etak) {
    if (etak == zero3f) return fresnel_dielectric(cosw, eta);

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

// Schlick approximation of the Fresnel term
vec3f fresnel_schlick(const vec3f& ks, float cosw) {
    if (ks == zero3f) return zero3f;
    return ks + (vec3f{1, 1, 1} - ks) * pow(clamp(1.0f - cosw, 0.0f, 1.0f), 5.0f);
}
vec3f fresnel_schlick(const vec3f& ks, float cosw, float rs) {
    if (ks == zero3f) return zero3f;
    auto fks = fresnel_schlick(ks, cosw);
    return ks + (fks - ks) * (1 - sqrt(clamp(rs, 0.0f, 1.0f)));
}

// Evaluates the GGX pdf
float sample_ggx_pdf(float rs, float ndh) {
    auto alpha2 = rs * rs;
    auto di     = (ndh * ndh) * (alpha2 - 1) + 1;
    auto d      = alpha2 / (pif * di * di);
    return d * ndh;
}

// Sample the GGX distribution
vec3f sample_ggx(float rs, const vec2f& rn) {
    auto tan2 = rs * rs * rn.y / (1 - rn.y);
    auto rz   = sqrt(1 / (tan2 + 1));
    auto rr   = sqrt(1 - rz * rz);
    auto rphi = 2 * pif * rn.x;
    // set to wh
    auto wh_local = vec3f{rr * cos(rphi), rr * sin(rphi), rz};
    return wh_local;
}

}  // namespace ygl

// -----------------------------------------------------------------------------
// NUMERICAL TESTS FOR MONTE CARLO INTEGRATION
// -----------------------------------------------------------------------------
namespace ygl {

float integrate_func_base(
    function<float(float)> f, float a, float b, int nsamples, rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = get_random_float(rng);
        auto x = a + r * (b - a);
        integral += f(x) * (b - a);
    }
    integral /= nsamples;
    return integral;
}

float integrate_func_stratified(
    function<float(float)> f, float a, float b, int nsamples, rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = (i + get_random_float(rng)) / nsamples;
        auto x = a + r * (b - a);
        integral += f(x) * (b - a);
    }
    integral /= nsamples;
    return integral;
}

float integrate_func_importance(function<float(float)> f,
    function<float(float)> pdf, function<float(float)> warp, int nsamples,
    rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = get_random_float(rng);
        auto x = warp(r);
        integral += f(x) / pdf(x);
    }
    integral /= nsamples;
    return integral;
}

// compute the integral and error using different Monte Carlo scehems
// example 1: ---------
// auto f = [](double x) { return 1.0 - (3.0 / 4.0) * x * x; };
// auto a = 0.0, b = 1.0;
// auto expected = 3.0 / 4.0;
// auto nsamples = 10000
// example 2: ---------
// auto f = [](double x) { return sin(x); }
// auto a = 0.0, b = (double)M_PI;
// auto expected = (double)M_PI;
void print_integrate_func_test(function<float(float)> f, float a, float b,
    float expected, int nsamples, function<float(float)> pdf,
    function<float(float)> warp) {
    auto rng = rng_state();
    printf("nsamples base base-err stratified-err importance-err\n");
    for (auto ns = 10; ns < nsamples; ns += 10) {
        auto integral_base       = integrate_func_base(f, a, b, ns, rng);
        auto integral_stratified = integrate_func_stratified(f, a, b, ns, rng);
        auto integral_importance = integrate_func_importance(
            f, pdf, warp, ns, rng);
        auto error_base       = fabs(integral_base - expected) / expected;
        auto error_stratified = fabs(integral_stratified - expected) / expected;
        auto error_importance = fabs(integral_importance - expected) / expected;
        printf("%d %g %g %g %g\n", ns, integral_base, error_base,
            error_stratified, error_importance);
    }
}

float integrate_func2_base(
    function<float(vec2f)> f, vec2f a, vec2f b, int nsamples, rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = get_random_vec2f(rng);
        auto x = a + r * (b - a);
        integral += f(x) * (b.x - a.x) * (b.y - a.y);
    }
    integral /= nsamples;
    return integral;
}

float integrate_func2_stratified(
    function<float(vec2f)> f, vec2f a, vec2f b, int nsamples, rng_state& rng) {
    auto integral  = 0.0f;
    auto nsamples2 = (int)sqrt(nsamples);
    for (auto i = 0; i < nsamples2; i++) {
        for (auto j = 0; j < nsamples2; j++) {
            auto r = vec2f{(i + get_random_float(rng)) / nsamples2,
                (j + get_random_float(rng)) / nsamples2};
            auto x = a + r * (b - a);
            integral += f(x) * (b.x - a.x) * (b.y - a.y);
        }
    }
    integral /= nsamples2 * nsamples2;
    return integral;
}

float integrate_func2_importance(function<float(vec2f)> f,
    function<float(vec2f)> pdf, function<vec2f(vec2f)> warp, int nsamples,
    rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = get_random_vec2f(rng);
        auto x = warp(r);
        integral += f(x) / pdf(x);
    }
    integral /= nsamples;
    return integral;
}

// compute the integral and error using different Monte Carlo scehems
// example 1: ---------
// auto f = [](double x) { return 1.0 - (3.0 / 4.0) * x * x; };
// auto a = 0.0, b = 1.0;
// auto expected = 3.0 / 4.0;
// auto nsamples = 10000
void print_integrate_func2_test(function<float(vec2f)> f, vec2f a, vec2f b,
    float expected, int nsamples, function<float(vec2f)> pdf,
    function<vec2f(vec2f)> warp) {
    auto rng = rng_state();
    printf("nsamples base base-err stratified-err importance-err\n");
    for (auto ns = 10; ns < nsamples; ns += 10) {
        auto integral_base       = integrate_func2_base(f, a, b, ns, rng);
        auto integral_stratified = integrate_func2_stratified(f, a, b, ns, rng);
        auto integral_importance = integrate_func2_importance(
            f, pdf, warp, ns, rng);
        auto error_base       = fabs(integral_base - expected) / expected;
        auto error_stratified = fabs(integral_stratified - expected) / expected;
        auto error_importance = fabs(integral_importance - expected) / expected;
        printf("%d %g %g %g %g\n", ns, integral_base, error_base,
            error_stratified, error_importance);
    }
}

}  // namespace ygl
