//
// # Yocto/Shape: Tiny Library for shape operations for graphics
//
//
// Yocto/Shape is a collection of utilities for manipulating shapes in 3D
// graphics, with a focus on triangle and quad meshes. We support both low-level
// geometry operation and whole shape operations.
//
//
// ## Geometry functions
//
// The library supports basic geomtry functions such as computing
// line/triangle/quad normals and areas, picking points on triangles
// and the like. In these functions triangles are parameterized with us written
// w.r.t the (p1-p0) and (p2-p0) axis respectively. Quads are internally handled
// as pairs of two triangles p0,p1,p3 and p2,p3,p1, with the u/v coordinates
// of the second triangle corrected as 1-u and 1-v to produce a quad
// parametrization where u and v go from 0 to 1. Degenerate quads with p2==p3
// represent triangles correctly, an this convention is used throught the
// library. This is equivalent to Intel's Embree.
//
//
// ## Shape functions
//
// We provide a small number of utilities for shape manipulation for index
// triangle and quad meshes, indexed line and point sets and indexed beziers.
// The utliities collected here are written to support a global illumination
// rendering and not for generic geometry processing. We support operation for
// shape smoothing, shape subdivision (including Catmull-Clark subdivs), and
// example shape creation.
//
// 1. compute line tangents, and triangle and quad areas and normals with
//    `line_tangent()`, `triamgle_normal()`, `quad_normal()` and
//    `line_length()`, `triangle_area()` and `quad_normal()`
// 2. interpolate values over primitives with `interpolate_line()`,
//    `interpolate_triangle()` and `interpolate_quad()`
// 3. evaluate Bezier curves and derivatives with `interpolate_bezier()` and
//    `interpolate_bezier_derivative()`
// 4. compute smooth normals and tangents with `compute_vertex_normals()`
//   `compute_vertex_tangents()`
// 5. compute tangent frames from texture coordinates with
//    `compute_tangent_spaces()`
// 6. compute skinning with `compute_skinning()` and
//    `compute_matrix_skinning()`
// 6. create shapes with `make_cube_shape()`, `make_sphere_shape()`,
// `make_quad_shape()`,
//    `make_cube_fvshape()`, `make_hair_shape()`, `make_suzanne_shape()`,
//    `make_lines_shape()`, `make_points_shape()`, `make_sphere_cube_shape()`,
//    `make_cube_rounded_shape()`, `make_sphere_flipcap_shape()`,
//    `make_cylinder_shape()`, `make_cylinder_rounded_shape()`,
//    `make_disk_shape()`, `make_cylinder_side_shape()`,
//    `make_disk_quad_shape()`
// 7. merge element with `marge_lines()`, `marge_triangles()`, `marge_quads()`
// 8. shape sampling with `sample_points_element()`, `sample_lines_element()`,
//    `sample_triangles_element()`; initialize the sampling CDFs with
//    `sample_points_element_cdf()`, `sample_lines_element_cdf()`,
//    `sample_triangles_element_cdf()`
// 9.  sample a could of point over a surface with
// `sample_triangles_element_points()`
// 10. get edges and boundaries with `get_edges()`
// 11. convert quads to triangles with `convert_quads_to_triangles()`
// 12. convert face varying to vertex shared representations with
//     `convert_face_varying()`
// 13. subdivide elements by edge splits with `subdivide_lines()`,
//     `subdivide_triangles()`, `subdivide_quads()`, `subdivide_beziers()`
// 14. Catmull-Clark subdivision surface with `subdivide_catmullclark()`
//
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

#ifndef _YOCTO_SHAPE_H_
#define _YOCTO_SHAPE_H_

#ifndef YOCTO_QUADS_AS_TRIANGLES
#define YOCTO_QUADS_AS_TRIANGLES 1
#endif

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include "yocto_math.h"

// -----------------------------------------------------------------------------
// GEOMETRY UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Line properties.
inline vec3f line_tangent(const vec3f& p0, const vec3f& p1) {
    return normalize(p1 - p0);
}
inline float line_length(const vec3f& p0, const vec3f& p1) {
    return length(p1 - p0);
}

// Triangle properties.
inline vec3f triangle_normal(
    const vec3f& p0, const vec3f& p1, const vec3f& p2) {
    return normalize(cross(p1 - p0, p2 - p0));
}
inline float triangle_area(const vec3f& p0, const vec3f& p1, const vec3f& p2) {
    return length(cross(p1 - p0, p2 - p0)) / 2;
}

// Quad propeties.
inline vec3f quad_normal(
    const vec3f& p0, const vec3f& p1, const vec3f& p2, const vec3f& p3) {
    return normalize(triangle_normal(p0, p1, p3) + triangle_normal(p2, p3, p1));
}
inline float quad_area(
    const vec3f& p0, const vec3f& p1, const vec3f& p2, const vec3f& p3) {
    return triangle_area(p0, p1, p3) + triangle_area(p2, p3, p1);
}

// Triangle tangent and bitangent from uv
inline pair<vec3f, vec3f> triangle_tangents_fromuv(const vec3f& p0,
    const vec3f& p1, const vec3f& p2, const vec2f& uv0, const vec2f& uv1,
    const vec2f& uv2);

// Quad tangent and bitangent from uv. Note that we pass a current_uv since
// internally we may want to split the quad in two and we need to known where
// to do it. If not interested in the split, just pass zero2f here.
inline pair<vec3f, vec3f> quad_tangents_fromuv(const vec3f& p0, const vec3f& p1,
    const vec3f& p2, const vec3f& p3, const vec2f& uv0, const vec2f& uv1,
    const vec2f& uv2, const vec2f& uv3, const vec2f& current_uv);

// Interpolates values over a line parameterized from a to b by u. Same as lerp.
template <typename T>
inline T interpolate_line(const T& p0, const T& p1, float u) {
    return p0 * (1 - u) + p1 * u;
}
// Interpolates values over a triangle parameterized by u and v along the
// (p1-p0) and (p2-p0) directions. Same as barycentric interpolation.
template <typename T>
inline T interpolate_triangle(
    const T& p0, const T& p1, const T& p2, const vec2f& uv) {
    return p0 * (1 - uv.x - uv.y) + p1 * uv.x + p2 * uv.y;
}
// Interpolates values over a quad parameterized by u and v along the
// (p1-p0) and (p2-p1) directions. Same as bilinear interpolation.
template <typename T>
inline T interpolate_quad(
    const T& p0, const T& p1, const T& p2, const T& p3, const vec2f& uv) {
#if YOCTO_QUADS_AS_TRIANGLES
    if (uv.x + uv.y <= 1) {
        return interpolate_triangle(p0, p1, p3, uv);
    } else {
        return interpolate_triangle(p2, p3, p1, 1 - uv);
    }
#else
    return p0 * (1 - uv.x) * (1 - uv.y) + p1 * uv.x * (1 - uv.y) +
           p2 * uv.x * uv.y + p3 * (1 - uv.x) * uv.y;
#endif
}

// Interpolates values along a cubic Bezier segment parametrized by u.
template <typename T>
inline T interpolate_bezier(
    const T& p0, const T& p1, const T& p2, const T& p3, float u) {
    return p0 * (1 - u) * (1 - u) * (1 - u) + p1 * 3 * u * (1 - u) * (1 - u) +
           p2 * 3 * u * u * (1 - u) + p3 * u * u * u;
}
// Computes the derivative of a cubic Bezier segment parametrized by u.
template <typename T>
inline T interpolate_bezier_derivative(
    const T& p0, const T& p1, const T& p2, const T& p3, float u) {
    return (p1 - p0) * 3 * (1 - u) * (1 - u) + (p2 - p1) * 6 * u * (1 - u) +
           (p3 - p2) * 3 * u * u;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// COMPUTATION OF PER_VERTEX PROPETIES
// -----------------------------------------------------------------------------
namespace yocto {

// Compute per-vertex normals/tangents for lines/triangles/quads.
vector<vec3f> compute_vertex_tangents(
    const vector<vec2i>& lines, const vector<vec3f>& positions);
vector<vec3f> compute_vertex_normals(
    const vector<vec3i>& triangles, const vector<vec3f>& positions);
vector<vec3f> compute_vertex_normals(
    const vector<vec4i>& quads, const vector<vec3f>& positions);

// Compute per-vertex tangent space for triangle meshes.
// Tangent space is defined by a four component vector.
// The first three components are the tangent with respect to the u texcoord.
// The fourth component is the sign of the tangent wrt the v texcoord.
// Tangent frame is useful in normal mapping.
vector<vec4f> compute_tangent_spaces(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3f>& normals,
    const vector<vec2f>& texturecoords);

// Apply skinning to vertex position and normals.
tuple<vector<vec3f>, vector<vec3f>> compute_skinning(
    const vector<vec3f>& positions, const vector<vec3f>& normals,
    const vector<vec4f>& weights, const vector<vec4i>& joints,
    const vector<frame3f>& xforms);
// Apply skinning as specified in Khronos glTF.
tuple<vector<vec3f>, vector<vec3f>> compute_matrix_skinning(
    const vector<vec3f>& positions, const vector<vec3f>& normals,
    const vector<vec4f>& weights, const vector<vec4i>& joints,
    const vector<mat4f>& xforms);

}  // namespace yocto

// -----------------------------------------------------------------------------
// EDGE AND GRID DATA STRUCTURES
// -----------------------------------------------------------------------------
namespace yocto {

// Dictionary to store edge information. edge_index is the index to the edge
// array, edges the array of edges and adj_faces the adjacent faces. We store
// only bidirectional edges to keep the dictionary small. Use the functions
// below to access this data.
struct edge_map {
    unordered_map<vec2i, int> edge_index = {};
    vector<vec2i>             edges      = {};
    vector<bool>              boundary   = {};
};

// initializes an edge map
edge_map make_edge_map(const vector<vec3i>& triangles);
edge_map make_edge_map(const vector<vec4i>& quads);
// Initialize an edge map with elements.
void insert_edges(edge_map& emap, const vector<vec3i>& triangles);
void insert_edges(edge_map& emap, const vector<vec4i>& quads);
// Insert an edge and return its index
int insert_edge(edge_map& emap, const vec2i& edge);
// Get the edge index / insertion count
int get_edge_index(const edge_map& emap, const vec2i& edge);
// Get list of edges / boundary edges
int           get_num_edges(const edge_map& emap);
vector<vec2i> get_edges(const edge_map& emap);
vector<vec2i> get_boundary(const edge_map& emap);
vector<vec2i> get_edges(const vector<vec3i>& triangles);
vector<vec2i> get_edges(const vector<vec4i>& quads);

// A sparse grid of cells, containing list of points. Cells are stored in
// a dictionary to get sparsing. Helpful for nearest neighboor lookups.
struct hash_grid {
    float                             cell_size     = 0;
    float                             cell_inv_size = 0;
    vector<vec3f>                     positions     = {};
    unordered_map<vec3i, vector<int>> cells         = {};
};

// Create a hash_grid
hash_grid make_hash_grid(float cell_size);
hash_grid make_hash_grid(const vector<vec3f>& positions, float cell_size);
// Inserts a point into the grid
int insert_vertex(hash_grid& grid, const vec3f& position);
// Finds the nearest neighboors within a given radius
vector<int> find_nearest_neightbors(
    const hash_grid& grid, const vec3f& position, float max_radius);
vector<int> find_nearest_neightbors(
    const hash_grid& grid, int vertex_id, float max_radius);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHAPE ELEMENT CONVERSION AND GROUPING
// -----------------------------------------------------------------------------
namespace yocto {

// Convert quads to triangles
vector<vec3i> convert_quads_to_triangles(const vector<vec4i>& quads);
// Convert quads to triangles with a diamond-like topology.
// Quads have to be consecutive one row after another.
vector<vec3i> convert_quads_to_triangles(
    const vector<vec4i>& quads, int row_length);
// Convert triangles to quads by creating degenerate quads
vector<vec4i> convert_triangles_to_quads(const vector<vec3i>& triangles);

// Convert beziers to lines using 3 lines for each bezier.
vector<vec2i> convert_bezier_to_lines(const vector<vec4i>& beziers);

// Convert face-varying data to single primitives. Returns the quads indices
// and face ids and filled vectors for pos, norm and texcoord. When used
// with ids, it also plits the faces per id.
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
convert_face_varying(const vector<vec4i>& quads_positions,
    const vector<vec4i>&                  quads_normals,
    const vector<vec4i>& quads_texturecoords, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords);

// Split primitives per id
vector<vector<vec2i>> ungroup_lines(
    const vector<vec2i>& lines, const vector<int>& ids);
vector<vector<vec3i>> ungroup_triangles(
    const vector<vec3i>& triangles, const vector<int>& ids);
vector<vector<vec4i>> ungroup_quads(
    const vector<vec4i>& quads, const vector<int>& ids);

// Weld vertices within a threshold.
tuple<vector<vec3f>, vector<int>> weld_vertices(
    const vector<vec3f>& positions, float threshold);
tuple<vector<vec3i>, vector<vec3f>> weld_triangles(
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    float threshold);
tuple<vector<vec4i>, vector<vec3f>> weld_quads(const vector<vec4i>& quads,
    const vector<vec3f>& positions, float threshold);

// Merge shape elements
void merge_lines(
    vector<vec2i>& lines, const vector<vec2i>& merge_lines, int num_verts);
void merge_triangles(vector<vec3i>& triangles,
    const vector<vec2i>& merge_triangles, int num_verts);
void merge_quads(
    vector<vec4i>& quads, const vector<vec4i>& merge_quads, int num_verts);
void merge_lines(vector<vec2i>& lines, vector<vec3f>& positions,
    vector<vec3f>& tangents, vector<vec2f>& texturecoords,
    vector<float>& radius, const vector<vec2i>& merge_lines,
    const vector<vec3f>& merge_positions, const vector<vec3f>& merge_tangents,
    const vector<vec2f>& merge_texturecoords,
    const vector<float>& merge_radius);
void merge_triangles(vector<vec3i>& triangles, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    const vector<vec2i>& merge_triangles, const vector<vec3f>& merge_positions,
    const vector<vec3f>& merge_normals,
    const vector<vec2f>& merge_texturecoords);
void merge_quads(vector<vec4i>& quads, vector<vec3f>& positions,
    vector<vec3f>& normals, vector<vec2f>& texturecoords,
    const vector<vec4i>& merge_quads, const vector<vec3f>& merge_positions,
    const vector<vec3f>& merge_normals,
    const vector<vec2f>& merge_texturecoords);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHAPE SUBDIVISION
// -----------------------------------------------------------------------------
namespace yocto {

// Subdivide lines by splitting each line in half.
tuple<vector<vec2i>, vector<float>> subdivide_lines(
    const vector<vec2i>& lines, const vector<float>& vert);
tuple<vector<vec2i>, vector<vec2f>> subdivide_lines(
    const vector<vec2i>& lines, const vector<vec2f>& vert);
tuple<vector<vec2i>, vector<vec3f>> subdivide_lines(
    const vector<vec2i>& lines, const vector<vec3f>& vert);
tuple<vector<vec2i>, vector<vec4f>> subdivide_lines(
    const vector<vec2i>& lines, const vector<vec4f>& vert);
// Subdivide triangle by splitting each triangle in four, creating new
// vertices for each edge.
tuple<vector<vec3i>, vector<float>> subdivide_triangles(
    const vector<vec3i>& triangles, const vector<float>& vert);
tuple<vector<vec3i>, vector<vec2f>> subdivide_triangles(
    const vector<vec3i>& triangles, const vector<vec2f>& vert);
tuple<vector<vec3i>, vector<vec3f>> subdivide_triangles(
    const vector<vec3i>& triangles, const vector<vec3f>& vert);
tuple<vector<vec3i>, vector<vec4f>> subdivide_triangles(
    const vector<vec3i>& triangles, const vector<vec4f>& vert);
// Subdivide quads by splitting each quads in four, creating new
// vertices for each edge and for each face.
tuple<vector<vec4i>, vector<float>> subdivide_quads(
    const vector<vec4i>& quads, const vector<float>& vert);
tuple<vector<vec4i>, vector<vec2f>> subdivide_quads(
    const vector<vec4i>& quads, const vector<vec2f>& vert);
tuple<vector<vec4i>, vector<vec3f>> subdivide_quads(
    const vector<vec4i>& quads, const vector<vec3f>& vert);
tuple<vector<vec4i>, vector<vec4f>> subdivide_quads(
    const vector<vec4i>& quads, const vector<vec4f>& vert);
// Subdivide beziers by splitting each segment in two.
tuple<vector<vec4i>, vector<float>> subdivide_beziers(
    const vector<vec4i>& beziers, const vector<float>& vert);
tuple<vector<vec4i>, vector<vec2f>> subdivide_beziers(
    const vector<vec4i>& beziers, const vector<vec2f>& vert);
tuple<vector<vec4i>, vector<vec3f>> subdivide_beziers(
    const vector<vec4i>& beziers, const vector<vec3f>& vert);
tuple<vector<vec4i>, vector<vec4f>> subdivide_beziers(
    const vector<vec4i>& beziers, const vector<vec4f>& vert);
// Subdivide quads using Carmull-Clark subdivision rules.
tuple<vector<vec4i>, vector<float>> subdivide_catmullclark(
    const vector<vec4i>& quads, const vector<float>& vert,
    bool lock_boundary = false);
tuple<vector<vec4i>, vector<vec2f>> subdivide_catmullclark(
    const vector<vec4i>& quads, const vector<vec2f>& vert,
    bool lock_boundary = false);
tuple<vector<vec4i>, vector<vec3f>> subdivide_catmullclark(
    const vector<vec4i>& quads, const vector<vec3f>& vert,
    bool lock_boundary = false);
tuple<vector<vec4i>, vector<vec4f>> subdivide_catmullclark(
    const vector<vec4i>& quads, const vector<vec4f>& vert,
    bool lock_boundary = false);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHAPE SAMPLING
// -----------------------------------------------------------------------------
namespace yocto {

// Pick a point in a point set uniformly.
int           sample_points_element(int npoints, float re);
vector<float> sample_points_element_cdf(int npoints);
int           sample_points_element(const vector<float>& cdf, float re);

// Pick a point on lines uniformly.
vector<float> sample_lines_element_cdf(
    const vector<vec2i>& lines, const vector<vec3f>& positions);
pair<int, float> sample_lines_element(
    const vector<float>& cdf, float re, float ru);

// Pick a point on a triangle mesh uniformly.
vector<float> sample_triangles_element_cdf(
    const vector<vec3i>& triangles, const vector<vec3f>& positions);
pair<int, vec2f> sample_triangles_element(
    const vector<float>& cdf, float re, const vec2f& ruv);

// Pick a point on a quad mesh uniformly.
vector<float> sample_quads_element_cdf(
    const vector<vec4i>& quads, const vector<vec3f>& positions);
pair<int, vec2f> sample_quads_element(
    const vector<float>& cdf, float re, const vec2f& ruv);
pair<int, vec2f> sample_quads_element(const vector<vec4i>& quads,
    const vector<float>& cdf, float re, const vec2f& ruv);

// Samples a set of points over a triangle/quad mesh uniformly. Returns pos,
// norm and texcoord of the sampled points.
tuple<vector<vec3f>, vector<vec3f>, vector<vec2f>> sample_triangles_points(
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    int npoints, int seed = 7);
tuple<vector<vec3f>, vector<vec3f>, vector<vec2f>> sample_quads_points(
    const vector<vec4i>& quads, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    int npoints, int seed = 7);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHAPE GEODESICS
// -----------------------------------------------------------------------------
namespace yocto {

// Data structure used for geodesic computation
struct geodesic_solver {
    struct arc_ {
        int   node   = 0;
        float length = 0;
    };
    struct index_ {
        int node  = 0;
        int index = 0;
    };
    vector<vector<arc_>>   graph      = {};
    vector<vector<index_>> edge_index = {};
    vector<vec3f>          positions  = {};
    vector<vec2i>          edges      = {};
};

// Construct an edge graph
geodesic_solver make_geodesic_solver(
    const vector<vec3i>& triangles, const vector<vec3f>& positions);
vector<float> compute_geodesic_distances(
    geodesic_solver& graph, const vector<int>& sources);
vector<vec4f> convert_distance_to_color(const vector<float>& distances);

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHAPE EXAMPLES
// -----------------------------------------------------------------------------
namespace yocto {

// Make examples shapes that are not watertight (besides quads).
// Return (triangles, quads, pos, norm, texcoord)
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_quad_shape(
    const vec2i& steps, const vec2f& size, const vec2f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_quad_stack_shape(
    const vec3i& steps, const vec3f& size, const vec2f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_floor_shape(
    const vec2i& steps, const vec2f& size, const vec2f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_floor_bent_shape(const vec2i& steps, const vec2f& size,
    const vec2f& uvsize, float radius, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_cube_shape(
    const vec3i& steps, const vec3f& size, const vec3f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_cube_rounded_shape(const vec3i& steps, const vec3f& size,
    const vec3f& uvsize, float radius, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_sphere_shape(
    const vec2i& steps, float size, const vec2f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_sphere_cube_shape(int steps, float size, float uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_sphere_flipcap_shape(const vec2i& steps, float size, const vec2f& uvsize,
    const vec2f& zflip, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_disk_shape(
    const vec2i& steps, float size, const vec2f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_disk_quad_shape(int steps, float size, float uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_disk_bulged_shape(
    int steps, float size, float uvsize, float height, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_cylinder_side_shape(
    const vec2i& steps, const vec2f& size, const vec2f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_cylinder_shape(
    const vec3i& steps, const vec2f& size, const vec3f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
                                                   make_cylinder_rounded_shape(const vec3i& steps, const vec2f& size,
                                                       const vec3f& uvsize, float radius, bool flip_v);
tuple<vector<vec3i>, vector<vec3f>, vector<vec3f>> make_geodesic_sphere_shape(
    int tesselation, float size, bool flip_v);

// Make examples shapes with are watertight (good for subdivs).
tuple<vector<vec4i>, vector<vec3f>> make_suzanne_shape(float size);
tuple<vector<vec4i>, vector<vec3f>> make_cube_shape(const vec3f& size);

// Make facevarying example shapes that are watertight (good for subdivs).
tuple<vector<vec4i>, vector<vec4i>, vector<vec4i>, vector<vec3f>, vector<vec3f>,
    vector<vec2f>>
make_cube_facevarying_shape(
    const vec3i& steps, const vec3f& size, const vec3f& uvsize, bool flip_v);
tuple<vector<vec4i>, vector<vec3f>> make_cube_posonly_shape(
    const vec3i& steps, const vec3f& size, const vec3f& uvsize);
tuple<vector<vec4i>, vector<vec4i>, vector<vec4i>, vector<int>, vector<vec3f>,
    vector<vec3f>, vector<vec2f>>
make_cube_multiplematerials_shape(
    const vec3i& steps, const vec3f& size, const vec3f& uvsize, bool flip_v);

// Generate lines set along a quad. Returns lines, pos, norm, texcoord, radius.
tuple<vector<vec2i>, vector<vec3f>, vector<vec3f>, vector<vec2f>, vector<float>>
make_lines_shape(const vec2i& steps, const vec2f& size, const vec2f& uvsize,
    const vec2f& line_radius = {0.001f, 0.001f});

// Make point primitives. Returns points, pos, norm, texcoord, radius.
tuple<vector<int>, vector<vec3f>, vector<vec3f>, vector<vec2f>, vector<float>>
make_point_shape(float point_radius = 0.001f);
tuple<vector<int>, vector<vec3f>, vector<vec3f>, vector<vec2f>, vector<float>>
make_points_shape(int num, float uvsize, float point_radius = 0.001f);
tuple<vector<int>, vector<vec3f>, vector<vec3f>, vector<vec2f>, vector<float>>
make_random_points_shape(int num, const vec3f& size, float uvsize,
    float point_radius = 0.001f, uint64_t seed = 0);

// Make a bezier circle. Returns bezier, pos.
tuple<vector<vec4i>, vector<vec3f>> make_bezier_circle_shape(float size);

// Make a hair ball around a shape.  Returns lines, pos, norm, texcoord, radius.
// length: minimum and maximum length
// rad: minimum and maximum radius from base to tip
// noise: noise added to hair (strength/scale)
// clump: clump added to hair (number/strength)
// rotation: rotation added to hair (angle/strength)
tuple<vector<vec2i>, vector<vec3f>, vector<vec3f>, vector<vec2f>, vector<float>>
make_hair_shape(const vec2i& steps, const vector<vec3i>& striangles,
    const vector<vec4i>& squads, const vector<vec3f>& spos,
    const vector<vec3f>& snorm, const vector<vec2f>& stexcoord,
    const vec2f& length = {0.1f, 0.1f}, const vec2f& rad = {0.001f, 0.001f},
    const vec2f& noise = zero2f, const vec2f& clump = zero2f,
    const vec2f& rotation = zero2f, int seed = 7);

// Thickens a shape by copy9ing the shape content, rescaling it and flipping its
// normals. Note that this is very much not robust and only useful for trivial
// cases.
tuple<vector<vec4i>, vector<vec3f>, vector<vec3f>, vector<vec2f>>
make_shell_shape(const vector<vec4i>& quads, const vector<vec3f>& positions,
    const vector<vec3f>& normals, const vector<vec2f>& texturecoords,
    float thickness);

}  // namespace yocto

// ---------------------------------------------------------------------------//
//                                                                            //
//                             IMPLEMENTATION                                 //
//                                                                            //
// ---------------------------------------------------------------------------//

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF GEOMETRY UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

// Triangle tangent and bitangent from uv
inline pair<vec3f, vec3f> triangle_tangents_fromuv(const vec3f& p0,
    const vec3f& p1, const vec3f& p2, const vec2f& uv0, const vec2f& uv1,
    const vec2f& uv2) {
    // Follows the definition in http://www.terathon.com/code/tangent.html and
    // https://gist.github.com/aras-p/2843984
    // normal points up from texture space
    auto p   = p1 - p0;
    auto q   = p2 - p0;
    auto s   = vec2f{uv1.x - uv0.x, uv2.x - uv0.x};
    auto t   = vec2f{uv1.y - uv0.y, uv2.y - uv0.y};
    auto div = s.x * t.y - s.y * t.x;

    if (div != 0) {
        auto tu = vec3f{t.y * p.x - t.x * q.x, t.y * p.y - t.x * q.y,
                      t.y * p.z - t.x * q.z} /
                  div;
        auto tv = vec3f{s.x * q.x - s.y * p.x, s.x * q.y - s.y * p.y,
                      s.x * q.z - s.y * p.z} /
                  div;
        return {tu, tv};
    } else {
        return {{1, 0, 0}, {0, 1, 0}};
    }
}

// Quad tangent and bitangent from uv.
inline pair<vec3f, vec3f> quad_tangents_fromuv(const vec3f& p0, const vec3f& p1,
    const vec3f& p2, const vec3f& p3, const vec2f& uv0, const vec2f& uv1,
    const vec2f& uv2, const vec2f& uv3, const vec2f& current_uv) {
#if YOCTO_QUADS_AS_TRIANGLES
    if (current_uv.x + current_uv.y <= 1) {
        return triangle_tangents_fromuv(p0, p1, p3, uv0, uv1, uv3);
    } else {
        return triangle_tangents_fromuv(p2, p3, p1, uv2, uv3, uv1);
    }
#else
    return triangle_tangents_fromuv(p0, p1, p3, uv0, uv1, uv3);
#endif
}

}  // namespace yocto

#endif
