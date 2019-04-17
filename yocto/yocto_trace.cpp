//
// Implementation for Yocto/Trace.
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

#include "yocto_trace.h"

#include <unordered_map>

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

using std::pair;
using std::unordered_map;

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {

// Set non-rigid frames as default
constexpr bool trace_non_rigid_frames = true;

// Trace stats.
atomic<uint64_t> _trace_npaths{0};
atomic<uint64_t> _trace_nrays{0};

// Emission lobe
enum struct esdf_type {
    diffuse_emission,
};
struct esdf {
    esdf_type type      = esdf_type::diffuse_emission;
    vec3f     albedo    = zero3f;
    vec3f     specular  = zero3f;
    float     roughness = 1;
};

// Material lobe
enum struct bsdf_type {
    diffuse_reflection,
    diffuse_transmission,
    specular_reflection,
    metallic_reflection,
    specular_transmission,
    specular_refraction,
    opacity_passthrough,
};
struct bsdf {
    bsdf_type type      = bsdf_type::diffuse_reflection;
    vec3f     albedo    = zero3f;
    vec3f     specular  = zero3f;
    float     roughness = 1;
};

// Volume lobe
enum struct vsdf_type {
    phase_hg,
};
struct vsdf {
    vsdf_type type     = vsdf_type::phase_hg;
    vec3f     emission = zero3f;
    vec3f     albedo   = zero3f;
    vec3f     density  = zero3f;
    float     phaseg   = 0.0f;
};

// Brdf as a list of lobes
template <typename T, int N = 8>
struct short_vector {
    short_vector() {}
    short_vector(int n) {
        _size = n;
        for (auto i = 0; i < n; i++) _data[i] = T{};
    }
    short_vector(initializer_list<T> values) {
        _size = 0;
        for (auto& value : values) {
            _data[_size++] = value;
        }
    }

    int  size() const { return _size; }
    bool empty() const { return _size == 0; }

    void push_back(const T& value) { _data[_size++] = value; }
    void pop_back() { _size--; }

    T&       operator[](int i) { return _data[i]; }
    const T& operator[](int i) const { return _data[i]; }
    T*       begin() { return _data; }
    const T* begin() const { return _data; }
    T*       end() { return _data + _size; }
    const T* end() const { return _data + _size; }
    T*       data() { return _data; }
    const T* data() const { return _data; }
    T&       front() { return _data[0]; }
    const T& front() const { return _data[0]; }
    T&       back() { return _data[_size - 1]; }
    const T& back() const { return _data[_size - 1]; }

   private:
    T   _data[N];
    int _size = 0;
};

bool is_volume_homogeneus(const yocto_material& material) {
    return material.volume_density_texture < 0;
}
bool is_volume_colored(const yocto_material& material) {
    return !(material.volume_density.x == material.volume_density.y &&
             material.volume_density.y == material.volume_density.z);
}

bool is_delta_bsdf(const bsdf& brdf) {
    return brdf.roughness == 0 || brdf.type == bsdf_type::opacity_passthrough;
}

float get_bsdf_sampling_weight(
    const bsdf& brdf, const vec3f& normal, const vec3f& outgoing) {
    auto fresnel = evaluate_fresnel_schlick(
        brdf.specular, abs(dot(outgoing, normal)));
    return (brdf.type == bsdf_type::specular_reflection ||
               brdf.type == bsdf_type::metallic_reflection)
               ? max(brdf.albedo * fresnel)
               : max(brdf.albedo * (1 - fresnel));
}

float get_vsdf_sampling_weight(const vsdf& vsdf, const vec3f& outgoing) {
    return max(vsdf.albedo);
}

void compute_scattering_functions(short_vector<esdf>& esdfs,
    short_vector<bsdf>& bsdfs, short_vector<vsdf>& vsdfs,
    const material_point& material, const vec4f& shape_color) {
    auto emission = material.emission;
    auto diffuse  = !material.base_metallic
                       ? material.diffuse * xyz(shape_color)
                       : material.diffuse * (1 - material.specular);
    auto specular = !material.base_metallic
                        ? material.specular * xyz(shape_color)
                        : material.diffuse * material.specular +
                              0.04f * (1 - material.specular);
    auto transmission = material.transmission;
    auto opacity      = material.opacity * shape_color.w;
    auto roughness    = material.roughness * material.roughness;
    auto refract      = material.refract;
    if (diffuse != zero3f) {
        roughness = clamp(roughness, 0.03f * 0.03f, 1.0f);
    } else if (roughness <= 0.03f * 0.03f) {
        roughness = 0;
    }
    if (opacity > 0.999f) opacity = 1;
    bsdfs = {};
    esdfs = {};
    vsdfs = {};
    if (diffuse != zero3f) {
        bsdfs.push_back(
            {bsdf_type::diffuse_reflection, opacity * diffuse, specular, 1});
    }
    if (specular != zero3f) {
        bsdfs.push_back({bsdf_type::specular_reflection, opacity * vec3f{1},
            specular, roughness});
    }
    if (transmission != zero3f) {
        bsdfs.push_back({refract ? bsdf_type::specular_refraction
                                 : bsdf_type::specular_transmission,
            opacity * transmission, specular, roughness});
    }
    if (opacity < 0.999f) {
        bsdfs.push_back(
            {bsdf_type::opacity_passthrough, vec3f{1 - opacity}, zero3f, 0});
    }
    if (emission != zero3f) {
        esdfs.push_back({esdf_type::diffuse_emission,
            opacity * material.emission * xyz(shape_color), zero3f, 1});
    }
    if (material.volume_density != zero3f) {
        vsdfs.push_back({vsdf_type::phase_hg, material.volume_emission,
            material.volume_albedo, material.volume_density,
            material.volume_phaseg});
    }
}

// Trace point
struct trace_point {
    vec3f              position = zero3f;
    vec3f              normal   = zero3f;
    short_vector<esdf> esdfs    = {};
    short_vector<bsdf> bsdfs    = {};
    short_vector<vsdf> vsdfs    = {};
};

// Make a trace point
void make_trace_point(trace_point& point, const yocto_scene& scene,
    const bvh_intersection& intersection, const vec3f& shading_direction) {
    auto& instance = scene.instances[intersection.instance_id];
    auto& shape    = scene.shapes[instance.shape];
    auto& material = scene.materials[instance.material];
    point.position = evaluate_instance_position(
        scene, instance, intersection.element_id, intersection.element_uv);
    point.normal      = evaluate_instance_normal(scene, instance,
        intersection.element_id, intersection.element_uv,
        trace_non_rigid_frames);
    auto texturecoord = evaluate_shape_texturecoord(
        shape, intersection.element_id, intersection.element_uv);
    auto color = evaluate_shape_color(
        shape, intersection.element_id, intersection.element_uv);
    auto material_point = evaluate_material_point(
        scene, material, texturecoord);
    compute_scattering_functions(
        point.esdfs, point.bsdfs, point.vsdfs, material_point, color);
    if (!shape.lines.empty()) {
        point.normal = orthonormalize(-shading_direction, point.normal);
    } else if (!shape.points.empty()) {
        point.normal = -shading_direction;
    } else {
        if (material.normal_texture >= 0) {
            point.normal = evaluate_instance_perturbed_normal(scene, instance,
                intersection.element_id, intersection.element_uv,
                material_point.normalmap, trace_non_rigid_frames);
        }
    }
}

// Intersects a ray and returns a point
bool trace_ray(const yocto_scene& scene, const bvh_scene& bvh, const ray3f& ray,
    bvh_intersection& intersection) {
    _trace_nrays += 1;
    return intersect_scene_bvh(scene, bvh, ray, intersection);
}

// Intersects a ray and returns a point
bool trace_ray(const yocto_scene& scene, const bvh_scene& bvh,
    const vec3f& position, const vec3f& direction,
    bvh_intersection& intersection) {
    _trace_nrays += 1;
    return intersect_scene_bvh(
        scene, bvh, make_ray(position, direction), intersection);
}

// Sample camera
ray3f sample_camera_ray(const yocto_camera& camera, const vec2i& ij,
    const vec2i& image_size, const vec2f& puv, const vec2f& luv) {
    return evaluate_camera_ray(
        camera, ij, image_size, puv, sample_disk_point(luv));
}

vec3f evaluate_emission(const short_vector<esdf>& esdfs, const vec3f& normal,
    const vec3f& outgoing) {
    auto emission = zero3f;
    for (auto& esdf : esdfs) emission += esdf.albedo;
    return emission;
}

vec3f evaluate_emission(
    const short_vector<vsdf>& vsdfs, const vec3f& outgoing) {
    auto emission = zero3f;
    for (auto& vsdf : vsdfs) emission += vsdf.emission;
    return emission;
}

#if YOCTO_TRACE_THINSHEET

// Schlick approximation of the Fresnel term
vec3f evaluate_fresnel_schlick(
    const vec3f& specular, const vec3f& half_vector, const vec3f& incoming) {
    if (specular == zero3f) return zero3f;
    return specular +
           (vec3f{1, 1, 1} - specular) *
               pow(clamp(1.0f - fabs(dot(half_vector, incoming)), 0.0f, 1.0f),
                   5.0f);
}
vec3f evaluate_fresnel_schlick(const vec3f& specular, const vec3f& half_vector,
    const vec3f& incoming, float roughness) {
    if (specular == zero3f) return zero3f;
    auto fks = evaluate_fresnel_schlick(specular, half_vector, incoming);
    return specular +
           (fks - specular) * (1 - sqrt(clamp(roughness, 0.0f, 1.0f)));
}

// Evaluates the GGX distribution and geometric term
float evaluate_ggx_distribution(
    float roughness, const vec3f& normal, const vec3f& half_vector) {
    auto di = (dot(normal, half_vector) * dot(normal, half_vector)) *
                  (roughness * roughness - 1) +
              1;
    return roughness * roughness / (pif * di * di);
}
float evaluate_ggx_shadowing(float roughness, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
#if 0
    // evaluate G from Heitz
    auto lambda_o = (-1 + sqrt(1 + alpha2 * (1 - ndo * ndo) / (ndo * ndo))) / 2;
    auto lambda_i = (-1 + sqrt(1 + alpha2 * (1 - ndi * ndi) / (ndi * ndi))) / 2;
    auto g = 1 / (1 + lambda_o + lambda_i);
#else
    auto Go = (2 * fabs(dot(normal, outgoing))) /
              (fabs(dot(normal, outgoing)) +
                  sqrt(roughness * roughness + (1 - roughness * roughness) *
                                                   dot(normal, outgoing) *
                                                   dot(normal, outgoing)));
    auto Gi = (2 * fabs(dot(normal, incoming))) /
              (fabs(dot(normal, incoming)) +
                  sqrt(roughness * roughness + (1 - roughness * roughness) *
                                                   dot(normal, incoming) *
                                                   dot(normal, incoming)));
    return Go * Gi;
#endif
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

// Evaluates the BRDF scaled by the cosine of the incoming direction.
// - ggx from [Heitz 2014] and [Walter 2007] and [Lagarde 2014]
// "Understanding the Masking-Shadowing Function in Microfacet-Based
// BRDFs" http://jcgt.org/published/0003/02/03/
// - "Microfacet Models for Refraction through Rough Surfaces" EGSR 07
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
vec3f evaluate_brdf_cosine(const microfacet_brdf& brdf, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
    if (is_brdf_delta(brdf)) return zero3f;
    auto brdf_cosine = zero3f;

    // diffuse
    if (brdf.diffuse != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto half_vector = normalize(incoming + outgoing);
        auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(brdf.specular,
                                            half_vector, outgoing)
                                      : brdf.specular;
        brdf_cosine += brdf.diffuse * (1 - fresnel) / pif;
    }

    // specular
    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto half_vector = normalize(incoming + outgoing);
        auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(brdf.specular,
                                            half_vector, outgoing)
                                      : brdf.specular;
        auto D = evaluate_ggx_distribution(brdf.roughness, normal, half_vector);
        auto G = evaluate_ggx_shadowing(
            brdf.roughness, normal, outgoing, incoming);
        brdf_cosine +=
            fresnel * D * G /
            (4 * fabs(dot(normal, outgoing)) * fabs(dot(normal, incoming)));
    }

    // transmission (thin sheet)
    if (brdf.transmission != zero3f && !brdf.refract &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto ir = (dot(normal, outgoing) >= 0) ? reflect(-incoming, normal)
                                               : reflect(-incoming, -normal);
        auto half_vector = normalize(ir + outgoing);
        auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(brdf.specular,
                                            half_vector, outgoing)
                                      : brdf.specular;
        auto D = evaluate_ggx_distribution(brdf.roughness, normal, half_vector);
        auto G = evaluate_ggx_shadowing(brdf.roughness, normal, outgoing, ir);
        brdf_cosine +=
            brdf.transmission * (1 - fresnel) * D * G /
            (4 * fabs(dot(normal, outgoing)) * fabs(dot(normal, ir)));
    }

    // refraction through rough surface
    if (brdf.transmission != zero3f && brdf.refract &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto eta            = convert_specular_to_eta(brdf.specular);
        auto halfway_vector = dot(normal, outgoing) > 0
                                  ? -(outgoing + eta * incoming)
                                  : (eta * outgoing + incoming);
        auto halfway = normalize(halfway_vector);

        auto fresnel = evaluate_fresnel_schlick(
            brdf.specular, halfway, outgoing);
        auto D = evaluate_ggx_distribution(brdf.roughness, normal, halfway);
        auto G = evaluate_ggx_shadowing(
            brdf.roughness, normal, outgoing, incoming);

        auto dots = dot(outgoing, halfway) * dot(incoming, halfway) /
                    (dot(outgoing, normal) * dot(incoming, normal));

        auto numerator   = (1 - fresnel) * D * G;
        auto denominator = dot(halfway_vector, halfway_vector);

        brdf_cosine += abs(dots) * numerator / denominator;
    }
    return brdf_cosine * abs(dot(normal, incoming));
}

// Evaluates the BRDF assuming that it is called only from the directions
// generated by sample_brdf_direction.
vec3f evaluate_delta_brdf_cosine(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (!is_brdf_delta(brdf)) return zero3f;
    auto microfacet_brdf = zero3f;

    // specular
    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(
                                            brdf.specular, normal, outgoing)
                                      : brdf.specular;
        microfacet_brdf += fresnel;
    }

    // transmission (thin sheet)
    if (brdf.transmission != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(
                                            brdf.specular, normal, outgoing)
                                      : brdf.specular;
        microfacet_brdf += brdf.transmission * (1 - fresnel);
    }

    return microfacet_brdf;
}

// Picks a direction based on the BRDF
vec3f sample_brdf_direction(const microfacet_brdf& brdf, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
    if (is_brdf_delta(brdf)) return zero3f;
    auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(
                                        brdf.specular, normal, outgoing)
                                  : brdf.specular;
    auto prob = vec3f{max(brdf.diffuse * (1 - fresnel)), max(fresnel),
        max(brdf.transmission * (1 - fresnel))};
    if (prob == zero3f) return zero3f;
    prob /= prob.x + prob.y + prob.z;

    // sample according to diffuse
    if (brdf.diffuse != zero3f && rnl < prob.x) {
        auto rz = sqrtf(rn.y), rr = sqrtf(1 - rz * rz), rphi = 2 * pif * rn.x;
        auto il = vec3f{rr * cosf(rphi), rr * sinf(rphi), rz};
        auto fp = dot(normal, outgoing) >= 0
                      ? make_frame_fromz(zero3f, normal)
                      : make_frame_fromz(zero3f, -normal);
        return transform_direction(fp, il);
    }
    // sample according to specular GGX
    else if (brdf.specular != zero3f && rnl < prob.x + prob.y) {
        auto hl = sample_ggx(brdf.roughness, rn);
        auto fp = dot(normal, outgoing) >= 0
                      ? make_frame_fromz(zero3f, normal)
                      : make_frame_fromz(zero3f, -normal);
        auto half_vector = transform_direction(fp, hl);
        return reflect(outgoing, half_vector);
    }
    // transmission hack
    else if (brdf.transmission != zero3f && !brdf.refract &&
             rnl < prob.x + prob.y + prob.z) {
        auto hl = sample_ggx(brdf.roughness, rn);
        auto fp = dot(normal, outgoing) >= 0
                      ? make_frame_fromz(zero3f, normal)
                      : make_frame_fromz(zero3f, -normal);
        auto half_vector = transform_direction(fp, hl);
        auto ir          = reflect(outgoing, half_vector);
        return dot(normal, outgoing) >= 0 ? reflect(-ir, -normal)
                                          : reflect(-ir, normal);
    }
    // sample according to rough refraction
    else if (brdf.transmission != zero3f && brdf.refract &&
             rnl < prob.x + prob.y + prob.z) {
        auto hl = sample_ggx(brdf.roughness, rn);
        auto fp = dot(normal, outgoing) >= 0
                      ? make_frame_fromz(zero3f, normal)
                      : make_frame_fromz(zero3f, -normal);
        auto halfway = transform_direction(fp, hl);

        auto eta = convert_specular_to_eta(brdf.specular);
        return refract(
            outgoing, halfway, dot(normal, outgoing) >= 0 ? 1 / eta : eta);
    }

    else {
        return zero3f;
    }
}

// Picks a direction based on the BRDF
vec3f sample_delta_brdf_direction(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
    if (!is_brdf_delta(brdf)) return zero3f;
    auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(
                                        brdf.specular, normal, outgoing)
                                  : brdf.specular;
    auto prob = vec3f{0, max(fresnel), max(brdf.transmission * (1 - fresnel))};
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
            return refract(
                outgoing, normal, 1 / convert_specular_to_eta(brdf.specular));
        } else {
            return refract(
                outgoing, -normal, convert_specular_to_eta(brdf.specular));
        }
    }
    // no sampling
    else {
        return zero3f;
    }
}

// Compute the weight for sampling the BRDF
float sample_brdf_direction_pdf(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (is_brdf_delta(brdf)) return 0;
    auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(
                                        brdf.specular, normal, outgoing)
                                  : brdf.specular;
    auto prob = vec3f{max(brdf.diffuse * (1 - fresnel)), max(fresnel),
        max(brdf.transmission * (1 - fresnel))};
    if (prob == zero3f) return 0;
    prob /= prob.x + prob.y + prob.z;

    auto pdf = 0.0f;

    if (brdf.diffuse != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        pdf += prob.x * fabs(dot(normal, incoming)) / pif;
    }
    if (brdf.specular != zero3f &&
        dot(normal, outgoing) * dot(normal, incoming) > 0) {
        auto half_vector = normalize(incoming + outgoing);
        auto d = sample_ggx_pdf(brdf.roughness, fabs(dot(normal, half_vector)));
        pdf += prob.y * d / (4 * fabs(dot(outgoing, half_vector)));
    }
    if (brdf.transmission != zero3f && !brdf.refract &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto ir = (dot(normal, outgoing) >= 0) ? reflect(-incoming, normal)
                                               : reflect(-incoming, -normal);
        auto half_vector = normalize(ir + outgoing);
        auto d = sample_ggx_pdf(brdf.roughness, fabs(dot(normal, half_vector)));
        pdf += prob.z * d / (4 * fabs(dot(outgoing, half_vector)));
    }

    // refraction through rough surface
    if (brdf.transmission != zero3f && brdf.refract &&
        dot(normal, outgoing) * dot(normal, incoming) < 0) {
        auto eta            = convert_specular_to_eta(brdf.specular);
        auto outgoing_up    = dot(normal, outgoing) > 0;
        auto halfway_vector = outgoing_up ? -(outgoing + eta * incoming)
                                          : (eta * outgoing + incoming);
        auto halfway = normalize(halfway_vector);

        auto d = sample_ggx_pdf(brdf.roughness, fabs(dot(normal, halfway)));

        auto jacobian = fabs(dot(halfway, incoming)) /
                        dot(halfway_vector, halfway_vector);
        pdf += prob.z * d * jacobian;
    }

    return pdf;
}

// Compute the weight for sampling the BRDF
float sample_delta_brdf_direction_pdf(const microfacet_brdf& brdf,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
    if (!is_brdf_delta(brdf)) return 0;
    auto fresnel = (brdf.fresnel) ? evaluate_fresnel_schlick(
                                        brdf.specular, normal, outgoing)
                                  : brdf.specular;
    auto prob = vec3f{0, max(fresnel), max(brdf.transmission * (1 - fresnel))};
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

#else

// Schlick approximation of the Fresnel term
vec3f evaluate_fresnel_schlick(const vec3f& specular, float direction_cosine) {
    if (specular == zero3f) return zero3f;
    return specular +
           (1 - specular) *
               pow(clamp(1 - abs(direction_cosine), 0.0f, 1.0f), 5.0f);
}
vec3f evaluate_fresnel_schlick(
    const vec3f& specular, float direction_cosine, float roughness) {
    if (specular == zero3f) return zero3f;
    auto fks = evaluate_fresnel_schlick(specular, direction_cosine);
    return specular +
           (fks - specular) * (1 - sqrt(clamp(roughness, 0.0f, 1.0f)));
}

// Evaluates the GGX distribution and geometric term
float evaluate_microfacet_distribution(
    float roughness, const vec3f& normal, const vec3f& half_vector, bool ggx) {
    auto cosine = dot(normal, half_vector);
    if (cosine <= 0) return 0;
    auto roughness_square = roughness * roughness;
    auto cosine_square    = cosine * cosine;
    auto tangent_square   = clamp01(1 - cosine_square) / cosine_square;
    if (ggx) {
        return roughness_square / (pif * cosine_square * cosine_square *
                                      (roughness_square + tangent_square) *
                                      (roughness_square + tangent_square));
    } else {
        return exp(-tangent_square / roughness_square) /
               (pif * roughness_square * cosine_square * cosine_square);
    }
}
float evaluate_microfacet_shadowing_term(float roughness, const vec3f& normal,
    const vec3f& half_vector, const vec3f& direction, bool ggx) {
    auto cosine = dot(normal, direction);
    if (dot(half_vector, direction) * cosine <= 0) return 0;
    auto roughness_square = roughness * roughness;
    auto cosine_square    = cosine * cosine;
    auto tangent_square   = clamp01(1 - cosine_square) / cosine_square;
    if (ggx) {
        return 2 / (1 + sqrt(1.0f + roughness_square * tangent_square));
    } else {
        auto tangent       = sqrt(tangent_square);
        auto inv_rt        = 1 / (roughness * tangent);
        auto inv_rt_square = 1 / (roughness_square * tangent_square);
        if (inv_rt < 1.6f) {
            return (3.535f * inv_rt + 2.181f * inv_rt_square) /
                   (1.0f + 2.276f * inv_rt + 2.577f * inv_rt_square);
        } else {
            return 1.0f;
        }
    }
}
float evaluate_microfacet_shadowing(float roughness, const vec3f& normal,
    const vec3f& half_vector, const vec3f& outgoing, const vec3f& incoming,
    bool ggx) {
    return evaluate_microfacet_shadowing_term(
               roughness, normal, half_vector, outgoing, ggx) *
           evaluate_microfacet_shadowing_term(
               roughness, normal, half_vector, incoming, ggx);
}
vec3f sample_microfacet_distribution(
    float roughness, const vec3f& normal, const vec2f& rn, bool ggx) {
    auto phi              = 2 * pif * rn.x;
    auto roughness_square = roughness * roughness;
    auto tangent_square   = 0.0f;
    if (ggx) {
        tangent_square = -roughness_square * log(1 - rn.y);
    } else {
        tangent_square = roughness_square * rn.y / (1 - rn.y);
    }
    auto cosine_square     = 1 / (1 + tangent_square);
    auto cosine            = 1 / sqrt(1 + tangent_square);
    auto radius            = sqrt(clamp01(1 - cosine_square));
    auto local_half_vector = vec3f{
        cos(phi) * radius, sin(phi) * radius, cosine};
    return transform_direction(make_basis_fromz(normal), local_half_vector);
}
float sample_microfacet_distribution_pdf(
    float roughness, const vec3f& normal, const vec3f& half_vector, bool ggx) {
    auto cosine = dot(normal, half_vector);
    if (cosine < 0) return 0;
    return evaluate_microfacet_distribution(
               roughness, normal, half_vector, ggx) *
           cosine;
}

// Evaluates the BRDF scaled by the cosine of the incoming direction.
// - ggx from [Heitz 2014] and [Walter 2007] and [Lagarde 2014]
// "Understanding the Masking-Shadowing Function in Microfacet-Based
// BRDFs" http://jcgt.org/published/0003/02/03/
// - "Microfacet Models for Refraction through Rough Surfaces" EGSR 07
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
//
// The following code for evaluation and sampling of bsdfs
// assumes that the input shading normal (normal_) can be back facing with
// respect to the outgoing direction. This lets us understand if the ray is
// entering or exiting the surface. However, at the beginning of each function,
// the normal is flipped if needed so that it is forward facing. In other words,
// for the processed normal, dot(outgoing, normal) > 0 always holds true.
//
//                                                  giacomo, 28-3-2019
vec3f evaluate_brdf_cosine(const short_vector<bsdf>& bsdfs,
    const vec3f& normal_, const vec3f& outgoing, const vec3f& incoming,
    bool delta) {
    // orientation
    auto outgoing_up = dot(outgoing, normal_) > 0;
    auto incoming_up = dot(incoming, normal_) > 0;
    auto normal      = outgoing_up ? normal_ : -normal_;

    auto brdf_cosine = zero3f;
    for (auto& brdf : bsdfs) {
        if (delta != is_delta_bsdf(brdf)) continue;

        // diffuse
        if (brdf.type == bsdf_type::diffuse_reflection &&
            outgoing_up == incoming_up) {
            auto halfway = normalize(incoming + outgoing);
            auto fresnel = evaluate_fresnel_schlick(
                brdf.specular, dot(halfway, outgoing));
            brdf_cosine += brdf.albedo * (1 - fresnel) / pif *
                           abs(dot(normal, incoming));
        }
        // specular
        else if (brdf.type == bsdf_type::specular_reflection &&
                 !is_delta_bsdf(brdf) && outgoing_up == incoming_up) {
            auto halfway = normalize(incoming + outgoing);
            auto fresnel = evaluate_fresnel_schlick(
                brdf.specular, dot(halfway, outgoing));
            auto D = evaluate_microfacet_distribution(
                brdf.roughness, normal, halfway);
            auto G = evaluate_microfacet_shadowing(
                brdf.roughness, normal, halfway, outgoing, incoming);
            brdf_cosine +=
                brdf.albedo * fresnel * D * G /
                fabs(4 * dot(normal, outgoing) * dot(normal, incoming)) *
                abs(dot(normal, incoming));
        }
        // specular
        else if (brdf.type == bsdf_type::specular_reflection &&
                 is_delta_bsdf(brdf) && outgoing_up == incoming_up) {
            auto fresnel = evaluate_fresnel_schlick(
                brdf.specular, dot(normal, outgoing));
            brdf_cosine += brdf.albedo * fresnel;
        }
        // transmission through rough thin surface
        else if (brdf.type == bsdf_type::specular_transmission &&
                 !is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            auto ir      = reflect(-incoming, normal);
            auto halfway = normalize(ir + outgoing);
            auto F       = evaluate_fresnel_schlick(
                brdf.specular, dot(halfway, outgoing));
            auto D = evaluate_microfacet_distribution(
                brdf.roughness, normal, halfway);
            auto G = evaluate_microfacet_shadowing(
                brdf.roughness, normal, halfway, outgoing, ir);
            brdf_cosine +=
                brdf.albedo * (1 - F) * D * G /
                fabs(4 * dot(normal, outgoing) * dot(normal, incoming)) *
                abs(dot(normal, incoming));
        }
        // transmission (thin surface)
        else if (brdf.type == bsdf_type::specular_transmission &&
                 is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            auto fresnel = evaluate_fresnel_schlick(
                brdf.specular, dot(normal, outgoing));
            brdf_cosine += brdf.albedo * (1 - fresnel);
        }
        // refraction through rough surface
        else if (brdf.type == bsdf_type::specular_refraction &&
                 !is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            auto eta            = convert_specular_to_eta(brdf.specular);
            auto halfway_vector = outgoing_up ? -(outgoing + eta * incoming)
                                              : (eta * outgoing + incoming);
            auto halfway = normalize(halfway_vector);

            auto F = evaluate_fresnel_schlick(
                brdf.specular, dot(halfway, outgoing));
            auto D = evaluate_microfacet_distribution(
                brdf.roughness, normal, halfway);
            auto G = evaluate_microfacet_shadowing(
                brdf.roughness, normal, halfway, outgoing, incoming);

            auto dot_terms = (dot(outgoing, halfway) * dot(incoming, halfway)) /
                             (dot(outgoing, normal) * dot(incoming, normal));

            auto numerator   = brdf.albedo * (1 - F) * D * G;
            auto denominator = dot(halfway_vector, halfway_vector);

            // [Walter 2007] equation 21
            brdf_cosine += abs(dot_terms) * numerator / denominator *
                           abs(dot(normal, incoming));
        }
        // refraction
        else if (brdf.type == bsdf_type::specular_refraction &&
                 is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            auto fresnel = evaluate_fresnel_schlick(
                brdf.specular, dot(normal, outgoing));
            brdf_cosine += brdf.albedo * (1 - fresnel);
        }
        // opacity passthrought
        else if (brdf.type == bsdf_type::opacity_passthrough &&
                 is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            brdf_cosine += brdf.albedo;
        }
        // all other lobes return zero for now
        else {
            // no instructions here
        }
    }
    return brdf_cosine;
}

// Picks a direction based on the BRDF
vec3f sample_brdf_direction(const short_vector<bsdf>& bsdfs,
    const vec3f& normal_, const vec3f& outgoing, float rnl, const vec2f& rn,
    bool delta) {
    // orientation
    auto outgoing_up = dot(outgoing, normal_) > 0;
    auto normal      = outgoing_up ? normal_ : -normal_;

    // weights
    auto weights     = short_vector<float, 8>{};
    auto sum_weights = 0.0f;
    for (auto& brdf : bsdfs) {
        if (delta != is_delta_bsdf(brdf)) {
            weights.push_back(0);
        } else {
            weights.push_back(get_bsdf_sampling_weight(brdf, normal, outgoing));
        }
        sum_weights += weights.back();
    }
    if (sum_weights == 0) return zero3f;
    for (auto& weight : weights) weight /= sum_weights;

    // pick lobe
    auto id  = 0;
    auto sum = 0.0f;
    for (id = 0; id < weights.size(); id++) {
        sum += weights[id];
        if (sum >= rnl) break;
    }
    id         = clamp(id, 0, weights.size() - 1);
    auto& brdf = bsdfs[id];

    // sample according to diffuse
    if (brdf.type == bsdf_type::diffuse_reflection) {
        return sample_hemisphere_direction(normal, rn);
    }
    // sample according to specular GGX
    else if (brdf.type == bsdf_type::specular_reflection &&
             !is_delta_bsdf(brdf)) {
        auto halfway = sample_microfacet_distribution(
            brdf.roughness, normal, rn);
        return reflect(outgoing, halfway);
    }
    // sample according to specular mirror
    else if (brdf.type == bsdf_type::specular_reflection &&
             is_delta_bsdf(brdf)) {
        return reflect(outgoing, normal);
    }
    // sample according to transmission
    else if (brdf.type == bsdf_type::specular_transmission &&
             !is_delta_bsdf(brdf)) {
        auto halfway = sample_microfacet_distribution(
            brdf.roughness, normal, rn);
        auto ir = reflect(outgoing, halfway);
        return -reflect(ir, normal);
    }
    // sample according to transmission
    else if (brdf.type == bsdf_type::specular_transmission &&
             is_delta_bsdf(brdf)) {
        return -outgoing;
    }
    // sample according to rough refraction
    else if (brdf.type == bsdf_type::specular_refraction &&
             !is_delta_bsdf(brdf)) {
        auto halfway = sample_microfacet_distribution(
            brdf.roughness, normal, rn);
        auto eta = convert_specular_to_eta(brdf.specular);
        return refract(outgoing, halfway, outgoing_up ? 1 / eta : eta);
    }
    // sample according to perfect refraction
    else if (brdf.type == bsdf_type::specular_refraction &&
             is_delta_bsdf(brdf)) {
        auto eta = convert_specular_to_eta(brdf.specular);
        return refract(outgoing, normal, outgoing_up ? 1 / eta : eta);
    }
    // sample according to opacity
    else if (brdf.type == bsdf_type::opacity_passthrough &&
             is_delta_bsdf(brdf)) {
        return -outgoing;
    }
    // other lobes
    else {
        return zero3f;
    }
}

// Compute the weight for sampling the BRDF
float sample_brdf_direction_pdf(const short_vector<bsdf>& bsdfs,
    const vec3f& normal_, const vec3f& outgoing, const vec3f& incoming,
    bool delta) {
    // orientation
    auto outgoing_up = dot(outgoing, normal_) > 0;
    auto incoming_up = dot(incoming, normal_) > 0;
    auto normal      = outgoing_up ? normal_ : -normal_;

    // weights
    auto weights     = short_vector<float, 8>{};
    auto sum_weights = 0.0f;
    for (auto& brdf : bsdfs) {
        if (delta != is_delta_bsdf(brdf)) {
            weights.push_back(0);
        } else {
            weights.push_back(get_bsdf_sampling_weight(brdf, normal, outgoing));
        }
        sum_weights += weights.back();
    }
    if (sum_weights == 0) return 0;
    for (auto& weight : weights) weight /= sum_weights;

    auto pdf = 0.0f;
    for (auto i = 0; i < bsdfs.size(); i++) {
        auto& brdf = bsdfs[i];
        if (delta != is_delta_bsdf(brdf)) continue;

        // diffuse
        if (brdf.type == bsdf_type::diffuse_reflection &&
            outgoing_up == incoming_up) {
            pdf += weights[i] * fabs(dot(normal, incoming)) / pif;
        }
        // specular reflection
        else if (brdf.type == bsdf_type::specular_reflection &&
                 !is_delta_bsdf(brdf) && outgoing_up == incoming_up) {
            auto halfway = normalize(incoming + outgoing);
            auto d       = sample_microfacet_distribution_pdf(
                brdf.roughness, normal, halfway);
            auto jacobian = 0.25f / fabs(dot(outgoing, halfway));
            pdf += weights[i] * d * jacobian;
        }
        // specular reflection
        else if (brdf.type == bsdf_type::specular_reflection &&
                 is_delta_bsdf(brdf) && outgoing_up == incoming_up) {
            pdf += weights[i];
        }
        // transmission through thin surface
        else if (brdf.type == bsdf_type::specular_transmission &&
                 !is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            auto ir      = reflect(-incoming, normal);
            auto halfway = normalize(ir + outgoing);
            auto d       = sample_microfacet_distribution_pdf(
                brdf.roughness, normal, halfway);
            auto jacobian = 0.25f / fabs(dot(outgoing, halfway));
            pdf += weights[i] * d * jacobian;
            // print("pdf transmission\n");
        }
        // transmission
        else if (brdf.type == bsdf_type::specular_transmission &&
                 is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            pdf += weights[i];
        }
        // refraction through rough surface
        else if (brdf.type == bsdf_type::specular_refraction &&
                 !is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            auto eta            = convert_specular_to_eta(brdf.specular);
            auto halfway_vector = outgoing_up ? -(outgoing + eta * incoming)
                                              : (eta * outgoing + incoming);
            auto halfway = normalize(halfway_vector);

            auto d = sample_microfacet_distribution_pdf(
                brdf.roughness, normal, halfway);

            // [Walter 2007] equation 17
            auto jacobian = fabs(dot(halfway, incoming)) /
                            dot(halfway_vector, halfway_vector);
            pdf += weights[i] * d * jacobian;
            // print("pdf refraction\n");
        }
        // refraction
        else if (brdf.type == bsdf_type::specular_refraction &&
                 is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            pdf += weights[i];
        }
        // opacity
        else if (brdf.type == bsdf_type::opacity_passthrough &&
                 is_delta_bsdf(brdf) && outgoing_up != incoming_up) {
            pdf += weights[i];
        }
        // other lobes
        else {
        }
    }

    return pdf;
}

#endif

// Sample pdf for an environment.
float sample_environment_direction_pdf(const yocto_scene& scene,
    const trace_lights& lights, int environment_id, const vec3f& incoming) {
    auto& environment = scene.environments[environment_id];
    if (environment.emission_texture >= 0) {
        auto& elements_cdf =
            lights.environment_texture_cdf[environment.emission_texture];
        auto& emission_texture = scene.textures[environment.emission_texture];
        auto  size             = evaluate_texture_size(emission_texture);
        auto  texcoord         = evaluate_environment_texturecoord(
            environment, incoming);
        auto i    = clamp((int)(texcoord.x * size.x), 0, size.x - 1);
        auto j    = clamp((int)(texcoord.y * size.y), 0, size.y - 1);
        auto prob = sample_discrete_distribution_pdf(
                        elements_cdf, j * size.x + i) /
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
    const trace_lights& lights, int environment_id, float rel,
    const vec2f& ruv) {
    auto& environment = scene.environments[environment_id];
    if (environment.emission_texture >= 0) {
        auto& elements_cdf =
            lights.environment_texture_cdf[environment.emission_texture];
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

// Picks a point on a light.
vec3f sample_instance_direction(const yocto_scene& scene,
    const trace_lights& lights, int instance_id, const vec3f& p, float rel,
    const vec2f& ruv) {
    auto& instance                = scene.instances[instance_id];
    auto& shape                   = scene.shapes[instance.shape];
    auto& elements_cdf            = lights.shape_elements_cdf[instance.shape];
    auto [element_id, element_uv] = sample_shape_element(
        shape, elements_cdf, rel, ruv);
    return normalize(
        evaluate_instance_position(scene, instance, element_id, element_uv) -
        p);
}

// Sample pdf for a light point.
float sample_instance_direction_pdf(const yocto_scene& scene,
    const trace_lights& lights, int instance_id, const bvh_scene& bvh,
    const vec3f& position, const vec3f& direction) {
    auto& instance = scene.instances[instance_id];
    auto& material = scene.materials[instance.material];
    if (material.emission == zero3f) return 0;
    auto& elements_cdf = lights.shape_elements_cdf[instance.shape];
    // check all intersection
    auto pdf           = 0.0f;
    auto next_position = position;
    for (auto bounce = 0; bounce < 100; bounce++) {
        auto isec = bvh_intersection{};
        if (!intersect_instance_bvh(scene, bvh, instance_id,
                make_ray(next_position, direction), isec))
            break;
        // accumulate pdf
        auto& instance       = scene.instances[isec.instance_id];
        auto  light_position = evaluate_instance_position(
            scene, instance, isec.element_id, isec.element_uv);
        auto light_normal = evaluate_instance_normal(scene, instance,
            isec.element_id, isec.element_uv, trace_non_rigid_frames);
        // prob triangle * area triangle = area triangle mesh
        auto area = elements_cdf.back();
        pdf += distance_squared(light_position, position) /
               (abs(dot(light_normal, direction)) * area);
        // continue
        next_position = light_position + direction * 1e-3f;
    }
    return pdf;
}

// Sample lights wrt solid angle
vec3f sample_lights_direction(const yocto_scene& scene,
    const trace_lights& lights, const bvh_scene& bvh, const vec3f& position,
    float rl, float rel, const vec2f& ruv) {
    auto light_id = sample_uniform_index(
        lights.instances.size() + lights.environments.size(), rl);
    if (light_id < lights.instances.size()) {
        auto instance = lights.instances[light_id];
        return sample_instance_direction(
            scene, lights, instance, position, rel, ruv);
    } else {
        auto environment =
            lights.environments[light_id - (int)lights.instances.size()];
        return sample_environment_direction(
            scene, lights, environment, rel, ruv);
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
    pdf *= sample_uniform_index_pdf<float>(
        lights.instances.size() + lights.environments.size());
    return pdf;
}

inline vec3f get_russian_roulette_albedo(const short_vector<bsdf>& bsdfs) {
    auto albedo = zero3f;
    for (auto& brdf : bsdfs) {
        albedo += brdf.albedo;
    }
    return albedo;
}

inline vec3f get_russian_roulette_albedo(const short_vector<vsdf>& vsdfs) {
    auto albedo = zero3f;
    for (auto& vsdf : vsdfs) {
        albedo += vsdf.albedo;
    }
    return albedo;
}

// Russian roulette mode: 0: albedo, 1: weight, 2: pbrt
#define YOCTO_RUSSIAN_ROULETTE_MODE 2
/*
   Russian roulette on weight causes extremely high noise with prefectly
   transmissive materials because when reaching transparent surfaces (where you
   just want to continue path tracing) a path may still have low weight, hence
   high probability of beign terminated by Russian roulette.
   This causes to throw away almost every path that passes through multiple
   transparent surfaces: the path gets terminated very often, but it doesn't
   carry any radiance yet (since we don't split to trace lights) hence gives no
   contribution.
   The current fix is to sample Russian roulette on albedo, which is a local
   property and does not depend on the path. In this way we ensure that a path
   has always low propabilty of being terminated on transmissive surfaces.

                                                    giacomo, 3-04-2019
*/

// Russian roulette. Returns whether to stop and its pdf.
pair<bool, float> sample_russian_roulette(const vec3f& albedo,
    const vec3f& weight, int bounce, float rr, int min_bounce = 4,
    float rr_threadhold = 1) {
#if YOCTO_RUSSIAN_ROULETTE_MODE == 0
    if (bounce <= min_bounce) return {false, 1};
    auto rr_prob = clamp(1.0f - max(albedo), 0.0f, 0.95f);
    return {rr < rr_prob, 1 - rr_prob};
#elif YOCTO_RUSSIAN_ROULETTE_MODE == 1
    if (bounce <= min_bounce) return {false, 1};
    auto rr_prob = 1.0f - min(max(weight), 0.95f);
    return rr < rr_prob;
#elif YOCTO_RUSSIAN_ROULETTE_MODE == 2
    // from pbrt
    // Spectrum rrBeta = beta * etaScale;
    // if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
    //     Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
    //     if (sampler.Get1D() < q) break;
    //     beta /= 1 - q;
    //     DCHECK(!std::isinf(beta.y()));
    // }
    if (max(weight) < rr_threadhold && bounce > min_bounce) {
        auto rr_prob = max((float)0.05, 1 - max(weight));
        return {rr < rr_prob, 1 - rr_prob};
    } else {
        return {false, 1};
    }
#endif
}

vec3f evaluate_transmission(const yocto_scene& scene,
    const yocto_material& material, const vec3f& from, const vec3f& dir,
    float distance, int channel, rng_state& rng) {
    auto& vd = material.volume_density;
    if (is_volume_homogeneus(material))
        return vec3f{exp(-distance * vd.x), exp(-distance * vd.y),
            exp(-distance * vd.z)};

    // ratio tracking
    auto tr = 1.0f, t = 0.0f;
    auto pos = from;
    while (true) {
        auto step = -log(1 - get_random_float(rng)) / vd[channel];
        t += step;
        if (t >= distance) break;
        pos += dir * step;
        auto density = material.volume_density;
        if (material.volume_density_texture >= 0) {
            auto& volume_density_texture =
                scene.voltextures[material.volume_density_texture];
            density *= evaluate_voltexture(volume_density_texture, pos);
        }
        tr *= 1.0f - max(0.0f, density[channel] / vd[channel]);
    }
    return {tr, tr, tr};
}

pair<float, int> sample_volume_distance(
    const short_vector<vsdf>& vsdfs, float rl, float rd) {
    if (vsdfs.size() != 1) throw runtime_error("unsupported size");
    auto channel = clamp((int)(rl * 3), 0, 2);
    auto density = vsdfs[0].density[channel];
    if (density == 0 || rd == 0)
        return {float_max, channel};
    else
        return {-log(rd) / density, channel};
}

float sample_volume_distance_pdf(
    const short_vector<vsdf>& vsdfs, float distance, int channel) {
    if (vsdfs.size() != 1) throw runtime_error("unsupported size");
    auto density = vsdfs[0].density[channel];
    return exp(-density * distance);
}

vec3f evaluate_volume_transmission(
    const short_vector<vsdf>& vsdfs, float distance) {
    if (vsdfs.size() != 1) throw runtime_error("unsupported size");
    return exp(-vsdfs[0].density * distance);
}

float sample_distance(const yocto_scene& scene, const yocto_material& material,
    const vec3f& from, const vec3f& dir, int channel, rng_state& rng) {
    auto pos      = from;
    auto majorant = material.volume_density[channel];
    if (majorant == 0) return float_max;

    // delta tracking
    auto distance = 0.0f;
    while (true) {
        auto r = get_random_float(rng);
        if (r == 0) return float_max;
        auto step = -log(r) / majorant;
        if (is_volume_homogeneus(material)) return step;

        pos += dir * step;
        distance += step;
        auto density = material.volume_density;
        if (material.volume_density_texture >= 0) {
            auto& volume_density_texture =
                scene.voltextures[material.volume_density_texture];
            density *= evaluate_voltexture(volume_density_texture, pos);
        }
        if (density[channel] / majorant >= get_random_float(rng))
            return distance;

        // Escape from volume.
        if (pos.x > 1 || pos.y > 1 || pos.z > 1) return float_max;
        if (pos.x < -1 || pos.y < -1 || pos.z < -1) return float_max;
    }
}

float sample_distance(const yocto_scene& scene, const yocto_instance& instance,
    const bbox3f& bbox, const vec3f& from, const vec3f& dir, int channel,
    rng_state& rng) {
    auto& material = scene.materials[instance.material];
    if (material.volume_density == zero3f) return float_max;

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

vec3f evaluate_vsdf_cosine(const short_vector<vsdf>& vsdfs,
    const vec3f& outgoing, const vec3f& incoming) {
    auto phase_function_cosine = zero3f;
    for (auto& vsdf : vsdfs) {
        auto cos_theta = dot(outgoing, incoming);
        phase_function_cosine += vsdf.albedo * evaluate_phase_function(
                                                   cos_theta, vsdf.phaseg);
    }
    return phase_function_cosine;
}

vec3f sample_vsdf_direction(const short_vector<vsdf>& vsdfs,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
    // weights
    auto weights     = short_vector<float, 8>{};
    auto sum_weights = 0.0f;
    for (auto& vsdf : vsdfs) {
        weights.push_back(get_vsdf_sampling_weight(vsdf, outgoing));
        sum_weights += weights.back();
    }
    if (sum_weights == 0) return zero3f;
    for (auto& weight : weights) weight /= sum_weights;

    // pick lobe
    auto id  = 0;
    auto sum = 0.0f;
    for (id = 0; id < weights.size(); id++) {
        sum += weights[id];
        if (sum >= rnl) break;
    }
    id         = clamp(id, 0, weights.size() - 1);
    auto& vsdf = vsdfs[id];

    // sample accirding to phase_g
    auto direction = sample_phase_function(vsdf.phaseg, rn);
    return make_basis_fromz(-outgoing) * direction;
}

float sample_vsdf_direction_pdf(const short_vector<vsdf>& vsdfs,
    const vec3f& outgoing, const vec3f& incoming) {
    // weights
    auto weights     = short_vector<float, 8>{};
    auto sum_weights = 0.0f;
    for (auto& vsdf : vsdfs) {
        weights.push_back(get_vsdf_sampling_weight(vsdf, outgoing));
        sum_weights += weights.back();
    }
    if (sum_weights == 0) return 0;
    for (auto& weight : weights) weight /= sum_weights;

    auto pdf = 0.0f;
    for (auto i = 0; i < vsdfs.size(); i++) {
        auto& vsdf = vsdfs[i];
        pdf += weights[i] *
               evaluate_phase_function(dot(outgoing, incoming), vsdf.phaseg);
    }
    return pdf;
}

tuple<vec3f, vec3f, float> sample_next_direction(const yocto_scene& scene,
    const trace_lights& lights, const bvh_scene& bvh, const vec3f& position,
    const vec3f& normal, const short_vector<bsdf>& bsdfs, const vec3f& outgoing,
    rng_state& rng, bool mis, bool include_smooth = true,
    bool include_delta = true) {
    // choose delta or non-delta
    auto weight_smooth = 0.0f, weight_delta = 0.0f;
    for (auto& brdf : bsdfs) {
        if (!is_delta_bsdf(brdf) && include_smooth) {
            weight_smooth += get_bsdf_sampling_weight(brdf, normal, outgoing);
        }
        if (is_delta_bsdf(brdf) && include_delta) {
            weight_delta += get_bsdf_sampling_weight(brdf, normal, outgoing);
        }
    }
    if (weight_smooth == 0 && weight_delta == 0) return {zero3f, zero3f, 0};
    auto weight_sum = weight_smooth + weight_delta;
    weight_smooth /= weight_sum;
    weight_delta /= weight_sum;
    auto delta    = get_random_float(rng) < weight_delta;
    auto mode_pdf = !delta ? weight_smooth : weight_delta;
    // continue path
    auto incoming     = zero3f;
    auto brdf_cosine  = zero3f;
    auto incoming_pdf = 0.0f;
    if (mis && !delta) {
        if (get_random_float(rng) < 0.5f) {
            incoming = sample_brdf_direction(bsdfs, normal, outgoing,
                get_random_float(rng), get_random_vec2f(rng), delta);
        } else {
            incoming = sample_lights_direction(scene, lights, bvh, position,
                get_random_float(rng), get_random_float(rng),
                get_random_vec2f(rng));
        }
        brdf_cosine = evaluate_brdf_cosine(
            bsdfs, normal, outgoing, incoming, delta);
        incoming_pdf = 0.5f * sample_brdf_direction_pdf(
                                  bsdfs, normal, outgoing, incoming, delta) +
                       0.5f * sample_lights_direction_pdf(
                                  scene, lights, bvh, position, incoming);
    } else {
        incoming    = sample_brdf_direction(bsdfs, normal, outgoing,
            get_random_float(rng), get_random_vec2f(rng), delta);
        brdf_cosine = evaluate_brdf_cosine(
            bsdfs, normal, outgoing, incoming, delta);
        incoming_pdf = sample_brdf_direction_pdf(
            bsdfs, normal, outgoing, incoming, delta);
    }
    incoming_pdf *= mode_pdf;
    if (incoming == zero3f || incoming_pdf == 0) {
        return {zero3f, zero3f, 0};
    } else {
        return {brdf_cosine, incoming, incoming_pdf};
    }
}

tuple<vec3f, vec3f, float> sample_next_direction_volume(
    const yocto_scene& scene, const trace_lights& lights, const bvh_scene& bvh,
    const vec3f& position, const vec3f& outgoing,
    const short_vector<vsdf>& vsdfs, rng_state& rng, bool mis) {
    // continue path
    auto incoming     = zero3f;
    auto vsdf_cosine  = zero3f;
    auto incoming_pdf = 0.0f;
    if (mis) {
        if (get_random_float(rng) < 0.5f) {
            incoming = sample_vsdf_direction(
                vsdfs, outgoing, get_random_float(rng), get_random_vec2f(rng));
        } else {
            incoming = sample_lights_direction(scene, lights, bvh, position,
                get_random_float(rng), get_random_float(rng),
                get_random_vec2f(rng));
        }
        vsdf_cosine  = evaluate_vsdf_cosine(vsdfs, outgoing, incoming);
        incoming_pdf = 0.5f * sample_vsdf_direction_pdf(
                                  vsdfs, outgoing, incoming) +
                       0.5f * sample_lights_direction_pdf(
                                  scene, lights, bvh, position, incoming);
    } else {
        incoming = sample_vsdf_direction(
            vsdfs, outgoing, get_random_float(rng), get_random_vec2f(rng));
        vsdf_cosine  = evaluate_vsdf_cosine(vsdfs, outgoing, incoming);
        incoming_pdf = sample_vsdf_direction_pdf(vsdfs, outgoing, incoming);
    }
    if (incoming == zero3f || incoming_pdf == 0) {
        return {zero3f, zero3f, 0};
    } else {
        return {vsdf_cosine, incoming, incoming_pdf};
    }
}

tuple<vec3f, float, float> sample_next_volume_distance(const vec3f& position,
    const vec3f& outgoing, float max_distance, const short_vector<vsdf>& vsdfs,
    rng_state& rng) {
    // clamp ray if inside a volume
    auto [distance, channel] = sample_volume_distance(
        vsdfs, get_random_float(rng), get_random_float(rng));

    // compute volume transmission
    distance          = min(distance, max_distance);
    auto pdf          = sample_volume_distance_pdf(vsdfs, distance, channel);
    auto transmission = evaluate_volume_transmission(vsdfs, distance);

    // done
    return {transmission, distance, pdf};
}

// Recursive path tracing.
pair<vec3f, bool> trace_path(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, const trace_image_options& options) {
    // initialize
    auto radiance     = zero3f;
    auto weight       = vec3f{1, 1, 1};
    auto ray          = make_ray(position, direction);
    auto volume_stack = short_vector<short_vector<vsdf>>{};
    auto hit          = false;

    // trace  path
    for (auto bounce = 0; bounce < options.max_bounces; bounce++) {
        if (volume_stack.empty()) {
            // intersect next point
            auto intersection = bvh_intersection{};
            if (!trace_ray(scene, bvh, ray, intersection)) {
                radiance += weight *
                            evaluate_environment_emission(scene, ray.d);
                break;
            }
            hit = true;

            // prepare shading point
            auto outgoing = -ray.d;
            auto point    = trace_point{};
            make_trace_point(point, scene, intersection, ray.d);

            // accumulate emission
            radiance += weight *
                        evaluate_emission(point.esdfs, point.normal, outgoing);
            if (point.bsdfs.empty()) break;

            // russian roulette
            auto [rr_stop, rr_pdf] = sample_russian_roulette(
                get_russian_roulette_albedo(point.bsdfs), weight, bounce,
                get_random_float(rng));
            if (rr_stop) break;
            weight /= rr_pdf;
            if (weight == zero3f) break;

            // continue path
            auto [brdf_cosine, incoming, incoming_pdf] = sample_next_direction(
                scene, lights, bvh, point.position, point.normal, point.bsdfs,
                outgoing, rng, true);

            // exit if no hit
            if (incoming == zero3f || incoming_pdf == 0 ||
                brdf_cosine == zero3f)
                break;
            weight *= brdf_cosine / incoming_pdf;
            if (weight == zero3f) break;

            // update volume_stack
            if (!point.vsdfs.empty() && dot(incoming, point.normal) > 0 !=
                                            dot(outgoing, point.normal) > 0) {
                if (volume_stack.empty())
                    volume_stack.push_back(point.vsdfs);
                else
                    volume_stack.pop_back();
            }

            // setup next iteration
            ray = make_ray(point.position, incoming);
        } else {
#if 0
            // grab vsdfs
            auto& vsdfs = volume_stack.back();

            // clamp ray if inside a volume
            auto [distance_, channel] = sample_volume_distance(
                vsdfs, get_random_float(rng), get_random_float(rng));
            ray.tmax = distance_;

            // intersect next point
            auto intersection = bvh_intersection{};
            auto hit_surface  = trace_ray(scene, bvh, ray, intersection);

            // compute volume transmission
            auto distance = hit_surface ? intersection.distance : ray.tmax;
            weight /= sample_volume_distance_pdf(vsdfs, distance, channel);
            weight *= evaluate_volume_transmission(vsdfs, distance);
#else
            // intersect next point
            auto intersection = bvh_intersection{};
            if (!trace_ray(scene, bvh, ray, intersection)) {
                break;  // bad for now
            }

            // clamp ray if inside a volume
            auto& vsdfs = volume_stack.back();
            auto [transmission, distance, distance_pdf] =
                sample_next_volume_distance(
                    ray.o, -ray.d, intersection.distance, vsdfs, rng);
            weight *= transmission / distance_pdf;

            // check hit
            auto hit_surface = distance >= intersection.distance;
#endif

            if (!hit_surface) {
                // prepare shading point
                auto outgoing = -ray.d;
                auto position = ray.o + ray.d * distance;

                // emission
                radiance += weight * evaluate_emission(vsdfs, outgoing);

                // russian roulette
                auto [rr_stop, rr_pdf] = sample_russian_roulette(
                    get_russian_roulette_albedo(vsdfs), weight, bounce,
                    get_random_float(rng));
                if (rr_stop) break;
                weight /= rr_pdf;
                if (weight == zero3f) break;

                // continue path
                auto [vsdf_cosine, incoming, incoming_pdf] =
                    sample_next_direction_volume(scene, lights, bvh, position,
                        outgoing, vsdfs, rng, true);
                if (incoming == zero3f || incoming_pdf == 0 ||
                    vsdf_cosine == zero3f)
                    break;
                weight *= vsdf_cosine / incoming_pdf;
                if (weight == zero3f) break;

                // setup next iteration
                ray = make_ray(position, incoming);
                continue;
            } else {
                // prepare shading point
                auto outgoing = -ray.d;
                auto point    = trace_point{};
                make_trace_point(point, scene, intersection, ray.d);

                // accumulate emission
                radiance += weight * evaluate_emission(
                                         point.esdfs, point.normal, outgoing);
                if (point.bsdfs.empty()) break;

                // russian roulette
                auto [rr_stop, rr_pdf] = sample_russian_roulette(
                    get_russian_roulette_albedo(point.bsdfs), weight, bounce,
                    get_random_float(rng));
                if (rr_stop) break;
                weight /= rr_pdf;
                if (weight == zero3f) break;

                // continue path
                auto [brdf_cosine, incoming, incoming_pdf] =
                    sample_next_direction(scene, lights, bvh, point.position,
                        point.normal, point.bsdfs, outgoing, rng, true);

                // exit if no hit
                if (incoming == zero3f || incoming_pdf == 0 ||
                    brdf_cosine == zero3f)
                    break;
                weight *= brdf_cosine / incoming_pdf;
                if (weight == zero3f) break;

                // update volume_stack
                if (!point.vsdfs.empty() &&
                    dot(incoming, point.normal) > 0 !=
                        dot(outgoing, point.normal) > 0) {
                    if (volume_stack.empty())
                        volume_stack.push_back(point.vsdfs);
                    else
                        volume_stack.pop_back();
                }

                // setup next iteration
                ray = make_ray(point.position, incoming);
            }
        }
    }

    return {radiance, hit};
}

// Recursive path tracing.
pair<vec3f, bool> trace_naive(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction,
    rng_state& rng, const trace_image_options& options) {
    // initialize
    auto radiance = zero3f;
    auto weight   = vec3f{1, 1, 1};
    auto ray      = make_ray(position, direction);
    auto hit      = false;

    // trace  path
    for (auto bounce = 0; bounce < options.max_bounces; bounce++) {
        // intersect next point
        auto intersection = bvh_intersection{};
        if (!trace_ray(scene, bvh, ray, intersection)) {
            radiance += weight * evaluate_environment_emission(scene, ray.d);
            break;
        }
        hit = true;

        // prepare shading point
        auto outgoing = -ray.d;
        auto point    = trace_point{};
        make_trace_point(point, scene, intersection, ray.d);

        // accumulate emission
        radiance += weight *
                    evaluate_emission(point.esdfs, point.normal, outgoing);
        if (point.bsdfs.empty()) break;

        // russian roulette
        auto [rr_stop, rr_pdf] = sample_russian_roulette(
            get_russian_roulette_albedo(point.bsdfs), weight, bounce,
            get_random_float(rng));
        if (rr_stop) break;
        weight /= rr_pdf;
        if (weight == zero3f) break;

        // continue path
        auto [brdf_cosine, incoming, incoming_pdf] = sample_next_direction(
            scene, lights, bvh, point.position, point.normal, point.bsdfs,
            outgoing, rng, false);

        // exit if no hit
        if (incoming == zero3f || incoming_pdf == 0 || brdf_cosine == zero3f)
            break;
        weight *= brdf_cosine / incoming_pdf;
        if (weight == zero3f) break;

        // setup next iteration
        ray = make_ray(point.position, incoming);
    }

    return {radiance, hit};
}

// Eyelight for quick previewing.
pair<vec3f, bool> trace_eyelight(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const vec3f& position_, const vec3f& direction_,
    rng_state& rng, const trace_image_options& options) {
    // initialize
    auto radiance  = zero3f;
    auto weight    = vec3f{1, 1, 1};
    auto position  = position_;
    auto direction = direction_;
    auto hit       = false;

    // trace  path
    for (auto bounce = 0; bounce < max(options.max_bounces, 4); bounce++) {
        // intersect next point
        auto intersection = bvh_intersection{};
        if (!trace_ray(scene, bvh, position, direction, intersection)) {
            radiance += weight *
                        evaluate_environment_emission(scene, direction);
            break;
        }
        hit = true;

        // prepare shading point
        auto outgoing = -direction;
        auto point    = trace_point{};
        make_trace_point(point, scene, intersection, direction);

        // accumulate emission
        radiance += weight *
                    evaluate_emission(point.esdfs, point.normal, outgoing);
        if (point.bsdfs.empty()) break;

        // brdf * light
        radiance += weight *
                    evaluate_brdf_cosine(
                        point.bsdfs, point.normal, outgoing, outgoing, false) *
                    pif;

        // exit if needed
        if (weight == zero3f) break;

        // continue path
        auto [brdf_cosine, incoming, incoming_pdf] = sample_next_direction(
            scene, lights, bvh, point.position, point.normal, point.bsdfs,
            outgoing, rng, false, false, true);

        // exit if no hit
        if (incoming == zero3f || incoming_pdf == 0 || brdf_cosine == zero3f)
            break;
        weight *= brdf_cosine / incoming_pdf;
        if (weight == zero3f) break;

        // setup next iteration
        position  = point.position;
        direction = incoming;
    }

    return {radiance, hit};
}

// False color rendering
pair<vec3f, bool> trace_falsecolor(const yocto_scene& scene,
    const bvh_scene& bvh, const trace_lights& lights, const vec3f& position,
    const vec3f& direction, rng_state& rng,
    const trace_image_options& options) {
    // intersect next point
    auto intersection = bvh_intersection{};
    if (!trace_ray(scene, bvh, position, direction, intersection)) {
        return {zero3f, false};
    }

    // get scene elements
    auto& instance = scene.instances[intersection.instance_id];
    auto& shape    = scene.shapes[instance.shape];
    // auto& material = scene.materials[instance.material];

    // prepare shading point
    auto point = trace_point{};
    make_trace_point(point, scene, intersection, direction);

    switch (options.falsecolor_type) {
        case trace_falsecolor_type::normal: {
            return {point.normal * 0.5f + 0.5f, 1};
        }
        case trace_falsecolor_type::frontfacing: {
            auto outgoing    = -direction;
            auto frontfacing = dot(point.normal, outgoing) > 0 ? vec3f{0, 1, 0}
                                                               : vec3f{1, 0, 0};
            return {frontfacing, 1};
        }
        case trace_falsecolor_type::gnormal: {
            auto normal = evaluate_instance_element_normal(
                scene, instance, intersection.element_id, true);
            return {normal * 0.5f + 0.5f, 1};
        }
        case trace_falsecolor_type::gfrontfacing: {
            auto normal = evaluate_instance_element_normal(
                scene, instance, intersection.element_id, true);
            auto outgoing    = -direction;
            auto frontfacing = dot(normal, outgoing) > 0 ? vec3f{0, 1, 0}
                                                         : vec3f{1, 0, 0};
            return {frontfacing, 1};
        }
        case trace_falsecolor_type::albedo: {
            auto albedo = zero3f;
            for (auto& brdf : point.bsdfs) albedo += brdf.albedo;
            return {albedo, 1};
        }
        case trace_falsecolor_type::texcoord: {
            auto texturecoord = evaluate_shape_texturecoord(
                shape, intersection.element_id, intersection.element_uv);
            return {{texturecoord.x, texturecoord.y, 0}, 1};
        }
        case trace_falsecolor_type::color: {
            auto color = evaluate_shape_color(
                shape, intersection.element_id, intersection.element_uv);
            return {xyz(color), 1};
        }
        case trace_falsecolor_type::emission: {
            auto emission = zero3f;
            for (auto& esdf : point.esdfs) emission += esdf.albedo;
            return {emission, 1};
        }
        case trace_falsecolor_type::diffuse: {
            auto diffuse = zero3f;
            for (auto& brdf : point.bsdfs)
                if (brdf.type == bsdf_type::diffuse_reflection)
                    diffuse += brdf.albedo;
            return {diffuse, 1};
        }
        case trace_falsecolor_type::specular: {
            auto specular = zero3f;
            for (auto& brdf : point.bsdfs)
                if (brdf.type == bsdf_type::specular_reflection)
                    specular += brdf.albedo;
            return {specular, 1};
        }
        case trace_falsecolor_type::transmission: {
            auto transmission = zero3f;
            for (auto& brdf : point.bsdfs)
                if (brdf.type == bsdf_type::specular_transmission)
                    transmission += brdf.albedo;
            return {transmission, 1};
        }
        case trace_falsecolor_type::roughness: {
            auto roughness = 0.f;
            for (auto& brdf : point.bsdfs)
                roughness = min(roughness, brdf.roughness);
            return {vec3f{roughness}, 1};
        }
        case trace_falsecolor_type::material: {
            auto hashed = std::hash<int>()(instance.material);
            auto rng_   = make_rng(trace_default_seed, hashed);
            return {pow(0.5f + 0.5f * get_random_vec3f(rng_), 2.2f), 1};
        }
        case trace_falsecolor_type::shape: {
            auto hashed = std::hash<int>()(instance.shape);
            auto rng_   = make_rng(trace_default_seed, hashed);
            return {pow(0.5f + 0.5f * get_random_vec3f(rng_), 2.2f), 1};
        }
        case trace_falsecolor_type::instance: {
            auto hashed = std::hash<int>()(intersection.instance_id);
            auto rng_   = make_rng(trace_default_seed, hashed);
            return {pow(0.5f + 0.5f * get_random_vec3f(rng_), 2.2f), 1};
        }
        case trace_falsecolor_type::highlight: {
            auto emission = zero3f;
            for (auto& esdf : point.esdfs) emission += esdf.albedo;
            auto outgoing = -direction;
            if (emission == zero3f) emission = {0.2f, 0.2f, 0.2f};
            return {emission * abs(dot(outgoing, point.normal)), 1};
        }
        default: {
            return {zero3f, false};
        }
    }
}

// Trace a single ray from the camera using the given algorithm.
using trace_sampler_func = pair<vec3f, bool> (*)(const yocto_scene& scene,
    const bvh_scene& bvh, const trace_lights& lights, const vec3f& position,
    const vec3f& direction, rng_state& rng, const trace_image_options& options);
trace_sampler_func get_trace_sampler_func(const trace_image_options& options) {
    switch (options.sampler_type) {
        case trace_sampler_type::path: return trace_path;
        case trace_sampler_type::naive: return trace_naive;
        case trace_sampler_type::eyelight: return trace_eyelight;
        case trace_sampler_type::falsecolor: return trace_falsecolor;
        default: {
            throw runtime_error("sampler unknown");
            return nullptr;
        }
    }
}

// Check is a sampler requires lights
bool is_trace_sampler_lit(const trace_image_options& options) {
    switch (options.sampler_type) {
        case trace_sampler_type::path: return true;
        case trace_sampler_type::naive: return true;
        case trace_sampler_type::eyelight: return true;
        case trace_sampler_type::falsecolor: return true;
        default: {
            throw runtime_error("sampler unknown");
            return false;
        }
    }
}

// Get trace pixel
trace_pixel& get_trace_pixel(trace_state& state, int i, int j) {
    return state.pixels[j * state.image_size.x + i];
}

// Trace a block of samples
void trace_image_region(image<vec4f>& image, trace_state& state,
    const yocto_scene& scene, const bvh_scene& bvh, const trace_lights& lights,
    const image_region& region, int num_samples,
    const trace_image_options& options) {
    auto& camera  = scene.cameras.at(options.camera_id);
    auto  sampler = get_trace_sampler_func(options);
    for (auto j = region.min.y; j < region.max.y; j++) {
        for (auto i = region.min.x; i < region.max.x; i++) {
            auto& pixel = get_trace_pixel(state, i, j);
            for (auto s = 0; s < num_samples; s++) {
                if (options.cancel_flag && *options.cancel_flag) return;
                _trace_npaths += 1;
                auto ray = sample_camera_ray(camera, {i, j}, image.size(),
                    get_random_vec2f(pixel.rng), get_random_vec2f(pixel.rng));
                auto [radiance, hit] = sampler(
                    scene, bvh, lights, ray.o, ray.d, pixel.rng, options);
                if (!hit) {
                    if (options.environments_hidden ||
                        scene.environments.empty()) {
                        radiance = zero3f;
                        hit      = false;
                    } else {
                        hit = true;
                    }
                }
                if (!isfinite(radiance)) {
                    // printf("NaN detected\n");
                    radiance = zero3f;
                }
                if (max(radiance) > options.pixel_clamp)
                    radiance = radiance * (options.pixel_clamp / max(radiance));
                pixel.radiance += radiance;
                pixel.hits += hit ? 1 : 0;
                pixel.samples += 1;
            }
            auto radiance = pixel.hits ? pixel.radiance / pixel.hits : zero3f;
            auto coverage = (float)pixel.hits / (float)pixel.samples;
            image[{i, j}] = {radiance.x, radiance.y, radiance.z, coverage};
        }
    }
}

// Init a sequence of random number generators.
void init_trace_state(
    trace_state& state, const vec2i& image_size, uint64_t seed) {
    state    = trace_state{image_size,
        vector<trace_pixel>(image_size.x * image_size.y, trace_pixel{})};
    auto rng = make_rng(1301081);
    for (auto j = 0; j < state.image_size.y; j++) {
        for (auto i = 0; i < state.image_size.x; i++) {
            auto& pixel = get_trace_pixel(state, i, j);
            pixel.rng   = make_rng(seed, get_random_int(rng, 1 << 31) / 2 + 1);
        }
    }
}

// Init trace lights
void init_trace_lights(trace_lights& lights, const yocto_scene& scene) {
    lights = {};

    lights.shape_elements_cdf.resize(scene.shapes.size());
    lights.environment_texture_cdf.resize(scene.textures.size());

    for (auto instance_id = 0; instance_id < scene.instances.size();
         instance_id++) {
        auto& instance = scene.instances[instance_id];
        auto& shape    = scene.shapes[instance.shape];
        auto& material = scene.materials[instance.material];
        if (material.emission == zero3f) continue;
        if (shape.triangles.empty() && shape.quads.empty()) continue;
        lights.instances.push_back(instance_id);
        compute_shape_elements_cdf(
            shape, lights.shape_elements_cdf[instance.shape]);
    }

    for (auto environment_id = 0; environment_id < scene.environments.size();
         environment_id++) {
        auto& environment = scene.environments[environment_id];
        if (environment.emission == zero3f) continue;
        lights.environments.push_back(environment_id);
        if (environment.emission_texture >= 0) {
            compute_environment_texels_cdf(scene, environment,
                lights.environment_texture_cdf[environment.emission_texture]);
        }
    }
}

// Progressively compute an image by calling trace_samples multiple times.
image<vec4f> trace_image(const yocto_scene& scene, const bvh_scene& bvh,
    const trace_lights& lights, const trace_image_options& options) {
    auto image_size = get_camera_image_size(
        scene.cameras.at(options.camera_id), options.image_size);
    auto image = yocto::image{image_size, zero4f};
    auto state = trace_state{};
    init_trace_state(state, image_size, options.random_seed);
    auto regions = vector<image_region>{};
    make_image_regions(regions, image.size(), options.region_size, true);

    parallel_foreach(regions, [&image, &state, &scene, &bvh, &lights, &options](
                                  const image_region& region) {
        trace_image_region(image, state, scene, bvh, lights, region,
            options.num_samples, options);
    });

    return image;
}

// Progressively compute an image by calling trace_samples multiple times.
int trace_image_samples(image<vec4f>& image, trace_state& state,
    const yocto_scene& scene, const bvh_scene& bvh, const trace_lights& lights,
    int current_sample, const trace_image_options& options) {
    auto regions = vector<image_region>{};
    make_image_regions(regions, image.size(), options.region_size, true);
    auto num_samples = min(
        options.samples_per_batch, options.num_samples - current_sample);
    parallel_foreach(
        regions, [&image, &state, &scene, &bvh, &lights, num_samples, &options](
                     const image_region& region) {
            trace_image_region(
                image, state, scene, bvh, lights, region, num_samples, options);
        });
    return current_sample + num_samples;
}

// Starts an anyncrhounous renderer.
void trace_image_async_start(image<vec4f>& image, trace_state& state,
    const yocto_scene& scene, const bvh_scene& bvh, const trace_lights& lights,
    vector<future<void>>& futures, atomic<int>& current_sample,
    concurrent_queue<image_region>& queue, const trace_image_options& options) {
    auto& camera     = scene.cameras.at(options.camera_id);
    auto  image_size = get_camera_image_size(camera, options.image_size);
    image            = {image_size, zero4f};
    state            = trace_state{};
    init_trace_state(state, image_size, options.random_seed);
    auto regions = vector<image_region>{};
    make_image_regions(regions, image.size(), options.region_size, true);
    if (options.cancel_flag) *options.cancel_flag = false;

    futures.clear();
    futures.emplace_back(async([options, regions, &current_sample, &image,
                                   &scene, &lights, &bvh, &state, &queue]() {
        for (auto sample = 0; sample < options.num_samples;
             sample += options.samples_per_batch) {
            if (options.cancel_flag && *options.cancel_flag) return;
            current_sample   = sample;
            auto num_samples = min(options.samples_per_batch,
                options.num_samples - current_sample);
            parallel_foreach(
                regions,
                [num_samples, &options, &image, &scene, &lights, &bvh, &state,
                    &queue](const image_region& region) {
                    trace_image_region(image, state, scene, bvh, lights, region,
                        num_samples, options);
                    queue.push(region);
                },
                options.cancel_flag, options.run_serially);
        }
        current_sample = options.num_samples;
    }));
}

// Stop the asynchronous renderer.
void trace_image_async_stop(vector<future<void>>& futures,
    concurrent_queue<image_region>& queue, const trace_image_options& options) {
    if (options.cancel_flag) *options.cancel_flag = true;
    for (auto& f : futures) f.get();
    futures.clear();
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

}  // namespace yocto
// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING SUPPORT FUNCTION
// -----------------------------------------------------------------------------
namespace yocto {

// Phong exponent to roughness.
float convert_specular_exponent_to_roughness(float exponent) {
    return sqrtf(2 / (exponent + 2));
}

// Specular to fresnel eta.
void compute_fresnel_from_specular(
    const vec3f& specular, vec3f& es, vec3f& esk) {
    es  = {(1 + sqrt(specular.x)) / (1 - sqrt(specular.x)),
        (1 + sqrt(specular.y)) / (1 - sqrt(specular.y)),
        (1 + sqrt(specular.z)) / (1 - sqrt(specular.z))};
    esk = {0, 0, 0};
}

// Specular to  eta.
float convert_specular_to_eta(const vec3f& specular) {
    auto f0 = (specular.x + specular.y + specular.z) / 3;
    return (1 + sqrt(f0)) / (1 - sqrt(f0));
}

// Compute the fresnel term for dielectrics. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3f evaluate_fresnel_dielectric(float cosw, const vec3f& eta_) {
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

    auto roughness = (vec3f{cosw, cosw, cosw} - t1) /
                     (vec3f{cosw, cosw, cosw} + t1);
    auto rp = (t0 - t2) / (t0 + t2);

    return (roughness * roughness + rp * rp) / 2.0f;
}

// Compute the fresnel term for metals. Implementation from
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3f evaluate_fresnel_metal(float cosw, const vec3f& eta, const vec3f& etak) {
    if (etak == zero3f) return evaluate_fresnel_dielectric(cosw, eta);

    cosw       = clamp(cosw, (float)-1, (float)1);
    auto cos2  = cosw * cosw;
    auto sin2  = clamp(1 - cos2, (float)0, (float)1);
    auto eta2  = eta * eta;
    auto etak2 = etak * etak;

    auto t0         = eta2 - etak2 - vec3f{sin2, sin2, sin2};
    auto a2plusb2_2 = t0 * t0 + 4.0f * eta2 * etak2;
    auto a2plusb2   = vec3f{
        sqrt(a2plusb2_2.x), sqrt(a2plusb2_2.y), sqrt(a2plusb2_2.z)};
    auto t1        = a2plusb2 + vec3f{cos2, cos2, cos2};
    auto a_2       = (a2plusb2 + t0) / 2.0f;
    auto a         = vec3f{sqrt(a_2.x), sqrt(a_2.y), sqrt(a_2.z)};
    auto t2        = 2.0f * a * cosw;
    auto roughness = (t1 - t2) / (t1 + t2);

    auto t3 = vec3f{cos2, cos2, cos2} * a2plusb2 +
              vec3f{sin2, sin2, sin2} * vec3f{sin2, sin2, sin2};
    auto t4 = t2 * sin2;
    auto rp = roughness * (t3 - t4) / (t3 + t4);

    return (rp + roughness) / 2.0f;
}

// Tabulated ior for metals
// https://github.com/tunabrain/tungsten
const unordered_map<string, pair<vec3f, vec3f>> metal_ior_table = {
    {"a-C", {{2.9440999183f, 2.2271502925f, 1.9681668794f},
                {0.8874329109f, 0.7993216383f, 0.8152862927f}}},
    {"Ag", {{0.1552646489f, 0.1167232965f, 0.1383806959f},
               {4.8283433224f, 3.1222459278f, 2.1469504455f}}},
    {"Al", {{1.6574599595f, 0.8803689579f, 0.5212287346f},
               {9.2238691996f, 6.2695232477f, 4.8370012281f}}},
    {"AlAs", {{3.6051023902f, 3.2329365777f, 2.2175611545f},
                 {0.0006670247f, -0.0004999400f, 0.0074261204f}}},
    {"AlSb", {{-0.0485225705f, 4.1427547893f, 4.6697691348f},
                 {-0.0363741915f, 0.0937665154f, 1.3007390124f}}},
    {"Au", {{0.1431189557f, 0.3749570432f, 1.4424785571f},
               {3.9831604247f, 2.3857207478f, 1.6032152899f}}},
    {"Be", {{4.1850592788f, 3.1850604423f, 2.7840913457f},
               {3.8354398268f, 3.0101260162f, 2.8690088743f}}},
    {"Cr", {{4.3696828663f, 2.9167024892f, 1.6547005413f},
               {5.2064337956f, 4.2313645277f, 3.7549467933f}}},
    {"CsI", {{2.1449030413f, 1.7023164587f, 1.6624194173f},
                {0.0000000000f, 0.0000000000f, 0.0000000000f}}},
    {"Cu", {{0.2004376970f, 0.9240334304f, 1.1022119527f},
               {3.9129485033f, 2.4528477015f, 2.1421879552f}}},
    {"Cu2O", {{3.5492833755f, 2.9520622449f, 2.7369202137f},
                 {0.1132179294f, 0.1946659670f, 0.6001681264f}}},
    {"CuO", {{3.2453822204f, 2.4496293965f, 2.1974114493f},
                {0.5202739621f, 0.5707372756f, 0.7172250613f}}},
    {"d-C", {{2.7112524747f, 2.3185812849f, 2.2288565009f},
                {0.0000000000f, 0.0000000000f, 0.0000000000f}}},
    {"Hg", {{2.3989314904f, 1.4400254917f, 0.9095512090f},
               {6.3276269444f, 4.3719414152f, 3.4217899270f}}},
    {"HgTe", {{4.7795267752f, 3.2309984581f, 2.6600252401f},
                 {1.6319827058f, 1.5808189339f, 1.7295753852f}}},
    {"Ir", {{3.0864098394f, 2.0821938440f, 1.6178866805f},
               {5.5921510077f, 4.0671757150f, 3.2672611269f}}},
    {"K", {{0.0640493070f, 0.0464100621f, 0.0381842017f},
              {2.1042155920f, 1.3489364357f, 0.9132113889f}}},
    {"Li", {{0.2657871942f, 0.1956102432f, 0.2209198538f},
               {3.5401743407f, 2.3111306542f, 1.6685930000f}}},
    {"MgO", {{2.0895885542f, 1.6507224525f, 1.5948759692f},
                {0.0000000000f, -0.0000000000f, 0.0000000000f}}},
    {"Mo", {{4.4837010280f, 3.5254578255f, 2.7760769438f},
               {4.1111307988f, 3.4208716252f, 3.1506031404f}}},
    {"Na", {{0.0602665320f, 0.0561412435f, 0.0619909494f},
               {3.1792906496f, 2.1124800781f, 1.5790940266f}}},
    {"Nb", {{3.4201353595f, 2.7901921379f, 2.3955856658f},
               {3.4413817900f, 2.7376437930f, 2.5799132708f}}},
    {"Ni", {{2.3672753521f, 1.6633583302f, 1.4670554172f},
               {4.4988329911f, 3.0501643957f, 2.3454274399f}}},
    {"Rh", {{2.5857954933f, 1.8601866068f, 1.5544279524f},
               {6.7822927110f, 4.7029501026f, 3.9760892461f}}},
    {"Se-e", {{5.7242724833f, 4.1653992967f, 4.0816099264f},
                 {0.8713747439f, 1.1052845009f, 1.5647788766f}}},
    {"Se", {{4.0592611085f, 2.8426947380f, 2.8207582835f},
               {0.7543791750f, 0.6385150558f, 0.5215872029f}}},
    {"SiC", {{3.1723450205f, 2.5259677964f, 2.4793623897f},
                {0.0000007284f, -0.0000006859f, 0.0000100150f}}},
    {"SnTe", {{4.5251865890f, 1.9811525984f, 1.2816819226f},
                 {0.0000000000f, 0.0000000000f, 0.0000000000f}}},
    {"Ta", {{2.0625846607f, 2.3930915569f, 2.6280684948f},
               {2.4080467973f, 1.7413705864f, 1.9470377016f}}},
    {"Te-e", {{7.5090397678f, 4.2964603080f, 2.3698732430f},
                 {5.5842076830f, 4.9476231084f, 3.9975145063f}}},
    {"Te", {{7.3908396088f, 4.4821028985f, 2.6370708478f},
               {3.2561412892f, 3.5273908133f, 3.2921683116f}}},
    {"ThF4", {{1.8307187117f, 1.4422274283f, 1.3876488528f},
                 {0.0000000000f, 0.0000000000f, 0.0000000000f}}},
    {"TiC", {{3.7004673762f, 2.8374356509f, 2.5823030278f},
                {3.2656905818f, 2.3515586388f, 2.1727857800f}}},
    {"TiN", {{1.6484691607f, 1.1504482522f, 1.3797795097f},
                {3.3684596226f, 1.9434888540f, 1.1020123347f}}},
    {"TiO2-e", {{3.1065574823f, 2.5131551146f, 2.5823844157f},
                   {0.0000289537f, -0.0000251484f, 0.0001775555f}}},
    {"TiO2", {{3.4566203131f, 2.8017076558f, 2.9051485020f},
                 {0.0001026662f, -0.0000897534f, 0.0006356902f}}},
    {"VC", {{3.6575665991f, 2.7527298065f, 2.5326814570f},
               {3.0683516659f, 2.1986687713f, 1.9631816252f}}},
    {"VN", {{2.8656011588f, 2.1191817791f, 1.9400767149f},
               {3.0323264950f, 2.0561075580f, 1.6162930914f}}},
    {"V", {{4.2775126218f, 3.5131538236f, 2.7611257461f},
              {3.4911844504f, 2.8893580874f, 3.1116965117f}}},
    {"W", {{4.3707029924f, 3.3002972445f, 2.9982666528f},
              {3.5006778591f, 2.6048652781f, 2.2731930614f}}},
};

// Get a complex ior table with keys the metal name and values (eta, etak)
bool get_metal_ior(const string& name, vec3f& eta, vec3f& etak) {
    if (metal_ior_table.find(name) == metal_ior_table.end()) return false;
    auto value = metal_ior_table.at(name);
    eta        = value.first;
    etak       = value.second;
    return true;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// NUMERICAL TESTS FOR MONTE CARLO INTEGRATION
// -----------------------------------------------------------------------------
namespace yocto {

template <typename Func>
float integrate_func_base(
    const Func& f, float a, float b, int nsamples, rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = get_random_float(rng);
        auto x = a + r * (b - a);
        integral += f(x) * (b - a);
    }
    integral /= nsamples;
    return integral;
}

template <typename Func>
float integrate_func_stratified(
    const Func& f, float a, float b, int nsamples, rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = (i + get_random_float(rng)) / nsamples;
        auto x = a + r * (b - a);
        integral += f(x) * (b - a);
    }
    integral /= nsamples;
    return integral;
}

template <typename Func>
float integrate_func_importance(const Func& f, const Func& pdf,
    const Func& warp, int nsamples, rng_state& rng) {
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
template <typename Func>
void print_integrate_func_test(const Func& f, float a, float b, float expected,
    int nsamples, const Func& pdf, const Func& warp) {
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

template <typename Func>
float integrate_func2_base(
    const Func& f, vec2f a, vec2f b, int nsamples, rng_state& rng) {
    auto integral = 0.0f;
    for (auto i = 0; i < nsamples; i++) {
        auto r = get_random_vec2f(rng);
        auto x = a + r * (b - a);
        integral += f(x) * (b.x - a.x) * (b.y - a.y);
    }
    integral /= nsamples;
    return integral;
}

template <typename Func>
float integrate_func2_stratified(
    const Func& f, vec2f a, vec2f b, int nsamples, rng_state& rng) {
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

template <typename Func, typename Func2>
float integrate_func2_importance(const Func& f, const Func& pdf,
    const Func2& warp, int nsamples, rng_state& rng) {
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
template <typename Func, typename Func2>
void print_integrate_func2_test(const Func& f, vec2f a, vec2f b, float expected,
    int nsamples, const Func& pdf, const Func2& warp) {
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

}  // namespace yocto
