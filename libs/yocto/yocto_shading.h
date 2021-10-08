//
// # Yocto/Shading: Shading routines
//
// Yocto/Shading defines shading and sampling functions useful to write path
// tracing algorithms. Yocto/Shading is implemented in `yocto_shading.h`.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
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

#ifndef _YOCTO_SHADING_H_
#define _YOCTO_SHADING_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "yocto_compensation.h"
#include "yocto_math.h"
#include "yocto_sampling.h"

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

// using directives
using std::pair;
using std::string;
using std::vector;

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHADING FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Check if on the same side of the hemisphere
inline bool same_hemisphere(
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Schlick approximation of the Fresnel term.
inline vec3f fresnel_schlick(
    const vec3f& specular, const vec3f& normal, const vec3f& outgoing);
// Compute the fresnel term for dielectrics.
inline float fresnel_dielectric(
    float eta, const vec3f& normal, const vec3f& outgoing);
// Compute the fresnel term for metals.
inline vec3f fresnel_conductor(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing);

// Convert eta to reflectivity
inline vec3f eta_to_reflectivity(const vec3f& eta);
// Convert reflectivity to  eta.
inline vec3f reflectivity_to_eta(const vec3f& reflectivity);
// Convert conductor eta to reflectivity.
inline vec3f eta_to_reflectivity(const vec3f& eta, const vec3f& etak);
// Convert eta to edge tint parametrization.
inline pair<vec3f, vec3f> eta_to_edgetint(const vec3f& eta, const vec3f& etak);
// Convert reflectivity and edge tint to eta.
inline pair<vec3f, vec3f> edgetint_to_eta(
    const vec3f& reflectivity, const vec3f& edgetint);

// Get tabulated ior for conductors
inline pair<vec3f, vec3f> conductor_eta(const string& name);

// Evaluates the microfacet distribution.
inline float microfacet_distribution(float roughness, const vec3f& normal,
    const vec3f& halfway, bool ggx = true);
// Evaluates the microfacet shadowing.
inline float microfacet_shadowing(float roughness, const vec3f& normal,
    const vec3f& halfway, const vec3f& outgoing, const vec3f& incoming,
    bool ggx = true);

// Samples a microfacet distribution.
inline vec3f sample_microfacet(
    float roughness, const vec3f& normal, const vec2f& rn, bool ggx = true);
// Pdf for microfacet distribution sampling.
inline float sample_microfacet_pdf(float roughness, const vec3f& normal,
    const vec3f& halfway, bool ggx = true);

// Samples a microfacet distribution with the distribution of visible normals.
inline vec3f sample_microfacet(float roughness, const vec3f& normal,
    const vec3f& outgoing, const vec2f& rn, bool ggx = true);
// Pdf for microfacet distribution sampling with the distribution of visible
// normals.
inline float sample_microfacet_pdf(float roughness, const vec3f& normal,
    const vec3f& halfway, const vec3f& outgoing, bool ggx = true);

// Microfacet energy compensation (E(cos(w)))
inline float microfacet_cosintegral(
    float roughness, const vec3f& normal, const vec3f& outgoing);
// Approximate microfacet compensation for metals with Schlick's Fresnel
inline vec3f microfacet_compensation(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing);

// Evaluates a diffuse BRDF lobe.
inline vec3f eval_matte(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);
// Sample a diffuse BRDF lobe.
inline vec3f sample_matte(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec2f& rn);
// Pdf for diffuse BRDF lobe sampling.
inline float sample_matte_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);

// Evaluates a specular BRDF lobe.
inline vec3f eval_glossy(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a specular BRDF lobe.
inline vec3f sample_glossy(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec2f& rn);
// Pdf for specular BRDF lobe sampling.
inline float sample_glossy_pdf(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Evaluates a metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec2f& rn);
// Pdf for metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Evaluate a delta metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);
// Sample a delta metal BRDF lobe.
inline vec3f sample_metallic(
    const vec3f& color, const vec3f& normal, const vec3f& outgoing);
// Pdf for delta metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);

// Evaluate a delta metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a delta metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing);
// Pdf for delta metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Evaluate a delta metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a delta metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing);
// Pdf for delta metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Evaluates a specular BRDF lobe.
inline vec3f eval_gltfpbr(const vec3f& color, float ior, float roughness,
    float metallic, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming);
// Sample a specular BRDF lobe.
inline vec3f sample_gltfpbr(const vec3f& color, float ior, float roughness,
    float metallic, const vec3f& normal, const vec3f& outgoing, float rnl,
    const vec2f& rn);
// Pdf for specular BRDF lobe sampling.
inline float sample_gltfpbr_pdf(const vec3f& color, float ior, float roughness,
    float metallic, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming);

// Evaluates a transmission BRDF lobe.
inline vec3f eval_transparent(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a transmission BRDF lobe.
inline vec3f sample_transparent(float ior, float roughness, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn);
// Pdf for transmission BRDF lobe sampling.
inline float sample_tranparent_pdf(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming);

// Evaluate a delta transmission BRDF lobe.
inline vec3f eval_transparent(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a delta transmission BRDF lobe.
inline vec3f sample_transparent(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, float rnl);
// Pdf for delta transmission BRDF lobe sampling.
inline float sample_tranparent_pdf(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Evaluates a refraction BRDF lobe.
inline vec3f eval_refractive(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);
// Sample a refraction BRDF lobe.
inline vec3f sample_refractive(float ior, float roughness, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn);
// Pdf for refraction BRDF lobe sampling.
inline float sample_refractive_pdf(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming);

// Evaluate a delta refraction BRDF lobe.
inline vec3f eval_refractive(const vec3f& color, float ior, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);
// Sample a delta refraction BRDF lobe.
inline vec3f sample_refractive(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, float rnl);
// Pdf for delta refraction BRDF lobe sampling.
inline float sample_refractive_pdf(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming);

// Evaluate a translucent BRDF lobe.
inline vec3f eval_translucent(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);
// Pdf for translucency BRDF lobe sampling.
inline float sample_translucent_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);
// Sample a translucency BRDF lobe.
inline vec3f sample_translucent(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec2f& rn);

// Evaluate a passthrough BRDF lobe.
inline vec3f eval_passthrough(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);
// Sample a passthrough BRDF lobe.
inline vec3f sample_passthrough(
    const vec3f& color, const vec3f& normal, const vec3f& outgoing);
// Pdf for passthrough BRDF lobe sampling.
inline float sample_passthrough_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming);

// Convert mean-free-path to transmission
inline vec3f mfp_to_transmission(const vec3f& mfp, float depth);

// Evaluate transmittance
inline vec3f eval_transmittance(const vec3f& density, float distance);
// Sample a distance proportionally to transmittance
inline float sample_transmittance(
    const vec3f& density, float max_distance, float rl, float rd);
// Pdf for distance sampling
inline float sample_transmittance_pdf(
    const vec3f& density, float distance, float max_distance);

// Evaluate phase function
inline float eval_phasefunction(
    float anisotropy, const vec3f& outgoing, const vec3f& incoming);
// Sample phase function
inline vec3f sample_phasefunction(
    float anisotropy, const vec3f& outgoing, const vec2f& rn);
// Pdf for phase function sampling
inline float sample_phasefunction_pdf(
    float anisotropy, const vec3f& outgoing, const vec3f& incoming);

}  // namespace yocto

// -----------------------------------------------------------------------------
//
//
// IMPLEMENTATION
//
//
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SHADING FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto {

// Check if on the same side of the hemisphere
inline bool same_hemisphere(
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  return dot(normal, outgoing) * dot(normal, incoming) >= 0;
}

// Schlick approximation of the Fresnel term
inline vec3f fresnel_schlick(
    const vec3f& specular, const vec3f& normal, const vec3f& outgoing) {
  if (specular == zero3f) return zero3f;
  auto cosine = dot(normal, outgoing);
  return specular +
         (1 - specular) * pow(clamp(1 - abs(cosine), 0.0f, 1.0f), 5.0f);
}

// Compute the fresnel term for dielectrics.
inline float fresnel_dielectric(
    float eta, const vec3f& normal, const vec3f& outgoing) {
  // Implementation from
  // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
  auto cosw = abs(dot(normal, outgoing));

  auto sin2 = 1 - cosw * cosw;
  auto eta2 = eta * eta;

  auto cos2t = 1 - sin2 / eta2;
  if (cos2t < 0) return 1;  // tir

  auto t0 = sqrt(cos2t);
  auto t1 = eta * t0;
  auto t2 = eta * cosw;

  auto rs = (cosw - t1) / (cosw + t1);
  auto rp = (t0 - t2) / (t0 + t2);

  return (rs * rs + rp * rp) / 2;
}

// Compute the fresnel term for metals.
inline vec3f fresnel_conductor(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing) {
  // Implementation from
  // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
  auto cosw = dot(normal, outgoing);
  if (cosw <= 0) return zero3f;

  cosw       = clamp(cosw, (float)-1, (float)1);
  auto cos2  = cosw * cosw;
  auto sin2  = clamp(1 - cos2, (float)0, (float)1);
  auto eta2  = eta * eta;
  auto etak2 = etak * etak;

  auto t0       = eta2 - etak2 - sin2;
  auto a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
  auto t1       = a2plusb2 + cos2;
  auto a        = sqrt((a2plusb2 + t0) / 2);
  auto t2       = 2 * a * cosw;
  auto rs       = (t1 - t2) / (t1 + t2);

  auto t3 = cos2 * a2plusb2 + sin2 * sin2;
  auto t4 = t2 * sin2;
  auto rp = rs * (t3 - t4) / (t3 + t4);

  return (rp + rs) / 2;
}

// Convert eta to reflectivity
inline vec3f eta_to_reflectivity(const vec3f& eta) {
  return ((eta - 1) * (eta - 1)) / ((eta + 1) * (eta + 1));
}
inline float eta_to_reflectivity(float eta) {
  return ((eta - 1) * (eta - 1)) / ((eta + 1) * (eta + 1));
}
// Convert reflectivity to  eta.
inline vec3f reflectivity_to_eta(const vec3f& reflectivity_) {
  auto reflectivity = clamp(reflectivity_, 0.0f, 0.99f);
  return (1 + sqrt(reflectivity)) / (1 - sqrt(reflectivity));
}
// Convert conductor eta to reflectivity
inline vec3f eta_to_reflectivity(const vec3f& eta, const vec3f& etak) {
  return ((eta - 1) * (eta - 1) + etak * etak) /
         ((eta + 1) * (eta + 1) + etak * etak);
}
// Convert eta to edge tint parametrization
inline pair<vec3f, vec3f> eta_to_edgetint(const vec3f& eta, const vec3f& etak) {
  auto reflectivity = eta_to_reflectivity(eta, etak);
  auto numer        = (1 + sqrt(reflectivity)) / (1 - sqrt(reflectivity)) - eta;
  auto denom        = (1 + sqrt(reflectivity)) / (1 - sqrt(reflectivity)) -
               (1 - reflectivity) / (1 + reflectivity);
  auto edgetint = numer / denom;
  return {reflectivity, edgetint};
}
// Convert reflectivity and edge tint to eta.
inline pair<vec3f, vec3f> edgetint_to_eta(
    const vec3f& reflectivity, const vec3f& edgetint) {
  auto r = clamp(reflectivity, 0.0f, 0.99f);
  auto g = edgetint;

  auto r_sqrt = sqrt(r);
  auto n_min  = (1 - r) / (1 + r);
  auto n_max  = (1 + r_sqrt) / (1 - r_sqrt);

  auto n  = lerp(n_max, n_min, g);
  auto k2 = ((n + 1) * (n + 1) * r - (n - 1) * (n - 1)) / (1 - r);
  k2      = max(k2, 0.0f);
  auto k  = sqrt(k2);
  return {n, k};
}

// Evaluate microfacet distribution
inline float microfacet_distribution(
    float roughness, const vec3f& normal, const vec3f& halfway, bool ggx) {
  // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
  // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
  auto cosine = dot(normal, halfway);
  if (cosine <= 0) return 0;
  auto roughness2 = roughness * roughness;
  auto cosine2    = cosine * cosine;
  if (ggx) {
    return roughness2 / (pif * (cosine2 * roughness2 + 1 - cosine2) *
                            (cosine2 * roughness2 + 1 - cosine2));
  } else {
    return exp((cosine2 - 1) / (roughness2 * cosine2)) /
           (pif * roughness2 * cosine2 * cosine2);
  }
}

// Evaluate the microfacet shadowing1
inline float microfacet_shadowing1(float roughness, const vec3f& normal,
    const vec3f& halfway, const vec3f& direction, bool ggx) {
  // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
  // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
  auto cosine  = dot(normal, direction);
  auto cosineh = dot(halfway, direction);
  if (cosine * cosineh <= 0) return 0;
  auto roughness2 = roughness * roughness;
  auto cosine2    = cosine * cosine;
  if (ggx) {
    return 2 * abs(cosine) /
           (abs(cosine) + sqrt(cosine2 - roughness2 * cosine2 + roughness2));
  } else {
    auto ci = abs(cosine) / (roughness * sqrt(1 - cosine2));
    return ci < 1.6f ? (3.535f * ci + 2.181f * ci * ci) /
                           (1.0f + 2.276f * ci + 2.577f * ci * ci)
                     : 1.0f;
  }
}

// Evaluate microfacet shadowing
inline float microfacet_shadowing(float roughness, const vec3f& normal,
    const vec3f& halfway, const vec3f& outgoing, const vec3f& incoming,
    bool ggx) {
  return microfacet_shadowing1(roughness, normal, halfway, outgoing, ggx) *
         microfacet_shadowing1(roughness, normal, halfway, incoming, ggx);
}

// Sample a microfacet ditribution.
inline vec3f sample_microfacet(
    float roughness, const vec3f& normal, const vec2f& rn, bool ggx) {
  auto phi   = 2 * pif * rn.x;
  auto theta = 0.0f;
  if (ggx) {
    theta = atan(roughness * sqrt(rn.y / (1 - rn.y)));
  } else {
    auto roughness2 = roughness * roughness;
    theta           = atan(sqrt(-roughness2 * log(1 - rn.y)));
  }
  auto local_half_vector = vec3f{
      cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)};
  return transform_direction(basis_fromz(normal), local_half_vector);
}

// Pdf for microfacet distribution sampling.
inline float sample_microfacet_pdf(
    float roughness, const vec3f& normal, const vec3f& halfway, bool ggx) {
  auto cosine = dot(normal, halfway);
  if (cosine < 0) return 0;
  return microfacet_distribution(roughness, normal, halfway, ggx) * cosine;
}

// Sample a microfacet distribution with the distribution of visible normals.
inline vec3f sample_microfacet(float roughness, const vec3f& normal,
    const vec3f& outgoing, const vec2f& rn, bool ggx) {
  // http://jcgt.org/published/0007/04/01/
  if (ggx) {
    // move to local coordinate system
    auto basis   = basis_fromz(normal);
    auto Ve      = transform_direction(transpose(basis), outgoing);
    auto alpha_x = roughness, alpha_y = roughness;
    // Section 3.2: transforming the view direction to the hemisphere
    // configuration
    auto Vh = normalize(vec3f{alpha_x * Ve.x, alpha_y * Ve.y, Ve.z});
    // Section 4.1: orthonormal basis (with special case if cross product is
    // zero)
    auto lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    auto T1    = lensq > 0 ? vec3f{-Vh.y, Vh.x, 0} * (1 / sqrt(lensq))
                           : vec3f{1, 0, 0};
    auto T2    = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    auto r   = sqrt(rn.y);
    auto phi = 2 * pif * rn.x;
    auto t1  = r * cos(phi);
    auto t2  = r * sin(phi);
    auto s   = 0.5f * (1 + Vh.z);
    t2       = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    auto Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1 - t1 * t1 - t2 * t2)) * Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    auto Ne = normalize(vec3f{alpha_x * Nh.x, alpha_y * Nh.y, max(0.0f, Nh.z)});
    // move to world coordinate
    auto local_halfway = Ne;
    return transform_direction(basis, local_halfway);
  } else {
    throw std::invalid_argument{"not implemented yet"};
  }
}

// Pdf for microfacet distribution sampling with the distribution of visible
// normals.
inline float sample_microfacet_pdf(float roughness, const vec3f& normal,
    const vec3f& halfway, const vec3f& outgoing, bool ggx) {
  // http://jcgt.org/published/0007/04/01/
  if (dot(normal, halfway) < 0) return 0;
  if (dot(halfway, outgoing) < 0) return 0;
  return microfacet_distribution(roughness, normal, halfway, ggx) *
         microfacet_shadowing1(roughness, normal, halfway, outgoing, ggx) *
         max(0.0f, dot(halfway, outgoing)) / abs(dot(normal, outgoing));
}

// Microfacet energy compensation (E(cos(w)))
inline float microfacet_cosintegral_fit(
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  // https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
  const float S[5] = {-0.170718f, 4.07985f, -11.5295f, 18.4961f, -9.23618f};
  const float T[5] = {0.0632331f, 3.1434f, -7.47567f, 13.0482f, -7.0401f};
  auto        m    = abs(dot(normal, outgoing));
  auto        r    = roughness;
  auto        s = S[0] * sqrt(m) + S[1] * r + S[2] * r * r + S[3] * r * r * r +
           S[4] * r * r * r * r;
  auto t = T[0] * m + T[1] * r + T[2] * r * r + T[3] * r * r * r +
           T[4] * r * r * r * r;
  return 1 - pow(s, 6.0f) * pow(m, 3.0f / 4.0f) / (pow(t, 6.0f) + pow(m, 2.0f));
}
// Approximate microfacet compensation for metals with Schlick's Fresnel
inline vec3f microfacet_compensation_fit(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing) {
  // https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
  auto E = microfacet_cosintegral_fit(sqrt(roughness), normal, outgoing);
  return 1 + color * (1 - E) / E;
}

// Microfacet energy compensation (E(cos(w)))
inline float microfacet_cosintegral_fit1(
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  // https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2021x.md.html
  auto alpha     = roughness;
  auto cos_theta = abs(dot(normal, outgoing));
  return 1 - 1.4594f * alpha * cos_theta *
                 (-0.20277f +
                     alpha * (2.772f + alpha * (-2.6175f + 0.73343f * alpha))) *
                 (3.09507f +
                     cos_theta *
                         (-9.11369f +
                             cos_theta *
                                 (15.8884f +
                                     cos_theta *
                                         (-13.70343 + 4.51786f * cos_theta))));
}
// Approximate microfacet compensation for metals with Schlick's Fresnel
inline vec3f microfacet_compensation_fit1(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing) {
  // https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
  auto E = microfacet_cosintegral_fit1(roughness, normal, outgoing);
  return 1 + color * (1 - E) / E;
}

inline float interpolate2d(const vector<float>& lut, const vec2f& uv) {
  // get coordinates normalized
  auto s = uv.x * (ALBEDO_LUT_SIZE - 1);
  auto t = uv.y * (ALBEDO_LUT_SIZE - 1);

  // get image coordinates and residuals
  auto i = (int)s, j = (int)t;
  auto ii = min(i + 1, ALBEDO_LUT_SIZE - 1);
  auto jj = min(j + 1, ALBEDO_LUT_SIZE - 1);
  auto u = s - i, v = t - j;

  // handle interpolation
  return lut[j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) +
         lut[jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v +
         lut[j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) +
         lut[jj * ALBEDO_LUT_SIZE + ii] * u * v;
}

inline float interpolate3d(const vector<float>& lut, const vec3f& uvw) {
  // get coordinates normalized
  auto s = uvw.x * (ALBEDO_LUT_SIZE - 1);
  auto t = uvw.y * (ALBEDO_LUT_SIZE - 1);
  auto r = uvw.z * (ALBEDO_LUT_SIZE - 1);

  // get image coordinates and residuals
  auto i = (int)s, j = (int)t, k = (int)r;
  auto ii = min(i + 1, ALBEDO_LUT_SIZE - 1);
  auto jj = min(j + 1, ALBEDO_LUT_SIZE - 1);
  auto kk = min(k + 1, ALBEDO_LUT_SIZE - 1);
  auto u = s - i, v = t - j, w = r - k;

  // trilinear interpolation
  auto size2 = ALBEDO_LUT_SIZE * ALBEDO_LUT_SIZE;

  return lut[k * size2 + j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) *
             (1 - w) +
         lut[k * size2 + j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) * (1 - w) +
         lut[k * size2 + jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v * (1 - w) +
         lut[kk * size2 + j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) * w +
         lut[kk * size2 + jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v * w +
         lut[kk * size2 + j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) * w +
         lut[k * size2 + jj * ALBEDO_LUT_SIZE + ii] * u * v * (1 - w) +
         lut[kk * size2 + jj * ALBEDO_LUT_SIZE + ii] * u * v * w;
}

// Microfacet energy compensation (E(cos(w)))
inline vec3f microfacet_compensation_conductors(const vector<float>& E_lut,
    const vec3f& color, float roughness, const vec3f& normal,
    const vec3f& outgoing) {
  auto E = interpolate2d(E_lut, {abs(dot(normal, outgoing)), sqrt(roughness)});
  return 1 + color * (1 - E) / E;
}

inline float microfacet_compensation_dielectrics(const vector<float>& E_lut,
    float ior, float roughness, const vec3f& normal, const vec3f& outgoing) {
  const auto minF0 = 0.0125f, maxF0 = 0.25f;

  auto F0 = eta_to_reflectivity(ior);
  auto w  = (clamp(F0, minF0, maxF0) - minF0) / (maxF0 - minF0);

  auto E = interpolate3d(
      E_lut, {abs(dot(normal, outgoing)), sqrt(roughness), w});
  return 1 / E;
}

inline float eval_ratpoly2d(const float coef[], float x, float y) {
  auto x2 = x * x, y2 = y * y;
  auto x3 = x2 * x, y3 = y2 * y;

  return (coef[0] + coef[1] * x + coef[2] * y + coef[3] * x2 + coef[4] * x * y +
             coef[5] * y2 + coef[6] * x3 + coef[7] * x2 * y + coef[8] * x * y2 +
             coef[9] * y3) /
         (1 + coef[10] * x + coef[11] * y + coef[12] * x2 + coef[13] * x * y +
             coef[14] * y2 + coef[15] * x3 + coef[16] * x2 * y +
             coef[17] * x * y2 + coef[18] * y3);
}

inline float eval_ratpoly3d(const float coef[], float x, float y, float z) {
  auto x2 = x * x, y2 = y * y, z2 = z * z;
  auto x3 = x2 * x, y3 = y2 * y, z3 = z2 * z;

  return (coef[0] + coef[1] * x + coef[2] * y + coef[3] * z + coef[4] * x2 +
             coef[5] * x * y + coef[6] * x * z + coef[7] * y2 +
             coef[8] * y * z + coef[9] * z2 + coef[10] * x3 +
             coef[11] * x2 * y + coef[12] * x2 * z + coef[13] * x * y2 +
             coef[14] * x * y * z + coef[15] * x * z2 + coef[16] * y3 +
             coef[17] * y2 * z + coef[18] * y * z2 + coef[19] * z3) /
         (1 + coef[20] * x + coef[21] * y + coef[22] * z + coef[23] * x2 +
             coef[24] * x * y + coef[25] * x * z + coef[26] * y2 +
             coef[27] * y * z + coef[28] * z2 + coef[29] * x3 +
             coef[30] * x2 * y + coef[31] * x2 * z + coef[32] * x * y2 +
             coef[33] * x * y * z + coef[34] * x * z2 + coef[35] * y3 +
             coef[36] * y2 * z + coef[37] * y * z2 + coef[38] * z3);
}

inline float eval_ratpoly3d_deg4(
    const float coef[], float x, float y, float z) {
  auto x2 = x * x, y2 = y * y, z2 = z * z;
  auto x3 = x2 * x, y3 = y2 * y, z3 = z2 * z;
  auto x4 = x3 * x, y4 = y3 * y, z4 = z3 * z;

  return (coef[0] + coef[1] * x + coef[2] * y + coef[3] * z + coef[4] * x2 +
             coef[5] * x * y + coef[6] * x * z + coef[7] * y2 +
             coef[8] * y * z + coef[9] * z2 + coef[10] * x3 +
             coef[11] * x2 * y + coef[12] * x2 * z + coef[13] * x * y2 +
             coef[14] * x * y * z + coef[15] * x * z2 + coef[16] * y3 +
             coef[17] * y2 * z + coef[18] * y * z2 + coef[19] * z3 +
             coef[20] * x4 + coef[21] * x3 * y + coef[22] * x3 * z +
             coef[23] * x2 * y2 + coef[24] * x2 * y * z + coef[25] * x2 * z2 +
             coef[26] * x * y3 + coef[27] * x * y2 * z + coef[28] * x * y * z2 +
             coef[29] * x * z3 + coef[30] * y4 + coef[31] * y3 * z +
             coef[32] * y2 * z2 + coef[33] * y * z3 + coef[34] * z4) /
         (1 + coef[35] * x + coef[36] * y + coef[37] * z + coef[38] * x2 +
             coef[39] * x * y + coef[40] * x * z + coef[41] * y2 +
             coef[42] * y * z + coef[43] * z2 + coef[44] * x3 +
             coef[45] * x2 * y + coef[46] * x2 * z + coef[47] * x * y2 +
             coef[48] * x * y * z + coef[49] * x * z2 + coef[50] * y3 +
             coef[51] * y2 * z + coef[52] * y * z2 + coef[53] * z3 +
             coef[54] * x4 + coef[55] * x3 * y + coef[56] * x3 * z +
             coef[57] * x2 * y2 + coef[58] * x2 * y * z + coef[59] * x2 * z2 +
             coef[60] * x * y3 + coef[61] * x * y2 * z + coef[62] * x * y * z2 +
             coef[63] * x * z3 + coef[64] * y4 + coef[65] * y3 * z +
             coef[66] * y2 * z2 + coef[67] * y * z3 + coef[68] * z4);
}

inline vec3f microfacet_compensation_conductors_myfit(const vec3f& color,
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  const float coef[19] = {1.01202782, -11.1084138, 13.68932726, 46.63441392,
      -56.78561075, 17.38577426, -29.2033844, 30.94339247, -5.38305905,
      -4.72530367, -10.45175028, 13.88865122, 43.49596666, -57.01339516,
      16.76996746, -21.80566626, 32.0408972, -5.48296756, -4.29104947};

  auto alpha     = sqrt(roughness);
  auto cos_theta = abs(dot(normal, outgoing));

  auto E = eval_ratpoly2d(coef, alpha, cos_theta);

  return 1 + color * (1 - E) / E;
}

inline float microfacet_compensation_dielectrics_fit(const float coef[],
    float ior, float roughness, const vec3f& normal, const vec3f& outgoing) {
  const auto minF0 = 0.0125f, maxF0 = 0.25f;

  auto F0 = eta_to_reflectivity(ior);
  auto x  = clamp(F0, minF0, maxF0);

  auto E = eval_ratpoly3d(coef, x, sqrt(roughness), abs(dot(normal, outgoing)));

  return 1 / E;
}

// Evaluate a diffuse BRDF lobe.
inline vec3f eval_matte(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  return color / pif * abs(dot(normal, incoming));
}

// Sample a diffuse BRDF lobe.
inline vec3f sample_matte(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return sample_hemisphere_cos(up_normal, rn);
}

// Pdf for diffuse BRDF lobe sampling.
inline float sample_matte_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return sample_hemisphere_cos_pdf(up_normal, incoming);
}

// Evaluate a specular BRDF lobe.
inline vec3f eval_glossy(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto F1        = fresnel_dielectric(ior, up_normal, outgoing);
  auto halfway   = normalize(incoming + outgoing);
  auto F         = fresnel_dielectric(ior, halfway, incoming);
  auto D         = microfacet_distribution(roughness, up_normal, halfway);
  auto G         = microfacet_shadowing(
      roughness, up_normal, halfway, outgoing, incoming);
  return color * (1 - F1) / pif * abs(dot(up_normal, incoming)) +
         vec3f{1, 1, 1} * F * D * G /
             (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
             abs(dot(up_normal, incoming));
}

inline vec3f eval_glossy_comp(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto E1        = interpolate3d(
      glossy_albedo_lut, {abs(dot(up_normal, outgoing)), sqrt(roughness),
                             eta_to_reflectivity(ior)});
  auto halfway = normalize(incoming + outgoing);
  auto C       = microfacet_compensation_conductors(
      my_albedo_lut, {1, 1, 1}, roughness, normal, outgoing);
  auto F = fresnel_dielectric(ior, halfway, incoming);
  auto D = microfacet_distribution(roughness, up_normal, halfway);
  auto G = microfacet_shadowing(
      roughness, up_normal, halfway, outgoing, incoming);
  return color * (1 - E1) / pif * abs(dot(up_normal, incoming)) +
         vec3f{1, 1, 1} * C * F * D * G /
             (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
             abs(dot(up_normal, incoming));
}

inline vec3f eval_glossy_comp_fit(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const float coef[39] = {1.3871118e-03, 2.6386258e+02, -1.8921787e-02,
      -2.4023395e-02, -5.1012723e+02, -2.9148245e+02, -4.0365756e+02,
      2.0257600e-01, -4.0661687e-01, 6.5557098e-01, 2.5328067e+02,
      6.1129333e+02, 1.0740126e+03, 1.5193332e+02, 2.4832379e+02, 3.8414221e+02,
      1.5020353e-01, -1.2501916e+00, 1.8174797e+00, -1.2472963e+00,
      2.5489156e+02, -2.0209291e+01, 1.0437813e+01, -4.0524969e+02,
      1.3936530e+02, 5.0904727e+02, 1.4095477e+02, -2.7545721e+02,
      1.5878168e+02, 1.6524709e+02, 2.1102951e+02, 2.4955217e+02, 3.2012157e+01,
      3.3603104e+02, 7.3130522e+00, -6.0488926e+01, 1.6286488e+02,
      1.4251903e+02, 8.1229729e+01};

  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto E1      = eval_ratpoly3d(coef, eta_to_reflectivity(ior), sqrt(roughness),
      abs(dot(up_normal, outgoing)));
  auto halfway = normalize(incoming + outgoing);
  auto C       = microfacet_compensation_conductors_myfit(
      {1, 1, 1}, roughness, normal, outgoing);
  auto F = fresnel_dielectric(ior, halfway, incoming);
  auto D = microfacet_distribution(roughness, up_normal, halfway);
  auto G = microfacet_shadowing(
      roughness, up_normal, halfway, outgoing, incoming);
  return color * (1 - E1) / pif * abs(dot(up_normal, incoming)) +
         vec3f{1, 1, 1} * C * F * D * G /
             (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
             abs(dot(up_normal, incoming));
}

// Sample a specular BRDF lobe.
inline vec3f sample_glossy(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (rnl < fresnel_dielectric(ior, up_normal, outgoing)) {
    auto halfway  = sample_microfacet(roughness, up_normal, rn);
    auto incoming = reflect(outgoing, halfway);
    if (!same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
  } else {
    return sample_hemisphere_cos(up_normal, rn);
  }
}

// Pdf for specular BRDF lobe sampling.
inline float sample_glossy_pdf(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = normalize(outgoing + incoming);
  auto F         = fresnel_dielectric(ior, up_normal, outgoing);
  return F * sample_microfacet_pdf(roughness, up_normal, halfway) /
             (4 * abs(dot(outgoing, halfway))) +
         (1 - F) * sample_hemisphere_cos_pdf(up_normal, incoming);
}

// Evaluate a metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = normalize(incoming + outgoing);
  auto F         = fresnel_conductor(
      reflectivity_to_eta(color), {0, 0, 0}, halfway, incoming);
  auto D = microfacet_distribution(roughness, up_normal, halfway);
  auto G = microfacet_shadowing(
      roughness, up_normal, halfway, outgoing, incoming);
  return F * D * G / (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
         abs(dot(up_normal, incoming));
}

inline vec3f eval_metallic_comp_fit(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto C = microfacet_compensation_fit(color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

inline vec3f eval_metallic_comp_fit1(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto C = microfacet_compensation_fit1(color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

inline vec3f eval_metallic_comp_tab(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto C = microfacet_compensation_conductors(
      albedo_lut, color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

inline vec3f eval_metallic_comp_mytab(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto C = microfacet_compensation_conductors(
      my_albedo_lut, color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

inline vec3f eval_metallic_comp_myfit(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto C = microfacet_compensation_conductors_myfit(
      color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

// Sample a metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  auto incoming  = reflect(outgoing, halfway);
  if (!same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
  return incoming;
}

// Pdf for metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = normalize(outgoing + incoming);
  return sample_microfacet_pdf(roughness, up_normal, halfway) /
         (4 * abs(dot(outgoing, halfway)));
}

// Evaluate a metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& eta, const vec3f& etak, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = normalize(incoming + outgoing);
  auto F         = fresnel_conductor(eta, etak, halfway, incoming);
  auto D         = microfacet_distribution(roughness, up_normal, halfway);
  auto G         = microfacet_shadowing(
      roughness, up_normal, halfway, outgoing, incoming);
  return F * D * G / (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
         abs(dot(up_normal, incoming));
}

// Sample a metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& eta, const vec3f& etak,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  return reflect(outgoing, halfway);
}

// Pdf for metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& eta, const vec3f& etak,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = normalize(outgoing + incoming);
  return sample_microfacet_pdf(roughness, up_normal, halfway) /
         (4 * abs(dot(outgoing, halfway)));
}

// Evaluate a delta metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return fresnel_conductor(
      reflectivity_to_eta(color), {0, 0, 0}, up_normal, outgoing);
}

// Sample a delta metal BRDF lobe.
inline vec3f sample_metallic(
    const vec3f& color, const vec3f& normal, const vec3f& outgoing) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return reflect(outgoing, up_normal);
}

// Pdf for delta metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  return 1;
}

// Evaluate a delta metal BRDF lobe.
inline vec3f eval_metallic(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return fresnel_conductor(eta, etak, up_normal, outgoing);
}

// Sample a delta metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return reflect(outgoing, up_normal);
}

// Pdf for delta metal BRDF lobe sampling.
inline float sample_metallic_pdf(const vec3f& eta, const vec3f& etak,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  return 1;
}

// Evaluate a specular BRDF lobe.
inline vec3f eval_gltfpbr(const vec3f& color, float ior, float roughness,
    float metallic, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return zero3f;
  auto reflectivity = lerp(
      eta_to_reflectivity(vec3f{ior, ior, ior}), color, metallic);
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto F1        = fresnel_schlick(reflectivity, up_normal, outgoing);
  auto halfway   = normalize(incoming + outgoing);
  auto F         = fresnel_schlick(reflectivity, halfway, incoming);
  auto D         = microfacet_distribution(roughness, up_normal, halfway);
  auto G         = microfacet_shadowing(
      roughness, up_normal, halfway, outgoing, incoming);
  return color * (1 - metallic) * (1 - F1) / pif *
             abs(dot(up_normal, incoming)) +
         F * D * G / (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
             abs(dot(up_normal, incoming));
}

// Sample a specular BRDF lobe.
inline vec3f sample_gltfpbr(const vec3f& color, float ior, float roughness,
    float metallic, const vec3f& normal, const vec3f& outgoing, float rnl,
    const vec2f& rn) {
  auto up_normal    = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto reflectivity = lerp(
      eta_to_reflectivity(vec3f{ior, ior, ior}), color, metallic);
  if (rnl < mean(fresnel_schlick(reflectivity, up_normal, outgoing))) {
    auto halfway  = sample_microfacet(roughness, up_normal, rn);
    auto incoming = reflect(outgoing, halfway);
    if (!same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
  } else {
    return sample_hemisphere_cos(up_normal, rn);
  }
}

// Pdf for specular BRDF lobe sampling.
inline float sample_gltfpbr_pdf(const vec3f& color, float ior, float roughness,
    float metallic, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) <= 0) return 0;
  auto up_normal    = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway      = normalize(outgoing + incoming);
  auto reflectivity = lerp(
      eta_to_reflectivity(vec3f{ior, ior, ior}), color, metallic);
  auto F = mean(fresnel_schlick(reflectivity, up_normal, outgoing));
  return F * sample_microfacet_pdf(roughness, up_normal, halfway) /
             (4 * abs(dot(outgoing, halfway))) +
         (1 - F) * sample_hemisphere_cos_pdf(up_normal, incoming);
}

// Evaluate a transmission BRDF lobe.
inline vec3f eval_transparent(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    auto halfway = normalize(incoming + outgoing);
    auto F       = fresnel_dielectric(ior, halfway, outgoing);
    auto D       = microfacet_distribution(roughness, up_normal, halfway);
    auto G       = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, incoming);
    return vec3f{1, 1, 1} * F * D * G /
           (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
           abs(dot(up_normal, incoming));
  } else {
    auto reflected = reflect(-incoming, up_normal);
    auto halfway   = normalize(reflected + outgoing);
    auto F         = fresnel_dielectric(ior, halfway, outgoing);
    auto D         = microfacet_distribution(roughness, up_normal, halfway);
    auto G         = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, reflected);
    return color * (1 - F) * D * G /
           (4 * dot(up_normal, outgoing) * dot(up_normal, reflected)) *
           (abs(dot(up_normal, reflected)));
  }
}

inline vec3f eval_transparent_comp(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  auto C = microfacet_compensation_conductors(
      my_albedo_lut, {1, 1, 1}, roughness, normal, outgoing);
  return C *
         eval_transparent(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_transparent_comp_fit(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  auto C = microfacet_compensation_conductors_myfit(
      {1, 1, 1}, roughness, normal, outgoing);
  return C *
         eval_transparent(color, ior, roughness, normal, outgoing, incoming);
}

// Sample a transmission BRDF lobe.
inline vec3f sample_transparent(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  if (rnl < fresnel_dielectric(ior, halfway, outgoing)) {
    auto incoming = reflect(outgoing, halfway);
    if (!same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
  } else {
    auto reflected = reflect(outgoing, halfway);
    auto incoming  = -reflect(reflected, up_normal);
    if (same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
  }
}

// Pdf for transmission BRDF lobe sampling.
inline float sample_tranparent_pdf(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    auto halfway = normalize(incoming + outgoing);
    return fresnel_dielectric(ior, halfway, outgoing) *
           sample_microfacet_pdf(roughness, up_normal, halfway) /
           (4 * abs(dot(outgoing, halfway)));
  } else {
    auto reflected = reflect(-incoming, up_normal);
    auto halfway   = normalize(reflected + outgoing);
    auto d         = (1 - fresnel_dielectric(ior, halfway, outgoing)) *
             sample_microfacet_pdf(roughness, up_normal, halfway);
    return d / (4 * abs(dot(outgoing, halfway)));
  }
}

// Evaluate a delta transmission BRDF lobe.
inline vec3f eval_transparent(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return vec3f{1, 1, 1} * fresnel_dielectric(ior, up_normal, outgoing);
  } else {
    return color * (1 - fresnel_dielectric(ior, up_normal, outgoing));
  }
}

// Sample a delta transmission BRDF lobe.
inline vec3f sample_transparent(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, float rnl) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (rnl < fresnel_dielectric(ior, up_normal, outgoing)) {
    return reflect(outgoing, up_normal);
  } else {
    return -outgoing;
  }
}

// Pdf for delta transmission BRDF lobe sampling.
inline float sample_tranparent_pdf(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return fresnel_dielectric(ior, up_normal, outgoing);
  } else {
    return 1 - fresnel_dielectric(ior, up_normal, outgoing);
  }
}

// Evaluate a refraction BRDF lobe.
inline vec3f eval_refractive(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto rel_ior   = entering ? ior : (1 / ior);
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    auto halfway = normalize(incoming + outgoing);
    auto F       = fresnel_dielectric(rel_ior, halfway, outgoing);
    auto D       = microfacet_distribution(roughness, up_normal, halfway);
    auto G       = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, incoming);
    return vec3f{1, 1, 1} * F * D * G /
           abs(4 * dot(normal, outgoing) * dot(normal, incoming)) *
           abs(dot(normal, incoming));
  } else {
    auto halfway = -normalize(rel_ior * incoming + outgoing) *
                   (entering ? 1.0f : -1.0f);
    auto F = fresnel_dielectric(rel_ior, halfway, outgoing);
    auto D = microfacet_distribution(roughness, up_normal, halfway);
    auto G = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, incoming);
    // [Walter 2007] equation 21
    return vec3f{1, 1, 1} *
           abs((dot(outgoing, halfway) * dot(incoming, halfway)) /
               (dot(outgoing, normal) * dot(incoming, normal))) *
           (1 - F) * D * G /
           pow(rel_ior * dot(halfway, incoming) + dot(halfway, outgoing), 2) *
           abs(dot(normal, incoming));
  }
}

inline vec3f eval_refractive_comp(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  auto E_lut = dot(normal, outgoing) >= 0 ? enter_eta2 : leave_eta2;
  auto C     = microfacet_compensation_dielectrics(
      E_lut, ior, roughness, normal, outgoing);
  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_refractive_comp_fit_3d(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const float coef_enter[39] = {1.0335013, 7.450409, -3.319554, 18.44608,
      -30.58467, 33.400505, 327.8045, 4.6029525, -30.953785, 36.854183,
      342.75027, -90.9943, -73.366615, -18.440683, -254.84488, -89.319405,
      -1.7787099, 17.897514, -22.77419, 13.296596, 8.721737, -2.0361824,
      18.254898, -36.243607, 31.847424, 323.02133, -0.5822772, -27.898998,
      36.163277, 364.78198, -80.00294, -83.136086, -14.092524, -218.99208,
      -92.53893, 4.594424, 20.955828, -28.854736, 14.729479};
  const float coef_leave[39] = {9.67145145e-01, 2.16013241e+01, -2.02096558e+00,
      -2.01874876e+00, -5.43549538e+00, -7.87079477e+00, -2.86914902e+01,
      1.92847443e+00, 1.09270012e+00, 3.73462844e+00, -5.97717094e+01,
      1.93938808e+01, 1.33008852e+01, -2.35681677e+00, 1.80178809e+00,
      1.33748035e+01, -8.85033965e-01, 1.34998655e+00, -2.96786380e+00,
      -9.18334782e-01, 2.21405087e+01, -1.81398928e+00, -2.36214828e+00,
      -3.70193338e+00, -6.65679741e+00, -3.13353672e+01, 2.26810932e+00,
      4.21848953e-01, 4.52370930e+00, -5.91011772e+01, 1.16894159e+01,
      1.24615879e+01, 3.95525575e+00, 5.39931841e-02, 1.54366455e+01,
      -8.92938375e-01, 1.19969201e+00, -2.58653760e+00, -1.40058172e+00};

  auto coef = dot(normal, outgoing) >= 0 ? coef_enter : coef_leave;
  auto C    = microfacet_compensation_dielectrics_fit(
      coef, ior, roughness, normal, outgoing);

  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_refractive_comp_fit_slices(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const float coef_enter[114] = {};
  const float coef_leave[114] = {};

  const auto min_ior = 1.05f, max_ior = 3.0f;
  auto       x = (clamp(ior, min_ior, max_ior) - min_ior) / (max_ior - min_ior);

  auto  coef = dot(normal, outgoing) >= 0 ? coef_enter : coef_leave;
  float coef1[19];

  for (auto i = 0; i < 19; i++) {
    auto j   = i * 6;
    coef1[i] = coef[j] + coef[j + 1] * x + coef[j + 2] * x * x +
               coef[j + 3] * x * x * x + coef[j + 4] * x * x * x * x +
               coef[j + 5] * x * x * x * x * x;
  }

  // Ratio of polynomials
  // auto x2 = x * x, x3 = x * x * x;

  // for (auto i = 0; i < 19; i++) {
  //   auto j   = i * 7;
  //   coef1[i] = (coef_leave[j] + coef_leave[j + 1] * x +
  //                   coef_leave[j + 2] * x2 + coef_leave[j + 3] * x3) /
  //               (1 + coef_leave[j + 4] * x + coef_leave[j + 5] * x2 +
  //                   coef_leave[j + 6] * x3);
  // }

  auto C = 1 /
           eval_ratpoly2d(coef1, sqrt(roughness), abs(dot(normal, outgoing)));

  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_refractive_comp_fit_slices2(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const auto num_coef   = 19;
  const auto num_slices = 32;
  const auto min_ior = 1.25f, max_ior = 3.0f;

  const float coef_enter[608] = {9.82499301e-01, 3.72620106e+00, 3.23144722e+00,
      -9.02183628e+00, -2.33354306e+00, 2.92274933e+01, 4.59039831e+00,
      1.01330483e+00, -2.80003815e+01, 6.09541988e+00, 5.65234709e+00,
      4.06026363e+00, -8.66300392e+00, -4.69757414e+00, 4.95434189e+01,
      3.74326253e+00, 4.28073835e+00, -4.61370926e+01, 7.11859179e+00,
      9.45301771e-01, 8.09553452e+01, 9.43695679e+01, 3.29477478e+02,
      7.85417080e+00, 2.52776672e+02, -3.44333588e+02, 3.04873260e+02,
      -1.86942139e+02, 5.06203674e+02, 7.91898346e+01, 9.10786972e+01,
      5.14301392e+02, 1.95650940e+02, 5.71992126e+02, -1.12093521e+02,
      1.73230698e+02, -3.46710510e+02, 7.70378662e+02, 9.38130736e-01,
      2.47041855e+01, 3.29518700e+01, 1.48844742e+02, -3.16202621e+01,
      8.48196945e+01, -1.45176743e+02, 1.48626938e+02, -3.42517929e+01,
      1.59967316e+02, 2.28848076e+01, 3.22320061e+01, 2.34111374e+02,
      1.98311100e+01, 2.11593246e+02, -5.55982513e+01, 1.42004318e+02,
      -7.85294952e+01, 2.60738495e+02, 9.33640122e-01, 1.18790321e+01,
      1.82653370e+01, 9.55051651e+01, -3.15858898e+01, 4.75520592e+01,
      -8.90876465e+01, 9.68329468e+01, -3.37474608e+00, 7.21697540e+01,
      1.02606421e+01, 1.82748508e+01, 1.53412689e+02, -1.24724245e+01,
      1.28454254e+02, -3.59040489e+01, 1.16979813e+02, -1.93650436e+01,
      1.23091461e+02, 9.31089342e-01, 6.57067299e+00, 1.17192869e+01,
      6.83322220e+01, -2.73909149e+01, 3.19130325e+01, -6.15485306e+01,
      6.90430222e+01, 5.15146875e+00, 3.48343010e+01, 5.18426323e+00,
      1.21254129e+01, 1.12428085e+02, -2.19855194e+01, 9.21730804e+01,
      -2.49939537e+01, 9.78470154e+01, 2.26487398e-01, 6.00615501e+01,
      9.29579556e-01, 3.82536006e+00, 8.02297115e+00, 5.15667725e+01,
      -2.30995464e+01, 2.36688023e+01, -4.51025314e+01, 5.15906525e+01,
      6.25250578e+00, 1.60792999e+01, 2.66421461e+00, 8.69095898e+00,
      8.70302582e+01, -2.47823620e+01, 7.21295471e+01, -1.82395687e+01,
      8.26835098e+01, 5.17368364e+00, 2.59003239e+01, 9.28517580e-01,
      2.25821280e+00, 5.69573021e+00, 4.02028580e+01, -1.94695225e+01,
      1.88543930e+01, -3.43068199e+01, 3.97539482e+01, 4.64477777e+00,
      5.98761177e+00, 1.29964340e+00, 6.54853010e+00, 6.97967148e+01,
      -2.50896645e+01, 5.98345108e+01, -1.40121069e+01, 7.02937164e+01,
      3.63764000e+00, 6.10699129e+00, 9.27593112e-01, 1.33312261e+00,
      4.13585377e+00, 3.19982510e+01, -1.65594959e+01, 1.58811216e+01,
      -2.67761383e+01, 3.12935104e+01, 2.23965359e+00, 3.78775299e-01,
      5.50617397e-01, 5.12359142e+00, 5.74601936e+01, -2.43834286e+01,
      5.18586388e+01, -1.14773798e+01, 5.98238411e+01, -6.58379555e-01,
      -5.69043159e+00, 9.23538983e-01, 1.00264597e+00, 3.27584028e+00,
      3.09484119e+01, -1.18050089e+01, 1.51084967e+01, -2.46197453e+01,
      1.53203793e+01, 3.67965622e+01, -1.80847816e+01, 1.62788749e-01,
      4.14526701e+00, 5.63176231e+01, -2.26481400e+01, 5.31129379e+01,
      -7.99154615e+00, 3.63134155e+01, 8.92799301e+01, -5.45024681e+01,
      9.25773442e-01, 5.00020564e-01, 2.28668690e+00, 2.12491360e+01,
      -1.26067181e+01, 1.27276649e+01, -1.73843079e+01, 2.04776878e+01,
      -1.93001485e+00, -4.44964647e+00, -1.05944006e-02, 3.46530771e+00,
      4.17943077e+01, -2.22825336e+01, 4.28593330e+01, -9.48835087e+00,
      4.34497452e+01, -1.01196556e+01, -1.67279377e+01, 9.24788594e-01,
      3.39306146e-01, 1.78069723e+00, 1.80330334e+01, -1.14692869e+01,
      1.19124594e+01, -1.46710367e+01, 1.73062706e+01, -3.23306489e+00,
      -5.24569225e+00, -8.28173384e-02, 3.03971744e+00, 3.73353462e+01,
      -2.15608368e+01, 4.05203018e+01, -9.27349472e+00, 3.80301361e+01,
      -1.37279530e+01, -1.87283688e+01, 9.23650622e-01, 2.43082732e-01,
      1.47249961e+00, 1.59632063e+01, -1.07910433e+01, 1.14044867e+01,
      -1.29167471e+01, 1.52419662e+01, -4.13772821e+00, -5.52912855e+00,
      -1.25574350e-01, 2.80726886e+00, 3.46055069e+01, -2.12265625e+01,
      3.92215309e+01, -9.22554111e+00, 3.45777168e+01, -1.66770229e+01,
      -1.95240440e+01, 9.22314763e-01, 1.71795473e-01, 1.30356777e+00,
      1.47229519e+01, -1.04415226e+01, 1.11233130e+01, -1.18293896e+01,
      1.39421711e+01, -4.83109760e+00, -5.54634905e+00, -1.73093557e-01,
      2.70821142e+00, 3.31078072e+01, -2.12062988e+01, 3.87276382e+01,
      -9.23232555e+00, 3.26714554e+01, -1.93427639e+01, -1.96605701e+01,
      9.20790076e-01, 1.08706690e-01, 1.22469819e+00, 1.40265369e+01,
      -1.03025703e+01, 1.10141144e+01, -1.11692162e+01, 1.31204634e+01,
      -5.43386698e+00, -5.44046307e+00, -2.34060973e-01, 2.69220901e+00,
      3.24104500e+01, -2.13905544e+01, 3.88292084e+01, -9.25660610e+00,
      3.18057308e+01, -2.19559708e+01, -1.94621487e+01, 9.19114947e-01,
      4.60826717e-02, 1.20172882e+00, 1.36730547e+01, -1.02871504e+01,
      1.10374489e+01, -1.07743406e+01, 1.25838604e+01, -6.00964785e+00,
      -5.27814674e+00, -3.09043050e-01, 2.72526860e+00, 3.22086563e+01,
      -2.16915913e+01, 3.93592415e+01, -9.29132175e+00, 3.16007500e+01,
      -2.46138821e+01, -1.90763340e+01, 9.17330205e-01, -1.91755854e-02,
      1.21164203e+00, 1.35349293e+01, -1.03374510e+01, 1.11655531e+01,
      -1.05446234e+01, 1.22125940e+01, -6.58346844e+00, -5.09317064e+00,
      -3.95918071e-01, 2.78511405e+00, 3.23089142e+01, -2.20507565e+01,
      4.01936951e+01, -9.33942699e+00, 3.18018379e+01, -2.73201008e+01,
      -1.85800648e+01, 9.15471077e-01, -8.87579694e-02, 1.23909926e+00,
      1.35329962e+01, -1.04152699e+01, 1.13785791e+01, -1.04191179e+01,
      1.19352074e+01, -7.16600561e+00, -4.90096903e+00, -4.92789656e-01,
      2.85730481e+00, 3.25895233e+01, -2.24310856e+01, 4.12416077e+01,
      -9.40663052e+00, 3.22511139e+01, -3.00483398e+01, -1.80102825e+01,
      9.13567483e-01, -1.63848072e-01, 1.27307451e+00, 1.36155252e+01,
      -1.04937983e+01, 1.16605358e+01, -1.03588848e+01, 1.17076159e+01,
      -7.76047659e+00, -4.70820379e+00, -5.98253846e-01, 2.93183899e+00,
      3.29689369e+01, -2.28088646e+01, 4.24287567e+01, -9.49918747e+00,
      3.28469009e+01, -3.27569008e+01, -1.73847694e+01, 9.11642492e-01,
      -2.45092422e-01, 1.30573404e+00, 1.37493410e+01, -1.05557575e+01,
      1.19989882e+01, -1.03396072e+01, 1.15047350e+01, -8.36788082e+00,
      -4.51794100e+00, -7.11165369e-01, 3.00175619e+00, 3.33940125e+01,
      -2.31707516e+01, 4.36975365e+01, -9.62221622e+00, 3.35271530e+01,
      -3.54065018e+01, -1.67149963e+01, 9.09714103e-01, -3.32799613e-01,
      1.33134365e+00, 1.39126778e+01, -1.05904856e+01, 1.23834543e+01,
      -1.03458147e+01, 1.13132305e+01, -8.98825932e+00, -4.33136988e+00,
      -8.30559194e-01, 3.06210041e+00, 3.38291740e+01, -2.35104294e+01,
      4.50006027e+01, -9.77900410e+00, 3.42538872e+01, -3.79631004e+01,
      -1.60090179e+01, 9.07796204e-01, -4.26948547e-01, 1.34570599e+00,
      1.40914059e+01, -1.05926714e+01, 1.28046408e+01, -1.03677864e+01,
      1.11271658e+01, -9.62061977e+00, -4.14886427e+00, -9.55522835e-01,
      3.10938549e+00, 3.42510986e+01, -2.38271503e+01, 4.62980003e+01,
      -9.97068214e+00, 3.50031967e+01, -4.03977661e+01, -1.52734594e+01,
      9.05899942e-01, -5.27282655e-01, 1.34586453e+00, 1.42767429e+01,
      -1.05614853e+01, 1.32540398e+01, -1.03996544e+01, 1.09456348e+01,
      -1.02631750e+01, -3.97033668e+00, -1.08518195e+00, 3.14132786e+00,
      3.46451759e+01, -2.41249561e+01, 4.75556946e+01, -1.01960611e+01,
      3.57605896e+01, -4.26874428e+01, -1.45137959e+01, 9.04033124e-01,
      -6.33336782e-01, 1.32988179e+00, 1.44636507e+01, -1.04990311e+01,
      1.37233210e+01, -1.04381781e+01, 1.07707777e+01, -1.09140263e+01,
      -3.79492188e+00, -1.21871054e+00, 3.15664625e+00, 3.50034447e+01,
      -2.44100857e+01, 4.87436371e+01, -1.04522696e+01, 3.65165367e+01,
      -4.48157692e+01, -1.37330990e+01, 9.02203619e-01, -7.44460166e-01,
      1.29647791e+00, 1.46491051e+01, -1.04104309e+01, 1.42044058e+01,
      -1.04813700e+01, 1.06056099e+01, -1.15679178e+01, -3.62272000e+00,
      -1.35518157e+00, 3.15475011e+00, 3.53215828e+01, -2.46927814e+01,
      4.98347092e+01, -1.07348022e+01, 3.72614784e+01, -4.67628174e+01,
      -1.29365149e+01, 9.00417328e-01, -8.59878540e-01, 1.24510479e+00,
      1.48317318e+01, -1.03023624e+01, 1.46890106e+01, -1.05283203e+01,
      1.04541311e+01, -1.22194862e+01, -3.45296454e+00, -1.49369371e+00,
      3.13580537e+00, 3.55986824e+01, -2.49841576e+01, 5.08041153e+01,
      -1.10385227e+01, 3.79871597e+01, -4.85134926e+01, -1.21260805e+01,
      8.98679078e-01, -9.78705645e-01, 1.17590976e+00, 1.50108290e+01,
      -1.01826439e+01, 1.51690826e+01, -1.05785284e+01, 1.03201122e+01,
      -1.28617048e+01, -3.28535295e+00, -1.63330936e+00, 3.10065770e+00,
      3.58357048e+01, -2.52962570e+01, 5.16309738e+01, -1.13583641e+01,
      3.86852684e+01, -5.00528336e+01, -1.13049059e+01, 8.96993160e-01,
      -1.09991479e+00, 1.08968532e+00, 1.51859856e+01, -1.00596056e+01,
      1.56369658e+01, -1.06317034e+01, 1.02068939e+01, -1.34867735e+01,
      -3.11976504e+00, -1.77300823e+00, 3.05075407e+00, 3.60350494e+01,
      -2.56409225e+01, 5.22988510e+01, -1.16898689e+01, 3.93476410e+01,
      -5.13677788e+01, -1.04764967e+01, 8.95362437e-01, -1.22241712e+00,
      9.87871468e-01, 1.53568659e+01, -9.94159508e+00, 1.60857105e+01,
      -1.06876478e+01, 1.01172781e+01, -1.40866489e+01, -2.95636749e+00,
      -1.91176450e+00, 2.98811626e+00, 3.62000351e+01, -2.60292206e+01,
      5.27968330e+01, -1.20296412e+01, 3.99674644e+01, -5.24482803e+01,
      -9.64531612e+00, 8.93788636e-01, -1.34504664e+00, 8.72487426e-01,
      1.55230570e+01, -9.83646011e+00, 1.65094852e+01, -1.07461796e+01,
      1.00533772e+01, -1.46537209e+01, -2.79564953e+00, -2.04851151e+00,
      2.91521835e+00, 3.63346558e+01, -2.64704857e+01, 5.31206017e+01,
      -1.23754930e+01, 4.05396614e+01, -5.32886963e+01, -8.81706333e+00,
      8.92272115e-01, -1.46660519e+00, 7.46072948e-01, 1.56842251e+01,
      -9.75131130e+00, 1.69040661e+01, -1.08072443e+01, 1.00167027e+01,
      -1.51814947e+01, -2.63852239e+00, -2.18220019e+00, 2.83489037e+00,
      3.64437256e+01, -2.69720039e+01, 5.32735443e+01, -1.27265253e+01,
      4.10618935e+01, -5.38893051e+01, -7.99906063e+00, 8.90810966e-01,
      -1.58592272e+00, 6.11615658e-01, 1.58403511e+01, -9.69211769e+00,
      1.72674980e+01, -1.08710327e+01, 1.00082245e+01, -1.56657610e+01,
      -2.48623085e+00, -2.31187367e+00, 2.75016785e+00, 3.65330200e+01,
      -2.75382614e+01, 5.32683449e+01, -1.30829153e+01, 4.15355148e+01,
      -5.42591743e+01, -7.20004225e+00, 8.89404058e-01, -1.70197976e+00,
      4.71927881e-01, 1.59907007e+01, -9.66337204e+00, 1.75980053e+01,
      -1.09372988e+01, 1.00281696e+01, -1.61029148e+01, -2.33955407e+00,
      -2.43665552e+00, 2.66394687e+00, 3.66065826e+01, -2.81719131e+01,
      5.31173134e+01, -1.34455318e+01, 4.19621620e+01, -5.44072380e+01,
      -6.42675734e+00};
  const float coef_leave[608] = {9.99094725e-01, -7.50803888e-01,
      -3.30241346e+00, -5.98110408e-02, 3.32929087e+00, 2.62241578e+00,
      -5.41002750e-02, 6.73881948e-01, -3.51727867e+00, 2.44636431e-01,
      -2.90576488e-01, -3.54028654e+00, 1.23303525e-01, 1.56944597e+00,
      3.54935098e+00, 1.47785783e-01, 1.10685937e-01, -1.81998456e+00,
      -6.56003475e-01, 9.93133008e-01, -8.23270082e-01, -3.06456423e+00,
      1.65509842e-02, 3.03649688e+00, 2.30954027e+00, -3.46905105e-02,
      3.07869732e-01, -2.75666332e+00, 1.35273859e-01, -3.83129865e-01,
      -3.33753896e+00, 8.87568668e-02, 1.51071846e+00, 3.23666453e+00,
      8.80776644e-02, 3.17344666e-02, -1.42568946e+00, -6.81839108e-01,
      9.86499727e-01, -8.62023115e-01, -2.89510798e+00, 5.16995750e-02,
      2.80007291e+00, 2.10740137e+00, -1.56880114e-02, 8.75662118e-02,
      -2.21845508e+00, 5.04499264e-02, -4.31714147e-01, -3.21339202e+00,
      3.29588018e-02, 1.45631635e+00, 3.06326652e+00, 6.00287579e-02,
      2.36139800e-02, -1.17813218e+00, -7.13931501e-01, 9.83043611e-01,
      -8.67429912e-01, -2.78804326e+00, 8.03491026e-02, 2.62021279e+00,
      1.98838937e+00, -2.04254091e-02, 1.61702055e-02, -1.94575155e+00,
      7.47671537e-03, -4.53101069e-01, -3.12029028e+00, 2.51329280e-02,
      1.39189970e+00, 2.93106127e+00, 4.32822146e-02, 1.66786201e-02,
      -1.04160810e+00, -7.13407636e-01, 9.80188727e-01, -7.97672987e-01,
      -2.72483420e+00, 6.64640591e-02, 2.40064383e+00, 1.90688789e+00,
      -3.76495831e-02, 5.88394701e-02, -1.79613853e+00, 7.43001979e-03,
      -3.84948999e-01, -3.07561898e+00, 3.60999745e-03, 1.19362915e+00,
      2.87448621e+00, 3.21145281e-02, 4.85449508e-02, -9.04332995e-01,
      -7.18163788e-01, 9.73533273e-01, -7.87938178e-01, -2.65552855e+00,
      8.52754191e-02, 2.23380637e+00, 1.86046886e+00, -3.84659283e-02,
      3.61383939e-03, -1.56824076e+00, -5.16133234e-02, -3.83749515e-01,
      -3.04356480e+00, -1.00123696e-02, 1.12259936e+00, 2.85460925e+00,
      2.27254592e-02, 5.13831228e-02, -8.00966501e-01, -7.54031658e-01,
      9.68655646e-01, -8.35574746e-01, -2.58609843e+00, 1.29007906e-01,
      2.16907358e+00, 1.82082295e+00, -3.12829874e-02, -8.61634016e-02,
      -1.38580561e+00, -1.15106538e-01, -4.53801513e-01, -2.98683906e+00,
      8.89140589e-04, 1.19582200e+00, 2.78207827e+00, 2.04792190e-02,
      2.21894365e-02, -7.72774756e-01, -7.57481635e-01, 9.64338660e-01,
      -7.23216057e-01, -2.56259847e+00, 9.76659954e-02, 1.94227707e+00,
      1.79530525e+00, -4.78870235e-02, -1.37652699e-02, -1.30080318e+00,
      -1.09859906e-01, -3.40479761e-01, -2.98611856e+00, -1.26532083e-02,
      9.42777991e-01, 2.79634452e+00, 1.29091488e-02, 4.82515767e-02,
      -6.40979111e-01, -7.75430501e-01, 9.58329499e-01, -7.77651310e-01,
      -2.51706815e+00, 1.41911715e-01, 1.90270436e+00, 1.80605769e+00,
      -3.64630520e-02, -1.02092788e-01, -1.14984119e+00, -1.90364838e-01,
      -4.21136290e-01, -2.95255637e+00, 4.83086566e-03, 1.04707515e+00,
      2.76535130e+00, 1.31742852e-02, 1.40166143e-02, -6.42032564e-01,
      -7.90972590e-01, 9.49225068e-01, -6.29118145e-01, -2.52049780e+00,
      1.07955687e-01, 1.59935760e+00, 1.86021698e+00, -4.83060554e-02,
      -3.72732617e-02, -1.01803875e+00, -2.34870508e-01, -2.76822150e-01,
      -2.99363422e+00, -1.92096434e-03, 7.27671921e-01, 2.86751962e+00,
      6.11324189e-03, 2.92246770e-02, -4.72499967e-01, -8.54555130e-01,
      9.53172624e-01, -6.26633763e-01, -2.48049402e+00, 1.13544673e-01,
      1.58989286e+00, 1.77231324e+00, -5.63348792e-02, -2.53082402e-02,
      -1.02066648e+00, -1.91134289e-01, -2.83049792e-01, -2.92600870e+00,
      5.48511557e-03, 7.29337931e-01, 2.73405862e+00, 7.37896515e-03,
      2.20676772e-02, -4.68356371e-01, -7.89926410e-01, 9.33483243e-01,
      -5.43464005e-01, -2.51218414e+00, 1.21952273e-01, 1.28384256e+00,
      1.99758422e+00, -3.93517129e-02, -7.38154203e-02, -7.59420276e-01,
      -3.90145332e-01, -2.37070724e-01, -3.01107121e+00, 1.87659375e-02,
      5.72587669e-01, 2.94974279e+00, 4.34542634e-03, -3.15929926e-03,
      -3.44294161e-01, -9.29403841e-01, 9.26011562e-01, -4.13956404e-01,
      -2.52960134e+00, 8.76578316e-02, 1.02632511e+00, 2.07102847e+00,
      -4.14600037e-02, -2.95171011e-02, -6.41459465e-01, -4.40275103e-01,
      -1.22829758e-01, -3.04742455e+00, 1.08981179e-02, 3.27103853e-01,
      3.03460097e+00, 1.22593902e-03, 7.01207528e-03, -2.14928538e-01,
      -9.78972852e-01, 9.29419219e-01, -3.85809600e-01, -2.50791407e+00,
      6.81766644e-02, 9.94096041e-01, 2.00734615e+00, -4.60233726e-02,
      1.69250311e-03, -6.46086514e-01, -3.99518490e-01, -9.76626575e-02,
      -3.01186013e+00, -6.31058181e-04, 2.82722175e-01, 2.96068287e+00,
      1.84389192e-03, 2.00764406e-02, -1.97388962e-01, -9.40410554e-01,
      9.28349376e-01, -4.99486744e-01, -2.48802185e+00, 1.05760038e-01,
      1.12354231e+00, 2.02654767e+00, -2.85774134e-02, -7.63106495e-02,
      -6.28504157e-01, -4.50176626e-01, -2.43202463e-01, -2.96752977e+00,
      1.43932980e-02, 5.51959872e-01, 2.88591790e+00, 4.60150512e-03,
      -6.15263171e-03, -3.11695904e-01, -9.13698614e-01, 9.14265990e-01,
      -3.02708477e-01, -2.55579686e+00, 8.42722505e-02, 7.03972757e-01,
      2.24685264e+00, -2.99737770e-02, -4.56193574e-02, -4.13585216e-01,
      -5.93312025e-01, -7.59954080e-02, -3.05515027e+00, 2.36927383e-02,
      1.85009882e-01, 3.08106256e+00, 1.93516957e-03, -1.51630305e-02,
      -1.12711914e-01, -1.02278519e+00, 9.10931349e-01, -2.39203393e-01,
      -2.56721258e+00, 6.10664934e-02, 5.82691371e-01, 2.28606343e+00,
      -2.90229265e-02, -2.08645314e-02, -3.59273732e-01, -6.18229628e-01,
      -2.74607409e-02, -3.07110119e+00, 1.64843071e-02, 8.44818428e-02,
      3.11696243e+00, 7.64869445e-04, -7.45631987e-03, -6.13492951e-02,
      -1.04300368e+00, 9.09815729e-01, -1.89105302e-01, -2.57093477e+00,
      4.23391126e-02, 4.92311209e-01, 2.29782510e+00, -2.88707931e-02,
      -2.59565597e-04, -3.21794689e-01, -6.25038326e-01, 1.47502329e-02,
      -3.07550359e+00, 9.35270824e-03, -5.29405894e-04, 3.12745357e+00,
      2.25598371e-04, 6.28783528e-05, -1.90158952e-02, -1.04919553e+00,
      9.12930429e-01, -1.80036053e-01, -2.55683136e+00, 2.70035584e-02,
      4.91964906e-01, 2.25303769e+00, -3.07733845e-02, 2.01044790e-02,
      -3.34133685e-01, -5.96447706e-01, 2.24877298e-02, -3.05217171e+00,
      -5.38109853e-05, -9.81961749e-03, 3.07865310e+00, 8.14659579e-04,
      9.89111979e-03, -1.80595182e-02, -1.02360165e+00, 9.27595913e-01,
      -3.76652539e-01, -2.48360872e+00, 7.60483071e-02, 8.61652136e-01,
      2.05746651e+00, -3.47634517e-02, -2.73516383e-02, -5.02253056e-01,
      -4.89019841e-01, -1.67660400e-01, -2.92486715e+00, 1.89854391e-02,
      3.71933609e-01, 2.81727457e+00, 3.53458780e-03, -1.07151447e-02,
      -2.08187938e-01, -8.89669180e-01, 9.11880434e-01, -2.37822190e-01,
      -2.56779742e+00, 6.22837991e-02, 5.26790321e-01, 2.32863355e+00,
      -1.99648216e-02, -3.79042514e-02, -2.93406785e-01, -6.67583585e-01,
      -7.08617046e-02, -3.02098823e+00, 1.80680975e-02, 1.56870082e-01,
      3.02687263e+00, 2.61343713e-03, -1.45323230e-02, -8.70146602e-02,
      -1.00479531e+00, 9.08806920e-01, -1.86307207e-01, -2.58891296e+00,
      4.39250581e-02, 4.27755415e-01, 2.38828874e+00, -1.88797135e-02,
      -1.87276583e-02, -2.48340398e-01, -7.03610361e-01, -3.84689830e-02,
      -3.03905916e+00, 1.49740484e-02, 8.96285921e-02, 3.06607151e+00,
      1.80652342e-03, -1.13497758e-02, -5.25359139e-02, -1.02608657e+00,
      9.08313870e-01, -1.57816350e-01, -2.60463381e+00, 2.90538128e-02,
      3.76097590e-01, 2.42536068e+00, -1.74321290e-02, -4.51003807e-03,
      -2.26456702e-01, -7.24720955e-01, -2.25952677e-02, -3.04820585e+00,
      1.13046207e-02, 5.76076247e-02, 3.08598995e+00, 1.25180325e-03,
      -7.70612527e-03, -3.65823209e-02, -1.03694367e+00, 9.00027752e-01,
      -9.59325805e-02, -2.59513974e+00, 1.52319185e-02, 2.57195473e-01,
      2.43418884e+00, -1.67410746e-02, 9.49971937e-03, -1.70706287e-01,
      -7.34913945e-01, 2.58358419e-02, -3.06000161e+00, 8.78109783e-03,
      -4.17807624e-02, 3.11116791e+00, 7.84700096e-04, -5.13403164e-03,
      1.42266164e-02, -1.05039167e+00, 9.03864622e-01, -6.22696131e-02,
      -2.62898111e+00, 3.31898429e-03, 1.95012450e-01, 2.49523020e+00,
      -1.55631406e-02, 2.08279155e-02, -1.43069237e-01, -7.66230881e-01,
      4.81867567e-02, -3.06897306e+00, 7.65443873e-03, -8.77160132e-02,
      3.13070679e+00, 4.85771190e-04, -4.21880977e-03, 3.77562940e-02,
      -1.06104112e+00, 8.95614743e-01, -4.41224650e-02, -2.59447241e+00,
      -5.60872909e-03, 1.64197996e-01, 2.44829941e+00, -1.52594680e-02,
      2.98546609e-02, -1.31088227e-01, -7.45353222e-01, 6.22775964e-02,
      -3.06686568e+00, 3.54702817e-03, -1.15020946e-01, 3.12626410e+00,
      3.37203062e-04, -2.26430511e-05, 5.08800596e-02, -1.05869186e+00,
      8.99854481e-01, -5.50983809e-02, -2.59910369e+00, -1.40020996e-02,
      1.93264171e-01, 2.44294691e+00, -1.48540195e-02, 3.83154377e-02,
      -1.49831414e-01, -7.39430308e-01, 5.10839298e-02, -3.05875754e+00,
      -7.34488305e-04, -8.98728892e-02, 3.10958314e+00, 4.36116301e-04,
      4.16414300e-03, 3.68692391e-02, -1.05010951e+00, 9.06752944e-01,
      -5.21679483e-02, -2.61345410e+00, -1.51721463e-02, 1.89650029e-01,
      2.45049906e+00, -1.60858817e-02, 4.17035446e-02, -1.49935693e-01,
      -7.39409447e-01, 5.47154397e-02, -3.04715633e+00, -1.76274194e-03,
      -9.69612673e-02, 3.08623672e+00, 9.35875287e-04, 5.06281899e-03,
      4.03308421e-02, -1.03837037e+00, 9.54770803e-01, -6.71819925e-01,
      -2.20851874e+00, 1.13596596e-01, 1.39886665e+00, 1.44119179e+00,
      -3.30680050e-02, -7.71174207e-02, -7.26564229e-01, -1.79418236e-01,
      -4.79131311e-01, -2.52338743e+00, 2.27199476e-02, 9.81826603e-01,
      2.01570463e+00, 9.07849427e-03, -1.92808062e-02, -5.02656817e-01,
      -4.91051823e-01, 9.79444563e-01, -1.04827106e+00, -1.84534228e+00,
      2.79097736e-01, 2.03927016e+00, 6.55083895e-01, -5.69573194e-02,
      -2.26325706e-01, -9.69834328e-01, 2.14651421e-01, -8.12159836e-01,
      -2.04476357e+00, 7.08583072e-02, 1.63569212e+00, 1.04752779e+00,
      1.91790108e-02, -6.99196681e-02, -8.19889009e-01, -2.18319334e-03,
      9.82450843e-01, -1.08352947e+00, -1.75147450e+00, 3.22893053e-01,
      2.08264399e+00, 4.66546446e-01, -7.63730407e-02, -2.43334055e-01,
      -9.84157801e-01, 3.05412143e-01, -8.36240828e-01, -1.93039978e+00,
      8.60961527e-02, 1.67864215e+00, 8.17023575e-01, 1.89721342e-02,
      -8.25787559e-02, -8.39642704e-01, 1.13805212e-01, 1.07250237e+00,
      -7.73365557e-01, 2.37530017e+00, 1.28005242e+00, 1.47353840e+00,
      -8.35273170e+00, -5.40786684e-01, -6.33106768e-01, -7.17929423e-01,
      4.91948414e+00, 1.96515620e-01, 2.44487619e+00, 4.21910405e-01,
      -4.39637542e-01, -8.10014439e+00, 8.44644904e-02, -3.76839638e-01,
      2.44069621e-01, 4.65732241e+00};

  auto w = (clamp(ior, min_ior, max_ior) - min_ior) / (max_ior - min_ior);

  auto s  = w * (num_slices - 1);
  auto i  = (int)s;
  auto ii = min(i + 1, num_slices - 1);
  auto u  = s - i;

  auto  coef = dot(normal, outgoing) >= 0 ? coef_enter : coef_leave;
  float coef1[num_coef];
  float coef2[num_coef];

  for (auto j = 0; j < num_coef; j++) {
    coef1[j] = coef[i * num_coef + j];
    coef2[j] = coef[ii * num_coef + j];
  }

  auto r = sqrt(roughness);
  auto t = abs(dot(normal, outgoing));

  auto E1 = eval_ratpoly2d(coef1, r, t);
  auto E2 = eval_ratpoly2d(coef2, r, t);

  auto E = E1 * (1 - u) + E2 * u;

  return eval_refractive(color, ior, roughness, normal, outgoing, incoming) / E;
}

// Sample a refraction BRDF lobe.
inline vec3f sample_refractive(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  if (rnl < fresnel_dielectric(entering ? ior : (1 / ior), halfway, outgoing)) {
    auto incoming = reflect(outgoing, halfway);
    if (!same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
  } else {
    auto incoming = refract(outgoing, halfway, entering ? (1 / ior) : ior);
    if (same_hemisphere(up_normal, outgoing, incoming)) return {0, 0, 0};
    return incoming;
  }
}

// Pdf for refraction BRDF lobe sampling.
inline float sample_refractive_pdf(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto rel_ior   = entering ? ior : (1 / ior);
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    auto halfway = normalize(incoming + outgoing);
    return fresnel_dielectric(rel_ior, halfway, outgoing) *
           sample_microfacet_pdf(roughness, up_normal, halfway) /
           (4 * abs(dot(outgoing, halfway)));
  } else {
    auto halfway = -normalize(rel_ior * incoming + outgoing) *
                   (entering ? 1.0f : -1.0f);
    // [Walter 2007] equation 17
    return (1 - fresnel_dielectric(rel_ior, halfway, outgoing)) *
           sample_microfacet_pdf(roughness, up_normal, halfway) *
           abs(dot(halfway, incoming)) /
           pow(rel_ior * dot(halfway, incoming) + dot(halfway, outgoing), 2);
  }
}

// Evaluate a delta refraction BRDF lobe.
inline vec3f eval_refractive(const vec3f& color, float ior, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (abs(ior - 1) < 1e-3)
    return dot(normal, incoming) * dot(normal, outgoing) <= 0 ? vec3f{1, 1, 1}
                                                              : vec3f{0, 0, 0};
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto rel_ior   = entering ? ior : (1 / ior);
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return vec3f{1, 1, 1} * fresnel_dielectric(rel_ior, up_normal, outgoing);
  } else {
    return vec3f{1, 1, 1} * (1 / (rel_ior * rel_ior)) *
           (1 - fresnel_dielectric(rel_ior, up_normal, outgoing));
  }
}

// Sample a delta refraction BRDF lobe.
inline vec3f sample_refractive(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, float rnl) {
  if (abs(ior - 1) < 1e-3) return -outgoing;
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto rel_ior   = entering ? ior : (1 / ior);
  if (rnl < fresnel_dielectric(rel_ior, up_normal, outgoing)) {
    return reflect(outgoing, up_normal);
  } else {
    return refract(outgoing, up_normal, 1 / rel_ior);
  }
}

// Pdf for delta refraction BRDF lobe sampling.
inline float sample_refractive_pdf(const vec3f& color, float ior,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (abs(ior - 1) < 1e-3)
    return dot(normal, incoming) * dot(normal, outgoing) < 0 ? 1.0f : 0.0f;
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto rel_ior   = entering ? ior : (1 / ior);
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return fresnel_dielectric(rel_ior, up_normal, outgoing);
  } else {
    return (1 - fresnel_dielectric(rel_ior, up_normal, outgoing));
  }
}

// Evaluate a translucent BRDF lobe.
inline vec3f eval_translucent(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  // TODO (fabio): fix me
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) return zero3f;
  return color / pif * abs(dot(normal, incoming));
}

// Sample a translucency BRDF lobe.
inline vec3f sample_translucent(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec2f& rn) {
  // TODO (fabio): fix me
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return sample_hemisphere_cos(-up_normal, rn);
}

// Pdf for translucency BRDF lobe sampling.
inline float sample_translucent_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  // TODO (fabio): fix me
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) return 0;
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  return sample_hemisphere_cos_pdf(-up_normal, incoming);
}

// Evaluate a passthrough BRDF lobe.
inline vec3f eval_passthrough(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return vec3f{0, 0, 0};
  } else {
    return vec3f{1, 1, 1};
  }
}

// Sample a passthrough BRDF lobe.
inline vec3f sample_passthrough(
    const vec3f& color, const vec3f& normal, const vec3f& outgoing) {
  return -outgoing;
}

// Pdf for passthrough BRDF lobe sampling.
inline float sample_passthrough_pdf(const vec3f& color, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (dot(normal, incoming) * dot(normal, outgoing) >= 0) {
    return 0;
  } else {
    return 1;
  }
}

// Convert mean-free-path to transmission
inline vec3f mfp_to_transmission(const vec3f& mfp, float depth) {
  return exp(-depth / mfp);
}

// Evaluate transmittance
inline vec3f eval_transmittance(const vec3f& density, float distance) {
  return exp(-density * distance);
}

// Sample a distance proportionally to transmittance
inline float sample_transmittance(
    const vec3f& density, float max_distance, float rl, float rd) {
  auto channel  = clamp((int)(rl * 3), 0, 2);
  auto distance = (density[channel] == 0) ? flt_max
                                          : -log(1 - rd) / density[channel];
  return min(distance, max_distance);
}

// Pdf for distance sampling
inline float sample_transmittance_pdf(
    const vec3f& density, float distance, float max_distance) {
  if (distance < max_distance) {
    return sum(density * exp(-density * distance)) / 3;
  } else {
    return sum(exp(-density * max_distance)) / 3;
  }
}

// Evaluate phase function
inline float eval_phasefunction(
    float anisotropy, const vec3f& outgoing, const vec3f& incoming) {
  auto cosine = -dot(outgoing, incoming);
  auto denom  = 1 + anisotropy * anisotropy - 2 * anisotropy * cosine;
  return (1 - anisotropy * anisotropy) / (4 * pif * denom * sqrt(denom));
}

// Sample phase function
inline vec3f sample_phasefunction(
    float anisotropy, const vec3f& outgoing, const vec2f& rn) {
  auto cos_theta = 0.0f;
  if (abs(anisotropy) < 1e-3f) {
    cos_theta = 1 - 2 * rn.y;
  } else {
    auto square = (1 - anisotropy * anisotropy) /
                  (1 + anisotropy - 2 * anisotropy * rn.y);
    cos_theta = (1 + anisotropy * anisotropy - square * square) /
                (2 * anisotropy);
  }

  auto sin_theta      = sqrt(max(0.0f, 1 - cos_theta * cos_theta));
  auto phi            = 2 * pif * rn.x;
  auto local_incoming = vec3f{
      sin_theta * cos(phi), sin_theta * sin(phi), cos_theta};
  return basis_fromz(-outgoing) * local_incoming;
}

// Pdf for phase function sampling
inline float sample_phasefunction_pdf(
    float anisotropy, const vec3f& outgoing, const vec3f& incoming) {
  return eval_phasefunction(anisotropy, outgoing, incoming);
}

// Conductor etas
inline pair<vec3f, vec3f> conductor_eta(const string& name) {
  static const vector<pair<string, pair<vec3f, vec3f>>> metal_ior_table = {
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
  for (auto& [ename, etas] : metal_ior_table) {
    if (ename == name) return etas;
  }
  return {zero3f, zero3f};
}

}  // namespace yocto

#endif
