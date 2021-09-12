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
  auto alpha        = sqrt(roughness);
  auto cos_theta    = abs(dot(normal, outgoing));
  auto reflectivity = eta_to_reflectivity(ior);

  auto E = eval_ratpoly3d(coef, reflectivity, alpha, cos_theta);

  return 1 / E;
}

inline float microfacet_compensation_dielectrics_fit2(const float coef[],
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  auto alpha     = sqrt(roughness);
  auto cos_theta = abs(dot(normal, outgoing));

  auto E = eval_ratpoly2d(coef, alpha, cos_theta);

  return 1 / E;
}

#define ALBEDO_LUT_SIZE 32

// https://github.com/DassaultSystemes-Technology/EnterprisePBRShadingModel/tree/master/res/GGX_E.exr
static const auto albedo_lut = vector<float>{0.9633789f, 0.99560547f,
    0.99853516f, 0.99902344f, 0.9995117f, 0.9995117f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.96240234f,
    0.99560547f, 0.99853516f, 0.99902344f, 0.9995117f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.9995117f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.9326172f, 0.9902344f, 0.99658203f, 0.99853516f, 0.99902344f, 0.9995117f,
    0.9995117f, 0.9995117f, 1.0f, 0.9995117f, 1.0f, 1.0f, 1.0f, 0.9995117f,
    1.0f, 0.9995117f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.8930664f, 0.9633789f,
    0.98583984f, 0.99316406f, 0.99609375f, 0.9975586f, 0.9980469f, 0.99853516f,
    0.99853516f, 0.99902344f, 0.99902344f, 0.99902344f, 0.9995117f, 0.9995117f,
    0.9995117f, 0.9995117f, 0.9995117f, 0.9995117f, 1.0f, 1.0f, 1.0f,
    0.9995117f, 0.9995117f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 0.8984375f, 0.9267578f, 0.9638672f, 0.97998047f, 0.9873047f,
    0.9916992f, 0.9941406f, 0.99560547f, 0.99658203f, 0.9975586f, 0.9975586f,
    0.9980469f, 0.99853516f, 0.99853516f, 0.99853516f, 0.99902344f, 0.99902344f,
    0.99902344f, 0.99902344f, 0.99902344f, 0.9995117f, 0.9995117f, 0.9995117f,
    0.9995117f, 0.9995117f, 0.9995117f, 0.9995117f, 0.9995117f, 0.9995117f,
    0.9995117f, 0.9995117f, 0.9995117f, 0.91503906f, 0.8979492f, 0.9350586f,
    0.9589844f, 0.9736328f, 0.9824219f, 0.9873047f, 0.9897461f, 0.9921875f,
    0.99365234f, 0.9946289f, 0.99560547f, 0.99658203f, 0.99658203f, 0.9970703f,
    0.9975586f, 0.9975586f, 0.9975586f, 0.9980469f, 0.99853516f, 0.9980469f,
    0.99853516f, 0.99853516f, 0.99902344f, 0.99853516f, 0.99902344f,
    0.99853516f, 0.99902344f, 0.99853516f, 0.99902344f, 0.99902344f,
    0.99902344f, 0.93310547f, 0.8911133f, 0.9082031f, 0.93359375f, 0.95214844f,
    0.9658203f, 0.9741211f, 0.97998047f, 0.9848633f, 0.9873047f, 0.9902344f,
    0.99121094f, 0.9926758f, 0.99365234f, 0.9946289f, 0.9951172f, 0.9951172f,
    0.99560547f, 0.99609375f, 0.99658203f, 0.99609375f, 0.9970703f, 0.9970703f,
    0.9970703f, 0.9975586f, 0.9975586f, 0.9975586f, 0.9980469f, 0.9980469f,
    0.9980469f, 0.9980469f, 0.9980469f, 0.9453125f, 0.8930664f, 0.8935547f,
    0.9111328f, 0.9301758f, 0.94628906f, 0.95654297f, 0.9658203f, 0.97216797f,
    0.9770508f, 0.98095703f, 0.984375f, 0.98583984f, 0.98876953f, 0.9897461f,
    0.9897461f, 0.99121094f, 0.9921875f, 0.9926758f, 0.99365234f, 0.9941406f,
    0.9946289f, 0.9946289f, 0.99560547f, 0.99560547f, 0.99560547f, 0.99609375f,
    0.99609375f, 0.99658203f, 0.99658203f, 0.99658203f, 0.9970703f, 0.95458984f,
    0.90234375f, 0.88623047f, 0.8955078f, 0.9086914f, 0.9243164f, 0.93652344f,
    0.9477539f, 0.95654297f, 0.9628906f, 0.9692383f, 0.97314453f, 0.9770508f,
    0.9794922f, 0.9819336f, 0.984375f, 0.98583984f, 0.9868164f, 0.98828125f,
    0.9892578f, 0.9897461f, 0.99121094f, 0.99121094f, 0.9916992f, 0.9926758f,
    0.9921875f, 0.99316406f, 0.99365234f, 0.99365234f, 0.9941406f, 0.9941406f,
    0.9941406f, 0.96191406f, 0.91064453f, 0.88671875f, 0.88378906f, 0.8925781f,
    0.90625f, 0.91748047f, 0.92871094f, 0.9379883f, 0.94628906f, 0.953125f,
    0.95996094f, 0.96435547f, 0.9692383f, 0.97216797f, 0.9741211f, 0.97753906f,
    0.9794922f, 0.9814453f, 0.9838867f, 0.984375f, 0.9848633f, 0.98535156f,
    0.9873047f, 0.9868164f, 0.98876953f, 0.98876953f, 0.9897461f, 0.9902344f,
    0.99072266f, 0.9902344f, 0.9902344f, 0.9663086f, 0.9199219f, 0.890625f,
    0.8833008f, 0.8828125f, 0.88964844f, 0.89990234f, 0.9086914f, 0.9189453f,
    0.9277344f, 0.93603516f, 0.94433594f, 0.94970703f, 0.95458984f, 0.9580078f,
    0.9614258f, 0.9658203f, 0.96875f, 0.9716797f, 0.97265625f, 0.97558594f,
    0.97802734f, 0.9790039f, 0.97998047f, 0.9814453f, 0.9824219f, 0.98291016f,
    0.98339844f, 0.984375f, 0.984375f, 0.98583984f, 0.98535156f, 0.9692383f,
    0.92578125f, 0.89746094f, 0.8828125f, 0.8774414f, 0.87890625f, 0.8852539f,
    0.8925781f, 0.8989258f, 0.9091797f, 0.91748047f, 0.9248047f, 0.93115234f,
    0.9370117f, 0.9423828f, 0.94677734f, 0.95214844f, 0.9550781f, 0.9584961f,
    0.96240234f, 0.96484375f, 0.9658203f, 0.9692383f, 0.96972656f, 0.97216797f,
    0.97314453f, 0.97509766f, 0.97558594f, 0.9765625f, 0.97753906f, 0.97802734f,
    0.9794922f, 0.97216797f, 0.93066406f, 0.9013672f, 0.88378906f, 0.87353516f,
    0.8725586f, 0.87060547f, 0.8774414f, 0.8833008f, 0.8911133f, 0.8989258f,
    0.9042969f, 0.9111328f, 0.91796875f, 0.9238281f, 0.93066406f, 0.93652344f,
    0.9394531f, 0.9423828f, 0.9477539f, 0.94970703f, 0.9526367f, 0.9555664f,
    0.95751953f, 0.9589844f, 0.96240234f, 0.9638672f, 0.9663086f, 0.9663086f,
    0.9682617f, 0.9692383f, 0.96972656f, 0.97265625f, 0.9345703f, 0.9038086f,
    0.8852539f, 0.8720703f, 0.86572266f, 0.8642578f, 0.86572266f, 0.86865234f,
    0.8754883f, 0.8798828f, 0.8857422f, 0.89404297f, 0.89941406f, 0.9038086f,
    0.9091797f, 0.91503906f, 0.9189453f, 0.9238281f, 0.9291992f, 0.93359375f,
    0.9355469f, 0.93896484f, 0.9428711f, 0.9453125f, 0.94677734f, 0.94970703f,
    0.95214844f, 0.95410156f, 0.9555664f, 0.95751953f, 0.9604492f, 0.9741211f,
    0.93603516f, 0.9067383f, 0.8847656f, 0.87060547f, 0.8618164f, 0.8569336f,
    0.85498047f, 0.85546875f, 0.8569336f, 0.8623047f, 0.8671875f, 0.8720703f,
    0.8779297f, 0.88378906f, 0.88964844f, 0.89404297f, 0.89941406f, 0.9038086f,
    0.9067383f, 0.9116211f, 0.9165039f, 0.9194336f, 0.92285156f, 0.92578125f,
    0.9296875f, 0.93115234f, 0.9355469f, 0.9379883f, 0.9394531f, 0.94189453f,
    0.94384766f, 0.9741211f, 0.9375f, 0.90771484f, 0.88623047f, 0.8691406f,
    0.8569336f, 0.8496094f, 0.84716797f, 0.8442383f, 0.8442383f, 0.8486328f,
    0.84814453f, 0.8540039f, 0.8569336f, 0.8618164f, 0.8666992f, 0.8701172f,
    0.875f, 0.8808594f, 0.8852539f, 0.8876953f, 0.8930664f, 0.89697266f,
    0.90185547f, 0.9067383f, 0.90966797f, 0.9116211f, 0.9169922f, 0.9189453f,
    0.92041016f, 0.9223633f, 0.92626953f, 0.97265625f, 0.93603516f, 0.9086914f,
    0.8857422f, 0.8671875f, 0.8535156f, 0.8442383f, 0.8388672f, 0.8330078f,
    0.82958984f, 0.83203125f, 0.83203125f, 0.83251953f, 0.83496094f, 0.8388672f,
    0.8466797f, 0.8486328f, 0.85302734f, 0.8564453f, 0.8613281f, 0.8652344f,
    0.8666992f, 0.87353516f, 0.8769531f, 0.87890625f, 0.8847656f, 0.8876953f,
    0.8911133f, 0.89501953f, 0.8984375f, 0.9013672f, 0.9013672f, 0.9716797f,
    0.93408203f, 0.90527344f, 0.88134766f, 0.86376953f, 0.85009766f, 0.8378906f,
    0.8286133f, 0.8227539f, 0.8183594f, 0.81591797f, 0.8173828f, 0.81591797f,
    0.8173828f, 0.81689453f, 0.8203125f, 0.8227539f, 0.8276367f, 0.8310547f,
    0.83251953f, 0.8354492f, 0.8442383f, 0.8466797f, 0.8491211f, 0.85595703f,
    0.8574219f, 0.8598633f, 0.86572266f, 0.8666992f, 0.87060547f, 0.87402344f,
    0.87646484f, 0.9711914f, 0.93408203f, 0.9038086f, 0.8774414f, 0.85791016f,
    0.8432617f, 0.8305664f, 0.8208008f, 0.8129883f, 0.80566406f, 0.8017578f,
    0.79833984f, 0.79833984f, 0.79589844f, 0.79541016f, 0.7973633f, 0.80029297f,
    0.80029297f, 0.8046875f, 0.8071289f, 0.80908203f, 0.8129883f, 0.8173828f,
    0.8183594f, 0.8222656f, 0.8256836f, 0.8300781f, 0.8334961f, 0.8378906f,
    0.83984375f, 0.8442383f, 0.84521484f, 0.9692383f, 0.92871094f, 0.8984375f,
    0.8730469f, 0.85302734f, 0.83740234f, 0.8203125f, 0.8105469f, 0.7998047f,
    0.7944336f, 0.78759766f, 0.7817383f, 0.7763672f, 0.7788086f, 0.77734375f,
    0.77490234f, 0.7739258f, 0.77783203f, 0.77734375f, 0.7788086f, 0.78271484f,
    0.78271484f, 0.7832031f, 0.7890625f, 0.7910156f, 0.79296875f, 0.79833984f,
    0.7998047f, 0.80322266f, 0.8041992f, 0.8095703f, 0.8120117f, 0.96777344f,
    0.92578125f, 0.8955078f, 0.8691406f, 0.84521484f, 0.828125f, 0.81347656f,
    0.79785156f, 0.7866211f, 0.7783203f, 0.77197266f, 0.7661133f, 0.76123047f,
    0.7573242f, 0.75146484f, 0.75341797f, 0.7504883f, 0.7495117f, 0.75097656f,
    0.7480469f, 0.75097656f, 0.75341797f, 0.7548828f, 0.75390625f, 0.75878906f,
    0.75927734f, 0.76123047f, 0.7636719f, 0.76904297f, 0.7680664f, 0.77246094f,
    0.77734375f, 0.9658203f, 0.9223633f, 0.8881836f, 0.8598633f, 0.8359375f,
    0.8173828f, 0.80078125f, 0.7871094f, 0.77685547f, 0.7631836f, 0.7558594f,
    0.74853516f, 0.74121094f, 0.7363281f, 0.73291016f, 0.7270508f, 0.7241211f,
    0.72216797f, 0.72314453f, 0.72021484f, 0.7216797f, 0.71875f, 0.71875f,
    0.7216797f, 0.7246094f, 0.7236328f, 0.72509766f, 0.72558594f, 0.7285156f,
    0.7314453f, 0.7314453f, 0.7324219f, 0.9633789f, 0.91748047f, 0.88134766f,
    0.8520508f, 0.82910156f, 0.80810547f, 0.7885742f, 0.77441406f, 0.7602539f,
    0.74658203f, 0.734375f, 0.72998047f, 0.7192383f, 0.7128906f, 0.70703125f,
    0.7036133f, 0.6982422f, 0.69433594f, 0.6923828f, 0.6904297f, 0.6904297f,
    0.68896484f, 0.6875f, 0.68847656f, 0.6875f, 0.68652344f, 0.6875f, 0.6875f,
    0.6894531f, 0.69091797f, 0.69091797f, 0.6928711f, 0.9604492f, 0.9121094f,
    0.87353516f, 0.84375f, 0.81884766f, 0.79345703f, 0.7758789f, 0.7636719f,
    0.7446289f, 0.7319336f, 0.71875f, 0.70996094f, 0.70166016f, 0.6928711f,
    0.68359375f, 0.6801758f, 0.671875f, 0.66796875f, 0.6665039f, 0.65771484f,
    0.6591797f, 0.6533203f, 0.6533203f, 0.6538086f, 0.6503906f, 0.64990234f,
    0.64941406f, 0.64746094f, 0.64941406f, 0.6489258f, 0.6489258f, 0.64941406f,
    0.9589844f, 0.90722656f, 0.86621094f, 0.8339844f, 0.8071289f, 0.7832031f,
    0.76171875f, 0.74560547f, 0.72802734f, 0.71191406f, 0.7001953f, 0.6899414f,
    0.6777344f, 0.6669922f, 0.66259766f, 0.65185547f, 0.6455078f, 0.64453125f,
    0.6352539f, 0.6323242f, 0.62597656f, 0.6220703f, 0.6191406f, 0.6142578f,
    0.6142578f, 0.61328125f, 0.6074219f, 0.60546875f, 0.6074219f, 0.60791016f,
    0.6064453f, 0.60546875f, 0.9555664f, 0.9003906f, 0.8588867f, 0.8251953f,
    0.7949219f, 0.7705078f, 0.7504883f, 0.72753906f, 0.70996094f, 0.6953125f,
    0.6791992f, 0.66748047f, 0.6591797f, 0.64697266f, 0.6401367f, 0.6279297f,
    0.61816406f, 0.6142578f, 0.60791016f, 0.6015625f, 0.59375f, 0.59033203f,
    0.5839844f, 0.57910156f, 0.578125f, 0.5722656f, 0.5708008f, 0.56640625f,
    0.5654297f, 0.5654297f, 0.56152344f, 0.5625f, 0.95214844f, 0.89501953f,
    0.8491211f, 0.81347656f, 0.78515625f, 0.7553711f, 0.73339844f, 0.7133789f,
    0.69140625f, 0.6772461f, 0.66015625f, 0.6484375f, 0.63183594f, 0.62353516f,
    0.61035156f, 0.6015625f, 0.5957031f, 0.5878906f, 0.578125f, 0.5673828f,
    0.5625f, 0.5576172f, 0.5527344f, 0.54785156f, 0.5410156f, 0.53564453f,
    0.52978516f, 0.52734375f, 0.5253906f, 0.5229492f, 0.5205078f, 0.5161133f,
    0.94970703f, 0.88720703f, 0.8408203f, 0.8017578f, 0.76904297f, 0.7421875f,
    0.7163086f, 0.69433594f, 0.6743164f, 0.6582031f, 0.6381836f, 0.6230469f,
    0.6088867f, 0.59716797f, 0.5859375f, 0.57666016f, 0.56152344f, 0.5551758f,
    0.5488281f, 0.5395508f, 0.5307617f, 0.5253906f, 0.52001953f, 0.51171875f,
    0.50634766f, 0.5004883f, 0.49438477f, 0.49023438f, 0.4819336f, 0.48486328f,
    0.4802246f, 0.47705078f, 0.9472656f, 0.8808594f, 0.8305664f, 0.79003906f,
    0.7573242f, 0.7260742f, 0.7001953f, 0.6777344f, 0.65478516f, 0.63427734f,
    0.6166992f, 0.6040039f, 0.5883789f, 0.57421875f, 0.5595703f, 0.54833984f,
    0.53808594f, 0.52734375f, 0.5175781f, 0.50683594f, 0.49975586f, 0.48950195f,
    0.4855957f, 0.47827148f, 0.47216797f, 0.46362305f, 0.45947266f, 0.45385742f,
    0.44750977f, 0.44384766f, 0.43920898f, 0.4321289f, 0.94384766f, 0.8745117f,
    0.8198242f, 0.7753906f, 0.7421875f, 0.7084961f, 0.68408203f, 0.6557617f,
    0.6352539f, 0.6166992f, 0.5957031f, 0.5800781f, 0.56152344f, 0.546875f,
    0.5341797f, 0.52441406f, 0.50683594f, 0.49926758f, 0.48754883f, 0.47802734f,
    0.46777344f, 0.4621582f, 0.45483398f, 0.4465332f, 0.4404297f, 0.42871094f,
    0.42578125f, 0.4140625f, 0.40966797f, 0.40454102f, 0.4008789f, 0.3972168f,
    0.9394531f, 0.8652344f, 0.8100586f, 0.765625f, 0.7265625f, 0.69433594f,
    0.6669922f, 0.63964844f, 0.6152344f, 0.59277344f, 0.5776367f, 0.5546875f,
    0.54052734f, 0.52490234f, 0.51123047f, 0.49682617f, 0.48486328f, 0.4716797f,
    0.46166992f, 0.45092773f, 0.43920898f, 0.43237305f, 0.42016602f,
    0.41430664f, 0.40600586f, 0.3972168f, 0.38916016f, 0.3828125f, 0.37402344f,
    0.37231445f, 0.36694336f, 0.36010742f, 0.9370117f, 0.85791016f, 0.7993164f,
    0.75439453f, 0.71484375f, 0.67871094f, 0.6489258f, 0.61816406f, 0.5961914f,
    0.5727539f, 0.5522461f, 0.5317383f, 0.51660156f, 0.49804688f, 0.48364258f,
    0.46899414f, 0.4580078f, 0.44335938f, 0.43139648f, 0.42260742f, 0.4111328f,
    0.39916992f, 0.39135742f, 0.3840332f, 0.37597656f, 0.3684082f, 0.359375f,
    0.35253906f, 0.34399414f, 0.33935547f, 0.33129883f, 0.3256836f};

static const auto my_albedo_lut = vector<float>{0.9987483f, 0.9995335f,
    0.999886f, 0.99994946f, 0.9999715f, 0.99998164f, 0.9999871f, 0.9999904f,
    0.99999255f, 0.99999404f, 0.99999505f, 0.9999958f, 0.9999964f, 0.9999969f,
    0.99999726f, 0.99999756f, 0.9999978f, 0.999998f, 0.99999815f, 0.99999833f,
    0.99999845f, 0.9999985f, 0.9999986f, 0.9999987f, 0.99999875f, 0.9999988f,
    0.99999887f, 0.9999989f, 0.999999f, 0.999999f, 0.99999905f, 0.99999905f,
    0.99852264f, 0.9994506f, 0.99986607f, 0.9999407f, 0.99996656f, 0.9999784f,
    0.99998486f, 0.9999888f, 0.9999913f, 0.999993f, 0.9999942f, 0.9999951f,
    0.9999958f, 0.99999636f, 0.9999968f, 0.99999714f, 0.99999744f, 0.9999977f,
    0.99999785f, 0.99999803f, 0.99999815f, 0.9999983f, 0.9999984f, 0.99999845f,
    0.99999857f, 0.9999986f, 0.9999987f, 0.99999875f, 0.9999988f, 0.99999887f,
    0.99999887f, 0.9999989f, 0.97428286f, 0.9900636f, 0.9976743f, 0.9989968f,
    0.99944216f, 0.99964374f, 0.9997517f, 0.9998162f, 0.9998577f, 0.9998861f,
    0.99990624f, 0.99992114f, 0.99993247f, 0.99994123f, 0.99994814f,
    0.99995375f, 0.99995834f, 0.9999621f, 0.9999653f, 0.999968f, 0.99997026f,
    0.9999722f, 0.9999739f, 0.9999754f, 0.9999767f, 0.9999779f, 0.9999789f,
    0.9999798f, 0.99998057f, 0.99998134f, 0.999982f, 0.9999826f, 0.9161118f,
    0.95421505f, 0.98729444f, 0.9945121f, 0.99698454f, 0.9980949f, 0.9986834f,
    0.99903166f, 0.99925435f, 0.9994053f, 0.9995123f, 0.9995909f, 0.9996503f,
    0.99969625f, 0.9997326f, 0.9997618f, 0.9997856f, 0.99980533f, 0.9998218f,
    0.99983567f, 0.99984753f, 0.99985766f, 0.9998665f, 0.9998742f, 0.9998809f,
    0.9998868f, 0.99989206f, 0.99989676f, 0.99990094f, 0.9999047f, 0.99990803f,
    0.9999111f, 0.8839908f, 0.9084669f, 0.96205616f, 0.9820045f, 0.989903f,
    0.9936095f, 0.9955992f, 0.99677855f, 0.99753124f, 0.9980395f, 0.9983983f,
    0.99866086f, 0.9988586f, 0.99901116f, 0.9991314f, 0.99922776f, 0.9993062f,
    0.9993709f, 0.9994248f, 0.99947035f, 0.99950904f, 0.9995423f, 0.99957097f,
    0.999596f, 0.9996179f, 0.9996371f, 0.9996542f, 0.9996694f, 0.9996829f,
    0.99969506f, 0.99970603f, 0.9997159f, 0.88281846f, 0.8844622f, 0.92777115f,
    0.9590807f, 0.97530717f, 0.98389596f, 0.98877746f, 0.99175316f, 0.9936787f,
    0.99498755f, 0.9959139f, 0.996592f, 0.9971023f, 0.99749565f, 0.997805f,
    0.99805254f, 0.9982537f, 0.9984193f, 0.9985572f, 0.9986733f, 0.99877197f,
    0.9988564f, 0.9989294f, 0.9989928f, 0.9990483f, 0.9990971f, 0.99914026f,
    0.9991786f, 0.9992128f, 0.9992435f, 0.9992711f, 0.99929607f, 0.89108163f,
    0.8807155f, 0.8992527f, 0.93051773f, 0.9531632f, 0.9675453f, 0.9766192f,
    0.98250335f, 0.98645675f, 0.9892072f, 0.99118257f, 0.9926416f, 0.9937461f,
    0.99460024f, 0.99527323f, 0.9958123f, 0.99625033f, 0.99661094f, 0.9969112f,
    0.9971638f, 0.9973782f, 0.99756175f, 0.99772006f, 0.9978575f, 0.9979777f,
    0.9980833f, 0.99817663f, 0.9982595f, 0.9983334f, 0.9983996f, 0.99845916f,
    0.9985129f, 0.89924264f, 0.88536507f, 0.8833017f, 0.9048411f, 0.92776746f,
    0.94584006f, 0.9589296f, 0.9682296f, 0.97487557f, 0.9796996f, 0.9832681f,
    0.9859593f, 0.9880269f, 0.989643f, 0.9909261f, 0.9919595f, 0.9928026f,
    0.9934986f, 0.99407923f, 0.9945683f, 0.9949838f, 0.9953397f, 0.9956467f,
    0.9959134f, 0.99614644f, 0.9963512f, 0.9965321f, 0.99669266f, 0.9968359f,
    0.9969641f, 0.9970794f, 0.9971833f, 0.9051134f, 0.8914494f, 0.87756497f,
    0.88697624f, 0.90469223f, 0.92250854f, 0.9376076f, 0.949589f, 0.95886546f,
    0.96600795f, 0.9715287f, 0.9758322f, 0.9792225f, 0.9819236f, 0.98410016f,
    0.98587334f, 0.98733294f, 0.9885462f, 0.9895639f, 0.9904248f, 0.9911587f,
    0.991789f, 0.9923338f, 0.9928078f, 0.9932225f, 0.99358726f, 0.9939098f,
    0.9941962f, 0.99445164f, 0.99468046f, 0.9948862f, 0.9950718f, 0.90862095f,
    0.89634943f, 0.8773364f, 0.87685066f, 0.8871351f, 0.90136176f, 0.9157365f,
    0.9286302f, 0.93957156f, 0.9486121f, 0.9559958f, 0.962007f, 0.9669086f,
    0.9709229f, 0.97423f, 0.97697276f, 0.97926354f, 0.9811904f, 0.9828223f,
    0.9842138f, 0.985408f, 0.98643905f, 0.98733443f, 0.98811626f, 0.98880243f,
    0.9894076f, 0.98994374f, 0.99042076f, 0.99084693f, 0.9912292f, 0.99157315f,
    0.99188375f, 0.9100671f, 0.8993535f, 0.8790211f, 0.87202406f, 0.87531334f,
    0.8844228f, 0.8959386f, 0.9077914f, 0.91891134f, 0.9288462f, 0.9374846f,
    0.9448847f, 0.95117635f, 0.95650995f, 0.9610316f, 0.96487224f, 0.9681447f,
    0.9709438f, 0.9733483f, 0.9754233f, 0.9772221f, 0.9787887f, 0.9801592f,
    0.9813635f, 0.98242617f, 0.9833678f, 0.98420537f, 0.9849532f, 0.9856233f,
    0.98622584f, 0.9867693f, 0.9872611f, 0.9097146f, 0.90038806f, 0.88054985f,
    0.8698843f, 0.86777693f, 0.8717926f, 0.87944925f, 0.88884914f, 0.89873457f,
    0.9083549f, 0.91731024f, 0.9254246f, 0.9326551f, 0.9390326f, 0.94462454f,
    0.9495131f, 0.95378244f, 0.9575126f, 0.96077615f, 0.96363735f, 0.9661521f,
    0.9683684f, 0.9703277f, 0.972065f, 0.9736103f, 0.97498906f, 0.976223f,
    0.97733074f, 0.9783281f, 0.9792286f, 0.980044f, 0.9807842f, 0.9077287f,
    0.8995387f, 0.8808933f, 0.8684735f, 0.86268824f, 0.86245f, 0.86620647f,
    0.8725072f, 0.8802071f, 0.8884857f, 0.8967966f, 0.90480065f, 0.912306f,
    0.91921985f, 0.92551345f, 0.93119717f, 0.9363039f, 0.94087803f, 0.94496834f,
    0.94862396f, 0.95189196f, 0.9548158f, 0.9574351f, 0.9597852f, 0.9618977f,
    0.96380025f, 0.96551734f, 0.9670703f, 0.9684779f, 0.9697564f, 0.9709203f,
    0.9719821f, 0.90419525f, 0.8968991f, 0.87958896f, 0.8665415f, 0.8584674f,
    0.85503435f, 0.8553944f, 0.8585726f, 0.863674f, 0.8699648f, 0.876884f,
    0.8840252f, 0.89110756f, 0.89794695f, 0.9044309f, 0.91049796f, 0.91612214f,
    0.9213011f, 0.926048f, 0.9303849f, 0.93433905f, 0.93793994f, 0.9412175f,
    0.94420063f, 0.946917f, 0.9493921f, 0.95164955f, 0.9537108f, 0.95559525f,
    0.9573205f, 0.95890224f, 0.9603546f, 0.89914805f, 0.89253515f, 0.87645006f,
    0.8633613f, 0.85396636f, 0.84829426f, 0.84595066f, 0.8463466f, 0.84885544f,
    0.85289925f, 0.85798687f, 0.8637223f, 0.86979926f, 0.8759884f, 0.88212436f,
    0.8880918f, 0.89381444f, 0.89924544f, 0.90435946f, 0.9091467f, 0.9136085f,
    0.91775346f, 0.9215951f, 0.9251498f, 0.92843556f, 0.93147093f, 0.9342744f,
    0.93686366f, 0.939256f, 0.94146746f, 0.943513f, 0.94540656f, 0.8925907f,
    0.88648534f, 0.8714148f, 0.858538f, 0.8484287f, 0.8412529f, 0.83687264f,
    0.8349674f, 0.8351352f, 0.83696187f, 0.8400605f, 0.8440916f, 0.84876966f,
    0.85386163f, 0.8591826f, 0.86458915f, 0.8699728f, 0.8752537f, 0.88037485f,
    0.88529754f, 0.88999695f, 0.8944592f, 0.8986785f, 0.902655f, 0.90639323f,
    0.90990067f, 0.9131869f, 0.9162627f, 0.9191395f, 0.9218291f, 0.9243432f,
    0.9266933f, 0.8845124f, 0.87876964f, 0.86447597f, 0.8518729f, 0.8413913f,
    0.83322066f, 0.8273488f, 0.82362133f, 0.8217994f, 0.82160676f, 0.82276195f,
    0.8249989f, 0.82807755f, 0.83178854f, 0.835954f, 0.8404259f, 0.8450832f,
    0.8498287f, 0.8545857f, 0.85929465f, 0.8639103f, 0.86839944f, 0.8727382f,
    0.87691045f, 0.88090634f, 0.8847206f, 0.8883519f, 0.8918014f, 0.89507276f,
    0.8981709f, 0.901102f, 0.9038728f, 0.87489957f, 0.86939996f, 0.85565126f,
    0.8432808f, 0.8325916f, 0.8237475f, 0.8167854f, 0.8116421f, 0.80818486f,
    0.8062396f, 0.8056136f, 0.80611134f, 0.8075457f, 0.809744f, 0.8125512f,
    0.8158312f, 0.8194663f, 0.8233564f, 0.82741725f, 0.8315791f, 0.83578444f,
    0.8399869f, 0.8441494f, 0.8482427f, 0.85224426f, 0.8561372f, 0.8599093f,
    0.863552f, 0.86705995f, 0.87043023f, 0.87366205f, 0.87675613f, 0.8637434f,
    0.8583885f, 0.84497255f, 0.832743f, 0.82189995f, 0.81256515f, 0.80478394f,
    0.7985362f, 0.79375124f, 0.7903233f, 0.78812546f, 0.78702086f, 0.78687125f,
    0.7875426f, 0.788909f, 0.79085505f, 0.7932763f, 0.79608005f, 0.7991846f,
    0.8025191f, 0.8060221f, 0.80964136f, 0.8133324f, 0.8170579f, 0.82078683f,
    0.82449347f, 0.8281569f, 0.8317602f, 0.8352902f, 0.8387364f, 0.842091f,
    0.8453483f, 0.8510459f, 0.845754f, 0.8324832f, 0.820283f, 0.80927473f,
    0.79953706f, 0.7911049f, 0.78397316f, 0.778103f, 0.77343f, 0.7698716f,
    0.7673343f, 0.76571935f, 0.7649271f, 0.7648603f, 0.7654263f, 0.7665385f,
    0.768117f, 0.77008915f, 0.7723893f, 0.7749588f, 0.7777455f, 0.7807035f,
    0.7837923f, 0.786977f, 0.79022694f, 0.793516f, 0.79682165f, 0.80012476f,
    0.8034092f, 0.8066615f, 0.8098703f, 0.8368233f, 0.83152705f, 0.81823915f,
    0.8059541f, 0.7947344f, 0.7846211f, 0.77563184f, 0.76776135f, 0.7609841f,
    0.7552577f, 0.75052696f, 0.74672747f, 0.74378926f, 0.74163973f, 0.7402059f,
    0.7394162f, 0.7392018f, 0.73949754f, 0.7402423f, 0.74137944f, 0.74285674f,
    0.7446267f, 0.74664587f, 0.74887526f, 0.75127965f, 0.75382745f, 0.7564906f,
    0.7592441f, 0.76206577f, 0.7649362f, 0.76783824f, 0.77075684f, 0.8211088f,
    0.8157528f, 0.80231047f, 0.7898334f, 0.7783412f, 0.767844f, 0.75834143f,
    0.74982125f, 0.7422605f, 0.73562676f, 0.7298793f, 0.7249717f, 0.72085303f,
    0.71747f, 0.71476793f, 0.7126926f, 0.7111906f, 0.71021026f, 0.7097022f,
    0.70961964f, 0.7099187f, 0.7105582f, 0.71150005f, 0.712709f, 0.7141526f,
    0.71580106f, 0.7176271f, 0.71960604f, 0.7217152f, 0.7239342f, 0.72624445f,
    0.7286294f, 0.803954f, 0.79849327f, 0.78478265f, 0.77201897f, 0.7601904f,
    0.7492834f, 0.7392809f, 0.7301616f, 0.72189987f, 0.7144655f, 0.7078245f,
    0.70194f, 0.69677246f, 0.69228095f, 0.6884237f, 0.6851589f, 0.6824451f,
    0.6802418f, 0.67850983f, 0.6772114f, 0.6763106f, 0.6757733f, 0.67556745f,
    0.6756627f, 0.67603076f, 0.67664516f, 0.6774815f, 0.67851675f, 0.67972994f,
    0.6811014f, 0.6826132f, 0.6842486f, 0.78542936f, 0.779828f, 0.76575756f,
    0.7526279f, 0.7404046f, 0.72905594f, 0.7185511f, 0.7088597f, 0.699951f,
    0.6917934f, 0.68435466f, 0.6776018f, 0.6715014f, 0.6660195f, 0.6611224f,
    0.65677637f, 0.6529485f, 0.64960635f, 0.6467185f, 0.6442547f, 0.6421857f,
    0.6404836f, 0.6391219f, 0.6380753f, 0.63732f, 0.63683337f, 0.63659424f,
    0.6365826f, 0.6367798f, 0.6371682f, 0.6377315f, 0.6384543f, 0.76562357f,
    0.7598543f, 0.7453533f, 0.7317949f, 0.71912855f, 0.70730865f, 0.6962937f,
    0.68604505f, 0.67652637f, 0.66770315f, 0.659542f, 0.65201086f, 0.6450784f,
    0.63871425f, 0.63288885f, 0.6275734f, 0.62274003f, 0.6183619f, 0.6144128f,
    0.61086774f, 0.60770273f, 0.60489464f, 0.6024215f, 0.60026234f, 0.5983972f,
    0.5968071f, 0.5954741f, 0.5943812f, 0.5935123f, 0.59285223f, 0.59238666f,
    0.59210205f, 0.744643f, 0.73868597f, 0.7237032f, 0.70967054f, 0.69652545f,
    0.6842123f, 0.6726811f, 0.6618864f, 0.6517868f, 0.6423441f, 0.6335229f,
    0.62529004f, 0.61761427f, 0.61046636f, 0.6038184f, 0.59764403f, 0.5919181f,
    0.5866167f, 0.58171713f, 0.5771976f, 0.57303756f, 0.5692172f, 0.5657179f,
    0.56252176f, 0.5596119f, 0.5569722f, 0.5545875f, 0.5524431f, 0.5505254f,
    0.54882145f, 0.54731876f, 0.5460058f, 0.72260916f, 0.7164517f, 0.7009545f,
    0.6864188f, 0.6727735f, 0.6599555f, 0.6479082f, 0.636581f, 0.6259279f,
    0.6159075f, 0.60648155f, 0.59761536f, 0.58927673f, 0.581436f, 0.57406545f,
    0.56713945f, 0.560634f, 0.5545265f, 0.5487959f, 0.54342216f, 0.53838664f,
    0.5336716f, 0.5292605f, 0.5251374f, 0.52128756f, 0.51769674f, 0.5143516f,
    0.5112396f, 0.50834876f, 0.5056677f, 0.5031857f, 0.50089264f, 0.6996572f,
    0.69329304f, 0.67726564f, 0.66221434f, 0.6480618f, 0.6347393f, 0.62218535f,
    0.61034507f, 0.599169f, 0.5886126f, 0.5786354f, 0.56920063f, 0.5602747f,
    0.551827f, 0.54382926f, 0.5362555f, 0.52908164f, 0.5222856f, 0.5158466f,
    0.5097456f, 0.5039645f, 0.4984867f, 0.49329647f, 0.48837918f, 0.48372105f,
    0.4793091f, 0.4751312f, 0.47117588f, 0.4674323f, 0.46389022f, 0.4605401f,
    0.45737273f, 0.6759327f, 0.6693614f, 0.6528038f, 0.6372396f, 0.6225868f,
    0.60877246f, 0.5957315f, 0.58340573f, 0.57174283f, 0.56069577f, 0.5502219f,
    0.54028285f, 0.5308434f, 0.52187175f, 0.5133387f, 0.50521755f, 0.49748376f,
    0.49011484f, 0.48308992f, 0.47638983f, 0.46999672f, 0.4638941f, 0.45806658f,
    0.45249993f, 0.44718078f, 0.44209668f, 0.43723598f, 0.4325878f, 0.42814186f,
    0.4238886f, 0.41981894f, 0.4159244f, 0.6515887f, 0.6448152f, 0.6277414f,
    0.6116808f, 0.5965479f, 0.5822665f, 0.5687687f, 0.55599374f, 0.543887f,
    0.5323996f, 0.5214871f, 0.5111094f, 0.5012302f, 0.49181625f, 0.48283753f,
    0.47426644f, 0.4660777f, 0.45824817f, 0.45075658f, 0.4435833f, 0.43671024f,
    0.43012065f, 0.42379907f, 0.4177311f, 0.4119034f, 0.40630358f, 0.40092006f,
    0.39574206f, 0.39075947f, 0.38596284f, 0.3813434f, 0.37689283f, 0.6267824f,
    0.6198164f, 0.6022529f, 0.58572483f, 0.5701437f, 0.5554308f, 0.5415162f,
    0.5283369f, 0.5158366f, 0.50396466f, 0.49267533f, 0.4819272f, 0.4716827f,
    0.4619077f, 0.45257112f, 0.44364452f, 0.435102f, 0.4269196f, 0.41907555f,
    0.4115497f, 0.40432343f, 0.39737967f, 0.39070258f, 0.38427743f, 0.37809068f,
    0.37212965f, 0.36638263f, 0.36083862f, 0.35548744f, 0.35031953f, 0.345326f,
    0.34049854f, 0.6016722f, 0.59452736f, 0.5765114f, 0.5595551f, 0.54356784f,
    0.5284687f, 0.5141858f, 0.5006546f, 0.48781732f, 0.47562188f, 0.46402133f,
    0.45297322f, 0.44243896f, 0.4323835f, 0.422775f, 0.41358423f, 0.40478456f,
    0.39635155f, 0.38826275f, 0.3804975f, 0.37303677f, 0.36586297f, 0.3589599f,
    0.3523125f, 0.34590682f, 0.3397299f, 0.33376974f, 0.3280151f, 0.3224555f,
    0.31708124f, 0.31188318f, 0.30685282f};

#define ALBEDO_LUT_SIZE_3D 16

static const auto entering_albedo_lut = vector<float>{0.97655153f, 0.9458529f,
    0.92380977f, 0.9146185f, 0.91065145f, 0.9088524f, 0.9079934f, 0.9075634f,
    0.90733963f, 0.90721977f, 0.90715444f, 0.9071188f, 0.90709966f, 0.90709f,
    0.021601094f, 0.9035749f, 0.9519571f, 0.9442338f, 0.92356455f, 0.9145406f,
    0.9106129f, 0.9088283f, 0.9079766f, 0.9075512f, 0.90733063f, 0.9072131f,
    0.90714955f, 0.9071153f, 0.90709734f, 0.9070885f, 0.33626214f, 0.90708447f,
    0.8668362f, 0.91948485f, 0.9191841f, 0.9131173f, 0.9099214f, 0.9084072f,
    0.90769064f, 0.9073471f, 0.90718174f, 0.90710396f, 0.9070702f, 0.9070588f,
    0.9070587f, 0.90706426f, 0.9070723f, 0.90707827f, 0.8590661f, 0.87579334f,
    0.9023281f, 0.9065863f, 0.90668f, 0.90646785f, 0.906399f, 0.9064378f,
    0.9065247f, 0.9066252f, 0.90672314f, 0.9068123f, 0.9068908f, 0.9069588f,
    0.907017f, 0.90705144f, 0.8360961f, 0.8493865f, 0.87654704f, 0.89173615f,
    0.8981708f, 0.9011702f, 0.90285265f, 0.9039501f, 0.9047346f, 0.90532506f,
    0.90578306f, 0.90614563f, 0.906437f, 0.9066739f, 0.9068675f, 0.9069788f,
    0.7751077f, 0.8156093f, 0.84763944f, 0.86951566f, 0.8828294f, 0.8907015f,
    0.89556015f, 0.89875156f, 0.9009705f, 0.9025845f, 0.90379924f, 0.90473795f,
    0.9054785f, 0.9060719f, 0.90655136f, 0.9068249f, 0.6802985f, 0.7540333f,
    0.80738235f, 0.8390833f, 0.859964f, 0.8738578f, 0.8832201f, 0.88968825f,
    0.8942947f, 0.8976752f, 0.90022415f, 0.9021912f, 0.90373945f, 0.9049772f,
    0.9059753f, 0.90654373f, 0.5692191f, 0.66800404f, 0.74902797f, 0.796726f,
    0.8279965f, 0.84955204f, 0.8647823f, 0.8757521f, 0.8838093f, 0.88984764f,
    0.89446294f, 0.898056f, 0.9008998f, 0.9031819f, 0.9050269f, 0.9060795f,
    0.46069577f, 0.5721242f, 0.675787f, 0.74144375f, 0.7856452f, 0.81677693f,
    0.8393666f, 0.85610753f, 0.8687313f, 0.8784034f, 0.8859273f, 0.89186496f,
    0.8966135f, 0.9004542f, 0.9035784f, 0.90536785f, 0.36613342f, 0.47981638f,
    0.5964265f, 0.67682964f, 0.73389196f, 0.7755173f, 0.8065885f, 0.8302171f,
    0.84846675f, 0.86275494f, 0.8740816f, 0.883165f, 0.890527f, 0.8965478f,
    0.9014897f, 0.90433854f, 0.28910667f, 0.39850459f, 0.51911646f, 0.6086484f,
    0.6759687f, 0.72730726f, 0.7670074f, 0.7981079f, 0.8227607f, 0.84251344f,
    0.8584975f, 0.8715506f, 0.8822993f, 0.8912113f, 0.89861286f, 0.90291655f,
    0.22860849f, 0.3304311f, 0.4488174f, 0.542039f, 0.61589473f, 0.6748157f,
    0.7221759f, 0.7605385f, 0.79185164f, 0.81760144f, 0.8389274f, 0.85670847f,
    0.87162465f, 0.884198f, 0.89479417f, 0.9010234f, 0.18189582f, 0.27494723f,
    0.38750172f, 0.48030338f, 0.55711824f, 0.62095535f, 0.6742323f, 0.71889436f,
    0.75650656f, 0.7883292f, 0.8153774f, 0.83846956f, 0.8582636f, 0.8752794f,
    0.8898767f, 0.89857644f, 0.1460185f, 0.23026232f, 0.33526173f, 0.42498553f,
    0.5019422f, 0.56817144f, 0.6253514f, 0.67487323f, 0.7178946f, 0.75538206f,
    0.78814375f, 0.81685495f, 0.84207696f, 0.86426204f, 0.8837042f, 0.8954883f,
    0.11841738f, 0.19438213f, 0.2912746f, 0.37642014f, 0.45156935f, 0.5181507f,
    0.5773454f, 0.63013947f, 0.67736197f, 0.7197139f, 0.75779027f, 0.7920961f,
    0.8230559f, 0.8510082f, 0.87612754f, 0.89166725f, 0.097061746f, 0.16550453f,
    0.25439933f, 0.33426526f, 0.40639588f, 0.47184977f, 0.5315024f, 0.5860841f,
    0.6362079f, 0.68239105f, 0.7250715f, 0.7646187f, 0.80133784f, 0.8354559f,
    0.86701334f, 0.88701797f, 0.95774376f, 0.88790256f, 0.82136756f,
    0.78139806f, 0.7572965f, 0.74268514f, 0.73377764f, 0.7283213f, 0.7249698f,
    0.7229134f, 0.72166055f, 0.7209103f, 0.7204763f, 0.7202422f, 0.50720066f,
    0.720108f, 0.9291044f, 0.8853209f, 0.8208607f, 0.7812465f, 0.75724465f,
    0.7426658f, 0.7337695f, 0.7283171f, 0.724967f, 0.72291106f, 0.7216586f,
    0.7209086f, 0.72047496f, 0.72024107f, 0.7201334f, 0.72010726f, 0.81926167f,
    0.8491504f, 0.8130317f, 0.7787755f, 0.75630724f, 0.7422602f, 0.73356813f,
    0.72820055f, 0.7248894f, 0.7228539f, 0.7216142f, 0.7208737f, 0.72044784f,
    0.7202207f, 0.7201188f, 0.7200959f, 0.78580105f, 0.78333765f, 0.78547096f,
    0.76826245f, 0.7518627f, 0.7401544f, 0.732447f, 0.7275311f, 0.7244474f,
    0.7225401f, 0.72138155f, 0.7206984f, 0.72031665f, 0.720125f, 0.72005224f,
    0.7200453f, 0.75692534f, 0.74331146f, 0.7476956f, 0.7468971f, 0.74074185f,
    0.7341887f, 0.72902673f, 0.72541547f, 0.72304404f, 0.7215572f, 0.72066903f,
    0.7201743f, 0.7199329f, 0.7198509f, 0.71986514f, 0.7199052f, 0.7199602f,
    0.71632653f, 0.71731573f, 0.72183144f, 0.72366476f, 0.7232294f, 0.7219804f,
    0.72075504f, 0.7198501f, 0.7192978f, 0.7190358f, 0.71898365f, 0.7190707f,
    0.71924186f, 0.71945405f, 0.7196001f, 0.6718054f, 0.68361413f, 0.6908644f,
    0.6978751f, 0.703978f, 0.7082858f, 0.711061f, 0.7128608f, 0.71412414f,
    0.71511304f, 0.71596116f, 0.71672827f, 0.7174366f, 0.7180902f, 0.7186797f,
    0.71902776f, 0.6113632f, 0.6381961f, 0.6585764f, 0.6716815f, 0.6819614f,
    0.6901894f, 0.69662654f, 0.70159984f, 0.70546037f, 0.70850855f, 0.7109696f,
    0.7130008f, 0.71470696f, 0.7161547f, 0.71737415f, 0.71806234f, 0.5414314f,
    0.5804111f, 0.6155898f, 0.63837314f, 0.65499824f, 0.6679839f, 0.6783926f,
    0.6867978f, 0.6936075f, 0.6991503f, 0.7036931f, 0.7074473f, 0.71057427f,
    0.7131902f, 0.7153582f, 0.716566f, 0.46769062f, 0.5148682f, 0.56276536f,
    0.5962051f, 0.6209207f, 0.6400752f, 0.6553986f, 0.6678873f, 0.6781737f,
    0.6867031f, 0.69381213f, 0.6997643f, 0.704766f, 0.70897037f, 0.7124607f,
    0.71440536f, 0.3961032f, 0.44754687f, 0.5042763f, 0.546908f, 0.57980615f,
    0.6058504f, 0.6269183f, 0.64424455f, 0.65866446f, 0.6707677f, 0.6809884f,
    0.6896572f, 0.6970284f, 0.70328814f, 0.70852786f, 0.71146333f, 0.3309572f,
    0.38347277f, 0.4449443f, 0.494029f, 0.53372115f, 0.5662271f, 0.5931663f,
    0.61572796f, 0.6347914f, 0.65101516f, 0.66490036f, 0.67683303f, 0.6871096f,
    0.69594234f, 0.70341754f, 0.7076395f, 0.27436224f, 0.32573324f, 0.3885641f,
    0.44118977f, 0.48555946f, 0.523206f, 0.5553426f, 0.58293307f, 0.606745f,
    0.62739193f, 0.64536506f, 0.6610567f, 0.67477375f, 0.68673176f, 0.6969867f,
    0.70283854f, 0.22670448f, 0.27559718f, 0.33739266f, 0.39110923f, 0.4380008f,
    0.47908702f, 0.51520926f, 0.54706776f, 0.5752483f, 0.6002416f, 0.622458f,
    0.64223677f, 0.65984726f, 0.67546827f, 0.6890852f, 0.696957f, 0.18734965f,
    0.23308873f, 0.29235584f, 0.34538084f, 0.39298916f, 0.43586347f,
    0.47457728f, 0.5096175f, 0.54139984f, 0.57027996f, 0.5965605f, 0.6204938f,
    0.6422749f, 0.6620078f, 0.67956114f, 0.6898779f, 0.1552127f, 0.19755763f,
    0.25348803f, 0.30466425f, 0.35165852f, 0.39495197f, 0.43495336f, 0.4720122f,
    0.50642806f, 0.5384571f, 0.56831497f, 0.59617597f, 0.6221602f, 0.64628637f,
    0.6682789f, 0.68147504f, 0.94732726f, 0.8560238f, 0.763836f, 0.70375633f,
    0.6643481f, 0.6383822f, 0.62123597f, 0.6099171f, 0.6024697f, 0.59760594f,
    0.5944721f, 0.5924995f, 0.59130687f, 0.59063774f, 0.59031963f, 0.59023863f,
    0.9169485f, 0.8529682f, 0.7631614f, 0.70353556f, 0.66426927f, 0.638355f,
    0.6212284f, 0.6099165f, 0.6024711f, 0.5976074f, 0.5944732f, 0.5924999f,
    0.5913068f, 0.5906373f, 0.59031785f, 0.5902378f, 0.7947068f, 0.8114941f,
    0.7533022f, 0.7001958f, 0.66297257f, 0.6378274f, 0.62101287f, 0.60982966f,
    0.60243565f, 0.5975905f, 0.5944615f, 0.5924886f, 0.5912945f, 0.59062415f,
    0.5903043f, 0.59022397f, 0.74342245f, 0.7337037f, 0.7195565f, 0.6867087f,
    0.6571811f, 0.63518786f, 0.6197517f, 0.6091957f, 0.60209334f, 0.5973855f,
    0.5943222f, 0.5923822f, 0.59120643f, 0.5905479f, 0.59023684f, 0.5901605f,
    0.70003605f, 0.6804526f, 0.67274815f, 0.6599609f, 0.6432172f, 0.62789994f,
    0.6158409f, 0.60700613f, 0.6007973f, 0.59656495f, 0.59376365f, 0.5919756f,
    0.5908944f, 0.59029967f, 0.590034f, 0.5899794f, 0.65666634f, 0.64434445f,
    0.63460106f, 0.62929666f, 0.62260246f, 0.6149937f, 0.60789895f, 0.60206336f,
    0.59763294f, 0.59446025f, 0.59230345f, 0.5909211f, 0.59010696f, 0.5896971f,
    0.58956325f, 0.5895729f, 0.6117296f, 0.6101887f, 0.6048499f, 0.60225993f,
    0.60058266f, 0.5985497f, 0.5961852f, 0.59387165f, 0.591902f, 0.5904031f,
    0.58938f, 0.5887748f, 0.58850497f, 0.58848655f, 0.58863723f, 0.58878857f,
    0.5627507f, 0.57155293f, 0.5753366f, 0.5771353f, 0.57896864f, 0.58064413f,
    0.58192635f, 0.5828291f, 0.58348465f, 0.5840291f, 0.5845573f, 0.585119f,
    0.5857282f, 0.5863761f, 0.5870286f, 0.58743674f, 0.50882256f, 0.5264096f,
    0.5406997f, 0.54930824f, 0.5556984f, 0.5609924f, 0.5654917f, 0.569304f,
    0.5725253f, 0.57526106f, 0.5776128f, 0.57966506f, 0.5814794f, 0.58309424f,
    0.5845077f, 0.5853171f, 0.45134175f, 0.4756174f, 0.49954268f, 0.51595575f,
    0.52819276f, 0.5379594f, 0.5460864f, 0.55299f, 0.55890846f, 0.5640056f,
    0.56841093f, 0.57223123f, 0.57555145f, 0.57843333f, 0.5808848f, 0.5822574f,
    0.39321786f, 0.4218979f, 0.45332974f, 0.476974f, 0.49542525f, 0.51032996f,
    0.52270913f, 0.53319585f, 0.54219496f, 0.5499795f, 0.55674607f, 0.5626434f,
    0.56778324f, 0.5722444f, 0.57602924f, 0.5781412f, 0.33755222f, 0.36852372f,
    0.40490016f, 0.43418828f, 0.4581763f, 0.47815105f, 0.49503297f, 0.5094796f,
    0.5219643f, 0.53283256f, 0.54234177f, 0.55068594f, 0.5580073f, 0.5644007f,
    0.56985277f, 0.5729054f, 0.2866696f, 0.31823844f, 0.35716584f, 0.39015123f,
    0.418337f, 0.44260946f, 0.46366203f, 0.48203963f, 0.49817196f, 0.5123989f,
    0.52498925f, 0.5361538f, 0.5460484f, 0.55477136f, 0.5622777f, 0.5665108f,
    0.24180496f, 0.27277339f, 0.3123442f, 0.34720424f, 0.37805194f, 0.40545368f,
    0.42987704f, 0.45171222f, 0.47128624f, 0.48887327f, 0.5047015f, 0.518956f,
    0.53177327f, 0.543228f, 0.55321485f, 0.5589089f, 0.20325606f, 0.2328745f,
    0.27174324f, 0.3070213f, 0.33912987f, 0.36842093f, 0.39519468f, 0.41970974f,
    0.44218984f, 0.46282786f, 0.4817874f, 0.4992008f, 0.51515716f, 0.52967894f,
    0.5425659f, 0.5500257f, 0.17070512f, 0.19856888f, 0.23588641f, 0.2705459f,
    0.30280566f, 0.33289078f, 0.360999f, 0.3873043f, 0.41195905f, 0.43509486f,
    0.45682114f, 0.47722036f, 0.49633068f, 0.51411176f, 0.53024566f, 0.5397689f,
    0.93973005f, 0.83360946f, 0.7246405f, 0.65155965f, 0.60200816f, 0.5681695f,
    0.54498357f, 0.5291013f, 0.51826715f, 0.51094204f, 0.5060652f, 0.5028998f,
    0.5009309f, 0.49979702f, 0.49924314f, 0.49936506f, 0.90826035f, 0.8302636f,
    0.7238611f, 0.651292f, 0.60190797f, 0.5681338f, 0.54497385f, 0.52910167f,
    0.51827073f, 0.510946f, 0.50606835f, 0.5029019f, 0.50093204f, 0.49979722f,
    0.49924263f, 0.49910092f, 0.77794534f, 0.7856146f, 0.71275604f, 0.64737445f,
    0.60033756f, 0.567487f, 0.5447196f, 0.5290162f, 0.5182542f, 0.5109531f,
    0.50607926f, 0.5029086f, 0.500932f, 0.49979058f, 0.49923027f, 0.499085f,
    0.7144759f, 0.7003531f, 0.67522365f, 0.6319467f, 0.5935624f, 0.56437916f,
    0.5432699f, 0.52834076f, 0.5179409f, 0.51080376f, 0.5059981f, 0.50285023f,
    0.50087637f, 0.49972975f, 0.49916193f, 0.4990105f, 0.6603246f, 0.6379967f,
    0.62253594f, 0.6016058f, 0.5775377f, 0.55599177f, 0.538835f, 0.52595454f,
    0.5166208f, 0.5100375f, 0.5055166f, 0.50251234f, 0.50060904f, 0.4994958f,
    0.49894214f, 0.49879217f, 0.6102168f, 0.5943094f, 0.5784849f, 0.5667617f,
    0.55416095f, 0.54140055f, 0.5299585f, 0.5205597f, 0.5132889f, 0.5079134f,
    0.50409687f, 0.501504f, 0.49984294f, 0.49887413f, 0.4984065f, 0.49828997f,
    0.56353116f, 0.5563594f, 0.5447244f, 0.5364711f, 0.52972454f, 0.52329016f,
    0.51718736f, 0.51175964f, 0.5072485f, 0.5037216f, 0.50112057f, 0.49932188f,
    0.4981814f, 0.4975565f, 0.49731532f, 0.49730003f, 0.5177008f, 0.5185462f,
    0.5144087f, 0.510171f, 0.5070222f, 0.50451595f, 0.5022983f, 0.50029516f,
    0.49856377f, 0.49717405f, 0.4961614f, 0.49552038f, 0.49521613f, 0.49519247f,
    0.49537775f, 0.49556762f, 0.47070068f, 0.47832394f, 0.48250705f,
    0.48382598f, 0.4845794f, 0.4853443f, 0.48614538f, 0.48692542f, 0.4876648f,
    0.48838034f, 0.48910177f, 0.48985347f, 0.4906485f, 0.49147958f, 0.49231315f,
    0.4928326f, 0.42211676f, 0.43508628f, 0.44692576f, 0.4544616f, 0.45991844f,
    0.46434546f, 0.46818212f, 0.47160015f, 0.4746731f, 0.47744602f, 0.47995764f,
    0.48224205f, 0.48432902f, 0.48622942f, 0.48792088f, 0.48889494f, 0.3730225f,
    0.38980576f, 0.40783256f, 0.4212021f, 0.4316304f, 0.4401601f, 0.44740018f,
    0.45370018f, 0.4592646f, 0.4642206f, 0.46865448f, 0.47262836f, 0.47619024f,
    0.4793602f, 0.48211035f, 0.48366156f, 0.3252282f, 0.34434724f, 0.36676252f,
    0.38486382f, 0.39981318f, 0.4124223f, 0.42325342f, 0.4326945f, 0.4410147f,
    0.44840342f, 0.45499688f, 0.4608928f, 0.4661631f, 0.47083738f, 0.47487378f,
    0.4771402f, 0.28048262f, 0.30067173f, 0.3257159f, 0.3471381f, 0.36565295f,
    0.38180304f, 0.39600545f, 0.40858302f, 0.4197866f, 0.42981175f, 0.4388106f,
    0.44689783f, 0.454159f, 0.46062556f, 0.4662304f, 0.46938756f, 0.23999543f,
    0.2602888f, 0.2864789f, 0.30983764f, 0.3307689f, 0.34959635f, 0.36658633f,
    0.38196093f, 0.39590672f, 0.40857995f, 0.42010975f, 0.43059713f,
    0.44011712f, 0.448684f, 0.4561847f, 0.4604497f, 0.20432949f, 0.22406691f,
    0.2502918f, 0.27442765f, 0.29668316f, 0.31723446f, 0.33623463f, 0.35381836f,
    0.3701042f, 0.3851956f, 0.39918017f, 0.41212428f, 0.42407155f, 0.43499827f,
    0.4447197f, 0.4503312f, 0.17351879f, 0.19229928f, 0.21780881f, 0.24185747f,
    0.2645398f, 0.28594542f, 0.30615887f, 0.325259f, 0.34331793f, 0.36039895f,
    0.37655428f, 0.3918171f, 0.40619588f, 0.41962123f, 0.43181983f, 0.43900105f,
    0.9334854f, 0.81597745f, 0.69525415f, 0.61365515f, 0.5576258f, 0.518744f,
    0.49160784f, 0.47264642f, 0.45944214f, 0.45032668f, 0.44413158f,
    0.44002923f, 0.43742794f, 0.43590215f, 0.43514404f, 0.43494782f,
    0.90123904f, 0.81244075f, 0.69441026f, 0.61335856f, 0.55751157f,
    0.51870185f, 0.49159575f, 0.4726466f, 0.4594466f, 0.45033193f, 0.4441361f,
    0.44003242f, 0.43742982f, 0.4359028f, 0.43514362f, 0.43494678f, 0.7651353f,
    0.7657465f, 0.68254024f, 0.6090828f, 0.55576134f, 0.5179672f, 0.49130422f,
    0.4725518f, 0.4594352f, 0.45035034f, 0.44415945f, 0.4400499f, 0.43743742f,
    0.4359f, 0.4351313f, 0.43492848f, 0.6929362f, 0.6755909f, 0.6426752f,
    0.5924481f, 0.5483413f, 0.5145212f, 0.48969147f, 0.47181338f, 0.45911336f,
    0.45021865f, 0.44410452f, 0.4400159f, 0.43739837f, 0.43584523f, 0.43505758f,
    0.4348409f, 0.6310752f, 0.60704696f, 0.5862504f, 0.5598335f, 0.5309559f,
    0.5053469f, 0.48482767f, 0.46921688f, 0.45771056f, 0.44943747f, 0.44363734f,
    0.4396974f, 0.43714097f, 0.43560386f, 0.4348098f, 0.43458024f, 0.575817f,
    0.5579517f, 0.53821373f, 0.52221787f, 0.50569636f, 0.48952746f, 0.47519052f,
    0.46337977f, 0.4541407f, 0.44719607f, 0.4421617f, 0.43865463f, 0.43633598f,
    0.43492347f, 0.43418777f, 0.4339713f, 0.52703357f, 0.51682514f, 0.50142413f,
    0.4895454f, 0.47946912f, 0.47011182f, 0.46150374f, 0.45396233f, 0.44769624f,
    0.4427396f, 0.43900025f, 0.43632153f, 0.43452492f, 0.4334364f, 0.43289512f,
    0.43275544f, 0.48217905f, 0.47866446f, 0.46995234f, 0.4620186f, 0.4556942f,
    0.45045337f, 0.4459055f, 0.44193807f, 0.4385686f, 0.43583035f, 0.4337252f,
    0.43221882f, 0.43124717f, 0.4307301f, 0.4305741f, 0.43060997f, 0.4387738f,
    0.44080102f, 0.43914464f, 0.4360922f, 0.43334514f, 0.4312134f, 0.42959204f,
    0.42834595f, 0.42739838f, 0.42671978f, 0.4263002f, 0.42613238f, 0.42619866f,
    0.4264694f, 0.4268929f, 0.4272148f, 0.39552537f, 0.40195352f, 0.4066776f,
    0.40886304f, 0.41013047f, 0.41117582f, 0.41223174f, 0.41334245f, 0.4144959f,
    0.41567475f, 0.4168682f, 0.41807318f, 0.41928366f, 0.42048728f, 0.4216446f,
    0.42234802f, 0.35245287f, 0.362131f, 0.3720526f, 0.37910724f, 0.38451555f,
    0.3889801f, 0.39287934f, 0.39640528f, 0.39965218f, 0.402667f, 0.4054748f,
    0.40809307f, 0.4105281f, 0.41277495f, 0.41478994f, 0.41595325f, 0.31045252f,
    0.32227087f, 0.33600047f, 0.34705228f, 0.35622582f, 0.36405748f, 0.3709022f,
    0.37699237f, 0.38247976f, 0.38746446f, 0.39201298f, 0.39617226f, 0.399967f,
    0.4034005f, 0.40641776f, 0.40812954f, 0.27069333f, 0.28367448f, 0.29984543f,
    0.3137944f, 0.3259851f, 0.33675778f, 0.34636652f, 0.35500288f, 0.36281228f,
    0.36990532f, 0.37636483f, 0.38225406f, 0.38760945f, 0.39243725f,
    0.39666304f, 0.39905295f, 0.23415306f, 0.24751985f, 0.26495227f,
    0.28070533f, 0.2950101f, 0.3080468f, 0.31996217f, 0.3308782f, 0.34089738f,
    0.35010558f, 0.35857216f, 0.36635333f, 0.37347934f, 0.37994513f,
    0.38564155f, 0.38888684f, 0.20140965f, 0.21460468f, 0.23238623f, 0.2490083f,
    0.26455572f, 0.27910122f, 0.29271066f, 0.30544487f, 0.31735957f, 0.3285047f,
    0.33892167f, 0.3486434f, 0.3576786f, 0.365995f, 0.3734302f, 0.3777317f,
    0.17264047f, 0.18530497f, 0.20279753f, 0.21957494f, 0.23563685f,
    0.25099352f, 0.26566276f, 0.279667f, 0.29303026f, 0.30577564f, 0.3179208f,
    0.32947603f, 0.3404258f, 0.3507051f, 0.3600862f, 0.36562723f, 0.92798793f,
    0.80113435f, 0.67183673f, 0.58469975f, 0.52475405f, 0.4829206f, 0.4534776f,
    0.43268794f, 0.41803727f, 0.40779284f, 0.400737f, 0.39600098f, 0.3929572f,
    0.39114815f, 0.3902378f, 0.38999966f, 0.89514774f, 0.7974685f, 0.6709553f,
    0.5843873f, 0.52463204f, 0.4828744f, 0.4534636f, 0.43268746f, 0.41804188f,
    0.40779856f, 0.4007421f, 0.39600477f, 0.39295948f, 0.39114898f, 0.3902374f,
    0.3899985f, 0.7546507f, 0.74942476f, 0.65863675f, 0.57991076f, 0.5227782f,
    0.48208416f, 0.45314375f, 0.43258056f, 0.41802782f, 0.4078194f, 0.40076983f,
    0.3960265f, 0.3929701f, 0.3911471f, 0.3902236f, 0.38997695f, 0.67598623f,
    0.6560211f, 0.6173917f, 0.56258297f, 0.514976f, 0.47842056f, 0.45140958f,
    0.43177938f, 0.41767865f, 0.40768015f, 0.40071666f, 0.39599636f, 0.3929323f,
    0.3910882f, 0.39013886f, 0.38987342f, 0.6086197f, 0.5833181f, 0.5587014f,
    0.52863294f, 0.49676627f, 0.46873173f, 0.4462291f, 0.42899448f, 0.416169f,
    0.4068412f, 0.40021783f, 0.39565635f, 0.3926527f, 0.39081714f, 0.38985f,
    0.3895629f, 0.5497272f, 0.5305802f, 0.5081699f, 0.48934174f, 0.47033978f,
    0.45209473f, 0.43602708f, 0.42277464f, 0.4123432f, 0.40442723f, 0.39861947f,
    0.39451554f, 0.3917562f, 0.39003944f, 0.38911733f, 0.3888321f, 0.49936926f,
    0.48727006f, 0.4694212f, 0.45517427f, 0.44296563f, 0.43178317f, 0.42164037f,
    0.4128147f, 0.405479f, 0.39964062f, 0.39518738f, 0.39194524f, 0.38972074f,
    0.38832524f, 0.3875845f, 0.38736445f, 0.4549751f, 0.44876185f, 0.4371463f,
    0.42682943f, 0.41845757f, 0.4114694f, 0.4054493f, 0.40025163f, 0.3958602f,
    0.3922791f, 0.3894874f, 0.38743144f, 0.3860328f, 0.3851983f, 0.38482466f,
    0.3847665f, 0.41382214f, 0.41235337f, 0.40699142f, 0.4011239f, 0.39610326f,
    0.39208484f, 0.38888705f, 0.38632593f, 0.38428617f, 0.3827061f, 0.3815496f,
    0.38078618f, 0.38038123f, 0.380291f, 0.38045472f, 0.38065755f, 0.37413386f,
    0.37640768f, 0.37656182f, 0.37526786f, 0.37378153f, 0.3725852f, 0.37178636f,
    0.37135124f, 0.37121475f, 0.37131998f, 0.37162548f, 0.37210125f,
    0.37272203f, 0.37345916f, 0.37426466f, 0.3747972f, 0.33529449f, 0.34036908f,
    0.3449917f, 0.3478731f, 0.34992838f, 0.35164988f, 0.35327345f, 0.35489362f,
    0.35653615f, 0.35819823f, 0.35986787f, 0.36153114f, 0.3631728f, 0.36477047f,
    0.36627346f, 0.36717394f, 0.29758182f, 0.30456972f, 0.31250733f, 0.3188126f,
    0.32405832f, 0.3286079f, 0.33268604f, 0.33642742f, 0.33991f, 0.34317797f,
    0.34625542f, 0.3491537f, 0.3518738f, 0.3544008f, 0.35667622f, 0.3579902f,
    0.26170638f, 0.26982135f, 0.2799585f, 0.28878042f, 0.29659244f, 0.3036065f,
    0.30997333f, 0.3158012f, 0.32116857f, 0.326132f, 0.33073214f, 0.3349956f,
    0.3389347f, 0.34253836f, 0.34573415f, 0.3475553f, 0.22839865f, 0.23700306f,
    0.24837244f, 0.25881624f, 0.26845625f, 0.27738166f, 0.285663f, 0.2933585f,
    0.30051702f, 0.3071794f, 0.3133786f, 0.31913817f, 0.32446754f, 0.32934847f,
    0.33368295f, 0.33616203f, 0.19817013f, 0.20679441f, 0.21863785f,
    0.22993447f, 0.24069111f, 0.25091505f, 0.26061687f, 0.26981023f, 0.2785108f,
    0.28673434f, 0.29449463f, 0.30179936f, 0.30864304f, 0.31498817f,
    0.32069743f, 0.32401454f, 0.17124784f, 0.1795803f, 0.19135836f, 0.20291565f,
    0.21419236f, 0.22514877f, 0.23576036f, 0.24601361f, 0.25590217f,
    0.26542336f, 0.2745747f, 0.28334853f, 0.29172274f, 0.29963654f, 0.3069051f,
    0.31122416f, 0.92294073f, 0.7880774f, 0.65237576f, 0.561776f, 0.49973226f,
    0.4564674f, 0.42594832f, 0.40429968f, 0.38894626f, 0.37812766f, 0.37061197f,
    0.36552054f, 0.3622169f, 0.3602343f, 0.35922697f, 0.35896128f, 0.8896247f,
    0.7843241f, 0.6514754f, 0.56145716f, 0.4996071f, 0.4564193f, 0.42593312f,
    0.40429854f, 0.38895053f, 0.3781333f, 0.37061712f, 0.3655244f, 0.36221924f,
    0.36023512f, 0.35922644f, 0.35895988f, 0.745662f, 0.73539627f, 0.63892615f,
    0.5568921f, 0.497707f, 0.45560026f, 0.42559445f, 0.4041798f, 0.3889296f,
    0.37815058f, 0.3706432f, 0.36554527f, 0.3622288f, 0.36023143f, 0.35920963f,
    0.3589344f, 0.66209817f, 0.6398676f, 0.59695494f, 0.5392342f, 0.48971725f,
    0.4518155f, 0.4237783f, 0.40332323f, 0.38854373f, 0.37798756f, 0.370574f,
    0.3655029f, 0.36217946f, 0.36015934f, 0.35910866f, 0.35881197f, 0.59082174f,
    0.56444854f, 0.5370345f, 0.5046171f, 0.47107774f, 0.44182515f, 0.41837963f,
    0.40037987f, 0.38691878f, 0.37706435f, 0.37001103f, 0.3651102f, 0.36185166f,
    0.3598394f, 0.35876676f, 0.35844445f, 0.5295125f, 0.5094098f, 0.4851183f,
    0.46446556f, 0.44402617f, 0.42469585f, 0.40778184f, 0.39384305f, 0.3828397f,
    0.37444687f, 0.3682447f, 0.3638241f, 0.36082202f, 0.35893217f, 0.35790208f,
    0.35757792f, 0.47821727f, 0.4648168f, 0.44530553f, 0.4295314f, 0.41603673f,
    0.40383863f, 0.39289552f, 0.3834293f, 0.37556708f, 0.36929762f, 0.3644867f,
    0.36095506f, 0.35850477f, 0.35694402f, 0.35609594f, 0.35583514f,
    0.43428105f, 0.4262562f, 0.41269585f, 0.40082294f, 0.39116055f, 0.38312608f,
    0.3762643f, 0.37039325f, 0.36545867f, 0.36144084f, 0.35829464f, 0.3559546f,
    0.35433412f, 0.3533349f, 0.3528511f, 0.35275015f, 0.39481425f, 0.3910022f,
    0.38315046f, 0.37540847f, 0.36889303f, 0.3636663f, 0.35948285f, 0.35612145f,
    0.35343298f, 0.351331f, 0.34975484f, 0.34865743f, 0.3479917f, 0.3477053f,
    0.3477334f, 0.34788132f, 0.3577446f, 0.3572095f, 0.3542596f, 0.35060254f,
    0.34724647f, 0.34454024f, 0.3425013f, 0.34104347f, 0.34006283f, 0.33947474f,
    0.33921328f, 0.33922938f, 0.33948305f, 0.33993495f, 0.34053218f,
    0.34097144f, 0.32205433f, 0.32397345f, 0.3249451f, 0.3249389f, 0.32466176f,
    0.3244725f, 0.32450733f, 0.32479668f, 0.32531935f, 0.32603934f, 0.32691795f,
    0.32792032f, 0.32901537f, 0.3301694f, 0.33132827f, 0.33205885f, 0.2876009f,
    0.29122534f, 0.29510832f, 0.29807162f, 0.30053112f, 0.30274013f, 0.3048327f,
    0.306879f, 0.30890745f, 0.31092504f, 0.3129269f, 0.3149012f, 0.31683236f,
    0.3186943f, 0.32042825f, 0.32145622f, 0.2547528f, 0.25942492f, 0.26527023f,
    0.2704228f, 0.2750763f, 0.27936158f, 0.28335673f, 0.28711528f, 0.2906704f,
    0.29404312f, 0.29724535f, 0.30028033f, 0.30314365f, 0.30581442f, 0.3082247f,
    0.30961353f, 0.22402875f, 0.22921236f, 0.23620395f, 0.24279088f,
    0.24901657f, 0.2549136f, 0.26049846f, 0.26578742f, 0.27079317f, 0.27552655f,
    0.27999637f, 0.28420514f, 0.28814828f, 0.2918008f, 0.29507655f, 0.2969588f,
    0.19585662f, 0.20115222f, 0.20864968f, 0.21602632f, 0.22323231f, 0.2302352f,
    0.23700571f, 0.24352685f, 0.24978718f, 0.25577974f, 0.26149988f,
    0.26693952f, 0.27208385f, 0.27689442f, 0.2812564f, 0.28380245f, 0.17047358f,
    0.1756123f, 0.1831581f, 0.19082941f, 0.19852245f, 0.20616788f, 0.2137104f,
    0.22111315f, 0.22834916f, 0.23539868f, 0.24224591f, 0.2488725f, 0.25525212f,
    0.2613302f, 0.26695734f, 0.27032492f, 0.91818345f, 0.77625054f, 0.63571334f,
    0.54314995f, 0.4803212f, 0.43672946f, 0.40603936f, 0.38425702f, 0.3687693f,
    0.357811f, 0.35015786f, 0.34494123f, 0.34153327f, 0.33947328f, 0.33841896f,
    0.1958626f, 0.88447315f, 0.77243954f, 0.63480747f, 0.5428316f, 0.48019636f,
    0.43668106f, 0.40602353f, 0.38425523f, 0.36877292f, 0.35781622f,
    0.35016268f, 0.34494483f, 0.3415354f, 0.3394739f, 0.3384182f, 0.3381374f,
    0.73769915f, 0.7229568f, 0.6221822f, 0.5382582f, 0.47829205f, 0.43585378f,
    0.40567434f, 0.38412595f, 0.36874276f, 0.35782593f, 0.35018283f,
    0.34496093f, 0.3415408f, 0.3394662f, 0.33839712f, 0.33810735f, 0.65036607f,
    0.6261049f, 0.57994825f, 0.5205289f, 0.47025645f, 0.43202063f, 0.40380836f,
    0.38322264f, 0.3683161f, 0.35762852f, 0.35008505f, 0.34489432f, 0.34146994f,
    0.3393737f, 0.33827552f, 0.33796364f, 0.57635725f, 0.549023f, 0.5195422f,
    0.48572367f, 0.4514748f, 0.42188817f, 0.39826953f, 0.38014883f, 0.36657557f,
    0.35660493f, 0.34943557f, 0.34442595f, 0.34107402f, 0.3389898f, 0.337871f,
    0.33753324f, 0.51357436f, 0.49269393f, 0.46706462f, 0.44531065f, 0.4242007f,
    0.404512f, 0.38740933f, 0.37335286f, 0.36225408f, 0.3537669f, 0.347471f,
    0.34296098f, 0.33988014f, 0.33792812f, 0.33685735f, 0.33651984f, 0.4619133f,
    0.44755083f, 0.42688978f, 0.41017237f, 0.39600897f, 0.3833838f, 0.37218678f,
    0.36256492f, 0.35459805f, 0.34824f, 0.34335214f, 0.33975056f, 0.3372395f,
    0.33563086f, 0.33475244f, 0.334484f, 0.41858584f, 0.4092741f, 0.39437413f,
    0.38149405f, 0.37107795f, 0.3624953f, 0.35525513f, 0.34912634f, 0.34401557f,
    0.33986926f, 0.33662733f, 0.334213f, 0.33253518f, 0.3314948f, 0.33098817f,
    0.330886f, 0.3805657f, 0.3751049f, 0.36552292f, 0.3565097f, 0.34903228f,
    0.343063f, 0.33831602f, 0.33452994f, 0.33152467f, 0.32918507f, 0.32743186f,
    0.32620305f, 0.32544148f, 0.32508913f, 0.32508f, 0.3252231f, 0.34558403f,
    0.34306473f, 0.33792916f, 0.3326168f, 0.3279779f, 0.32425186f, 0.32139882f,
    0.31928882f, 0.31778738f, 0.3167845f, 0.31619516f, 0.31595498f, 0.31601185f,
    0.31631792f, 0.3168175f, 0.3172229f, 0.31235808f, 0.31202558f, 0.31038788f,
    0.30831337f, 0.30637437f, 0.30482495f, 0.30373752f, 0.3030938f, 0.30283877f,
    0.30291042f, 0.30325007f, 0.3038078f, 0.30454013f, 0.3054048f, 0.30634567f,
    0.30697578f, 0.28045613f, 0.2816494f, 0.282591f, 0.28311688f, 0.28354222f,
    0.28403366f, 0.2846673f, 0.2854659f, 0.2864229f, 0.28751856f, 0.28872657f,
    0.29001993f, 0.29137057f, 0.29274476f, 0.2940846f, 0.29490817f, 0.24999619f,
    0.25214544f, 0.25483158f, 0.2572657f, 0.25957143f, 0.26181412f, 0.2640261f,
    0.26622054f, 0.2683992f, 0.2705581f, 0.27268803f, 0.27477735f, 0.27680987f,
    0.27875835f, 0.28055984f, 0.28161392f, 0.221329f, 0.22397198f, 0.2276965f,
    0.23138624f, 0.2350373f, 0.2386345f, 0.24216112f, 0.24560241f, 0.24894583f,
    0.25218174f, 0.2553004f, 0.2582918f, 0.26114133f, 0.26382044f, 0.2662537f,
    0.26765838f, 0.19480522f, 0.19759731f, 0.20181042f, 0.20620653f,
    0.21070303f, 0.21523225f, 0.21974137f, 0.2241903f, 0.2285487f, 0.23279418f,
    0.23690778f, 0.24087232f, 0.2446665f, 0.24825256f, 0.25153434f, 0.25345743f,
    0.17065778f, 0.17336687f, 0.1776784f, 0.18236373f, 0.18729508f, 0.19237547f,
    0.19753191f, 0.20270973f, 0.20786753f, 0.21297388f, 0.21800242f,
    0.22292903f, 0.22772469f, 0.2323405f, 0.23665567f, 0.23925716f, 0.9136233f,
    0.7653239f, 0.6211368f, 0.5277244f, 0.4650725f, 0.42195737f, 0.39175504f,
    0.37037072f, 0.3551702f, 0.34439954f, 0.33685613f, 0.3316942f, 0.32830602f,
    0.32624713f, 0.32518739f, 0.3249046f, 0.87957835f, 0.7614772f, 0.62023515f,
    0.52741134f, 0.46495044f, 0.42190987f, 0.39173907f, 0.37036827f, 0.3551731f,
    0.344404f, 0.33686033f, 0.33169731f, 0.32830775f, 0.3262474f, 0.32518628f,
    0.32490262f, 0.7304766f, 0.71168184f, 0.607646f, 0.52288723f, 0.463072f,
    0.42108935f, 0.39138567f, 0.37022984f, 0.3551324f, 0.34440365f, 0.33687153f,
    0.33170575f, 0.32830667f, 0.32623404f, 0.32516003f, 0.32486752f, 0.6402182f,
    0.61409825f, 0.5654841f, 0.5052717f, 0.45509169f, 0.4172609f, 0.38949442f,
    0.36928746f, 0.35466284f, 0.344165f, 0.33673647f, 0.33160633f, 0.32820684f,
    0.3261156f, 0.32501447f, 0.3247007f, 0.5643636f, 0.53613764f, 0.5051343f,
    0.47062284f, 0.4363739f, 0.40710235f, 0.38387424f, 0.3661064f, 0.3528082f,
    0.34303117f, 0.33598542f, 0.33104622f, 0.3277283f, 0.32565638f, 0.32454f,
    0.32420313f, 0.50083303f, 0.47928125f, 0.45270467f, 0.4303861f, 0.40916815f,
    0.3896596f, 0.37285215f, 0.35909766f, 0.34825557f, 0.33996356f, 0.33380306f,
    0.32937914f, 0.3263481f, 0.32442224f, 0.32336506f, 0.32303494f, 0.4492768f,
    0.43416172f, 0.41270196f, 0.3954668f, 0.3810807f, 0.3684669f, 0.35741884f,
    0.3480003f, 0.34023637f, 0.3340534f, 0.32930195f, 0.3257991f, 0.32335538f,
    0.3217913f, 0.32094342f, 0.3206936f, 0.40673792f, 0.39646918f, 0.3806302f,
    0.36713225f, 0.35634145f, 0.34757906f, 0.3403003f, 0.3342207f, 0.3292026f,
    0.32516247f, 0.3220207f, 0.31969213f, 0.31808347f, 0.31709784f, 0.31663647f,
    0.31656533f, 0.37006682f, 0.36339957f, 0.35258377f, 0.34271693f, 0.334644f,
    0.32827073f, 0.3232655f, 0.31932935f, 0.31624955f, 0.313886f, 0.31213832f,
    0.3109313f, 0.31019944f, 0.30988073f, 0.30990982f, 0.31008664f, 0.33685544f,
    0.3328836f, 0.3261635f, 0.3196789f, 0.31415617f, 0.3097577f, 0.30639333f,
    0.30389792f, 0.30210954f, 0.3008979f, 0.30015886f, 0.29981467f, 0.29980278f,
    0.30006865f, 0.30055425f, 0.30097055f, 0.3056413f, 0.30364722f, 0.30008888f,
    0.29650238f, 0.29335612f, 0.29083127f, 0.28894913f, 0.28765512f, 0.2868672f,
    0.28650466f, 0.2864911f, 0.28676438f, 0.28727102f, 0.2879619f, 0.28877932f,
    0.2893608f, 0.27579162f, 0.27517045f, 0.2739149f, 0.2726203f, 0.27152333f,
    0.27073103f, 0.2702741f, 0.27014115f, 0.27029914f, 0.27071f, 0.27133012f,
    0.27212092f, 0.27304512f, 0.27406338f, 0.27511877f, 0.2757984f, 0.2472429f,
    0.24748541f, 0.24777839f, 0.24814788f, 0.24866194f, 0.24934135f,
    0.25018257f, 0.25117016f, 0.25228244f, 0.25349858f, 0.25479314f,
    0.25614363f, 0.2575254f, 0.25890714f, 0.26023045f, 0.26102164f, 0.22021753f,
    0.22091594f, 0.2221328f, 0.22359127f, 0.22524855f, 0.22705783f, 0.22897592f,
    0.23096612f, 0.23299724f, 0.23504579f, 0.23708823f, 0.23910533f,
    0.24107566f, 0.24296874f, 0.24471834f, 0.24573252f, 0.19500291f,
    0.19585557f, 0.19751728f, 0.19959018f, 0.2019639f, 0.20454806f, 0.20727055f,
    0.21007502f, 0.21291706f, 0.21576388f, 0.21858591f, 0.22135922f,
    0.22405826f, 0.22664648f, 0.22904304f, 0.2304504f, 0.17182484f, 0.17263062f,
    0.17440237f, 0.1767363f, 0.17948338f, 0.18252936f, 0.1857863f, 0.18918703f,
    0.19267921f, 0.19622357f, 0.19978586f, 0.20333748f, 0.20684786f,
    0.21027309f, 0.2135148f, 0.21548325f, 0.90920424f, 0.7550911f, 0.6081835f,
    0.5147695f, 0.45300612f, 0.4109516f, 0.38171196f, 0.36110866f, 0.3465007f,
    0.33615696f, 0.32890633f, 0.32393432f, 0.32066077f, 0.3186639f, 0.31763157f,
    0.31735498f, 0.8748702f, 0.7512257f, 0.60729337f, 0.5144652f, 0.45288864f,
    0.41090584f, 0.38169616f, 0.36110562f, 0.34650272f, 0.33616054f, 0.3289097f,
    0.32393676f, 0.32066193f, 0.31866372f, 0.3176301f, 0.31735265f, 0.72381157f,
    0.7013018f, 0.5948228f, 0.51003146f, 0.45105717f, 0.41010264f, 0.3813432f,
    0.36095917f, 0.3464507f, 0.3361483f, 0.32890987f, 0.3239355f, 0.3206526f,
    0.3186434f, 0.31759793f, 0.31731218f, 0.63127613f, 0.6034347f, 0.5529787f,
    0.49266207f, 0.4432038f, 0.4063165f, 0.37944442f, 0.3599838f, 0.34593767f,
    0.33586413f, 0.32873172f, 0.32379723f, 0.3205186f, 0.31849504f, 0.31742597f,
    0.31712103f, 0.55425537f, 0.5251901f, 0.49308455f, 0.45841748f, 0.42469656f,
    0.39621457f, 0.37378556f, 0.3567133f, 0.3439718f, 0.33461428f, 0.32786915f,
    0.32313442f, 0.3199473f, 0.31795257f, 0.31687653f, 0.3165534f, 0.49053758f,
    0.46838635f, 0.441156f, 0.4186754f, 0.397769f, 0.37883466f, 0.36267492f,
    0.34952724f, 0.339199f, 0.331313f, 0.32545686f, 0.32125002f, 0.31836587f,
    0.31653365f, 0.31553197f, 0.3152253f, 0.43946093f, 0.4237333f, 0.40172863f,
    0.38428822f, 0.37001064f, 0.35772848f, 0.34712386f, 0.3381704f, 0.33083665f,
    0.3250193f, 0.32056147f, 0.31728274f, 0.31500265f, 0.31355292f, 0.31278223f,
    0.312571f, 0.39787585f, 0.38687146f, 0.37037426f, 0.35653654f, 0.34564912f,
    0.33697408f, 0.3299051f, 0.32409912f, 0.31937253f, 0.3156088f, 0.3127116f,
    0.3105874f, 0.3091424f, 0.30828354f, 0.30791876f, 0.3079029f, 0.36251712f,
    0.3549416f, 0.34323493f, 0.3328065f, 0.3244025f, 0.317868f, 0.31282654f,
    0.30894068f, 0.30596572f, 0.30373335f, 0.3021257f, 0.3010539f, 0.3004452f,
    0.30023474f, 0.3003583f, 0.30059743f, 0.33087325f, 0.32580972f, 0.31791916f,
    0.31059718f, 0.3044744f, 0.2996506f, 0.29599345f, 0.2933064f, 0.2914039f,
    0.29013303f, 0.2893755f, 0.28904015f, 0.28905484f, 0.28935906f, 0.2898931f,
    0.29035354f, 0.30135795f, 0.2981063f, 0.293103f, 0.28839076f, 0.28435874f,
    0.28113124f, 0.2786892f, 0.2769499f, 0.2758096f, 0.27516836f, 0.27493846f,
    0.27504608f, 0.2754291f, 0.27603248f, 0.276797f, 0.27736834f, 0.27319846f,
    0.27118915f, 0.26825404f, 0.26556954f, 0.26331282f, 0.2615448f, 0.26026177f,
    0.25942606f, 0.25898555f, 0.25888413f, 0.25906828f, 0.25948912f, 0.260102f,
    0.26086304f, 0.2617161f, 0.26229694f, 0.24620068f, 0.24496663f, 0.24340352f,
    0.24216828f, 0.24128646f, 0.24074462f, 0.24051219f, 0.24055207f,
    0.24082619f, 0.24129753f, 0.24193187f, 0.2426974f, 0.24356325f, 0.24449508f,
    0.24543843f, 0.24602059f, 0.22049178f, 0.21966565f, 0.21891183f,
    0.21861097f, 0.21869068f, 0.2190796f, 0.2197147f, 0.22054262f, 0.22151938f,
    0.22260803f, 0.22377786f, 0.22500175f, 0.22625351f, 0.22750136f,
    0.22868678f, 0.22937584f, 0.19631088f, 0.19562498f, 0.19525748f,
    0.19547653f, 0.19615065f, 0.19717163f, 0.19845288f, 0.19992569f,
    0.20153601f, 0.2032406f, 0.20500459f, 0.20679817f, 0.2085928f, 0.21035321f,
    0.2120107f, 0.21298271f, 0.17387879f, 0.17315805f, 0.17288816f, 0.17332661f,
    0.17430836f, 0.1757053f, 0.17741835f, 0.17937061f, 0.18150222f, 0.18376577f,
    0.18612309f, 0.1885415f, 0.19098943f, 0.19342682f, 0.19577293f, 0.19720727f,
    0.9048913f, 0.7454174f, 0.5965397f, 0.5037792f, 0.44343373f, 0.40286222f,
    0.37492284f, 0.35537088f, 0.34157103f, 0.33182418f, 0.32499763f,
    0.32031375f, 0.31722462f, 0.31533533f, 0.31435543f, 0.31409198f, 0.8703039f,
    0.74154687f, 0.5956663f, 0.50348616f, 0.44332194f, 0.40281868f, 0.37490743f,
    0.3553673f, 0.3415722f, 0.33182675f, 0.3250001f, 0.32031536f, 0.3172251f,
    0.31533462f, 0.31435356f, 0.31408936f, 0.7175818f, 0.69163764f, 0.5833757f,
    0.49917182f, 0.44155192f, 0.40203995f, 0.37455803f, 0.35521376f,
    0.34150842f, 0.33180156f, 0.32498786f, 0.3203031f, 0.31720638f, 0.31530648f,
    0.3143149f, 0.31404322f, 0.6232795f, 0.5938334f, 0.5420308f, 0.48214278f,
    0.43387514f, 0.39832193f, 0.37266415f, 0.35420984f, 0.34095174f,
    0.33146924f, 0.32476276f, 0.32012197f, 0.31703466f, 0.3151255f, 0.314115f,
    0.31382692f, 0.54561937f, 0.51576227f, 0.48288792f, 0.44847947f,
    0.41568086f, 0.38833398f, 0.36699653f, 0.35086292f, 0.33887738f,
    0.33010027f, 0.3237826f, 0.31934935f, 0.31636393f, 0.31449452f, 0.31348717f,
    0.31318754f, 0.4821514f, 0.45945436f, 0.43179718f, 0.4094616f, 0.38917956f,
    0.37110788f, 0.35584936f, 0.34352565f, 0.33389378f, 0.3265643f, 0.32113358f,
    0.3172383f, 0.31457192f, 0.3128831f, 0.3119681f, 0.3116969f, 0.43184772f,
    0.41560668f, 0.3932461f, 0.37583482f, 0.3619113f, 0.35019532f, 0.34024727f,
    0.33194643f, 0.3252045f, 0.31988978f, 0.31583822f, 0.31287408f, 0.3108275f,
    0.30954343f, 0.3088842f, 0.30872628f, 0.3913539f, 0.3797687f, 0.3628158f,
    0.3488428f, 0.33806276f, 0.3296706f, 0.32299164f, 0.31762075f, 0.31332815f,
    0.30996412f, 0.30741525f, 0.30558115f, 0.30436832f, 0.3036887f, 0.3034581f,
    0.3035152f, 0.35729492f, 0.34901938f, 0.33666563f, 0.32588673f, 0.31734374f,
    0.31082493f, 0.30590865f, 0.30221796f, 0.29947698f, 0.29748946f, 0.2961192f,
    0.295265f, 0.29484737f, 0.2947994f, 0.2950585f, 0.29538077f, 0.32708f,
    0.3211777f, 0.31241012f, 0.30449092f, 0.29797295f, 0.2929033f, 0.2891124f,
    0.28637457f, 0.28448305f, 0.28326386f, 0.28258386f, 0.2823398f, 0.28245097f,
    0.28285134f, 0.2834795f, 0.28400826f, 0.2990376f, 0.29481322f, 0.2887029f,
    0.28314182f, 0.27845597f, 0.27472395f, 0.2718943f, 0.26985854f, 0.2684974f,
    0.26769388f, 0.2673489f, 0.26737887f, 0.26771402f, 0.26829404f, 0.26905808f,
    0.26964763f, 0.27229792f, 0.2692033f, 0.2649585f, 0.2611942f, 0.25804007f,
    0.2555209f, 0.2536056f, 0.25223503f, 0.25134325f, 0.25085956f, 0.25072128f,
    0.2508715f, 0.2512596f, 0.2518378f, 0.2525504f, 0.25306627f, 0.24657449f,
    0.24417195f, 0.24113938f, 0.23863247f, 0.2366429f, 0.23513037f, 0.23404315f,
    0.23332666f, 0.2329314f, 0.23280779f, 0.23291367f, 0.23321056f, 0.23366278f,
    0.23423362f, 0.23487185f, 0.23528597f, 0.22192577f, 0.21987827f, 0.2175431f,
    0.21582714f, 0.21463415f, 0.21387354f, 0.21346663f, 0.21334681f, 0.2134611f,
    0.21376242f, 0.21421345f, 0.2147817f, 0.21543728f, 0.21614769f, 0.21686055f,
    0.21727467f, 0.19855647f, 0.19662336f, 0.19460796f, 0.19331774f,
    0.19260408f, 0.19234481f, 0.19244179f, 0.19281638f, 0.19340777f,
    0.19416478f, 0.19504756f, 0.19602223f, 0.19705802f, 0.19812073f,
    0.19915134f, 0.1997506f, 0.17668638f, 0.17471467f, 0.17276914f, 0.17164883f,
    0.17117551f, 0.17121033f, 0.17164513f, 0.17239508f, 0.173395f, 0.1745911f,
    0.17594135f, 0.1774103f, 0.17896542f, 0.18056972f, 0.18215609f, 0.18313277f,
    0.90066165f, 0.73621196f, 0.5859836f, 0.4943897f, 0.4358562f, 0.39706975f,
    0.3706657f, 0.35235047f, 0.339506f, 0.33047247f, 0.32416064f, 0.31983352f,
    0.3169783f, 0.31522936f, 0.3143201f, 0.314075f, 0.8658504f, 0.73234695f,
    0.5851311f, 0.4941095f, 0.43575087f, 0.39702883f, 0.37065083f, 0.35234642f,
    0.33950618f, 0.33047396f, 0.32416207f, 0.31983423f, 0.31697807f,
    0.31522804f, 0.31431776f, 0.314072f, 0.7117032f, 0.6825659f, 0.5730659f,
    0.48993483f, 0.4340519f, 0.3962791f, 0.37030697f, 0.35218632f, 0.33943045f,
    0.33043513f, 0.32413656f, 0.3198101f, 0.3169492f, 0.3151915f, 0.31427234f,
    0.31402016f, 0.6160435f, 0.5850967f, 0.53235483f, 0.47331196f, 0.42658484f,
    0.39264604f, 0.36842608f, 0.35115707f, 0.33883035f, 0.33005315f,
    0.32386228f, 0.3195836f, 0.3167374f, 0.31497616f, 0.3140434f, 0.31377828f,
    0.53816724f, 0.50755465f, 0.47418222f, 0.44035494f, 0.4087733f, 0.3828104f,
    0.3627698f, 0.34774303f, 0.33665016f, 0.32856375f, 0.32276106f, 0.31869692f,
    0.3159632f, 0.3142535f, 0.31333554f, 0.31306645f, 0.47528183f, 0.45208275f,
    0.42417854f, 0.40222442f, 0.38280022f, 0.36579984f, 0.35162178f,
    0.34027338f, 0.3314634f, 0.324794f, 0.31987226f, 0.3163544f, 0.31395555f,
    0.31244552f, 0.3116396f, 0.31141248f, 0.42597842f, 0.409298f, 0.38672608f,
    0.36952132f, 0.35613418f, 0.3451547f, 0.33601487f, 0.3284989f, 0.32246065f,
    0.31774312f, 0.3141759f, 0.3115893f, 0.30982524f, 0.30874303f, 0.3082196f,
    0.30812553f, 0.38668388f, 0.3746304f, 0.35737196f, 0.34341568f, 0.3328932f,
    0.32492626f, 0.31876674f, 0.31394678f, 0.31018725f, 0.30730882f,
    0.30518076f, 0.30369663f, 0.30276406f, 0.3023007f, 0.30223072f, 0.3023733f,
    0.35391897f, 0.34509373f, 0.3322709f, 0.32129642f, 0.31275535f, 0.3063811f,
    0.30170688f, 0.29831713f, 0.29590106f, 0.29423872f, 0.29317474f, 0.2925964f,
    0.29241794f, 0.29257083f, 0.29299465f, 0.2934146f, 0.3250316f, 0.31847298f,
    0.3090416f, 0.30070066f, 0.29393756f, 0.28875345f, 0.28494504f, 0.2822609f,
    0.2804699f, 0.27938217f, 0.27884907f, 0.278756f, 0.27901402f, 0.27955204f,
    0.28030723f, 0.28092152f, 0.29829195f, 0.29330117f, 0.28632993f,
    0.28012267f, 0.27495337f, 0.27086276f, 0.2677718f, 0.2655522f, 0.26406604f,
    0.26318562f, 0.26280034f, 0.26281777f, 0.26316106f, 0.263765f, 0.26456708f,
    0.26519564f, 0.27276477f, 0.26880544f, 0.26352167f, 0.25890544f,
    0.25504774f, 0.2519439f, 0.24954048f, 0.24776237f, 0.24652824f, 0.2457594f,
    0.24538407f, 0.24533871f, 0.24556737f, 0.24601895f, 0.24663785f, 0.2471143f,
    0.24809973f, 0.24475461f, 0.24053732f, 0.23700508f, 0.23412192f,
    0.23182568f, 0.23004767f, 0.22872068f, 0.2277825f, 0.22717716f, 0.22685505f,
    0.22677234f, 0.22688943f, 0.22716779f, 0.22755855f, 0.22783628f,
    0.22430758f, 0.22126263f, 0.21763481f, 0.21475998f, 0.2125223f, 0.21081549f,
    0.20954841f, 0.20864488f, 0.20804188f, 0.20768726f, 0.20753768f,
    0.20755637f, 0.2077106f, 0.20796728f, 0.208279f, 0.20845754f, 0.20157069f,
    0.19860704f, 0.19522825f, 0.19268686f, 0.19081998f, 0.18949409f,
    0.18860173f, 0.18805704f, 0.18779144f, 0.18774988f, 0.1878877f, 0.18816793f,
    0.18855792f, 0.18902434f, 0.1895158f, 0.18978922f, 0.18011217f, 0.17709565f,
    0.17374918f, 0.17132366f, 0.16962999f, 0.16852033f, 0.1678792f, 0.1676159f,
    0.16765893f, 0.16795132f, 0.16844724f, 0.16910887f, 0.16990283f,
    0.17079435f, 0.17172693f, 0.17230386f, 0.89650035f, 0.72741073f, 0.5763534f,
    0.48633128f, 0.42990166f, 0.3931119f, 0.3684013f, 0.35144442f, 0.33965084f,
    0.33140624f, 0.3256686f, 0.32174382f, 0.31915593f, 0.31756982f, 0.31674394f,
    0.3165209f, 0.8614899f, 0.7235601f, 0.57552475f, 0.4860651f, 0.42980325f,
    0.39307383f, 0.36838704f, 0.35143986f, 0.33965003f, 0.3314066f, 0.32566896f,
    0.3217436f, 0.3191549f, 0.31756794f, 0.31674117f, 0.31651753f, 0.70611554f,
    0.6739989f, 0.56371856f, 0.48204356f, 0.42818108f, 0.39235574f, 0.36804986f,
    0.3512736f, 0.33956227f, 0.33135378f, 0.32562968f, 0.321707f, 0.31911537f,
    0.31752264f, 0.31668884f, 0.3164599f, 0.6094332f, 0.5770811f, 0.52374065f,
    0.46587178f, 0.42094484f, 0.38881788f, 0.36618686f, 0.35022092f,
    0.33891898f, 0.3309215f, 0.32530499f, 0.32143375f, 0.31886226f, 0.31727195f,
    0.3164304f, 0.31619236f, 0.53167164f, 0.5003469f, 0.46669978f, 0.4337069f,
    0.40356135f, 0.3791588f, 0.3605547f, 0.34674603f, 0.33663523f, 0.32931155f,
    0.3240813f, 0.32043105f, 0.31798264f, 0.31645608f, 0.3156417f, 0.3154081f,
    0.46963564f, 0.4459723f, 0.41796625f, 0.39657703f, 0.37818408f, 0.3624036f,
    0.34942895f, 0.33915678f, 0.33125168f, 0.32531017f, 0.3209524f, 0.31785548f,
    0.31575745f, 0.31445014f, 0.31376863f, 0.31359157f, 0.42150652f,
    0.40444452f, 0.3817741f, 0.3649112f, 0.35219523f, 0.34207422f, 0.33384782f,
    0.3272057f, 0.32194638f, 0.3178885f, 0.3148572f, 0.3126898f, 0.311241f,
    0.310385f, 0.31001395f, 0.3099908f, 0.3834923f, 0.371055f, 0.3536048f,
    0.33977917f, 0.32962555f, 0.32218572f, 0.3166366f, 0.31244636f, 0.3092884f,
    0.30695334f, 0.30529457f, 0.3042004f, 0.30358037f, 0.30335814f, 0.3034667f,
    0.303703f, 0.35201532f, 0.34275177f, 0.3295924f, 0.31853747f, 0.31010205f,
    0.30396605f, 0.29961747f, 0.29660183f, 0.29457566f, 0.29329392f,
    0.29258364f, 0.29232085f, 0.29241443f, 0.29279438f, 0.29340273f, 0.2939302f,
    0.324376f, 0.31729582f, 0.30735877f, 0.29872632f, 0.29182908f, 0.2866271f,
    0.28288603f, 0.28032947f, 0.27870515f, 0.27780586f, 0.27746895f,
    0.27756897f, 0.27800936f, 0.2787143f, 0.27961951f, 0.28033122f, 0.29880616f,
    0.29320168f, 0.28555232f, 0.27884975f, 0.2733239f, 0.2689833f, 0.2657251f,
    0.26340303f, 0.26186463f, 0.26097035f, 0.26059982f, 0.26065242f,
    0.26104486f, 0.26170737f, 0.26257557f, 0.26325804f, 0.27432847f,
    0.26966858f, 0.26354778f, 0.2582503f, 0.25383443f, 0.250271f, 0.2474881f,
    0.2453967f, 0.2439044f, 0.24292383f, 0.24237612f, 0.24219193f, 0.24231073f,
    0.24267828f, 0.24323887f, 0.24369502f, 0.25055033f, 0.24643178f,
    0.24124354f, 0.23687162f, 0.23325667f, 0.23031855f, 0.22797456f,
    0.22614679f, 0.22476429f, 0.22376429f, 0.22309166f, 0.22269818f,
    0.22254108f, 0.2225801f, 0.22276837f, 0.22293428f, 0.2274515f, 0.22357783f,
    0.21887547f, 0.2150361f, 0.2119272f, 0.20942956f, 0.20744172f, 0.20587938f,
    0.2046727f, 0.20376417f, 0.20310605f, 0.20265815f, 0.20238546f, 0.20225441f,
    0.2022215f, 0.20219569f, 0.20520192f, 0.20137203f, 0.19684619f, 0.19325054f,
    0.1904101f, 0.18818176f, 0.18645035f, 0.1851238f, 0.18412822f, 0.1834043f,
    0.18290392f, 0.18258746f, 0.18242085f, 0.18237124f, 0.18239334f,
    0.18238066f, 0.18403226f, 0.18012819f, 0.17559084f, 0.17205471f,
    0.16932133f, 0.16723527f, 0.16567492f, 0.16454439f, 0.16376773f,
    0.16328444f, 0.16304576f, 0.16301182f, 0.16314843f, 0.16342238f, 0.1637854f,
    0.16401072f, 0.8923973f, 0.71896756f, 0.56752676f, 0.4793983f, 0.42528638f,
    0.39063668f, 0.36771962f, 0.35219485f, 0.3415089f, 0.334098f, 0.32897025f,
    0.32547572f, 0.3231761f, 0.3217673f, 0.32103318f, 0.32083464f, 0.8572086f,
    0.7151387f, 0.5667241f, 0.47914684f, 0.4251951f, 0.39060146f, 0.36770597f,
    0.35218978f, 0.3415071f, 0.33409727f, 0.3289695f, 0.32547456f, 0.32317424f,
    0.32176477f, 0.32102996f, 0.32083094f, 0.7007748f, 0.66587174f, 0.55520177f,
    0.47528696f, 0.42365286f, 0.38991618f, 0.36737597f, 0.35201752f,
    0.34140724f, 0.33403033f, 0.32891625f, 0.32542527f, 0.32312384f, 0.3217106f,
    0.3209706f, 0.3207675f, 0.6033473f, 0.56967944f, 0.5160297f, 0.45959532f,
    0.41665938f, 0.38647884f, 0.36553335f, 0.3509426f, 0.34072122f, 0.33354762f,
    0.32854074f, 0.3251047f, 0.3228289f, 0.32142413f, 0.32068256f, 0.32047448f,
    0.5259694f, 0.49397272f, 0.4602382f, 0.42827928f, 0.39973116f, 0.37700993f,
    0.35993254f, 0.3474107f, 0.33833587f, 0.33181787f, 0.32719457f, 0.32398555f,
    0.32184333f, 0.32051486f, 0.3198133f, 0.31961843f, 0.46498942f, 0.44089618f,
    0.4129073f, 0.39222547f, 0.37499145f, 0.3605337f, 0.34884298f, 0.33971038f,
    0.33276072f, 0.32758787f, 0.32382715f, 0.32117772f, 0.31940117f,
    0.31831166f, 0.3177644f, 0.3176412f, 0.41816646f, 0.40076974f, 0.3780903f,
    0.36167258f, 0.34972662f, 0.3405492f, 0.3333063f, 0.32759535f, 0.32316163f,
    0.31980163f, 0.31733763f, 0.3156149f, 0.31450155f, 0.31388694f, 0.31367877f,
    0.313731f, 0.38149038f, 0.36873466f, 0.35118002f, 0.33757088f, 0.32786685f,
    0.32102665f, 0.31614983f, 0.31264198f, 0.31012934f, 0.30837408f, 0.3072149f,
    0.3065356f, 0.30624813f, 0.30628264f, 0.306581f, 0.30691624f, 0.35129175f,
    0.34167427f, 0.32827863f, 0.31722942f, 0.30897585f, 0.30314547f,
    0.29918098f, 0.29659057f, 0.2949975f, 0.2941331f, 0.29380733f, 0.29388598f,
    0.2942727f, 0.2948966f, 0.2957021f, 0.29634327f, 0.32483414f, 0.31733412f,
    0.3070106f, 0.29818428f, 0.29123533f, 0.2860866f, 0.28247398f, 0.28009886f,
    0.27868745f, 0.27801645f, 0.27790976f, 0.27823183f, 0.27887884f,
    0.27977043f, 0.28084108f, 0.28165802f, 0.3003267f, 0.29422385f, 0.2860344f,
    0.27895027f, 0.27316338f, 0.2686542f, 0.26529855f, 0.26293534f, 0.26139778f,
    0.26053593f, 0.2602203f, 0.26034278f, 0.2608142f, 0.26156008f, 0.26251447f,
    0.26326132f, 0.27676725f, 0.2715313f, 0.26472676f, 0.25887758f, 0.2540142f,
    0.25008637f, 0.2470063f, 0.24467418f, 0.24298713f, 0.24185064f, 0.24117969f,
    0.2408998f, 0.24094644f, 0.24126223f, 0.24179126f, 0.24224144f, 0.25373796f,
    0.24897489f, 0.24297884f, 0.23790926f, 0.23368667f, 0.23021562f,
    0.22740175f, 0.2251588f, 0.22340783f, 0.22208078f, 0.22111796f, 0.22046748f,
    0.22008383f, 0.2199252f, 0.21994624f, 0.22001967f, 0.23120028f, 0.22662722f,
    0.22101791f, 0.21636368f, 0.2125177f, 0.20934972f, 0.20674938f, 0.20462635f,
    0.20290455f, 0.20152232f, 0.2004283f, 0.19957954f, 0.19893911f, 0.19847295f,
    0.19814068f, 0.19793567f, 0.20932001f, 0.2047506f, 0.19924498f, 0.19474778f,
    0.19107354f, 0.18807088f, 0.18561795f, 0.18361801f, 0.1819922f, 0.18067789f,
    0.17962404f, 0.17878881f, 0.1781369f, 0.17763573f, 0.17724472f, 0.1769777f,
    0.18833901f, 0.18366945f, 0.17810462f, 0.17360964f, 0.16997787f,
    0.16704738f, 0.16469091f, 0.16280872f, 0.16132066f, 0.16016324f,
    0.15928525f, 0.15864491f, 0.15820727f, 0.15794f, 0.15780121f, 0.15771376f,
    0.88834554f, 0.710848f, 0.559409f, 0.47343013f, 0.4217893f, 0.3893709f,
    0.368304f, 0.35424927f, 0.34469867f, 0.33814335f, 0.33364323f, 0.33059347f,
    0.32859346f, 0.3273703f, 0.32673293f, 0.3265604f, 0.8529971f, 0.7070469f,
    0.5586339f, 0.4731937f, 0.42170516f, 0.38933852f, 0.36829096f, 0.3542437f,
    0.3446959f, 0.33814147f, 0.33364138f, 0.33059132f, 0.32859087f, 0.32736713f,
    0.32672924f, 0.32655632f, 0.6956482f, 0.6581358f, 0.5474132f, 0.46950027f,
    0.42024386f, 0.3886862f, 0.36796814f, 0.35406542f, 0.34458396f, 0.3380604f,
    0.33357415f, 0.3305293f, 0.3285295f, 0.32730395f, 0.32666287f, 0.32648718f,
    0.59770876f, 0.5628096f, 0.5090988f, 0.45430583f, 0.41349858f, 0.3853511f,
    0.36614662f, 0.3529688f, 0.34385565f, 0.33752757f, 0.33314794f, 0.33016154f,
    0.32819274f, 0.32698187f, 0.32634544f, 0.32616895f, 0.52093273f,
    0.48830417f, 0.4546407f, 0.42387322f, 0.39703926f, 0.37607825f, 0.36058038f,
    0.34938222f, 0.34137046f, 0.3356794f, 0.33168033f, 0.32892662f, 0.32710183f,
    0.32597977f, 0.32539642f, 0.32524222f, 0.4611703f, 0.4366791f, 0.4088059f,
    0.38894236f, 0.3729594f, 0.35989255f, 0.34953403f, 0.34157607f, 0.33560798f,
    0.3312244f, 0.32807738f, 0.32588908f, 0.324445f, 0.32358164f, 0.32317418f,
    0.32310703f, 0.41575173f, 0.39805943f, 0.37544277f, 0.35954916f,
    0.34844428f, 0.34026766f, 0.33405188f, 0.3293052f, 0.32572234f, 0.32308018f,
    0.32119966f, 0.31993487f, 0.3191676f, 0.31880274f, 0.31876317f, 0.3188932f,
    0.38045242f, 0.36742976f, 0.3498388f, 0.3365107f, 0.32731462f, 0.3211237f,
    0.31695938f, 0.3141661f, 0.31232426f, 0.31116918f, 0.310526f, 0.31027514f,
    0.3103311f, 0.31063086f, 0.31112516f, 0.31156212f, 0.35151786f, 0.3416116f,
    0.32805663f, 0.31707782f, 0.3090618f, 0.3035847f, 0.30004424f, 0.29791242f,
    0.29677993f, 0.29635537f, 0.29643267f, 0.29686785f, 0.29756004f,
    0.29843768f, 0.2994476f, 0.3002058f, 0.32618427f, 0.3183423f, 0.30772322f,
    0.29877687f, 0.29183775f, 0.28679413f, 0.28335407f, 0.28119802f,
    0.28003132f, 0.27961546f, 0.27976167f, 0.28032467f, 0.28119388f, 0.2822849f,
    0.28353068f, 0.28445753f, 0.30264983f, 0.2961369f, 0.28751305f, 0.28013417f,
    0.27415866f, 0.26954216f, 0.2661413f, 0.2637822f, 0.2622844f, 0.26148856f,
    0.2612568f, 0.26147404f, 0.26204535f, 0.2628924f, 0.263947f, 0.2647654f,
    0.2799011f, 0.27418464f, 0.26681432f, 0.26051316f, 0.25528723f, 0.25106817f,
    0.24775422f, 0.24523675f, 0.24340338f, 0.24215358f, 0.24139678f,
    0.24105403f, 0.24105686f, 0.24134552f, 0.24186318f, 0.24231817f, 0.2575076f,
    0.25219968f, 0.2455222f, 0.2398651f, 0.23513138f, 0.2312125f, 0.2280035f,
    0.22541204f, 0.22335202f, 0.22175094f, 0.22054529f, 0.21968009f,
    0.21910739f, 0.21878414f, 0.21866584f, 0.21866211f, 0.23542306f,
    0.23025125f, 0.22386567f, 0.21851361f, 0.21403565f, 0.21029195f,
    0.20716454f, 0.20455831f, 0.20239165f, 0.2005995f, 0.19912758f, 0.19793057f,
    0.19696991f, 0.19621113f, 0.1956162f, 0.19525254f, 0.2138156f, 0.20860592f,
    0.20225182f, 0.19697325f, 0.19257545f, 0.1888999f, 0.18581888f, 0.18323183f,
    0.18105552f, 0.17922449f, 0.17768529f, 0.1763942f, 0.17531468f, 0.17441443f,
    0.1736564f, 0.17316115f, 0.19294143f, 0.18760252f, 0.18113929f, 0.17580563f,
    0.17138764f, 0.1677176f, 0.16466343f, 0.16212198f, 0.16000934f, 0.1582595f,
    0.15681899f, 0.15564439f, 0.1546998f, 0.1539536f, 0.1533686f, 0.15300027f,
    0.88434035f, 0.7030256f, 0.5519244f, 0.46829858f, 0.41923505f, 0.38909936f,
    0.3699066f, 0.35733315f, 0.3489241f, 0.3432291f, 0.33936107f, 0.3367606f,
    0.3350645f, 0.33403042f, 0.3334922f, 0.33336854f, 0.8488484f, 0.6992576f,
    0.55117804f, 0.4680773f, 0.419158f, 0.3890698f, 0.36989415f, 0.35732707f,
    0.34892035f, 0.34322608f, 0.33935815f, 0.33675748f, 0.33506107f,
    0.33402663f, 0.33348805f, 0.33334205f, 0.6907105f, 0.6507531f, 0.5402713f,
    0.4645523f, 0.41777727f, 0.38844985f, 0.36957815f, 0.35714275f, 0.3487965f,
    0.34313104f, 0.33927706f, 0.33668286f, 0.3349889f, 0.33395457f, 0.33341473f,
    0.33326727f, 0.5924572f, 0.5564074f, 0.5028506f, 0.44986314f, 0.41128045f,
    0.38521606f, 0.36777735f, 0.35602456f, 0.34802628f, 0.34254867f, 0.3388008f,
    0.3362684f, 0.3346107f, 0.33359724f, 0.33306834f, 0.33292434f, 0.5164615f,
    0.48324072f, 0.44978368f, 0.4203317f, 0.3952937f, 0.37613958f, 0.3622461f,
    0.35238427f, 0.34544283f, 0.3405839f, 0.33721346f, 0.33491933f, 0.33341596f,
    0.33250365f, 0.3320409f, 0.33192825f, 0.4580423f, 0.4331846f, 0.40550822f,
    0.38654888f, 0.3718816f, 0.36024693f, 0.3512454f, 0.3444752f, 0.3394967f,
    0.33590785f, 0.33337903f, 0.33165592f, 0.330548f, 0.3299139f, 0.3296486f,
    0.32963845f, 0.41409966f, 0.39614537f, 0.37364963f, 0.35833997f,
    0.34812587f, 0.34098583f, 0.33582163f, 0.33205307f, 0.3293304f, 0.32741255f,
    0.32612035f, 0.32531756f, 0.3249f, 0.3247878f, 0.32491904f, 0.3251278f,
    0.38020006f, 0.3669521f, 0.34937823f, 0.33637956f, 0.3277325f, 0.32222307f,
    0.3187964f, 0.31673306f, 0.31557372f, 0.31502724f, 0.31490645f, 0.31508875f,
    0.31549215f, 0.31606033f, 0.3167529f, 0.31729245f, 0.35251093f, 0.34236708f,
    0.32871228f, 0.3178524f, 0.31011406f, 0.30502287f, 0.30193406f, 0.30027935f,
    0.29962292f, 0.29965007f, 0.30013964f, 0.30093858f, 0.30194193f,
    0.30307767f, 0.30429515f, 0.30517164f, 0.32824916f, 0.32012582f, 0.3092815f,
    0.30027112f, 0.2933875f, 0.2884866f, 0.28525186f, 0.28333867f, 0.28243765f,
    0.28229403f, 0.28270704f, 0.28352237f, 0.284623f, 0.28592065f, 0.28734696f,
    0.2883861f, 0.30561152f, 0.29875717f, 0.28978074f, 0.2821739f, 0.27606493f,
    0.2713873f, 0.26798195f, 0.26565838f, 0.26422855f, 0.2635228f, 0.26339546f,
    0.26372477f, 0.26441061f, 0.26537076f, 0.26653498f, 0.2674295f, 0.28358352f,
    0.2774613f, 0.26961708f, 0.26294154f, 0.2574189f, 0.2529653f, 0.24946773f,
    0.246806f, 0.24486345f, 0.24353284f, 0.24271871f, 0.24233794f, 0.24231867f,
    0.24259862f, 0.24312004f, 0.24358778f, 0.2617324f, 0.25595778f, 0.24869809f,
    0.24253999f, 0.23737091f, 0.23307088f, 0.22952712f, 0.22663853f,
    0.22431624f, 0.22248305f, 0.22107211f, 0.22002569f, 0.2192938f, 0.21883208f,
    0.21859664f, 0.21852791f, 0.24001193f, 0.23432085f, 0.22726229f,
    0.22130506f, 0.21627846f, 0.21203434f, 0.20844898f, 0.20542048f,
    0.20286524f, 0.20071453f, 0.19891149f, 0.1974088f, 0.19616655f, 0.19514973f,
    0.19432214f, 0.19381672f, 0.21859801f, 0.21282692f, 0.20572883f,
    0.19976476f, 0.19473158f, 0.19046444f, 0.18683113f, 0.18372577f,
    0.18106338f, 0.17877546f, 0.17680633f, 0.17511067f, 0.17365092f, 0.1723947f,
    0.17130777f, 0.17060612f, 0.19776364f, 0.1918323f, 0.18457404f, 0.1784982f,
    0.17338423f, 0.1690593f, 0.16538745f, 0.16226108f, 0.15959437f, 0.1573186f,
    0.15537828f, 0.15372843f, 0.15233222f, 0.15115832f, 0.15017316f,
    0.14955068f};

static const auto leaving_albedo_lut = vector<float>{0.9978224f, 0.9997985f,
    0.9999442f, 0.9999739f, 0.9999995f, 1.0856267f, 1.0987046f, 1.1010973f,
    1.101869f, 1.1021783f, 1.1023175f, 1.1023837f, 1.1024158f, 1.1024309f,
    1.1024373f, 0.012349165f, 0.96965796f, 0.99714637f, 0.9992363f, 0.99964124f,
    0.9999638f, 1.0847045f, 1.0985509f, 1.1010247f, 1.1018237f, 1.1021489f,
    1.1022992f, 1.102372f, 1.1024083f, 1.1024259f, 1.102434f, 0.16703294f,
    0.8636023f, 0.9537997f, 0.9862083f, 0.9933386f, 0.9975992f, 1.0750487f,
    1.0962243f, 1.0998539f, 1.101083f, 1.1016726f, 1.1020045f, 1.1021831f,
    1.1022847f, 1.1023451f, 1.1023815f, 1.1023937f, 0.8217266f, 0.86581814f,
    0.9339412f, 0.96298677f, 0.98105943f, 1.0519403f, 1.0866339f, 1.094711f,
    1.0978231f, 1.0996215f, 1.1007347f, 1.1013671f, 1.1017501f, 1.1019955f,
    1.1021537f, 1.1022106f, 0.7518938f, 0.78643256f, 0.8476665f, 0.8951086f,
    0.9396095f, 1.01459f, 1.0630939f, 1.0809798f, 1.0891236f, 1.0942185f,
    1.0973657f, 1.0991914f, 1.1003203f, 1.1010574f, 1.1015418f, 1.1017178f,
    0.65255326f, 0.6945629f, 0.7521683f, 0.8090538f, 0.8796376f, 0.9623078f,
    1.0227227f, 1.0546093f, 1.0720286f, 1.0834452f, 1.0905095f, 1.0947057f,
    1.0973483f, 1.0990967f, 1.1002572f, 1.1006805f, 0.5391716f, 0.5849826f,
    0.65483594f, 0.7290044f, 0.81695056f, 0.903195f, 0.9709921f, 1.0161933f,
    1.0454909f, 1.0658569f, 1.0788643f, 1.0868975f, 1.0920925f, 1.0955923f,
    1.0979425f, 1.0988011f, 0.42691046f, 0.47356194f, 0.56307304f, 0.6605726f,
    0.75945f, 0.8463081f, 0.9170844f, 0.97124296f, 1.0114115f, 1.0413982f,
    1.0616941f, 1.0749385f, 1.0838376f, 1.0899907f, 1.0941923f, 1.0957285f,
    0.3272321f, 0.3752697f, 0.483742f, 0.6006075f, 0.7066518f, 0.7942627f,
    0.8663288f, 0.92541647f, 0.97327244f, 1.011451f, 1.0391473f, 1.0584425f,
    1.0720491f, 1.0817865f, 1.0885879f, 1.0910716f, 0.24580304f, 0.2965285f,
    0.41846448f, 0.5462355f, 0.65642226f, 0.74562764f, 0.8191788f, 0.8812125f,
    0.93387336f, 0.9779876f, 1.0121496f, 1.0375861f, 1.0565094f, 1.0706143f,
    1.0807427f, 1.0844213f, 0.18307285f, 0.23688167f, 0.3654291f, 0.49633545f,
    0.60781556f, 0.6986615f, 0.77413857f, 0.8385572f, 0.89441466f, 0.94261926f,
    0.981946f, 1.0130132f, 1.0373701f, 1.056325f, 1.0703566f, 1.0753782f,
    0.13650313f, 0.19281508f, 0.32203484f, 0.45067966f, 0.5609674f, 0.65263855f,
    0.72994685f, 0.79658574f, 0.8549817f, 0.9062311f, 0.94963604f, 0.98558277f,
    1.0150931f, 1.0390185f, 1.0572615f, 1.0635848f, 0.10262236f, 0.16036013f,
    0.28604156f, 0.40923226f, 0.51641303f, 0.6076871f, 0.6861782f, 0.7547146f,
    0.81533915f, 0.86916274f, 0.9159644f, 0.9561119f, 0.9903079f, 1.0190189f,
    1.0414532f, 1.0487576f, 0.07817747f, 0.13618581f, 0.25577044f, 0.37188286f,
    0.4746516f, 0.5643066f, 0.64304304f, 0.7128898f, 0.7754045f, 0.83154655f,
    0.8813657f, 0.9252339f, 0.9636564f, 0.99679816f, 1.0230976f, 1.0307161f,
    0.060546033f, 0.117818095f, 0.23000407f, 0.3384078f, 0.43598866f,
    0.5230203f, 0.6010323f, 0.6714395f, 0.73536247f, 0.7935415f, 0.84611255f,
    0.8933872f, 0.93568623f, 0.9728811f, 1.0025059f, 1.0094048f, 0.04776533f,
    0.10352366f, 0.2078533f, 0.3085052f, 0.40052867f, 0.48421663f, 0.5606589f,
    0.6308512f, 0.6955892f, 0.7554123f, 0.8104435f, 0.8608769f, 0.90681326f,
    0.94776005f, 0.98008925f, 0.9849015f, 0.99787813f, 0.9998144f, 0.9999524f,
    0.9999786f, 0.9999885f, 0.99999404f, 0.99999964f, 1.0000166f, 1.1383651f,
    1.3574244f, 1.379059f, 1.3857088f, 1.3882414f, 1.3892602f, 1.389638f,
    1.3897173f, 0.97039634f, 0.9973608f, 0.9993478f, 0.999711f, 0.9998445f,
    0.99991876f, 0.99999267f, 1.0002124f, 1.1247071f, 1.3569255f, 1.3789238f,
    1.3856469f, 1.3882053f, 1.3892357f, 1.3896213f, 1.3897046f, 0.8743477f,
    0.9574645f, 0.9882362f, 0.9948459f, 0.9972353f, 0.9985055f, 0.9996793f,
    1.0029123f, 1.1300589f, 1.3508228f, 1.3770237f, 1.3847177f, 1.3876381f,
    1.3888444f, 1.3893514f, 1.3894976f, 0.8620915f, 0.88473207f, 0.94540566f,
    0.9724383f, 0.9844284f, 0.99105173f, 0.9967911f, 1.0102558f, 1.1403171f,
    1.3317008f, 1.369729f, 1.3808837f, 1.3852186f, 1.3871561f, 1.3881885f,
    1.3886019f, 0.8390249f, 0.84111965f, 0.88431925f, 0.92488116f, 0.9514334f,
    0.9691799f, 0.9852418f, 1.0166997f, 1.1398169f, 1.2978914f, 1.3524895f,
    1.370987f, 1.3787681f, 1.3826348f, 1.385078f, 1.3861955f, 0.79552925f,
    0.80318755f, 0.8306042f, 0.8668289f, 0.9002489f, 0.92915136f, 0.9596002f,
    1.0116652f, 1.1243911f, 1.253776f, 1.3218691f, 1.351365f, 1.3654816f,
    1.3732697f, 1.3786006f, 1.3811485f, 0.7367077f, 0.7511849f, 0.77798146f,
    0.8092253f, 0.84277356f, 0.87860745f, 0.922806f, 0.9907186f, 1.0936363f,
    1.2026322f, 1.2775766f, 1.3193284f, 1.3426809f, 1.3569294f, 1.3670974f,
    1.3720744f, 0.66708624f, 0.6841485f, 0.7162161f, 0.749683f, 0.78568107f,
    0.82791185f, 0.88298213f, 0.9584391f, 1.0520394f, 1.1467894f, 1.2229754f,
    1.2750273f, 1.3090614f, 1.331959f, 1.3489087f, 1.3574373f, 0.59103733f,
    0.6073765f, 0.6448052f, 0.6859423f, 0.730168f, 0.7815901f, 0.84497267f,
    0.92169285f, 1.0063225f, 1.0900586f, 1.1637443f, 1.222024f, 1.2656691f,
    1.2978278f, 1.3227109f, 1.3357269f, 0.513028f, 0.5272822f, 0.56915945f,
    0.62038195f, 0.6766793f, 0.7391543f, 0.8089702f, 0.88443863f, 0.9614072f,
    1.0362701f, 1.1054493f, 1.1656536f, 1.2156788f, 1.2554556f, 1.2878594f,
    1.3056865f, 0.43723294f, 0.44954833f, 0.49553156f, 0.5570915f, 0.6258729f,
    0.69871294f, 0.7734854f, 0.84768134f, 0.9191891f, 0.98737663f, 1.051612f,
    1.1105875f, 1.1630372f, 1.2069492f, 1.2445905f, 1.2665474f, 0.36697543f,
    0.3782688f, 0.42841306f, 0.49911088f, 0.57821834f, 0.6590039f, 0.73743576f,
    0.81118333f, 0.87957054f, 0.9433742f, 1.003354f, 1.0594993f, 1.1110388f,
    1.1548995f, 1.1939765f, 1.2182002f, 0.30437574f, 0.31571636f, 0.36998424f,
    0.44774383f, 0.5339079f, 0.6197354f, 0.7006044f, 0.7746095f, 0.8417506f,
    0.903209f, 0.9601636f, 1.0131502f, 1.0616671f, 1.1016632f, 1.1376557f,
    1.1612505f, 0.25034547f, 0.26259804f, 0.32061082f, 0.4030215f, 0.49297392f,
    0.5812171f, 0.66336465f, 0.73789686f, 0.8049721f, 0.86561674f, 0.9208442f,
    0.97111994f, 1.0156866f, 1.0489569f, 1.0774646f, 1.096941f, 0.20484039f,
    0.21854386f, 0.27959186f, 0.3643368f, 0.45536354f, 0.54394794f, 0.6263147f,
    0.7012372f, 0.76877975f, 0.8295729f, 0.88416064f, 0.93251234f, 0.9730643f,
    0.99779326f, 1.0151201f, 1.026973f, 0.16720319f, 0.18259102f, 0.2457662f,
    0.33087498f, 0.42096153f, 0.5083734f, 0.5900404f, 0.6649577f, 0.7330224f,
    0.794427f, 0.84914243f, 0.89640135f, 0.93338305f, 0.94862664f, 0.95203537f,
    0.9532763f, 0.99789387f, 0.9998189f, 0.9999546f, 0.99998003f, 0.99998933f,
    0.9999941f, 0.9999974f, 1.000001f, 1.0000085f, 1.0000534f, 1.526494f,
    1.6664665f, 1.6922531f, 1.7001559f, 1.702679f, 1.7031637f, 0.97060394f,
    0.9974211f, 0.9993771f, 0.9997298f, 0.9998561f, 0.9999199f, 0.99996406f,
    1.0000117f, 1.0001099f, 1.0006961f, 1.5210186f, 1.6660131f, 1.6921155f,
    1.7000923f, 1.7026416f, 1.7031354f, 0.8773063f, 0.9584556f, 0.9887311f,
    0.99517274f, 0.99745303f, 0.9985712f, 0.9993151f, 1.0000812f, 1.0016028f,
    1.0102135f, 1.4774435f, 1.6602037f, 1.6901574f, 1.6991247f, 1.702047f,
    1.7026764f, 0.87309384f, 0.8896936f, 0.94806033f, 0.97426736f, 0.9857527f,
    0.9917268f, 0.9956698f, 0.9995221f, 1.006635f, 1.0404216f, 1.4177529f,
    1.6404649f, 1.6825372f, 1.6951139f, 1.6994952f, 1.7006884f, 0.86298823f,
    0.85544664f, 0.8929159f, 0.9311137f, 0.9562521f, 0.97231793f, 0.9839436f,
    0.9953405f, 1.0147318f, 1.0832163f, 1.3754352f, 1.6012523f, 1.6641434f,
    1.6847066f, 1.6926639f, 1.6953466f, 0.8361645f, 0.83278584f, 0.8508296f,
    0.8822972f, 0.9126794f, 0.93802196f, 0.9598995f, 0.9832209f, 1.0207582f,
    1.1160918f, 1.3421452f, 1.5451934f, 1.6303678f, 1.6637601f, 1.6784288f,
    1.6841723f, 0.7963672f, 0.8003991f, 0.81555784f, 0.8393256f, 0.86724025f,
    0.89612466f, 0.92622524f, 0.9624287f, 1.0187706f, 1.1266778f, 1.3058661f,
    1.4790374f, 1.5792518f, 1.6283451f, 1.6532743f, 1.6642278f, 0.746375f,
    0.7546181f, 0.7743198f, 0.7976296f, 0.8242195f, 0.85447687f, 0.8903294f,
    0.93732333f, 1.0071751f, 1.1156876f, 1.2618368f, 1.4075428f, 1.5125481f,
    1.5762824f, 1.6140922f, 1.6324974f, 0.6884673f, 0.697634f, 0.7224877f,
    0.7506562f, 0.78087837f, 0.81517124f, 0.85712993f, 0.9123994f, 0.9884279f,
    1.0900712f, 1.2113774f, 1.3337848f, 1.434992f, 1.5084326f, 1.5591933f,
    1.5864534f, 0.6250343f, 0.6330433f, 0.661599f, 0.69679767f, 0.7348741f,
    0.7771584f, 0.8270497f, 0.8887181f, 0.9654701f, 1.0571045f, 1.1585381f,
    1.2609743f, 1.3525445f, 1.4286642f, 1.4890393f, 1.5246925f, 0.5586985f,
    0.5646924f, 0.59573424f, 0.63843095f, 0.68655336f, 0.7393359f, 0.7982253f,
    0.86491984f, 0.9398947f, 1.0215869f, 1.106969f, 1.1920984f, 1.2705742f,
    1.3423537f, 1.4062084f, 1.4474102f, 0.49214423f, 0.49619293f, 0.52929544f,
    0.57928085f, 0.6377573f, 0.7013179f, 0.76875675f, 0.83939946f, 0.9122324f,
    0.98578995f, 1.058623f, 1.128952f, 1.1926936f, 1.2545025f, 1.3145825f,
    1.3564949f, 0.42782634f, 0.4305591f, 0.46588326f, 0.5226048f, 0.5903683f,
    0.6632404f, 0.73768044f, 0.8113775f, 0.882684f, 0.95044786f, 1.0139507f,
    1.0719435f, 1.1205602f, 1.1686026f, 1.2182534f, 1.2551869f, 0.36770672f,
    0.3699734f, 0.40788406f, 0.4704957f, 0.5456655f, 0.6255051f, 0.7049755f,
    0.7809361f, 0.8515083f, 0.91562766f, 0.97256196f, 1.0205243f, 1.0543588f,
    1.0864869f, 1.1207074f, 1.1474558f, 0.3131102f, 0.3157326f, 0.35650444f,
    0.4239076f, 0.50429255f, 0.58860433f, 0.6711862f, 0.7486687f, 0.81909156f,
    0.8812147f, 0.93377626f, 0.9737295f, 0.9934662f, 1.0087533f, 1.024503f,
    1.037345f, 0.2647195f, 0.26835138f, 0.31202993f, 0.38297224f, 0.4664411f,
    0.55299985f, 0.6370472f, 0.7153408f, 0.7858897f, 0.8471161f, 0.8969357f,
    0.9305615f, 0.9369798f, 0.93529993f, 0.9313284f, 0.92847025f, 0.9979021f,
    0.9998213f, 0.9999557f, 0.9999808f, 0.9999898f, 0.9999943f, 0.99999714f,
    0.9999997f, 1.0000031f, 1.0000107f, 1.0000527f, 1.7278849f, 1.9898651f,
    2.0274246f, 2.036931f, 2.0385644f, 0.97071135f, 0.99745256f, 0.9993924f,
    0.99973965f, 0.99986273f, 0.9999231f, 0.99996156f, 0.9999948f, 1.000039f,
    1.0001395f, 1.0006882f, 1.7152977f, 1.9892209f, 2.0272517f, 2.0368574f,
    2.0385163f, 0.87882704f, 0.9589686f, 0.9889862f, 0.9953396f, 0.99756765f,
    0.9986343f, 0.99929214f, 0.99984205f, 1.0005462f, 1.0021131f, 1.0103905f,
    1.6342249f, 1.9809531f, 2.0247705f, 2.0357206f, 2.0377448f, 0.87872213f,
    0.8922341f, 0.9494046f, 0.97517276f, 0.9864032f, 0.992136f, 0.9956885f,
    0.99856883f, 1.0020785f, 1.0095007f, 1.044279f, 1.5571395f, 1.9528562f,
    2.0150554f, 2.0309536f, 2.0344062f, 0.87523097f, 0.86273485f, 0.8972214f,
    0.9341447f, 0.9585327f, 0.9739174f, 0.98448634f, 0.9933231f, 1.0038056f,
    1.0244055f, 1.1013066f, 1.5161344f, 1.8972439f, 1.9914805f, 2.0184748f,
    2.0254233f, 0.8569984f, 0.8478538f, 0.8609709f, 0.88985276f, 0.9186145f,
    0.9425039f, 0.96221846f, 0.98058f, 1.0029362f, 1.0436673f, 1.1580373f,
    1.4870119f, 1.81881f, 1.9479171f, 1.9930944f, 2.0065992f, 0.8272595f,
    0.8257041f, 0.83471304f, 0.8544825f, 0.8795885f, 0.9058488f, 0.9321569f,
    0.9604764f, 0.9972428f, 1.0598565f, 1.1938953f, 1.4540774f, 1.7287241f,
    1.8814323f, 1.9494728f, 1.97304f, 0.78816676f, 0.79161423f, 0.804936f,
    0.82316524f, 0.8455889f, 0.8715987f, 0.9014793f, 0.9378035f, 0.9876462f,
    1.0669236f, 1.2033288f, 1.4109164f, 1.6341399f, 1.7936126f, 1.88368f,
    1.9200181f, 0.74120337f, 0.7465763f, 0.7654784f, 0.78789765f, 0.8125334f,
    0.8405875f, 0.8742093f, 0.91720855f, 0.9765469f, 1.0635748f, 1.1907419f,
    1.3572491f, 1.5375206f, 1.6895443f, 1.7948095f, 1.8442073f, 0.6878397f,
    0.69292f, 0.7160496f, 0.7449925f, 0.77607f, 0.81015486f, 0.8499861f,
    0.8996829f, 0.96482384f, 1.0516583f, 1.1636707f, 1.2964202f, 1.4405055f,
    1.5757519f, 1.6856611f, 1.7448807f, 0.6297897f, 0.63334054f, 0.65913093f,
    0.6949752f, 0.73487383f, 0.7781594f, 0.82665133f, 0.8833412f, 0.9515037f,
    1.0334579f, 1.1286608f, 1.232636f, 1.3452871f, 1.4584708f, 1.561913f,
    1.6244806f, 0.569016f, 0.5706332f, 0.5981455f, 0.6405825f, 0.6901121f,
    0.7440407f, 0.80225277f, 0.8656698f, 0.9351622f, 1.0106187f, 1.0899774f,
    1.169117f, 1.2538997f, 1.3425622f, 1.4302979f, 1.4881995f, 0.507589f,
    0.5074823f, 0.53648114f, 0.5850731f, 0.64394486f, 0.7082908f, 0.7757307f,
    0.8449836f, 0.91498685f, 0.9842468f, 1.0499552f, 1.1076256f, 1.1675042f,
    1.2310919f, 1.296898f, 1.3427695f, 0.4474822f, 0.44623882f, 0.47696173f,
    0.5312376f, 0.59842336f, 0.67177343f, 0.7468995f, 0.82076764f, 0.890995f,
    0.9552018f, 1.009747f, 1.0487947f, 1.08633f, 1.1255138f, 1.166274f,
    1.1950526f, 0.39037684f, 0.38874295f, 0.42161086f, 0.48100471f, 0.5550182f,
    0.6353401f, 0.7161722f, 0.7933856f, 0.86378187f, 0.92423844f, 0.96990126f,
    0.992626f, 1.0100048f, 1.0261832f, 1.0413986f, 1.0509708f, 0.33753088f,
    0.33635107f, 0.37163067f, 0.43543094f, 0.5145716f, 0.5996861f, 0.6842505f,
    0.7636423f, 0.8341893f, 0.8920266f, 0.9306916f, 0.93887216f, 0.9379219f,
    0.93289155f, 0.92400825f, 0.91497296f, 0.9979072f, 0.99982274f, 0.9999565f,
    0.9999813f, 0.99999017f, 0.99999446f, 0.99999714f, 0.99999934f, 1.0000017f,
    1.0000057f, 1.000017f, 1.0001129f, 2.1884086f, 2.3604934f, 2.3884077f,
    2.3925462f, 0.9707783f, 0.9974724f, 0.99940217f, 0.99974597f, 0.999867f,
    0.9999257f, 0.99996156f, 0.99999f, 1.0000215f, 1.0000743f, 1.0002236f,
    1.0014762f, 2.183499f, 2.36001f, 2.3882668f, 2.3924696f, 0.87977135f,
    0.95929015f, 0.9891473f, 0.9954454f, 0.99764156f, 0.99868f, 0.9992993f,
    0.9997746f, 1.0002843f, 1.0011204f, 1.0034504f, 1.0220164f, 2.1330779f,
    2.353447f, 2.3861454f, 2.3912528f, 0.8822072f, 0.89381826f, 0.9502463f,
    0.97573876f, 0.9868108f, 0.9924072f, 0.9957786f, 0.99831164f, 1.0009222f,
    1.0050316f, 1.01601f, 1.0889673f, 2.0220392f, 2.3294258f, 2.3774753f,
    2.3860302f, 0.8827967f, 0.86726165f, 0.899899f, 0.93601894f, 0.9599326f,
    0.97491807f, 0.9849739f, 0.9928477f, 1.0008923f, 1.0130318f, 1.043021f,
    1.1879854f, 1.9038229f, 2.2764819f, 2.3554327f, 2.372092f, 0.86987066f,
    0.85719943f, 0.8672642f, 0.89451003f, 0.92223537f, 0.9452509f, 0.9638751f,
    0.9802591f, 0.99785054f, 1.0238683f, 1.0815685f, 1.2753134f, 1.809025f,
    2.1910212f, 2.3122673f, 2.3431768f, 0.8463939f, 0.84142977f, 0.8466467f,
    0.8638899f, 0.8872064f, 0.9118968f, 0.9362702f, 0.96124417f, 0.9906324f,
    1.0342753f, 1.1200916f, 1.3280507f, 1.729453f, 2.0798402f, 2.2415648f,
    2.2922902f, 0.8142082f, 0.8147528f, 0.82421845f, 0.8393023f, 0.85915434f,
    0.88271874f, 0.90956026f, 0.94095826f, 0.9813227f, 1.0415214f, 1.1465468f,
    1.3443241f, 1.651967f, 1.953409f, 2.1407058f, 2.21324f, 0.7744023f,
    0.7775492f, 0.7930548f, 0.8121213f, 0.8335244f, 0.8581584f, 0.88746333f,
    0.92404675f, 0.9730778f, 1.0445445f, 1.155632f, 1.3305264f, 1.569486f,
    1.8193083f, 2.0120342f, 2.1026256f, 0.7279709f, 0.7314851f, 0.7518843f,
    0.7775626f, 0.8049117f, 0.8346284f, 0.8689449f, 0.91128486f, 0.9667927f,
    1.0429091f, 1.1484759f, 1.2952627f, 1.4808414f, 1.6819807f, 1.8620067f,
    1.9614623f, 0.6761118f, 0.67859066f, 0.7022139f, 0.73494965f, 0.77077f,
    0.8089638f, 0.8512076f, 0.90049934f, 0.9607309f, 1.0360078f, 1.1287382f,
    1.2464219f, 1.3881514f, 1.5447137f, 1.6990647f, 1.795462f, 0.620304f,
    0.62108374f, 0.64669406f, 0.6860854f, 0.7313535f, 0.7799282f, 0.8319323f,
    0.88879764f, 0.95229024f, 1.0233136f, 1.0999513f, 1.1898446f, 1.2942908f,
    1.4103545f, 1.5315003f, 1.613693f, 0.5622502f, 0.5612583f, 0.5882789f,
    0.63378143f, 0.6884663f, 0.7478245f, 0.8099182f, 0.8740277f, 0.9395764f,
    1.0048258f, 1.0648f, 1.1292783f, 1.2015113f, 1.2810674f, 1.3660972f,
    1.4263308f, 0.5037392f, 0.5013258f, 0.5297387f, 0.5808796f, 0.6443095f,
    0.71367186f, 0.78492945f, 0.85523117f, 0.92192745f, 0.9811583f, 1.025255f,
    1.0669419f, 1.1111373f, 1.1581547f, 1.2076768f, 1.2425928f, 0.44647464f,
    0.44324f, 0.47336397f, 0.5297127f, 0.60079134f, 0.67860806f, 0.7573522f,
    0.8324684f, 0.8996918f, 0.9533374f, 0.9828168f, 1.0041285f, 1.0237916f,
    1.0422047f, 1.0592787f, 1.0695175f, 0.39192f, 0.38855612f, 0.42082012f,
    0.48192137f, 0.559259f, 0.64360386f, 0.727883f, 0.8064175f, 0.87376785f,
    0.9225253f, 0.9386662f, 0.94163316f, 0.93972206f, 0.93338704f, 0.9225981f,
    0.91163033f, 0.9979107f, 0.9998238f, 0.99995697f, 0.9999816f, 0.99999034f,
    0.99999464f, 0.9999972f, 0.99999917f, 1.0000011f, 1.0000039f, 1.0000101f,
    1.0000354f, 1.0016166f, 2.6798525f, 2.753503f, 2.762374f, 0.97082406f,
    0.9974862f, 0.99940896f, 0.99975044f, 0.9998701f, 0.9999276f, 0.99996203f,
    0.99998796f, 1.000014f, 1.000051f, 1.0001327f, 1.0004636f, 1.0202594f,
    2.678384f, 2.75324f, 2.7622566f, 0.8804165f, 0.9595123f, 0.9892597f,
    0.9955197f, 0.99769413f, 0.99871397f, 0.9993103f, 0.9997467f, 1.0001713f,
    1.0007639f, 1.0020497f, 1.0071875f, 1.1927868f, 2.6597612f, 2.7493804f,
    2.760403f, 0.884584f, 0.894909f, 0.95083106f, 0.9761334f, 0.9870963f,
    0.9926017f, 0.9958627f, 0.9982111f, 1.0004201f, 1.0033872f, 1.0096232f,
    1.0332899f, 1.4008827f, 2.5984619f, 2.7340028f, 2.7525177f, 0.88794583f,
    0.87037134f, 0.9017517f, 0.93731713f, 0.96090144f, 0.975615f, 0.98534954f,
    0.9926955f, 0.9996103f, 1.008618f, 1.0266541f, 1.0879099f, 1.5296724f,
    2.4848192f, 2.6961877f, 2.7316995f, 0.8786192f, 0.8636107f, 0.8716112f,
    0.89772606f, 0.92472684f, 0.94713724f, 0.9650593f, 0.9803201f, 0.9955802f,
    1.0154862f, 1.0532217f, 1.1617094f, 1.5945998f, 2.338855f, 2.6255634f,
    2.6891332f, 0.85939825f, 0.852217f, 0.8549005f, 0.8704004f, 0.892463f,
    0.91605794f, 0.93916327f, 0.96216786f, 0.98767823f, 1.0220739f, 1.0839471f,
    1.2309511f, 1.6142082f, 2.1867063f, 2.516895f, 2.6156774f, 0.83194023f,
    0.83065516f, 0.8376199f, 0.85056233f, 0.86862993f, 0.8905023f, 0.9153437f,
    0.9437352f, 0.9786262f, 1.027439f, 1.1107384f, 1.2761664f, 1.5984703f,
    2.0386415f, 2.3725119f, 2.5043812f, 0.7971049f, 0.7989358f, 0.8123894f,
    0.82926834f, 0.84850085f, 0.87082756f, 0.8972955f, 0.9297702f, 0.9720629f,
    1.0316687f, 1.1273161f, 1.2910103f, 1.5542558f, 1.8934233f, 2.2004704f,
    2.353159f, 0.75560814f, 0.7583303f, 0.7773362f, 0.8010886f, 0.8260895f,
    0.8529579f, 0.8836168f, 0.92089474f, 0.9690362f, 1.034161f, 1.1310471f,
    1.278671f, 1.4883673f, 1.7482946f, 2.010786f, 2.1662455f, 0.70833385f,
    0.7104598f, 0.7333418f, 0.76456016f, 0.79803973f, 0.8330095f, 0.8709806f,
    0.91465163f, 0.9675621f, 1.0330353f, 1.1219515f, 1.2457137f, 1.4076495f,
    1.6028786f, 1.8129343f, 1.9534761f, 0.65644467f, 0.65715677f, 0.6825083f,
    0.72077596f, 0.76384854f, 0.80909014f, 0.8566133f, 0.90783465f, 0.9644507f,
    1.0261534f, 1.1012825f, 1.1983697f, 1.3180232f, 1.4587771f, 1.6149517f,
    1.7277359f, 0.60136926f, 0.60040635f, 0.6274037f, 0.672078f, 0.7248408f,
    0.7811171f, 0.8390458f, 0.8979754f, 0.9570918f, 1.012238f, 1.0707762f,
    1.1413921f, 1.2239162f, 1.318178f, 1.423169f, 1.5019206f, 0.54470885f,
    0.5422383f, 0.5706267f, 0.6211473f, 0.6830884f, 0.7500223f, 0.81792927f,
    0.8838161f, 0.94422704f, 0.99122953f, 1.0323597f, 1.0781739f, 1.1283476f,
    1.1829766f, 1.2421643f, 1.2866548f, 0.48809722f, 0.48457736f, 0.51451343f,
    0.570433f, 0.6406522f, 0.7170563f, 0.7936181f, 0.8651805f, 0.9258652f,
    0.9640234f, 0.9879843f, 1.0111619f, 1.0333209f, 1.054538f, 1.074913f,
    1.0892489f, 0.43305066f, 0.42909396f, 0.4609468f, 0.521851f, 0.59918284f,
    0.68346053f, 0.76681167f, 0.8426058f, 0.9028189f, 0.932004f, 0.93948317f,
    0.942194f, 0.94021773f, 0.9337871f, 0.92307407f, 0.9136976f, 0.9979132f,
    0.9998246f, 0.9999574f, 0.9999818f, 0.9999906f, 0.99999475f, 0.9999972f,
    0.99999905f, 1.0000007f, 1.000003f, 1.0000074f, 1.0000205f, 1.0001366f,
    2.9319475f, 3.128735f, 3.1458073f, 0.97085726f, 0.99749625f, 0.999414f,
    0.9997538f, 0.99987245f, 0.9999291f, 0.9999625f, 0.9999868f, 1.0000095f,
    1.0000391f, 1.0000961f, 1.0002692f, 1.0017884f, 2.9261925f, 3.1282551f,
    3.1456318f, 0.8808841f, 0.95967513f, 0.98934335f, 0.99557537f, 0.99773365f,
    0.9987399f, 0.99932015f, 0.99973124f, 1.0001056f, 1.0005804f, 1.0014836f,
    1.0041976f, 1.0270959f, 2.8622699f, 3.1213696f, 3.142883f, 0.88630426f,
    0.8957075f, 0.9512639f, 0.97642744f, 0.98730934f, 0.99274755f, 0.9959304f,
    0.9981569f, 1.0001247f, 1.0025318f, 1.006976f, 1.0199366f, 1.1141543f,
    2.7014136f, 3.0946667f, 3.1312919f, 0.89166546f, 0.8726444f, 0.9031196f,
    0.93828005f, 0.961619f, 0.9761297f, 0.98563194f, 0.9926221f, 0.99884033f,
    1.0062653f, 1.0194688f, 1.0556566f, 1.2545096f, 2.5023122f, 3.031568f,
    3.10104f, 0.88492703f, 0.8682928f, 0.8748171f, 0.90010625f, 0.9265647f,
    0.9485179f, 0.9659221f, 0.98040396f, 0.99417025f, 1.0108318f, 1.0395478f,
    1.1111075f, 1.390124f, 2.3281963f, 2.9205701f, 3.0402226f, 0.86876345f,
    0.86008996f, 0.8609926f, 0.87522364f, 0.89634424f, 0.9191033f, 0.94125545f,
    0.9628528f, 0.985749f, 1.0148839f, 1.0642301f, 1.1736914f, 1.4819238f,
    2.1834965f, 2.762542f, 2.9377284f, 0.84470975f, 0.84226286f, 0.8475395f,
    0.8589435f, 0.8756728f, 0.89624953f, 0.91956866f, 0.9457442f, 0.9766978f,
    1.018432f, 1.0883056f, 1.2256821f, 1.5223923f, 2.0518117f, 2.5692248f,
    2.7870584f, 0.8134778f, 0.81457126f, 0.8267726f, 0.84214234f, 0.8597717f,
    0.8803427f, 0.9046354f, 0.93397385f, 0.97098887f, 1.0224093f, 1.1067479f,
    1.2549016f, 1.5176415f, 1.9200414f, 2.354814f, 2.5892956f, 0.7756047f,
    0.7780276f, 0.79641217f, 0.8189734f, 0.842313f, 0.8670492f, 0.8948792f,
    0.9281049f, 0.96979946f, 1.026467f, 1.1159124f, 1.2581651f, 1.4775524f,
    1.7823619f, 2.1303787f, 2.3534086f, 0.7317705f, 0.7339854f, 0.75691f,
    0.787433f, 0.81939447f, 0.85200626f, 0.88660777f, 0.9254642f, 0.9710211f,
    1.0282248f, 1.113601f, 1.2381525f, 1.4121277f, 1.6385615f, 1.9038321f,
    2.0937834f, 0.6829252f, 0.68401456f, 0.7099746f, 0.74808395f, 0.7899337f,
    0.83279604f, 0.87664986f, 0.9225731f, 0.9711112f, 1.0245554f, 1.0989604f,
    1.1995124f, 1.32987f, 1.4913714f, 1.6814892f, 1.82653f, 0.630301f,
    0.6298728f, 0.657868f, 0.70286363f, 0.7548072f, 0.8089358f, 0.8632375f,
    0.9166992f, 0.96698046f, 1.013156f, 1.0723908f, 1.1468177f, 1.2372844f,
    1.3443342f, 1.4687026f, 1.5662152f, 0.57534677f, 0.57343f, 0.60301006f,
    0.6542407f, 0.71586263f, 0.7811625f, 0.84585595f, 0.90635467f, 0.956917f,
    0.9932477f, 1.035301f, 1.0839939f, 1.1391087f, 1.200679f, 1.2697382f,
    1.3238729f, 0.51961166f, 0.5165435f, 0.547701f, 0.60465235f, 0.675159f,
    0.7507144f, 0.8247926f, 0.8911826f, 0.9406245f, 0.96541935f, 0.98971814f,
    1.0143207f, 1.038793f, 1.0629696f, 1.0875783f, 1.1063592f, 0.4646072f,
    0.46091446f, 0.49391764f, 0.55616057f, 0.63450074f, 0.71885115f, 0.8007332f,
    0.87160355f, 0.91879976f, 0.931093f, 0.93788636f, 0.94053864f, 0.9389271f,
    0.93312263f, 0.9239127f, 0.91671216f, 0.9979151f, 0.9998251f, 0.9999577f,
    0.999982f, 0.99999064f, 0.9999948f, 0.99999726f, 0.999999f, 1.0000005f,
    1.0000024f, 1.0000058f, 1.0000147f, 1.0000612f, 2.7867408f, 3.5104294f,
    3.5409923f, 0.9708823f, 0.99750394f, 0.99941796f, 0.9997564f, 0.99987435f,
    0.9999302f, 0.9999629f, 0.99998593f, 1.0000063f, 1.0000318f, 1.0000767f,
    1.0001925f, 1.0008014f, 2.7261853f, 3.5095606f, 3.5407374f, 0.8812369f,
    0.9597996f, 0.98940796f, 0.9956188f, 0.9977646f, 0.99876016f, 0.9993278f,
    0.99971956f, 1.0000571f, 1.0004681f, 1.0011809f, 1.0030062f, 1.0124584f,
    2.4482627f, 3.4973748f, 3.5367723f, 0.8876014f, 0.89631695f, 0.95159835f,
    0.97665596f, 0.98747486f, 0.9928597f, 0.99598074f, 0.9981134f, 0.99990237f,
    1.002004f, 1.0055448f, 1.0143998f, 1.0575662f, 2.2817287f, 3.4515805f,
    3.5201826f, 0.8944653f, 0.87437874f, 0.9041753f, 0.93902653f, 0.9621737f,
    0.9765221f, 0.9858384f, 0.9925494f, 0.9982389f, 1.0047922f, 1.0154812f,
    1.0410546f, 1.1501379f, 2.2069905f, 3.3488455f, 3.4773772f, 0.889665f,
    0.8718637f, 0.87729f, 0.90194917f, 0.92798173f, 0.9495658f, 0.9665483f,
    0.9804039f, 0.99299806f, 1.0078468f, 1.0316075f, 1.0849407f, 1.2709997f,
    2.1524687f, 3.1817842f, 3.3928525f, 0.8757852f, 0.8660907f, 0.86569566f,
    0.8789615f, 0.89933884f, 0.92141604f, 0.94277805f, 0.963211f, 0.98396325f,
    1.0101101f, 1.052013f, 1.1391526f, 1.380811f, 2.0912726f, 2.9655666f,
    3.2540655f, 0.8542735f, 0.8511051f, 0.8552137f, 0.86546093f, 0.8811315f,
    0.900641f, 0.9226766f, 0.9469606f, 0.97449934f, 1.012165f, 1.0732212f,
    1.1902243f, 1.4517343f, 2.0127444f, 2.7239113f, 3.0567324f, 0.82573926f,
    0.8264823f, 0.8379382f, 0.85221505f, 0.86858207f, 0.8876977f, 0.9101273f,
    0.9366931f, 0.96880233f, 1.015543f, 1.0915082f, 1.2255106f, 1.4760022f,
    1.9131172f, 2.4734676f, 2.8072052f, 0.7905968f, 0.79305226f, 0.81129366f,
    0.8330906f, 0.85515237f, 0.87811875f, 0.90348375f, 0.93296945f, 0.9681412f,
    1.020198f, 1.1032565f, 1.2383288f, 1.4582686f, 1.7933304f, 2.2220485f,
    2.5203528f, 0.7493866f, 0.75198257f, 0.77541786f, 0.80569345f, 0.83656204f,
    0.86722344f, 0.8988166f, 0.93297195f, 0.9703179f, 1.0235687f, 1.1051303f,
    1.2278135f, 1.4077172f, 1.6576405f, 1.9738476f, 2.2149277f, 0.70291096f,
    0.70465976f, 0.73172784f, 0.77018356f, 0.81128895f, 0.8521949f, 0.89264494f,
    0.93300223f, 0.9715347f, 1.0219254f, 1.0947325f, 1.1963923f, 1.3336825f,
    1.5118526f, 1.7329564f, 1.9092653f, 0.6522608f, 0.652675f, 0.6822474f,
    0.72816455f, 0.7798273f, 0.8321973f, 0.8829282f, 0.930105f, 0.96834856f,
    1.0122076f, 1.0712061f, 1.1477506f, 1.2439985f, 1.3617958f, 1.5038924f,
    1.6184968f, 0.5987685f, 0.5977708f, 0.6292389f, 0.6818955f, 0.74378276f,
    0.8077438f, 0.8689718f, 0.9226042f, 0.95869464f, 0.99300545f, 1.0352801f,
    1.0858332f, 1.1448072f, 1.2125021f, 1.2908565f, 1.3532101f, 0.54390734f,
    0.5417346f, 0.5749415f, 0.6337472f, 0.70513713f, 0.77999187f, 0.85097325f,
    0.9100021f, 0.9420185f, 0.9645611f, 0.9888462f, 1.0144374f, 1.0408785f,
    1.0679739f, 1.0970134f, 1.1192766f, 0.4891652f, 0.48626396f, 0.5213663f,
    0.58584285f, 0.6657545f, 0.7502266f, 0.82958573f, 0.8926318f, 0.9189764f,
    0.92823994f, 0.9343848f, 0.93703014f, 0.9359597f, 0.93123174f, 0.92417467f,
    0.9185183f, 0.99791664f, 0.99982566f, 0.9999579f, 0.9999822f, 0.9999908f,
    0.99999493f, 0.99999726f, 0.9999989f, 1.0000004f, 1.000002f, 1.0000049f,
    1.0000117f, 1.0000389f, 1.0012856f, 3.8942044f, 3.9463842f, 0.9709017f,
    0.9975101f, 0.9994212f, 0.99975854f, 0.9998758f, 0.99993116f, 0.99996316f,
    0.99998516f, 1.0000039f, 1.0000268f, 1.0000645f, 1.0001522f, 1.0005096f,
    1.0165629f, 3.8926158f, 3.9460237f, 0.8815096f, 0.9598976f, 0.98945963f,
    0.9956536f, 0.99778926f, 0.99877584f, 0.9993327f, 0.9997075f, 1.0000205f,
    1.0003909f, 1.0009918f, 1.0023792f, 1.007975f, 1.2010014f, 3.870894f,
    3.9404464f, 0.88860744f, 0.896797f, 0.95186514f, 0.9768391f, 0.98760676f,
    0.99294686f, 0.9960146f, 0.9980633f, 0.9997326f, 1.001639f, 1.0046451f,
    1.0114369f, 1.0378739f, 1.5302434f, 3.7925553f, 3.9172764f, 0.8966379f,
    0.8757449f, 0.90501654f, 0.93962383f, 0.9626146f, 0.9768257f, 0.98598015f,
    0.99243724f, 0.9977711f, 1.0037618f, 1.0129353f, 1.0329272f, 1.1046624f,
    1.7772498f, 3.6292539f, 3.858178f, 0.8933337f, 0.8746764f, 0.87926114f,
    0.90342295f, 0.9291068f, 0.9503758f, 0.9669858f, 0.98025894f, 0.99205947f,
    1.0057197f, 1.0263971f, 1.0693775f, 1.2044762f, 1.9099851f, 3.3904674f,
    3.7436876f, 0.88121045f, 0.8708156f, 0.86944866f, 0.88195384f, 0.9017189f,
    0.9232078f, 0.94385993f, 0.96317935f, 0.98247004f, 1.0066152f, 1.0436625f,
    1.1167151f, 1.3111253f, 1.9588257f, 3.1147473f, 3.5609086f, 0.86165017f,
    0.85806197f, 0.8613496f, 0.870694f, 0.88548625f, 0.904063f, 0.9249232f,
    0.94733906f, 0.972536f, 1.0074022f, 1.0623336f, 1.1647712f, 1.3942888f,
    1.9463012f, 2.8334851f, 3.3100903f, 0.8351868f, 0.8358473f, 0.84689075f,
    0.86034274f, 0.87565786f, 0.89348227f, 0.91416967f, 0.9378961f, 0.96662676f,
    1.0100472f, 1.0797237f, 1.2020595f, 1.4367673f, 1.8862166f, 2.5578833f,
    3.004898f, 0.8021467f, 0.8048651f, 0.8232695f, 0.84456086f, 0.8655626f,
    0.88693476f, 0.90993905f, 0.93543017f, 0.96609795f, 1.014781f, 1.0925572f,
    1.220463f, 1.4361441f, 1.7891536f, 2.2886207f, 2.666609f, 0.7629701f,
    0.7661482f, 0.79038334f, 0.82066184f, 0.8506516f, 0.8795278f, 0.90815234f,
    0.9370272f, 0.9687007f, 1.018996f, 1.096957f, 1.2166147f, 1.3984237f,
    1.6650536f, 2.0256069f, 2.3175213f, 0.71835357f, 0.7209502f, 0.74942464f,
    0.7884915f, 0.8290673f, 0.8681469f, 0.9050988f, 0.9388188f, 0.970452f,
    1.0185843f, 1.0894654f, 1.1908444f, 1.3322481f, 1.523396f, 1.7712542f,
    1.9769855f, 0.6692855f, 0.67074007f, 0.7022305f, 0.7493793f, 0.8009869f,
    0.85166425f, 0.89851314f, 0.93766546f, 0.9676373f, 1.0099281f, 1.0681804f,
    1.1456892f, 1.2459581f, 1.3726168f, 1.5298429f, 1.6597029f, 0.61701006f,
    0.61716205f, 0.6509312f, 0.70539343f, 0.767787f, 0.8303777f, 0.8875314f,
    0.93173504f, 0.95790786f, 0.99109733f, 1.0331517f, 1.0848025f, 1.1466842f,
    1.2197223f, 1.3059009f, 1.3751808f, 0.5629394f, 0.56194204f, 0.59770155f,
    0.6588185f, 0.73133934f, 0.8053343f, 0.8722565f, 0.9204197f, 0.94048697f,
    0.9620043f, 0.9860381f, 1.0122565f, 1.0403054f, 1.0702337f, 1.102976f,
    1.1279685f, 0.508534f, 0.5067623f, 0.54455835f, 0.61179113f, 0.6935097f,
    0.7777994f, 0.8533006f, 0.90399134f, 0.9158415f, 0.92392015f, 0.92944396f,
    0.932081f, 0.9316125f, 0.9282956f, 0.9230843f, 0.9185473f, 0.9979178f,
    0.99982595f, 0.9999581f, 0.99998236f, 0.9999909f, 0.99999493f, 0.9999973f,
    0.99999887f, 1.0000002f, 1.0000018f, 1.0000043f, 1.0000098f, 1.0000287f,
    1.0003042f, 4.273909f, 4.360691f, 0.97091734f, 0.9975149f, 0.99942374f,
    0.99976027f, 0.99987704f, 0.99993193f, 0.99996316f, 0.99998415f, 1.000002f,
    1.0000231f, 1.000056f, 1.0001277f, 1.0003767f, 1.0039831f, 4.270904f,
    4.3601923f, 0.88172966f, 0.9599767f, 0.98950183f, 0.99568224f, 0.99780923f,
    0.99878794f, 0.999335f, 0.9996923f, 0.9999913f, 1.0003334f, 1.0008616f,
    1.0019943f, 1.0059118f, 1.0594388f, 4.2310877f, 4.3525143f, 0.88941234f,
    0.89718443f, 0.9520832f, 0.9769893f, 0.9877137f, 0.99301416f, 0.99603236f,
    0.9979944f, 0.9995961f, 1.0013663f, 1.0040215f, 1.0096014f, 1.0284113f,
    1.2372067f, 4.0961075f, 4.3208337f, 0.898366f, 0.876848f, 0.90570396f,
    0.94011307f, 0.9629715f, 0.9770607f, 0.98606294f, 0.99225456f, 0.9973897f,
    1.0029843f, 1.0111514f, 1.0277793f, 1.0806868f, 1.4891207f, 3.844518f,
    4.240992f, 0.896244f, 0.87694865f, 0.8808726f, 0.9046301f, 0.93001765f,
    0.9510043f, 0.9672598f, 0.9799122f, 0.99127704f, 1.0040897f, 1.0226759f,
    1.059127f, 1.164483f, 1.7009816f, 3.5262063f, 4.089484f, 0.88550323f,
    0.8746323f, 0.87252086f, 0.8844081f, 0.90364903f, 0.9246045f, 0.9445701f,
    0.9626739f, 0.9811842f, 1.0038756f, 1.037523f, 1.1010876f, 1.2629778f,
    1.8271517f, 3.2031236f, 3.8548436f, 0.867474f, 0.86367774f, 0.8663829f,
    0.874998f, 0.8890306f, 0.90674835f, 0.9264497f, 0.94676745f, 0.9707662f,
    1.0035489f, 1.0540025f, 1.1457827f, 1.3492591f, 1.8716009f, 2.8992045f,
    3.544642f, 0.8426331f, 0.84339976f, 0.85425377f, 0.8670567f, 0.88145f,
    0.89806235f, 0.91699255f, 0.93742573f, 0.96453035f, 1.0054018f, 1.0702212f,
    1.1831422f, 1.4022522f, 1.8493713f, 2.6119246f, 3.1815057f, 0.8112406f,
    0.8143848f, 0.83315f, 0.8540925f, 0.87415266f, 0.893992f, 0.91455126f,
    0.935241f, 0.96391606f, 1.0099056f, 1.0833111f, 1.2046738f, 1.4139625f,
    1.7756159f, 2.3339267f, 2.7927914f, 0.7736631f, 0.777564f, 0.80277777f,
    0.8331927f, 0.8623952f, 0.8895042f, 0.914955f, 0.93722665f, 0.9666525f,
    1.0144682f, 1.0891671f, 1.2054021f, 1.3866695f, 1.6643648f, 2.0621424f,
    2.4030237f, 0.730519f, 0.7340922f, 0.76415074f, 0.8039535f, 0.844062f,
    0.8812648f, 0.9143288f, 0.93937886f, 0.9686235f, 1.0147384f, 1.0836136f,
    1.183927f, 1.3274697f, 1.5282743f, 1.7985532f, 2.0313475f, 0.682721f,
    0.68534744f, 0.718956f, 0.7674769f, 0.81907046f, 0.86791134f, 0.91022575f,
    0.9384049f, 0.96590364f, 1.0066866f, 1.0639188f, 1.1416287f, 1.2445834f,
    1.3782399f, 1.5479798f, 1.6912553f, 0.63144743f, 0.63289857f, 0.6692148f,
    0.7256621f, 0.7885909f, 0.8495514f, 0.9016416f, 0.9323412f, 0.955914f,
    0.9879612f, 1.0295179f, 1.0817175f, 1.1457472f, 1.2231951f, 1.3156803f,
    1.3907586f, 0.57806194f, 0.5784216f, 0.61704224f, 0.68070203f, 0.7543744f,
    0.8271148f, 0.8885911f, 0.92048883f, 0.9376887f, 0.9581827f, 0.98180616f,
    1.0083742f, 1.0377266f, 1.0701479f, 1.1057612f, 1.1329076f, 0.5240007f,
    0.5235808f, 0.5644476f, 0.63472086f, 0.7182549f, 0.8018211f, 0.8716464f,
    0.9030739f, 0.9115884f, 0.9185296f, 0.9234439f, 0.9260583f, 0.9262192f,
    0.92432845f, 0.9205074f, 0.9168148f, 0.9979187f, 0.9998263f, 0.9999583f,
    0.9999824f, 0.999991f, 0.999995f, 0.99999726f, 0.99999875f, 1.0000001f,
    1.0000015f, 1.0000038f, 1.0000085f, 1.000023f, 1.0001549f, 4.6389117f,
    4.7828298f, 0.9709299f, 0.99751896f, 0.9994259f, 0.99976176f, 0.999878f,
    0.99993247f, 0.9999631f, 0.9999832f, 1.0000005f, 1.0000201f, 1.00005f,
    1.000111f, 1.0003022f, 1.0020319f, 4.632795f, 4.782153f, 0.88190687f,
    0.9600418f, 0.989537f, 0.99570614f, 0.9978257f, 0.9987971f, 0.99933404f,
    0.99967784f, 0.9999669f, 1.0002882f, 1.0007653f, 1.0017338f, 1.0047495f,
    1.0313218f, 4.5555267f, 4.7717733f, 0.89006364f, 0.8975033f, 0.9522647f,
    0.9771145f, 0.9878012f, 0.99306494f, 0.9960331f, 0.99792784f, 0.999481f,
    1.0011503f, 1.0035589f, 1.008351f, 1.0229685f, 1.1390909f, 4.320227f,
    4.729224f, 0.89976615f, 0.87775666f, 0.9062767f, 0.9405208f, 0.9632638f,
    0.9772394f, 0.9860868f, 0.9920711f, 0.99706393f, 1.0023633f, 1.0098165f,
    1.0242215f, 1.0662199f, 1.3352594f, 3.9540906f, 4.62337f, 0.8985971f,
    0.87882215f, 0.8822166f, 0.9056369f, 0.93076414f, 0.9514864f, 0.96737707f,
    0.9795432f, 0.99059576f, 1.00277f, 1.01985f, 1.0518562f, 1.1384428f,
    1.5511562f, 3.5715456f, 4.427026f, 0.88896537f, 0.8777802f, 0.8750873f,
    0.88645834f, 0.9052347f, 0.92568463f, 0.9449336f, 0.9620815f, 0.9800333f,
    1.0016129f, 1.0327553f, 1.089567f, 1.2285974f, 1.7146738f, 3.2301672f,
    4.132796f, 0.8721596f, 0.86830735f, 0.8705969f, 0.8786036f, 0.8919535f,
    0.908844f, 0.9273131f, 0.9459745f, 0.96911937f, 1.0002778f, 1.0473273f,
    1.1310666f, 1.3139379f, 1.7994878f, 2.925945f, 3.7586167f, 0.8486116f,
    0.8496198f, 0.8604346f, 0.8727046f, 0.8862529f, 0.9016719f, 0.9186917f,
    0.93651503f, 0.9624727f, 1.0013077f, 1.0622821f, 1.1675766f, 1.3726006f,
    1.8089836f, 2.6406395f, 3.3369708f, 0.81853044f, 0.8222169f, 0.8414692f,
    0.8621539f, 0.881327f, 0.8996147f, 0.9174453f, 0.93431425f, 0.96161073f,
    1.0053805f, 1.075146f, 1.1907188f, 1.3929384f, 1.7566652f, 2.3619287f,
    2.9001136f, 0.7822266f, 0.7869503f, 0.8132492f, 0.843861f, 0.8722904f,
    0.8975471f, 0.91934776f, 0.93633914f, 0.9642612f, 1.0099521f, 1.0817505f,
    1.1945304f, 1.3738647f, 1.6581078f, 2.0863373f, 2.4732077f, 0.74026006f,
    0.7448995f, 0.77664185f, 0.8172194f, 0.85682744f, 0.89197433f, 0.9204068f,
    0.9384834f, 0.96621317f, 1.0105139f, 1.0774347f, 1.1762598f, 1.3206214f,
    1.5281446f, 1.81689f, 2.074126f, 0.6934866f, 0.6973738f, 0.7332117f,
    0.78314054f, 0.83464366f, 0.88135004f, 0.9180388f, 0.9373282f, 0.96337765f,
    1.002728f, 1.0588189f, 1.1362352f, 1.2409155f, 1.3797312f, 1.5596951f,
    1.7146181f, 0.6430351f, 0.6458848f, 0.6848887f, 0.74337465f, 0.8067286f,
    0.8656205f, 0.91112435f, 0.93079436f, 0.9529886f, 0.9839045f, 1.0248082f,
    1.0771713f, 1.1427901f, 1.2235514f, 1.3210791f, 1.4009751f, 0.590232f,
    0.592069f, 0.6337353f, 0.70002484f, 0.7747134f, 0.84560525f, 0.8996056f,
    0.9180999f, 0.9339209f, 0.9534128f, 0.9765376f, 1.0032535f, 1.033695f,
    1.0680013f, 1.1058402f, 1.1346881f, 0.5364931f, 0.5375738f, 0.58174795f,
    0.6551882f, 0.7403811f, 0.82246864f, 0.8840225f, 0.899426f, 0.90644133f,
    0.9123197f, 0.9167847f, 0.9192653f, 0.92009795f, 0.9193218f, 0.916573f,
    0.9135375f, 0.99791956f, 0.99982655f, 0.9999584f, 0.99998254f, 0.999991f,
    0.999995f, 0.99999726f, 0.9999987f, 1.0f, 1.0000014f, 1.0000035f,
    1.0000076f, 1.0000194f, 1.0001016f, 4.9653425f, 5.2118955f, 0.97094023f,
    0.99752235f, 0.9994277f, 0.99976295f, 0.9998788f, 0.9999328f, 0.9999628f,
    0.9999823f, 0.9999991f, 1.0000178f, 1.0000452f, 1.000099f, 1.000255f,
    1.0013332f, 4.950954f, 5.210991f, 0.88205284f, 0.9600962f, 0.98956674f,
    0.9957263f, 0.99783915f, 0.9988035f, 0.9993292f, 0.99966455f, 0.99994576f,
    1.000251f, 1.0006905f, 1.001545f, 1.0040092f, 1.0207788f, 4.785313f,
    5.197167f, 0.8906006f, 0.89777017f, 0.9524184f, 0.97722024f, 0.987873f,
    0.99310136f, 0.9960129f, 0.9978657f, 0.99938077f, 1.0009723f, 1.0031979f,
    1.0074408f, 1.01946f, 1.0961657f, 4.379154f, 5.140867f, 0.90091956f,
    0.8785179f, 0.9067617f, 0.94086534f, 0.9635042f, 0.9773698f, 0.9860411f,
    0.99189675f, 0.9967767f, 1.0018479f, 1.0087677f, 1.0216048f, 1.0566242f,
    1.2501564f, 3.910389f, 5.002783f, 0.9005313f, 0.8803934f, 0.88335586f,
    0.90648836f, 0.9313798f, 0.95184374f, 0.9673158f, 0.9791818f, 0.9899851f,
    1.0016617f, 1.0176032f, 1.0464072f, 1.1203127f, 1.4482753f, 3.519974f,
    4.753089f, 0.89180344f, 0.88042194f, 0.8772668f, 0.88819534f, 0.90654665f,
    0.9264975f, 0.9449132f, 0.96147555f, 0.9789772f, 0.9996804f, 1.0288954f,
    1.0806836f, 1.2030784f, 1.6242833f, 3.2025242f, 4.39205f, 0.8759902f,
    0.8721921f, 0.8741844f, 0.8816675f, 0.8943825f, 0.9104428f, 0.9274542f,
    0.9451108f, 0.9675598f, 0.9974179f, 1.0417831f, 1.1192771f, 1.2857906f,
    1.7345648f, 2.9208708f, 3.9509072f, 0.85348743f, 0.8548347f, 0.8657112f,
    0.8775233f, 0.8902664f, 0.9044616f, 0.9191733f, 0.9354305f, 0.96044147f,
    0.9976139f, 1.0554578f, 1.1545019f, 1.3471782f, 1.7687149f, 2.649303f,
    3.4718447f, 0.82446384f, 0.8287761f, 0.8485929f, 0.8690669f, 0.88736326f,
    0.9040142f, 0.91846687f, 0.93306994f, 0.95921147f, 1.001124f, 1.0678029f,
    1.1783042f, 1.3734261f, 1.7349232f, 2.3762593f, 2.990112f, 0.7891865f,
    0.7948036f, 0.82224506f, 0.85306585f, 0.880684f, 0.9039178f, 0.92107385f,
    0.9349704f, 0.96160936f, 1.0054672f, 1.0746746f, 1.1841369f, 1.3607137f,
    1.6481369f, 2.1007023f, 2.5299006f, 0.74817044f, 0.75393826f, 0.7874124f,
    0.82874674f, 0.8677578f, 0.9005611f, 0.9229187f, 0.9369502f, 0.96335995f,
    1.0060359f, 1.0710855f, 1.1682187f, 1.3124363f, 1.5243204f, 1.8280221f,
    2.1070065f, 0.7022286f, 0.7074366f, 0.74555606f, 0.7968599f, 0.84811723f,
    0.8922564f, 0.9213153f, 0.9354892f, 0.96024746f, 0.9982614f, 1.0531526f,
    1.129967f, 1.2355814f, 1.3780081f, 1.566211f, 1.7311343f, 0.65245223f,
    0.65676653f, 0.69852966f, 0.75902426f, 0.8225964f, 0.8788185f, 0.9150551f,
    0.92840976f, 0.9493546f, 0.9791825f, 1.0193341f, 1.0716032f, 1.1382846f,
    1.2213998f, 1.322911f, 1.4067755f, 0.6001395f, 0.6035338f, 0.648349f,
    0.7172569f, 0.79271317f, 0.8609701f, 0.90402395f, 0.91485435f, 0.9294231f,
    0.9479575f, 0.9705249f, 0.997255f, 1.0285122f, 1.0641448f, 1.1036961f,
    1.1338838f, 0.5466904f, 0.5493719f, 0.5969961f, 0.67361945f, 0.76019084f,
    0.83981967f, 0.8887403f, 0.89496213f, 0.90063924f, 0.90556335f, 0.9093964f,
    0.91195035f, 0.913377f, 0.91340184f, 0.9114933f, 0.90898174f, 0.9979202f,
    0.9998268f, 0.9999585f, 0.9999826f, 0.9999911f, 0.999995f, 0.9999972f,
    0.9999986f, 0.9999999f, 1.0000012f, 1.0000031f, 1.0000069f, 1.0000169f,
    1.0000755f, 5.1755238f, 5.6471405f, 0.9709489f, 0.9975252f, 0.9994293f,
    0.999764f, 0.9998795f, 0.999933f, 0.9999622f, 0.9999815f, 0.99999785f,
    1.0000157f, 1.0000414f, 1.0000898f, 1.0002223f, 1.0009905f, 5.1294713f,
    5.645947f, 0.8821748f, 0.9601423f, 0.98959225f, 0.9957435f, 0.9978501f,
    0.99880743f, 0.99932027f, 0.9996521f, 0.999927f, 1.0002196f, 1.0006299f,
    1.0014012f, 1.0034972f, 1.0155151f, 4.7199383f, 5.627756f, 0.8910495f,
    0.8979965f, 0.9525501f, 0.97731036f, 0.9879319f, 0.99312466f, 0.9959716f,
    0.99780697f, 0.9992911f, 1.0008209f, 1.0029055f, 1.0067456f, 1.0170145f,
    1.0732857f, 4.1204486f, 5.5541644f, 0.90188307f, 0.8791646f, 0.90717775f,
    0.94115925f, 0.9637018f, 0.9774563f, 0.9859259f, 0.9917296f, 0.9965177f,
    1.0014064f, 1.0079129f, 1.0195897f, 1.049808f, 1.1988395f, 3.6857932f,
    5.3765416f, 0.90214366f, 0.8817302f, 0.8843347f, 0.9072159f, 0.9318878f,
    0.9520893f, 0.96707886f, 0.97882754f, 0.9894267f, 1.0007026f, 1.0157533f,
    1.0421501f, 1.1069968f, 1.3765999f, 3.38296f, 5.0644336f, 0.89416265f,
    0.88267195f, 0.8791433f, 0.8896831f, 0.907634f, 0.92707264f, 0.9445191f,
    0.96086246f, 0.97799265f, 0.99798346f, 1.0256684f, 1.0735875f, 1.1834407f,
    1.5528674f, 3.1324193f, 4.6302896f, 0.8791655f, 0.8755014f, 0.8772809f,
    0.8842998f, 0.89640605f, 0.9116012f, 0.9268975f, 0.94419897f, 0.96606827f,
    0.99485564f, 1.0370479f, 1.1095703f, 1.2629021f, 1.677922f, 2.891911f,
    4.1210217f, 0.8575182f, 0.85927457f, 0.87027913f, 0.8816799f, 0.89363f,
    0.9065227f, 0.9184806f, 0.93422204f, 0.95843506f, 0.9942157f, 1.0494597f,
    1.1433161f, 1.3252306f, 1.7304015f, 2.6427917f, 3.587121f, 0.82935756f,
    0.83435476f, 0.8547791f, 0.8750599f, 0.8924571f, 0.90731984f, 0.91767603f,
    0.9315927f, 0.9567477f, 0.997073f, 1.0611033f, 1.1671731f, 1.3554325f,
    1.7120402f, 2.3800304f, 3.064462f, 0.7949163f, 0.801476f, 0.83008325f,
    0.86109245f, 0.8878232f, 0.9087753f, 0.9201975f, 0.93324053f, 0.95876694f,
    1.0010175f, 1.067908f, 1.1742693f, 1.3475778f, 1.6357651f, 2.1073284f,
    2.5748532f, 0.7546744f, 0.76161265f, 0.79683036f, 0.8388661f, 0.8771368f,
    0.9071958f, 0.9219085f, 0.9349308f, 0.96017575f, 1.001383f, 1.0646682f,
    1.160047f, 1.3033736f, 1.5178072f, 1.8334088f, 2.1315064f, 0.7094119f,
    0.7159804f, 0.75639343f, 0.8089932f, 0.85979104f, 0.9007866f, 0.92004496f,
    0.9330634f, 0.95666224f, 0.9934279f, 1.0471122f, 1.1231533f, 1.2290077f,
    1.3738196f, 1.5685537f, 1.7419755f, 0.6601918f, 0.6660137f, 0.7105602f,
    0.7729764f, 0.8364871f, 0.8892591f, 0.9133282f, 0.92537814f, 0.94518733f,
    0.9739747f, 1.0133264f, 1.065348f, 1.1325649f, 1.2172714f, 1.3218749f,
    1.4089772f, 0.6082906f, 0.6132947f, 0.66130596f, 0.73275226f, 0.80863976f,
    0.8732555f, 0.9015987f, 0.91095173f, 0.924383f, 0.94200647f, 0.9639908f,
    0.99066687f, 1.0223945f, 1.0589174f, 1.0997651f, 1.1309997f, 0.5550958f,
    0.5594459f, 0.6105981f, 0.6903427f, 0.7779087f, 0.85383546f, 0.8853582f,
    0.8898835f, 0.89436877f, 0.8984219f, 0.90176076f, 0.9044235f, 0.90614414f,
    0.90673554f, 0.90549f, 0.9034041f, 0.99792075f, 0.99982697f, 0.99995863f,
    0.99998266f, 0.9999911f, 0.99999505f, 0.99999714f, 0.9999986f, 0.99999976f,
    1.0000011f, 1.0000029f, 1.0000063f, 1.0000151f, 1.0000603f, 4.6847277f,
    6.0879607f, 0.9709562f, 0.99752766f, 0.99943066f, 0.9997649f, 0.99988f,
    0.99993306f, 0.9999616f, 0.9999807f, 0.9999968f, 1.000014f, 1.000038f,
    1.0000825f, 1.0001984f, 1.0007912f, 4.351962f, 6.0864024f, 0.8822777f,
    0.9601818f, 0.9896143f, 0.99575824f, 0.997859f, 0.9988089f, 0.99931145f,
    0.9996404f, 0.99991f, 1.0001923f, 1.0005796f, 1.0012871f, 1.0031216f,
    1.0124288f, 3.6625226f, 6.0626864f, 0.8914289f, 0.8981908f, 0.9526642f,
    0.9773877f, 0.9879797f, 0.9931349f, 0.99592966f, 0.9977511f, 0.99920964f,
    1.0006889f, 1.0026615f, 1.0061922f, 1.0152106f, 1.0593994f, 3.44017f,
    5.967431f, 0.90269727f, 0.8797206f, 0.9075386f, 0.9414119f, 0.96386296f,
    0.9775004f, 0.9858065f, 0.9915686f, 0.9962803f, 1.0010196f, 1.0071958f,
    1.0179768f, 1.0447112f, 1.1653615f, 3.316065f, 5.741723f, 0.90350366f,
    0.8828817f, 0.885185f, 0.9078425f, 0.93230444f, 0.952228f, 0.9668259f,
    0.9784803f, 0.98890877f, 0.99985456f, 1.0141883f, 1.0387049f, 1.0967937f,
    1.3250475f, 3.1886992f, 5.3578124f, 0.89614725f, 0.88461274f, 0.8807771f,
    0.8909676f, 0.9085314f, 0.9274222f, 0.94408065f, 0.96024746f, 0.97706485f,
    0.9964635f, 1.0229027f, 1.067743f, 1.1678506f, 1.4962468f, 3.034343f,
    4.8456445f, 0.88182884f, 0.8783576f, 0.87998444f, 0.8865801f, 0.8980869f,
    0.91234374f, 0.9262423f, 0.9432568f, 0.9646336f, 0.9925207f, 1.0329151f,
    1.1013753f, 1.2439159f, 1.62908f, 2.8465161f, 4.269021f, 0.8608897f,
    0.8631057f, 0.8742799f, 0.88529557f, 0.8964431f, 0.9078965f, 0.9176037f,
    0.9329276f, 0.9564557f, 0.9910495f, 1.0440962f, 1.133566f, 1.3060977f,
    1.6948075f, 2.625265f, 3.6840968f, 0.83344066f, 0.83916456f, 0.8602154f,
    0.88029885f, 0.89674866f, 0.90958846f, 0.9165867f, 0.9299473f, 0.9542454f,
    0.9931928f, 1.054924f, 1.1570808f, 1.3388512f, 1.6890113f, 2.375795f,
    3.124861f, 0.7996866f, 0.80722296f, 0.836995f, 0.8681495f, 0.89388597f,
    0.91218483f, 0.9188911f, 0.93124384f, 0.9557904f, 0.9966152f, 1.0614256f,
    1.1648884f, 1.3346612f, 1.6219047f, 2.107914f, 2.6096761f, 0.7600805f,
    0.76821756f, 0.80516505f, 0.84782046f, 0.8851712f, 0.91193557f, 0.9203368f,
    0.932546f, 0.95675004f, 0.99662066f, 1.0582548f, 1.1518502f, 1.2937591f,
    1.5093615f, 1.8342385f, 2.1489537f, 0.71537685f, 0.72333175f, 0.76602155f,
    0.81980586f, 0.8698842f, 0.9069679f, 0.91809905f, 0.9301927f, 0.95274025f,
    0.98833966f, 1.04084f, 1.1159718f, 1.2215292f, 1.3677608f, 1.5675666f,
    1.748139f, 0.66661686f, 0.6739743f, 0.7212949f, 0.7855052f, 0.84861565f,
    0.896914f, 0.9108404f, 0.9218558f, 0.94062555f, 0.968423f, 1.00696f,
    1.0585943f, 1.1259158f, 1.2116079f, 1.3185534f, 1.4082668f, 0.61506104f,
    0.62170947f, 0.67292446f, 0.7467797f, 0.82268786f, 0.8823534f, 0.89836013f,
    0.9065559f, 0.9189478f, 0.9357091f, 0.9571082f, 0.9836454f, 1.0155458f,
    1.0526177f, 1.09442f, 1.1264625f, 0.5620872f, 0.5681513f, 0.62286305f,
    0.7056091f, 0.793698f, 0.864306f, 0.8811423f, 0.884358f, 0.88776976f,
    0.8910325f, 0.89391905f, 0.89647496f, 0.8985084f, 0.89948833f, 0.8987658f,
    0.89703095f, 0.9979212f, 0.9998271f, 0.9999587f, 0.9999827f, 0.9999912f,
    0.99999505f, 0.9999971f, 0.99999857f, 0.9999997f, 1.000001f, 1.0000027f,
    1.0000058f, 1.0000137f, 1.0000504f, 1.0106659f, 6.5338883f, 0.9709624f,
    0.99752975f, 0.9994318f, 0.9997657f, 0.99988043f, 0.99993294f, 0.9999611f,
    0.99998f, 0.99999577f, 1.0000124f, 1.0000354f, 1.0000765f, 1.00018f,
    1.0006623f, 1.1289786f, 6.531869f, 0.88236564f, 0.96021605f, 0.98963356f,
    0.9957709f, 0.99786603f, 0.9988078f, 0.9993027f, 0.9996292f, 0.99989444f,
    1.0001681f, 1.0005369f, 1.0011934f, 1.0028336f, 1.0104232f, 1.956228f,
    6.5011625f, 0.89175314f, 0.8983594f, 0.95276403f, 0.9774544f, 0.9880177f,
    0.9931314f, 0.99588794f, 0.9976977f, 0.9991346f, 1.0005718f, 1.0024536f,
    1.0057378f, 1.0138215f, 1.0501852f, 2.618211f, 6.37881f, 0.90339303f,
    0.8802038f, 0.9078545f, 0.94163024f, 0.9639923f, 0.97749954f, 0.9856862f,
    0.9914131f, 0.99606013f, 1.0006744f, 1.0065818f, 1.0166456f, 1.0407459f,
    1.1420958f, 2.8993225f, 6.095083f, 0.90466356f, 0.8838843f, 0.8859307f,
    0.90838516f, 0.9326416f, 0.9522548f, 0.96656597f, 0.97814035f, 0.9884236f,
    0.99909186f, 1.0128372f, 1.0358353f, 1.0887052f, 1.2866825f, 2.9710736f,
    5.6300173f, 0.89783525f, 0.8863055f, 0.8822133f, 0.89208317f, 0.90926373f,
    0.92753786f, 0.94361866f, 0.95963466f, 0.97618425f, 0.9950813f, 1.0204879f,
    1.0628034f, 1.1551421f, 1.4508069f, 2.9217284f, 5.036741f, 0.88408726f,
    0.88085127f, 0.882368f, 0.8885675f, 0.89947f, 0.9126574f, 0.9255298f,
    0.94229716f, 0.96324885f, 0.9903662f, 1.0292493f, 1.0943043f, 1.2278754f,
    1.5870259f, 2.7909214f, 4.3954434f, 0.86374027f, 0.8664508f, 0.87781876f,
    0.88846034f, 0.8987771f, 0.9085624f, 0.9166143f, 0.9315754f, 0.9545069f,
    0.9880728f, 1.0392396f, 1.1249217f, 1.2892448f, 1.6621208f, 2.6000724f,
    3.764254f, 0.8368834f, 0.84336185f, 0.8650409f, 0.8849072f, 0.90033895f,
    0.91078585f, 0.91530955f, 0.9281824f, 0.95172596f, 0.98946166f, 1.0491815f,
    1.1478261f, 1.3235401f, 1.6664109f, 2.3655887f, 3.1729503f, 0.8036995f,
    0.812234f, 0.8431521f, 0.8743925f, 0.89900136f, 0.91408646f, 0.91730994f,
    0.9290519f, 0.95272505f, 0.9922724f, 1.055213f, 1.1559358f, 1.3220751f,
    1.6071827f, 2.103817f, 2.6358147f, 0.76461965f, 0.7739723f, 0.8126173f,
    0.85579205f, 0.8920101f, 0.9146776f, 0.91840553f, 0.9298895f, 0.95315325f,
    0.99180055f, 1.0519028f, 1.1436794f, 1.283824f, 1.4995477f, 1.8314683f,
    2.16049f, 0.7203787f, 0.72973466f, 0.7746639f, 0.829498f, 0.8785556f,
    0.9106308f, 0.9157249f, 0.9269881f, 0.94857466f, 0.9830837f, 1.0344479f,
    1.1085327f, 1.2134054f, 1.3602979f, 1.5639348f, 1.7504613f, 0.67200124f,
    0.68090993f, 0.73097146f, 0.7968184f, 0.85913616f, 0.9015205f, 0.90788233f,
    0.91796565f, 0.9357784f, 0.96263725f, 1.0003755f, 1.0514655f, 1.1185706f,
    1.2047684f, 1.3134247f, 1.4052131f, 0.6207356f, 0.629049f, 0.683447f,
    0.75954396f, 0.8349942f, 0.8878795f, 0.89464116f, 0.9017956f, 0.91323364f,
    0.9291816f, 0.950018f, 0.9762924f, 1.0081441f, 1.0454968f, 1.0879713f,
    1.1206251f, 0.5679525f, 0.57575977f, 0.63402927f, 0.7196118f, 0.8076681f,
    0.8706994f, 0.87646663f, 0.8785153f, 0.8809581f, 0.8834964f, 0.8859859f,
    0.8884664f, 0.89057636f, 0.8918084f, 0.8914955f, 0.89005363f, 0.99792165f,
    0.9998272f, 0.99995875f, 0.9999828f, 0.9999912f, 0.999995f, 0.9999971f,
    0.9999985f, 0.99999964f, 1.0000008f, 1.0000025f, 1.0000055f, 1.0000126f,
    1.0000436f, 1.002263f, 6.984593f, 0.9709677f, 0.99753165f, 0.99943286f,
    0.99976635f, 0.9998808f, 0.9999327f, 0.9999605f, 0.9999793f, 0.9999949f,
    1.0000111f, 1.000033f, 1.0000715f, 1.0001656f, 1.0005727f, 1.0292823f,
    6.981989f, 0.8824413f, 0.96024597f, 0.9896506f, 0.99578184f, 0.9978714f,
    0.99880296f, 0.99929404f, 0.9996186f, 0.99987996f, 1.0001464f, 1.0004997f,
    1.0011148f, 1.002605f, 1.0090233f, 1.3717408f, 6.942411f, 0.89203274f,
    0.898507f, 0.95285195f, 0.9775122f, 0.9880469f, 0.9931095f, 0.99584657f,
    0.99764645f, 0.9990648f, 1.0004665f, 1.0022725f, 1.0053551f, 1.012715f,
    1.0436646f, 2.032357f, 6.786163f, 0.90399295f, 0.88062763f, 0.9081332f,
    0.9418194f, 0.96409285f, 0.9774402f, 0.98556584f, 0.9912626f, 0.995854f,
    1.0003622f, 1.0060451f, 1.01552f, 1.0375617f, 1.1250999f, 2.525734f,
    6.4329844f, 0.9056623f, 0.88476557f, 0.8865899f, 0.9088566f, 0.9329075f,
    0.9521374f, 0.96630245f, 0.9778079f, 0.9879658f, 0.9983972f, 1.0116491f,
    1.0333899f, 1.082113f, 1.2572211f, 2.7574487f, 5.8779488f, 0.89928484f,
    0.88779634f, 0.88348603f, 0.8930557f, 0.9098485f, 0.92735565f, 0.94314176f,
    0.9590271f, 0.9753443f, 0.99381036f, 1.0183438f, 1.0585415f, 1.1445479f,
    1.4137809f, 2.8049755f, 5.202731f, 0.8860207f, 0.8830504f, 0.8844866f,
    0.8903067f, 0.90058726f, 0.91242945f, 0.92477816f, 0.94132984f, 0.96191f,
    0.98835987f, 1.0259507f, 1.0880945f, 1.2141007f, 1.5506771f, 2.7299142f,
    4.501222f, 0.8661733f, 0.86940217f, 0.88097507f, 0.8912422f, 0.90068275f,
    0.90833807f, 0.9155451f, 0.9301866f, 0.9525921f, 0.9852569f, 1.0347916f,
    1.1171505f, 1.27425f, 1.6322455f, 2.5698016f, 3.8291636f, 0.83981353f,
    0.8470643f, 0.86936045f, 0.8889787f, 0.90329975f, 0.9106349f, 0.9138958f,
    0.9263353f, 0.9492067f, 0.9858657f, 1.0438058f, 1.1392622f, 1.309355f,
    1.6445518f, 2.3510041f, 3.2102678f, 0.8071063f, 0.81665164f, 0.8486846f,
    0.8799396f, 0.903261f, 0.91407883f, 0.9155258f, 0.9267199f, 0.9496066f,
    0.9880004f, 1.0492477f, 1.1473669f, 1.3098762f, 1.5920274f, 2.0961144f,
    2.6545436f, 0.7684654f, 0.7790425f, 0.8193393f, 0.86291933f, 0.8977596f,
    0.91486144f, 0.91620564f, 0.9270338f, 0.9494411f, 0.9869637f, 1.0456405f,
    1.135575f, 1.2737322f, 1.4887865f, 1.8258672f, 2.1670887f, 0.7246101f,
    0.73537403f, 0.7824905f, 0.8382224f, 0.88591605f, 0.9110169f, 0.913029f,
    0.9235369f, 0.9442393f, 0.9777281f, 1.0280056f, 1.1009269f, 1.2048359f,
    1.3517951f, 1.5582135f, 1.7496395f, 0.6765523f, 0.6870199f, 0.73977154f,
    0.80707496f, 0.8681536f, 0.90207946f, 0.90457165f, 0.91380465f, 0.93073225f,
    0.9567034f, 0.99366426f, 1.0440681f, 1.1107179f, 1.1970413f, 1.3068776f,
    1.4002837f, 0.6255312f, 0.6355209f, 0.6930601f, 0.7712028f, 0.8456465f,
    0.88855004f, 0.89056563f, 0.8967725f, 0.90733165f, 0.92251503f, 0.9428122f,
    0.9687009f, 1.0003388f, 1.0377614f, 1.0806737f, 1.1137769f, 0.5729125f,
    0.5824805f, 0.6442814f, 0.7325002f, 0.8198805f, 0.87140226f, 0.87145877f,
    0.8724528f, 0.87402207f, 0.8758993f, 0.8780385f, 0.8803561f, 0.8825407f,
    0.88382196f, 0.88382554f, 0.88263f};

static const auto glossy_albedo_lut = vector<float>{1.514151e-32f,
    3.8824543e-33f, 3.1378603e-34f, 4.591827e-34f, 6.1190753e-34f,
    9.264686e-34f, 5.3258475e-35f, 3.3897839e-34f, 6.785131e-36f,
    3.3925055e-36f, 1.4232474e-37f, 1.4280332e-37f, 1.3051271e-38f,
    2.846275e-39f, 4.85118e-40f, 0.0f, 1.976315e-31f, 4.9790726e-32f,
    4.0019764e-33f, 5.487033e-33f, 6.9920696e-33f, 9.378302e-33f,
    5.9367203e-34f, 2.2556403e-33f, 8.6011455e-35f, 4.390883e-35f, 1.86739e-36f,
    1.8715479e-36f, 1.7149798e-37f, 3.7300093e-38f, 6.345044e-39f, 0.0f,
    2.6410407e-30f, 6.118049e-31f, 4.605231e-32f, 3.2672172e-32f,
    2.7216012e-32f, 1.869936e-32f, 2.3658073e-33f, 2.0024886e-33f,
    9.0011175e-34f, 5.480293e-34f, 2.8673065e-35f, 2.802577e-35f,
    2.7080539e-36f, 5.832896e-37f, 9.949024e-38f, 0.0f, 6.627023e-30f,
    1.412244e-30f, 9.853361e-32f, 4.1885823e-32f, 1.9968912e-32f, 1.056777e-32f,
    2.8045486e-33f, 2.149074e-33f, 1.864728e-33f, 1.3199186e-33f,
    1.2406848e-34f, 1.1026045e-34f, 1.3118329e-35f, 2.8142785e-36f,
    4.8549826e-37f, 0.0f, 6.9911445e-30f, 1.1758194e-30f, 9.370827e-32f,
    4.0553136e-32f, 1.7534441e-32f, 8.425902e-33f, 3.127069e-33f,
    2.3275831e-33f, 2.0230306e-33f, 1.4995172e-33f, 2.8027991e-34f,
    2.1243954e-34f, 3.7399808e-35f, 8.213841e-36f, 1.4342951e-36f, 0.0f,
    4.6718453e-30f, 6.1804376e-31f, 6.633387e-32f, 3.208313e-32f,
    1.5486167e-32f, 7.490452e-33f, 3.2799e-33f, 2.3217385e-33f, 1.906071e-33f,
    1.3671898e-33f, 4.1980568e-34f, 2.8003942e-34f, 7.6119406e-35f,
    1.7709082e-35f, 3.1543986e-36f, 0.0f, 2.5359107e-30f, 2.9481925e-31f,
    4.4277164e-32f, 2.2894507e-32f, 1.222714e-32f, 6.350054e-33f,
    3.1888862e-33f, 2.1839587e-33f, 1.7723863e-33f, 1.2012101e-33f,
    4.959255e-34f, 3.1935736e-34f, 1.2174655e-34f, 3.063128e-35f,
    5.6551194e-36f, 0.0f, 1.2787799e-30f, 1.4867822e-31f, 2.92791e-32f,
    1.597798e-32f, 8.9914415e-33f, 5.0750393e-33f, 2.8732862e-33f,
    1.9342589e-33f, 1.611628e-33f, 1.0464224e-33f, 5.1475733e-34f,
    3.4067769e-34f, 1.6353708e-34f, 4.4522672e-35f, 8.658575e-36f, 0.0f,
    6.424581e-31f, 8.189878e-32f, 1.9462174e-32f, 1.0282928e-32f, 6.471513e-33f,
    3.926353e-33f, 2.4528297e-33f, 1.6408849e-33f, 1.4206653e-33f,
    9.0065835e-34f, 4.9579216e-34f, 3.432618e-34f, 1.9328989e-34f,
    5.6378747e-35f, 1.1657871e-35f, 0.0f, 3.3440259e-31f, 4.8669854e-32f,
    1.3129495e-32f, 7.92158e-33f, 4.682525e-33f, 3.0115706e-33f, 2.0005661e-33f,
    1.3588554e-33f, 1.2198973e-33f, 7.646626e-34f, 4.5535033e-34f,
    3.2868978e-34f, 2.077549e-34f, 6.4245634e-35f, 1.4133847e-35f, 0.0f,
    1.8389689e-31f, 3.06432e-32f, 9.129576e-33f, 5.6609402e-33f, 3.4434e-33f,
    2.321849e-33f, 1.6033258e-33f, 1.1148132e-33f, 1.0295777e-33f,
    6.423606e-34f, 4.055128e-34f, 3.0259706e-34f, 2.0845592e-34f, 6.778911e-35f,
    1.5779112e-35f, 0.0f, 1.0746664e-31f, 2.019636e-32f, 6.506946e-33f,
    4.1583564e-33f, 2.5836866e-33f, 1.8118892e-33f, 1.2801262e-33f,
    9.147566e-34f, 8.614561e-34f, 5.367769e-34f, 3.5465936e-34f, 2.7132641e-34f,
    1.995828e-34f, 6.781982e-35f, 1.6567395e-35f, 0.0f, 6.657692e-32f,
    1.3849185e-32f, 4.7676012e-33f, 3.0403234e-33f, 1.9803844e-33f,
    1.4359702e-33f, 1.0265134e-33f, 7.5475687e-34f, 7.19726e-34f,
    4.4871384e-34f, 3.0776166e-34f, 2.3969037e-34f, 1.8559563e-34f,
    6.554515e-35f, 1.6667203e-35f, 0.0f, 4.346997e-32f, 9.850533e-33f,
    3.587811e-33f, 2.3567192e-33f, 1.5505503e-33f, 1.1573724e-33f,
    8.304576e-34f, 6.2795707e-34f, 6.0383616e-34f, 3.7711188e-34f,
    2.6704936e-34f, 2.1050973e-34f, 1.6989372e-34f, 6.2058433e-35f,
    1.6311108e-35f, 0.0f, 2.9725228e-32f, 7.251233e-33f, 2.7565594e-33f,
    1.6213122e-33f, 1.2391678e-33f, 9.488285e-34f, 6.795017e-34f, 5.276672e-34f,
    5.4955403e-34f, 3.1985853e-34f, 2.3295e-34f, 1.8502716e-34f, 1.5456097e-34f,
    5.814293e-35f, 1.5706169e-35f, 0.0f, 2.1169542e-32f, 5.511056e-33f,
    2.1331108e-33f, 1.41825965e-33f, 1.009722e-33f, 7.907591e-34f,
    5.682377e-34f, 4.4825966e-34f, 4.368938e-34f, 2.7449288e-34f, 2.049427e-34f,
    1.6349357e-34f, 1.4063854e-34f, 5.427424e-35f, 1.5000582e-35f, 0.0f,
    0.8931297f, 0.6961296f, 0.49801707f, 0.3646168f, 0.27285516f, 0.20882602f,
    0.16376905f, 0.13196154f, 0.10955256f, 0.09388782f, 0.08310225f,
    0.07586512f, 0.07121549f, 0.06845292f, 0.067063175f, 0.06666678f,
    0.8703438f, 0.69324934f, 0.4971507f, 0.3643021f, 0.27274188f, 0.20879441f,
    0.16377103f, 0.13197613f, 0.10957028f, 0.09390447f, 0.08311614f,
    0.07587577f, 0.07122298f, 0.0684576f, 0.06706548f, 0.06666712f, 0.75975543f,
    0.6605459f, 0.486962f, 0.3603748f, 0.27119678f, 0.20826198f, 0.16368131f,
    0.13207103f, 0.109728254f, 0.09406822f, 0.08325999f, 0.07598994f,
    0.07130565f, 0.068510704f, 0.06709262f, 0.066672385f, 0.62532073f,
    0.5746478f, 0.45381767f, 0.34649155f, 0.26521575f, 0.20583072f, 0.1629031f,
    0.13205066f, 0.11002456f, 0.094460845f, 0.0836411f, 0.07631047f,
    0.071547486f, 0.06867178f, 0.067178756f, 0.066692844f, 0.4932404f,
    0.46830496f, 0.39597803f, 0.31769606f, 0.2512302f, 0.19931436f, 0.16017649f,
    0.13122611f, 0.110110156f, 0.09492936f, 0.08421415f, 0.07684604f,
    0.07197903f, 0.06897446f, 0.06735024f, 0.06674237f, 0.38917494f,
    0.37364212f, 0.33020684f, 0.27788657f, 0.22877339f, 0.18732287f,
    0.15422082f, 0.12864576f, 0.109343864f, 0.0950707f, 0.084743224f,
    0.07747136f, 0.07254425f, 0.0694029f, 0.06761173f, 0.06683313f, 0.31149372f,
    0.29947045f, 0.2711983f, 0.23651543f, 0.20203227f, 0.17110123f, 0.14499591f,
    0.12383732f, 0.107196406f, 0.09443823f, 0.08489894f, 0.077966616f,
    0.07311128f, 0.06988989f, 0.06794001f, 0.06696943f, 0.25432065f,
    0.24410011f, 0.22401407f, 0.20018722f, 0.17605567f, 0.15363325f,
    0.13392755f, 0.117289945f, 0.10367981f, 0.09284753f, 0.08445159f,
    0.07812816f, 0.07352915f, 0.070340574f, 0.06828941f, 0.06714391f,
    0.21236557f, 0.2034352f, 0.18819343f, 0.17102359f, 0.15373583f, 0.13745083f,
    0.122797996f, 0.1100668f, 0.09932133f, 0.09048504f, 0.08340131f,
    0.07787496f, 0.07369932f, 0.07067288f, 0.068608716f, 0.06733915f,
    0.18150233f, 0.17363678f, 0.16154853f, 0.1486162f, 0.13581972f, 0.12376716f,
    0.112811774f, 0.103130914f, 0.094781846f, 0.08774193f, 0.08193783f,
    0.07726664f, 0.07361082f, 0.07084855f, 0.06886056f, 0.067534454f,
    0.15866442f, 0.15169813f, 0.14181758f, 0.13171446f, 0.12192721f,
    0.112778485f, 0.10445076f, 0.097033024f, 0.09055146f, 0.08499033f,
    0.08030656f, 0.07644011f, 0.073321395f, 0.0708766f, 0.06903143f,
    0.06771369f, 0.14162706f, 0.13540992f, 0.12715171f, 0.11902289f,
    0.111314006f, 0.10418641f, 0.09772189f, 0.091951914f, 0.08687567f,
    0.0824717f, 0.0787056f, 0.07553524f, 0.0729143f, 0.07079479f, 0.06912873f,
    0.06786937f, 0.1287962f, 0.12319081f, 0.11616274f, 0.109462835f,
    0.103234686f, 0.09754523f, 0.09241711f, 0.0878468f, 0.083815336f,
    0.080294915f, 0.07725293f, 0.07465459f, 0.07246453f, 0.07064791f,
    0.069171004f, 0.06800167f, 0.11903569f, 0.113919914f, 0.107844315f,
    0.102209195f, 0.09706619f, 0.09242471f, 0.088272005f, 0.08458377f,
    0.08133014f, 0.07847911f, 0.075998485f, 0.07385698f, 0.07202483f,
    0.0704741f, 0.06917878f, 0.0681148f, 0.11153543f, 0.10680361f, 0.101476535f,
    0.096653566f, 0.092325635f, 0.08846553f, 0.085038714f, 0.082008615f,
    0.07933925f, 0.07699654f, 0.074948914f, 0.07316744f, 0.07162582f,
    0.070300266f, 0.06916932f, 0.068213634f, 0.1057154f, 0.101277694f,
    0.09654552f, 0.0923539f, 0.08865143f, 0.08538642f, 0.08251046f,
    0.079979666f, 0.07775488f, 0.075801425f, 0.07408877f, 0.07259003f,
    0.07128158f, 0.07014261f, 0.0691548f, 0.068301976f, 0.8899187f, 0.69469255f,
    0.5087427f, 0.38940197f, 0.30988148f, 0.2554149f, 0.21738341f, 0.19050887f,
    0.17142355f, 0.15790129f, 0.14842486f, 0.14193252f, 0.13766424f,
    0.13506547f, 0.13372493f, 0.13333353f, 0.8673354f, 0.6920336f, 0.50799966f,
    0.38914582f, 0.30979237f, 0.25539055f, 0.21738508f, 0.19052035f,
    0.17143768f, 0.15791489f, 0.14843655f, 0.14194179f, 0.13767101f,
    0.13506988f, 0.1337272f, 0.13333392f, 0.7597992f, 0.661708f, 0.4991567f,
    0.3858911f, 0.3085464f, 0.25496465f, 0.21731031f, 0.19059266f, 0.17156294f,
    0.1580485f, 0.14855759f, 0.14204115f, 0.13774565f, 0.1351198f, 0.133754f,
    0.13333993f, 0.6334943f, 0.58244157f, 0.46998802f, 0.3741332f, 0.30359468f,
    0.2529614f, 0.21665324f, 0.19055307f, 0.17178932f, 0.15836538f, 0.14887664f,
    0.14231907f, 0.13796306f, 0.1352703f, 0.13383819f, 0.13336205f, 0.51148045f,
    0.48588434f, 0.41901413f, 0.34942037f, 0.29179665f, 0.24748942f,
    0.21433471f, 0.18981275f, 0.17181668f, 0.1587319f, 0.14935263f, 0.14278209f,
    0.13835023f, 0.13555236f, 0.13400505f, 0.13341454f, 0.41700438f,
    0.40125772f, 0.36143667f, 0.31516856f, 0.27268583f, 0.23731202f,
    0.20924088f, 0.18755312f, 0.17109512f, 0.15880089f, 0.14978004f,
    0.14331912f, 0.13885613f, 0.13595092f, 0.13425891f, 0.13350971f, 0.3475354f,
    0.33575127f, 0.31014132f, 0.27964166f, 0.24988005f, 0.22348732f, 0.2013292f,
    0.18336695f, 0.16917005f, 0.15818667f, 0.14987092f, 0.14373426f,
    0.13936017f, 0.13640228f, 0.13457641f, 0.13365106f, 0.29700422f, 0.2872903f,
    0.26933524f, 0.24850473f, 0.22771811f, 0.20857981f, 0.19183016f,
    0.17768334f, 0.16605964f, 0.15673427f, 0.14942597f, 0.14384614f, 0.1397237f,
    0.13681589f, 0.13491125f, 0.1338288f, 0.26024726f, 0.2519269f, 0.23845828f,
    0.22352779f, 0.2086609f, 0.19475186f, 0.1822733f, 0.1714235f, 0.16222897f,
    0.1546148f, 0.14845154f, 0.1435855f, 0.1398575f, 0.13711366f, 0.13521133f,
    0.13402168f, 0.23338048f, 0.22612265f, 0.21553147f, 0.2043317f, 0.19333939f,
    0.18303576f, 0.17368582f, 0.16541356f, 0.15825143f, 0.15217397f,
    0.14712046f, 0.14301054f, 0.13975465f, 0.13726093f, 0.13543963f, 0.1342058f,
    0.2135922f, 0.20717223f, 0.19856392f, 0.18983677f, 0.18143252f, 0.17360184f,
    0.16647851f, 0.16012253f, 0.15454686f, 0.14973456f, 0.14565f, 0.14224647f,
    0.13947128f, 0.13726926f, 0.13558511f, 0.134365f, 0.19888146f, 0.19311939f,
    0.18594827f, 0.17893483f, 0.17231256f, 0.16620237f, 0.16066001f,
    0.15570255f, 0.15132369f, 0.14750314f, 0.14421229f, 0.14141804f,
    0.13908502f, 0.13717726f, 0.1356591f, 0.13449588f, 0.1878348f, 0.18257956f,
    0.17648657f, 0.17070661f, 0.16535072f, 0.16046424f, 0.15605718f,
    0.15212019f, 0.14863355f, 0.14557225f, 0.14290892f, 0.14061555f,
    0.13866456f, 0.13702922f, 0.13568409f, 0.1346051f, 0.17945512f, 0.17457926f,
    0.16931467f, 0.16445027f, 0.1600209f, 0.15602614f, 0.15244871f, 0.14926365f,
    0.14644308f, 0.14395876f, 0.14178339f, 0.1398912f, 0.13825826f, 0.13686247f,
    0.13568361f, 0.1347031f, 0.17303681f, 0.16843331f, 0.16381647f, 0.15964882f,
    0.15591504f, 0.15258563f, 0.14962664f, 0.14700395f, 0.14468518f,
    0.14264055f, 0.14084305f, 0.13926846f, 0.13789508f, 0.13670355f,
    0.13567661f, 0.13479881f, 0.1680774f, 0.16365665f, 0.15955321f, 0.15592685f,
    0.15272725f, 0.14990552f, 0.1474171f, 0.14522246f, 0.14328694f, 0.14158043f,
    0.1400767f, 0.13875295f, 0.13758941f, 0.13656873f, 0.1356758f, 0.13489735f,
    0.8810619f, 0.68048847f, 0.5033689f, 0.39806545f, 0.332149f, 0.28922203f,
    0.26041567f, 0.24066295f, 0.22693285f, 0.21733867f, 0.21066388f, 0.2060989f,
    0.20308909f, 0.20124438f, 0.20028377f, 0.20000029f, 0.85829f, 0.6780132f,
    0.50276136f, 0.3978821f, 0.33209527f, 0.28921276f, 0.2604218f, 0.24067375f,
    0.22694415f, 0.21734881f, 0.21067235f, 0.20610555f, 0.20309395f, 0.2012476f,
    0.2002855f, 0.20000066f, 0.7509292f, 0.6496863f, 0.49538103f, 0.39545134f,
    0.33127198f, 0.28898147f, 0.26041788f, 0.24075332f, 0.22704731f,
    0.21745004f, 0.21076092f, 0.2061774f, 0.20314813f, 0.2012845f, 0.20030612f,
    0.20000635f, 0.6299005f, 0.5764386f, 0.4705398f, 0.38626865f, 0.3277272f,
    0.28768307f, 0.26006395f, 0.24079795f, 0.22724757f, 0.21769372f,
    0.21099545f, 0.20637858f, 0.20330565f, 0.20139508f, 0.2003701f, 0.20002587f,
    0.5170175f, 0.48961622f, 0.4271929f, 0.36651215f, 0.31886125f, 0.2838165f,
    0.2585369f, 0.2403741f, 0.22733091f, 0.21798782f, 0.21134967f, 0.20671585f,
    0.20358723f, 0.20160271f, 0.20049675f, 0.20007072f, 0.43242106f,
    0.41576257f, 0.37903297f, 0.33911526f, 0.30422986f, 0.27633995f,
    0.25493923f, 0.23884548f, 0.22688249f, 0.21807669f, 0.21167503f,
    0.20710999f, 0.20395692f, 0.20189714f, 0.20068975f, 0.20015067f,
    0.37206188f, 0.360129f, 0.33696824f, 0.31099144f, 0.28676334f, 0.26606166f,
    0.24920751f, 0.23588043f, 0.22554882f, 0.21766806f, 0.21175677f,
    0.20741606f, 0.20432538f, 0.20223053f, 0.20093079f, 0.2002674f, 0.32930875f,
    0.31993982f, 0.30411693f, 0.28665176f, 0.2699002f, 0.25498086f, 0.2422819f,
    0.23179826f, 0.22333865f, 0.21664317f, 0.21144482f, 0.20749772f,
    0.20458746f, 0.20253257f, 0.20118178f, 0.20040995f, 0.29891846f,
    0.29120567f, 0.27965766f, 0.26735577f, 0.25551066f, 0.2447439f, 0.23531929f,
    0.22729163f, 0.22060117f, 0.21513186f, 0.21074614f, 0.20730457f,
    0.20467603f, 0.20274225f, 0.20139913f, 0.2005564f, 0.2771438f, 0.27059814f,
    0.26174545f, 0.25267667f, 0.24402253f, 0.23610805f, 0.22907765f,
    0.22296919f, 0.21775918f, 0.2133909f, 0.20979166f, 0.20688333f, 0.20458853f,
    0.20283411f, 0.20155276f, 0.20068368f, 0.26138225f, 0.2556818f, 0.24864256f,
    0.24168794f, 0.23514718f, 0.22917956f, 0.22384953f, 0.21916771f,
    0.21511433f, 0.21165325f, 0.20874031f, 0.20632833f, 0.2043702f, 0.20282054f,
    0.20163672f, 0.20077924f, 0.24984461f, 0.24475284f, 0.2389943f, 0.23348244f,
    0.22838196f, 0.22376022f, 0.21963371f, 0.2159924f, 0.21281254f, 0.21006398f,
    0.20771407f, 0.20573007f, 0.20408025f, 0.20273466f, 0.20166528f,
    0.20084618f, 0.24130295f, 0.23663723f, 0.23181587f, 0.22732614f,
    0.22323799f, 0.21956645f, 0.2163005f, 0.2134172f, 0.21088895f, 0.20868707f,
    0.20678368f, 0.20515265f, 0.20376976f, 0.20261292f, 0.20166205f, 0.2008989f,
    0.23491192f, 0.23052786f, 0.22641087f, 0.22266856f, 0.21931313f,
    0.21632877f, 0.21368861f, 0.21136254f, 0.20932072f, 0.20753516f,
    0.20598046f, 0.20463379f, 0.20347485f, 0.20248565f, 0.20165023f, 0.2009545f,
    0.23008554f, 0.22586739f, 0.22229093f, 0.21911003f, 0.21629946f,
    0.21382435f, 0.21164864f, 0.2097384f, 0.20806305f, 0.2065955f, 0.20531215f,
    0.20419243f, 0.20321846f, 0.20237471f, 0.20164768f, 0.20102558f,
    0.22641371f, 0.2222678f, 0.21911336f, 0.21636394f, 0.21396863f, 0.21188015f,
    0.2100568f, 0.20846282f, 0.20706773f, 0.20584562f, 0.20477456f, 0.20383583f,
    0.20301354f, 0.20229399f, 0.20166552f, 0.20111796f, 0.8690424f, 0.6608928f,
    0.49322188f, 0.40302223f, 0.3514702f, 0.32059497f, 0.301403f, 0.28911516f,
    0.28106782f, 0.27571732f, 0.2721389f, 0.2697621f, 0.26822537f, 0.2672937f,
    0.2668101f, 0.26666704f, 0.84595895f, 0.65861726f, 0.49275932f, 0.4029144f,
    0.35145274f, 0.32060128f, 0.30141416f, 0.28912574f, 0.28107652f,
    0.27572414f, 0.27214408f, 0.26976594f, 0.2682281f, 0.2672955f, 0.26681113f,
    0.2666674f, 0.73798865f, 0.6324294f, 0.48694023f, 0.40134203f, 0.35106537f,
    0.32057285f, 0.30148795f, 0.2892181f, 0.28116116f, 0.27579433f, 0.27219963f,
    0.2698084f, 0.26825932f, 0.26731694f, 0.26682395f, 0.26667246f, 0.621886f,
    0.5655923f, 0.46671462f, 0.3948571f, 0.3489822f, 0.32001215f, 0.301462f,
    0.28936574f, 0.2813475f, 0.27597028f, 0.27234912f, 0.269928f, 0.2683499f,
    0.26738054f, 0.26686266f, 0.2666881f, 0.51848716f, 0.48912597f, 0.4314812f,
    0.38031393f, 0.34318182f, 0.31783247f, 0.3007843f, 0.28930053f, 0.28151456f,
    0.2762059f, 0.27258357f, 0.2701326f, 0.26851395f, 0.26750094f, 0.26693937f,
    0.2667222f, 0.44432932f, 0.42671627f, 0.39330438f, 0.36013758f, 0.3332541f,
    0.31320363f, 0.29878685f, 0.2885778f, 0.28138956f, 0.27634275f, 0.2728159f,
    0.27037838f, 0.26873296f, 0.26767388f, 0.26705727f, 0.26678115f, 0.3936226f,
    0.38155177f, 0.36100304f, 0.33981624f, 0.32141006f, 0.3066754f, 0.2953855f,
    0.2869445f, 0.28072426f, 0.27618667f, 0.27290964f, 0.27057615f, 0.26895303f,
    0.26787037f, 0.2672044f, 0.2668646f, 0.3591102f, 0.35012305f, 0.33656043f,
    0.32264924f, 0.31014195f, 0.29965296f, 0.29121408f, 0.2846021f, 0.27951485f,
    0.27565548f, 0.2727666f, 0.2706386f, 0.2691068f, 0.26804408f, 0.2673534f,
    0.2669607f, 0.33546382f, 0.32840797f, 0.31888968f, 0.30936062f, 0.30069667f,
    0.29323724f, 0.28703436f, 0.28199914f, 0.2779849f, 0.27483243f, 0.27239254f,
    0.27053514f, 0.26915172f, 0.26815403f, 0.26747146f, 0.26704797f,
    0.31908467f, 0.31330684f, 0.3062908f, 0.29947144f, 0.29328382f, 0.2878917f,
    0.28331718f, 0.27951157f, 0.27639472f, 0.2738771f, 0.2718715f, 0.2702986f,
    0.26908937f, 0.2681852f, 0.26753727f, 0.2671053f, 0.30759394f, 0.30267167f,
    0.29729265f, 0.29221314f, 0.28764403f, 0.28365052f, 0.2802269f, 0.27733395f,
    0.27491874f, 0.2729249f, 0.27129808f, 0.26998848f, 0.26895195f, 0.2681502f,
    0.2675504f, 0.2671247f, 0.29942667f, 0.29506516f, 0.29080522f, 0.2868867f,
    0.2834022f, 0.28036407f, 0.2777491f, 0.27551994f, 0.27363515f, 0.27205417f,
    0.27073932f, 0.269657f, 0.2687775f, 0.26807517f, 0.26752788f, 0.26711658f,
    0.29355058f, 0.28953466f, 0.28606668f, 0.28295085f, 0.28021422f,
    0.27784073f, 0.27579823f, 0.27405012f, 0.27256086f, 0.2712979f, 0.27023235f,
    0.2693392f, 0.2685969f, 0.26798695f, 0.2674936f, 0.2671034f, 0.28927976f,
    0.28544745f, 0.2825559f, 0.28001294f, 0.27780703f, 0.2759067f, 0.27427563f,
    0.27287865f, 0.27168402f, 0.27066407f, 0.26979506f, 0.2690569f, 0.26843244f,
    0.2679074f, 0.26746964f, 0.26710892f, 0.28615376f, 0.28238025f, 0.2799185f,
    0.27779606f, 0.27597752f, 0.27442265f, 0.2730935f, 0.27195668f, 0.27098355f,
    0.27014995f, 0.26943567f, 0.26882374f, 0.2683001f, 0.2678528f, 0.26747203f,
    0.26714936f, 0.2838595f, 0.28004685f, 0.27791232f, 0.2761068f, 0.27457887f,
    0.2732831f, 0.27218118f, 0.2712414f, 0.2704378f, 0.26974902f, 0.2691575f,
    0.26864877f, 0.26821095f, 0.26783404f, 0.26750973f, 0.26723108f, 0.8541941f,
    0.637933f, 0.4820008f, 0.4084619f, 0.37186548f, 0.35308236f, 0.34330887f,
    0.33822832f, 0.33562776f, 0.33433822f, 0.33373162f, 0.33346912f,
    0.33336973f, 0.33333984f, 0.33333415f, 0.3333338f, 0.83076876f, 0.6358972f,
    0.4816919f, 0.40842834f, 0.37188175f, 0.35310256f, 0.34332415f, 0.33823845f,
    0.335634f, 0.33434185f, 0.3337337f, 0.3334703f, 0.33337042f, 0.3333403f,
    0.3333345f, 0.3333341f, 0.72214144f, 0.612218f, 0.47753167f, 0.40770873f,
    0.37190688f, 0.35325906f, 0.34346628f, 0.3383404f, 0.33570042f, 0.3343829f,
    0.3337583f, 0.333485f, 0.33337963f, 0.33334666f, 0.33333954f, 0.33333856f,
    0.6115845f, 0.5526396f, 0.46221045f, 0.40393564f, 0.37123314f, 0.35338873f,
    0.34374052f, 0.3385801f, 0.3358723f, 0.33449534f, 0.33382818f, 0.33352757f,
    0.33340612f, 0.33336455f, 0.33335322f, 0.33335033f, 0.51860076f,
    0.48750052f, 0.4355239f, 0.39466408f, 0.3684309f, 0.35281932f, 0.34386104f,
    0.3388494f, 0.33611777f, 0.33467728f, 0.3339511f, 0.3336072f, 0.33345792f,
    0.33340025f, 0.33338058f, 0.3333736f, 0.4556346f, 0.4372731f, 0.40771466f,
    0.38177493f, 0.36313257f, 0.3509403f, 0.34339f, 0.33889157f, 0.33630142f,
    0.33486328f, 0.33409867f, 0.33371386f, 0.3335333f, 0.33345544f, 0.33342436f,
    0.33341154f, 0.41506907f, 0.40302402f, 0.38540924f, 0.36928028f, 0.3568252f,
    0.34805152f, 0.34222832f, 0.33853266f, 0.33627704f, 0.3349531f, 0.33420894f,
    0.3338116f, 0.33361235f, 0.33351946f, 0.3334791f, 0.3334616f, 0.3890734f,
    0.38058403f, 0.36947224f, 0.35925892f, 0.3510549f, 0.3449734f, 0.34071338f,
    0.33786193f, 0.33602974f, 0.33489957f, 0.33423293f, 0.33385977f,
    0.33366397f, 0.33356926f, 0.3335278f, 0.3335113f, 0.37230572f, 0.36598215f,
    0.35860243f, 0.35191828f, 0.34645408f, 0.3422706f, 0.33922204f, 0.33709225f,
    0.3356625f, 0.3347416f, 0.33417538f, 0.33384636f, 0.3336689f, 0.333583f,
    0.3335484f, 0.33353943f, 0.3613731f, 0.3564081f, 0.35128695f, 0.34674796f,
    0.34302354f, 0.3401197f, 0.33794498f, 0.33637407f, 0.33527926f, 0.3345453f,
    0.33407515f, 0.333791f, 0.33363277f, 0.33355573f, 0.33352783f, 0.33352688f,
    0.35415858f, 0.35003847f, 0.34634697f, 0.3431502f, 0.34053832f, 0.33848548f,
    0.33692142f, 0.33576348f, 0.3349311f, 0.3343521f, 0.33396512f, 0.33371958f,
    0.33357513f, 0.3335003f, 0.33347124f, 0.33347034f, 0.34934473f, 0.3457236f,
    0.34297085f, 0.3406405f, 0.33875245f, 0.3372666f, 0.33612394f, 0.33526382f,
    0.33463058f, 0.334176f, 0.33385944f, 0.33364767f, 0.33351365f, 0.3334361f,
    0.33339828f, 0.33338735f, 0.3461079f, 0.34274298f, 0.34062594f, 0.33887202f,
    0.33746564f, 0.33636174f, 0.3355096f, 0.33486167f, 0.3343767f, 0.33401993f,
    0.33376282f, 0.33358234f, 0.33345988f, 0.33338073f, 0.33333328f,
    0.33330852f, 0.34392866f, 0.34064376f, 0.33896893f, 0.3376099f, 0.3365322f,
    0.33569032f, 0.33504027f, 0.33454353f, 0.33416802f, 0.3338873f, 0.33368027f,
    0.3335299f, 0.33342266f, 0.33334792f, 0.33329728f, 0.33326417f, 0.34247556f,
    0.33913895f, 0.33777955f, 0.3366988f, 0.33585188f, 0.33519435f, 0.33468777f,
    0.33430034f, 0.33400634f, 0.33378515f, 0.33362043f, 0.33349922f,
    0.33341128f, 0.33334848f, 0.33330446f, 0.3332742f, 0.341534f, 0.33804482f,
    0.33691606f, 0.33603734f, 0.33535755f, 0.3348341f, 0.33443284f, 0.33412698f,
    0.3338956f, 0.3337222f, 0.33359405f, 0.333501f, 0.33343527f, 0.3333905f,
    0.33336186f, 0.3333454f, 0.8362788f, 0.6125501f, 0.4718431f, 0.41673914f,
    0.3954854f, 0.38850328f, 0.38760647f, 0.38915512f, 0.39148253f, 0.39382944f,
    0.39586976f, 0.39748776f, 0.39866912f, 0.39944643f, 0.39987132f,
    0.40000054f, 0.8125457f, 0.61081135f, 0.47169462f, 0.41677552f, 0.3955309f,
    0.3885345f, 0.3876245f, 0.38916424f, 0.39148626f, 0.39383018f, 0.39586908f,
    0.3974866f, 0.39866802f, 0.39944568f, 0.39987108f, 0.40000084f, 0.70371133f,
    0.590175f, 0.46927828f, 0.41687378f, 0.39592418f, 0.38884538f, 0.3878185f,
    0.3892693f, 0.39153385f, 0.3938443f, 0.39586604f, 0.3974768f, 0.39865783f,
    0.39943877f, 0.39986926f, 0.40000483f, 0.59999496f, 0.5390249f, 0.4591268f,
    0.41574875f, 0.3965459f, 0.3895781f, 0.3883417f, 0.38957772f, 0.39168572f,
    0.39389747f, 0.39586464f, 0.39745128f, 0.39862785f, 0.39941624f,
    0.39986074f, 0.4000133f, 0.51876885f, 0.48641825f, 0.44135907f, 0.41166985f,
    0.39655676f, 0.39046055f, 0.38915774f, 0.39012802f, 0.39199048f,
    0.39402708f, 0.39588743f, 0.39742044f, 0.39858073f, 0.39937717f,
    0.39984438f, 0.4000273f, 0.46788827f, 0.44915006f, 0.4241658f, 0.4059399f,
    0.39563984f, 0.3911039f, 0.39005166f, 0.3908399f, 0.3924385f, 0.3942515f,
    0.39595842f, 0.39740422f, 0.3985296f, 0.39932847f, 0.39982262f, 0.400047f,
    0.43792567f, 0.42616504f, 0.41189203f, 0.40106565f, 0.39457294f,
    0.39157677f, 0.39091775f, 0.39161676f, 0.39297852f, 0.39455557f,
    0.39608195f, 0.39741367f, 0.39848456f, 0.3992757f, 0.3997957f, 0.40006799f,
    0.42061213f, 0.41278252f, 0.40433997f, 0.39792722f, 0.39398444f,
    0.39214584f, 0.39182046f, 0.39244834f, 0.39358777f, 0.39492467f, 0.3962532f,
    0.39744923f, 0.39844614f, 0.399216f, 0.39975566f, 0.4000768f, 0.41071662f,
    0.40521413f, 0.40007526f, 0.39625758f, 0.39392376f, 0.39287055f,
    0.39277932f, 0.3933325f, 0.3942613f, 0.39535868f, 0.39647642f, 0.3975156f,
    0.3984156f, 0.39914426f, 0.39968935f, 0.40005222f, 0.40513763f, 0.40102315f,
    0.3978285f, 0.3955463f, 0.39420396f, 0.39366013f, 0.39372516f, 0.39421275f,
    0.39496124f, 0.39584038f, 0.39675084f, 0.39762104f, 0.3984023f, 0.3990644f,
    0.39959106f, 0.39997655f, 0.40207177f, 0.39875782f, 0.3967432f, 0.3953815f,
    0.39464292f, 0.394418f, 0.39458182f, 0.39501786f, 0.3956264f, 0.39632633f,
    0.39705414f, 0.3977619f, 0.398415f, 0.39898965f, 0.39947087f, 0.39985055f,
    0.4004773f, 0.397579f, 0.3962985f, 0.3954968f, 0.3951232f, 0.39509052f,
    0.3953103f, 0.39570475f, 0.39620927f, 0.39677215f, 0.397353f, 0.3979211f,
    0.3984535f, 0.39893392f, 0.39935127f, 0.39969856f, 0.39975303f, 0.39700845f,
    0.3961939f, 0.39573774f, 0.39558387f, 0.39566213f, 0.39590752f, 0.39626548f,
    0.39669222f, 0.39715344f, 0.39762285f, 0.39808065f, 0.39851215f,
    0.39890686f, 0.39925742f, 0.39955908f, 0.39955187f, 0.3967759f, 0.3962625f,
    0.39602253f, 0.3960006f, 0.39613983f, 0.39639142f, 0.39671612f, 0.3970833f,
    0.3974695f, 0.39785716f, 0.3982332f, 0.39858812f, 0.3989152f, 0.39920983f,
    0.39946905f, 0.39967287f, 0.39673027f, 0.39641508f, 0.39631268f, 0.3963703f,
    0.3965414f, 0.39678884f, 0.39708397f, 0.3974051f, 0.3977361f, 0.3980651f,
    0.39838338f, 0.39868486f, 0.39896518f, 0.3992215f, 0.39945197f, 0.39999938f,
    0.39678854f, 0.3966068f, 0.39659402f, 0.39670056f, 0.39688802f, 0.39712763f,
    0.39739835f, 0.39768487f, 0.39797613f, 0.39826426f, 0.3985437f, 0.39881065f,
    0.3990625f, 0.39929768f, 0.3995152f, 0.8147333f, 0.5854851f, 0.46458924f,
    0.42970487f, 0.42386967f, 0.42805937f, 0.43520844f, 0.44257593f,
    0.44912845f, 0.45454165f, 0.4587892f, 0.46196508f, 0.46420446f, 0.46564868f,
    0.4664303f, 0.46666726f, 0.7907992f, 0.58411604f, 0.46460432f, 0.42980397f,
    0.42393848f, 0.42809808f, 0.43522754f, 0.4425835f, 0.44912976f, 0.45453984f,
    0.45878622f, 0.46196198f, 0.46420187f, 0.46564695f, 0.46642956f,
    0.46666756f, 0.6827293f, 0.5672049f, 0.46398744f, 0.43065533f, 0.42463973f,
    0.42852572f, 0.4354541f, 0.44268423f, 0.44915792f, 0.45452994f, 0.4587597f,
    0.46193185f, 0.46417582f, 0.46562922f, 0.46642223f, 0.46667132f,
    0.58781964f, 0.5259297f, 0.45920858f, 0.43203387f, 0.42639387f, 0.429749f,
    0.43616322f, 0.4430333f, 0.44928333f, 0.4545291f, 0.45869753f, 0.46184984f,
    0.46409956f, 0.46557382f, 0.46639577f, 0.4666775f, 0.52007204f, 0.4872292f,
    0.45062023f, 0.43292817f, 0.42893654f, 0.43186945f, 0.4375429f, 0.4437971f,
    0.4496229f, 0.45460755f, 0.45863438f, 0.4617272f, 0.46397135f, 0.46547386f,
    0.4663441f, 0.46668458f, 0.48227513f, 0.46368295f, 0.44412038f, 0.43404022f,
    0.43201044f, 0.4347159f, 0.43958613f, 0.4450539f, 0.45027682f, 0.45485497f,
    0.45863795f, 0.4616087f, 0.46381664f, 0.46534076f, 0.46626952f, 0.46669045f,
    0.46332893f, 0.4521885f, 0.44171065f, 0.43637115f, 0.43571883f, 0.4381527f,
    0.44218966f, 0.44678015f, 0.45127803f, 0.45532915f, 0.45876828f,
    0.46154404f, 0.4636698f, 0.46519306f, 0.46617684f, 0.466689f, 0.45475057f,
    0.44777444f, 0.44222248f, 0.4396535f, 0.4398266f, 0.44194216f, 0.44517875f,
    0.4488829f, 0.45259988f, 0.45604444f, 0.4590577f, 0.46156737f, 0.46355695f,
    0.46504393f, 0.4660648f, 0.46666563f, 0.45159233f, 0.44700482f, 0.44418985f,
    0.44320416f, 0.44384933f, 0.44568545f, 0.44825596f, 0.45117423f,
    0.45414686f, 0.45696938f, 0.459511f, 0.46169764f, 0.4634961f, 0.46490142f,
    0.46592718f, 0.46659827f, 0.45115244f, 0.44791627f, 0.44664958f,
    0.44654867f, 0.44744042f, 0.44905367f, 0.4511217f, 0.45341685f, 0.4557596f,
    0.45801675f, 0.46009505f, 0.46193433f, 0.46349993f, 0.46477672f,
    0.46576372f, 0.46646982f, 0.452001f, 0.44947895f, 0.449096f, 0.4494741f,
    0.45046884f, 0.45189905f, 0.45359826f, 0.4554282f, 0.45727998f, 0.4590711f,
    0.46074155f, 0.46224973f, 0.4635689f, 0.46468398f, 0.46558887f, 0.4662843f,
    0.45340177f, 0.4511865f, 0.45130724f, 0.4519314f, 0.45294243f, 0.45421427f,
    0.45563748f, 0.4571246f, 0.4586082f, 0.46003792f, 0.461377f, 0.4625996f,
    0.46368846f, 0.46463305f, 0.46542805f, 0.4660723f, 0.4549889f, 0.4528096f,
    0.4532136f, 0.45395157f, 0.4549311f, 0.4560625f, 0.45727205f, 0.45850277f,
    0.45971218f, 0.46086925f, 0.46195194f, 0.46294516f, 0.463839f, 0.4646276f,
    0.46530807f, 0.4658799f, 0.4565911f, 0.45426056f, 0.45481902f, 0.4555974f,
    0.4565224f, 0.45752987f, 0.4585697f, 0.4596047f, 0.46060804f, 0.4615608f,
    0.4624501f, 0.46326753f, 0.46400803f, 0.46466905f, 0.4652497f, 0.46575046f,
    0.45813727f, 0.45552084f, 0.45615909f, 0.45693874f, 0.4578017f, 0.4587019f,
    0.45960572f, 0.46048933f, 0.4613364f, 0.46213597f, 0.46288124f, 0.46356806f,
    0.46419457f, 0.46476012f, 0.4652653f, 0.46571127f, 0.4596063f, 0.4566039f,
    0.45727947f, 0.45804158f, 0.45884395f, 0.45965397f, 0.4604495f, 0.4612161f,
    0.46194473f, 0.46263f, 0.46326914f, 0.46386108f, 0.464406f, 0.46490484f,
    0.46535903f, 0.46577033f, 0.78868437f, 0.5576811f, 0.46227184f, 0.44916084f,
    0.4583602f, 0.47270036f, 0.4867775f, 0.49895114f, 0.5088832f, 0.5166902f,
    0.5226308f, 0.5269873f, 0.5300226f, 0.531967f, 0.5330161f, 0.533334f,
    0.7647613f, 0.55676824f, 0.46244755f, 0.44931215f, 0.4584453f, 0.47274292f,
    0.48679602f, 0.49895674f, 0.50888216f, 0.51668626f, 0.5226259f, 0.5269827f,
    0.530019f, 0.53196454f, 0.53301495f, 0.5333343f, 0.6591238f, 0.5444046f,
    0.46363387f, 0.45081824f, 0.45938078f, 0.4732453f, 0.48703516f, 0.4990462f,
    0.50889117f, 0.51665604f, 0.5225809f, 0.5269372f, 0.5299812f, 0.53193897f,
    0.53300375f, 0.5333381f, 0.5757919f, 0.51469773f, 0.46431231f, 0.45446125f,
    0.46205932f, 0.47482914f, 0.4878618f, 0.49940744f, 0.5089849f, 0.51660836f,
    0.5224704f, 0.5268121f, 0.5298705f, 0.53185946f, 0.53296447f, 0.53334314f,
    0.52363443f, 0.49137327f, 0.46497387f, 0.45994216f, 0.46675882f, 0.4779296f,
    0.489655f, 0.50031126f, 0.5093344f, 0.5166386f, 0.5223382f, 0.5266194f,
    0.529682f, 0.53171504f, 0.5328875f, 0.5333462f, 0.49997962f, 0.4822149f,
    0.4689989f, 0.46735948f, 0.47329158f, 0.48258182f, 0.49259284f, 0.50197047f,
    0.5101293f, 0.5168928f, 0.5222857f, 0.52642244f, 0.5294502f, 0.5315203f,
    0.5327753f, 0.5333435f, 0.49236596f, 0.48225158f, 0.4760296f, 0.4762504f,
    0.48114493f, 0.48848078f, 0.4965833f, 0.5044274f, 0.5114727f, 0.5174869f,
    0.5224156f, 0.5263f, 0.5292275f, 0.5313032f, 0.5326356f, 0.5333277f,
    0.49242708f, 0.48651987f, 0.48405555f, 0.48528382f, 0.4893029f, 0.4949518f,
    0.5012552f, 0.50752634f, 0.51333845f, 0.5184594f, 0.52278924f, 0.52631146f,
    0.5290582f, 0.5310881f, 0.53247166f, 0.5332826f, 0.4957217f, 0.49213952f,
    0.4916945f, 0.49343303f, 0.4968131f, 0.5012007f, 0.5060457f, 0.51092935f,
    0.515561f, 0.51975644f, 0.5234131f, 0.52648675f, 0.5289727f, 0.530892f,
    0.53228116f, 0.5331845f, 0.5000751f, 0.49772903f, 0.49835128f, 0.5002954f,
    0.50320214f, 0.50669676f, 0.51046205f, 0.5142513f, 0.5178844f, 0.5212371f,
    0.5242303f, 0.52681965f, 0.5289867f, 0.53073156f, 0.53206736f, 0.5330158f,
    0.5044954f, 0.5027296f, 0.50389296f, 0.5058646f, 0.5083967f, 0.5112531f,
    0.5142423f, 0.51721776f, 0.520072f, 0.5227289f, 0.5251372f, 0.527265f,
    0.52909523f, 0.5306222f, 0.53184843f, 0.5327827f, 0.5085802f, 0.5069846f,
    0.50839794f, 0.51030165f, 0.51252216f, 0.5149055f, 0.5173315f, 0.5197114f,
    0.5219815f, 0.5240973f, 0.526029f, 0.5277579f, 0.52927345f, 0.5305716f,
    0.5316529f, 0.5325214f, 0.51220816f, 0.5105155f, 0.51201916f, 0.51381034f,
    0.51576656f, 0.5177861f, 0.519793f, 0.52173305f, 0.5235688f, 0.5252753f,
    0.5268366f, 0.5282435f, 0.52949154f, 0.5305798f, 0.53150994f, 0.53228533f,
    0.5153841f, 0.5134124f, 0.51492125f, 0.5165846f, 0.5183161f, 0.5200491f,
    0.5217364f, 0.52334577f, 0.5248565f, 0.5262556f, 0.5275359f, 0.52869415f,
    0.52972996f, 0.53064483f, 0.5314416f, 0.5321241f, 0.51816297f, 0.51578265f,
    0.5172541f, 0.5187909f, 0.5203329f, 0.5218376f, 0.5232773f, 0.5246347f,
    0.52590007f, 0.5270682f, 0.5281376f, 0.52910906f, 0.52998453f, 0.53076714f,
    0.5314606f, 0.5320687f, 0.52061397f, 0.5177291f, 0.51914525f, 0.5205657f,
    0.52194965f, 0.52327234f, 0.5245194f, 0.5256839f, 0.5267633f, 0.52775824f,
    0.5286708f, 0.5295043f, 0.53026265f, 0.5309499f, 0.5315704f, 0.5321282f,
    0.7568832f, 0.5306808f, 0.46744922f, 0.47708708f, 0.5002799f, 0.52327573f,
    0.54285514f, 0.55862707f, 0.57096887f, 0.58041686f, 0.5874829f, 0.59260684f,
    0.5961517f, 0.59841335f, 0.59963155f, 0.6000008f, 0.73335385f, 0.5303202f,
    0.46777332f, 0.4772765f, 0.50037336f, 0.5233183f, 0.5428716f, 0.55863035f,
    0.5709659f, 0.58041126f, 0.5874768f, 0.5926013f, 0.5961473f, 0.5984105f,
    0.5996303f, 0.6000011f, 0.632862f, 0.52344245f, 0.4706836f, 0.4793005f,
    0.50145763f, 0.5238516f, 0.54310435f, 0.5587032f, 0.57095724f, 0.5803654f,
    0.5874189f, 0.59254575f, 0.5961023f, 0.5983802f, 0.5996169f, 0.60000527f,
    0.56493443f, 0.5071815f, 0.47670597f, 0.4848546f, 0.50480884f, 0.5256536f,
    0.5439792f, 0.5590505f, 0.571017f, 0.5802805f, 0.58727443f, 0.59239244f,
    0.59597015f, 0.59828633f, 0.5995704f, 0.6000104f, 0.5308889f, 0.50067484f,
    0.48637915f, 0.4943265f, 0.5111919f, 0.52943814f, 0.54602444f, 0.56001985f,
    0.5713538f, 0.5802685f, 0.58709294f, 0.5921538f, 0.5957441f, 0.59811527f,
    0.59947926f, 0.60001254f, 0.52241796f, 0.50633615f, 0.5003962f, 0.50723976f,
    0.5204981f, 0.535425f, 0.54957116f, 0.56192833f, 0.5722228f, 0.58051497f,
    0.5869984f, 0.5919053f, 0.59546447f, 0.59788394f, 0.599346f, 0.60000724f,
    0.5262681f, 0.517649f, 0.516097f, 0.5217687f, 0.5316875f, 0.54318315f,
    0.5545461f, 0.5648731f, 0.57378f, 0.5811762f, 0.5871215f, 0.5917437f,
    0.5951943f, 0.5976255f, 0.59918f, 0.5999859f, 0.53465515f, 0.5300412f,
    0.53079915f, 0.5356441f, 0.54307944f, 0.5516884f, 0.56043315f, 0.5686582f,
    0.57600325f, 0.58230937f, 0.5875433f, 0.591744f, 0.5949885f, 0.59737f,
    0.59898597f, 0.5999306f, 0.54392123f, 0.54141897f, 0.5433266f, 0.54758054f,
    0.55333775f, 0.55983f, 0.5664671f, 0.5728378f, 0.57868063f, 0.58384776f,
    0.58827275f, 0.5919439f, 0.5948847f, 0.59713924f, 0.59876275f, 0.599815f,
    0.55256087f, 0.55109096f, 0.5535047f, 0.5572796f, 0.56189895f, 0.56692016f,
    0.5720068f, 0.5769175f, 0.5814882f, 0.5856148f, 0.5892388f, 0.5923348f,
    0.59490126f, 0.5969529f, 0.59851503f, 0.59961975f, 0.56008375f, 0.55900997f,
    0.56158257f, 0.56493986f, 0.5687503f, 0.5727451f, 0.5767263f, 0.580554f,
    0.5841324f, 0.5873987f, 0.59031487f, 0.59286106f, 0.595031f, 0.59682834f,
    0.59826356f, 0.59935224f, 0.566444f, 0.5653764f, 0.5679283f, 0.57091576f,
    0.57412165f, 0.57737815f, 0.5805662f, 0.5836046f, 0.5864392f, 0.5890359f,
    0.5913744f, 0.59344447f, 0.59524333f, 0.59677327f, 0.5980404f, 0.5990536f,
    0.5717678f, 0.5704563f, 0.57290125f, 0.5755637f, 0.5783009f, 0.581008f,
    0.5836141f, 0.5860729f, 0.5883556f, 0.5904455f, 0.59233433f, 0.5940195f,
    0.59550214f, 0.59678644f, 0.59787804f, 0.5987839f, 0.57623076f, 0.57450634f,
    0.57680935f, 0.5791901f, 0.5815557f, 0.583843f, 0.5860118f, 0.5880381f,
    0.58990896f, 0.5916184f, 0.59316564f, 0.5945527f, 0.5957838f, 0.59686446f,
    0.59780073f, 0.5985992f, 0.5800057f, 0.57774824f, 0.5799022f, 0.58204204f,
    0.58411014f, 0.5860711f, 0.5879057f, 0.58960474f, 0.5911655f, 0.59258926f,
    0.59387976f, 0.5950423f, 0.5960827f, 0.5970073f, 0.59782237f, 0.5985342f,
    0.58324397f, 0.5803639f, 0.58237666f, 0.58431363f, 0.5861426f, 0.58784825f,
    0.58942544f, 0.59087515f, 0.5922016f, 0.5934109f, 0.5945101f, 0.5955062f,
    0.5964065f, 0.5972178f, 0.59794664f, 0.59859914f, 0.71760744f, 0.5072572f,
    0.48356223f, 0.51580656f, 0.55103076f, 0.5806087f, 0.6039231f, 0.6218867f,
    0.6355525f, 0.6458202f, 0.6534032f, 0.658856f, 0.6626082f, 0.6649948f,
    0.6662785f, 0.6666675f, 0.6951482f, 0.50754344f, 0.48400915f, 0.51601636f,
    0.5511241f, 0.5806478f, 0.6039364f, 0.6218877f, 0.635548f, 0.6458136f,
    0.6533964f, 0.65885f, 0.6626036f, 0.66499174f, 0.66627717f, 0.66666794f,
    0.6041896f, 0.5071307f, 0.48843852f, 0.51837844f, 0.5522625f, 0.58116883f,
    0.6041463f, 0.6219408f, 0.6355249f, 0.64575785f, 0.65333205f, 0.65879035f,
    0.66255605f, 0.66496027f, 0.6662635f, 0.66667277f, 0.5569308f, 0.50621307f,
    0.49937528f, 0.52535206f, 0.5559924f, 0.5830393f, 0.605003f, 0.62225276f,
    0.63555247f, 0.6456481f, 0.6531699f, 0.65862507f, 0.6624162f, 0.6648621f,
    0.6662158f, 0.6666793f, 0.5439077f, 0.5176902f, 0.5173327f, 0.53795356f,
    0.5634795f, 0.587178f, 0.6071328f, 0.62321615f, 0.6358587f, 0.64560425f,
    0.652962f, 0.6583668f, 0.66217685f, 0.66468334f, 0.666122f, 0.666684f,
    0.5514887f, 0.5381281f, 0.54026437f, 0.55520475f, 0.5747001f, 0.5939546f,
    0.61097634f, 0.62521464f, 0.6367364f, 0.6458316f, 0.6528426f, 0.65809613f,
    0.6618807f, 0.66444176f, 0.66598517f, 0.6666822f, 0.5665938f, 0.55998415f,
    0.5633795f, 0.57410276f, 0.588213f, 0.602864f, 0.616485f, 0.628385f,
    0.63837296f, 0.6465072f, 0.65295446f, 0.65791625f, 0.66159356f, 0.66417164f,
    0.66581464f, 0.6666647f, 0.58265775f, 0.57954746f, 0.58353883f, 0.5916198f,
    0.6018312f, 0.6126429f, 0.6230583f, 0.6325156f, 0.6407538f, 0.6476996f,
    0.6533875f, 0.6579073f, 0.6613727f, 0.66390306f, 0.6656137f, 0.6666114f,
    0.59713644f, 0.5957523f, 0.5998907f, 0.606309f, 0.6139403f, 0.6219619f,
    0.62980354f, 0.63710153f, 0.6436468f, 0.6493397f, 0.6541541f, 0.65811074f,
    0.6612576f, 0.6636574f, 0.66537917f, 0.6664922f, 0.6093413f, 0.60869163f,
    0.6127126f, 0.61799985f, 0.6239263f, 0.63002783f, 0.6359842f, 0.64158314f,
    0.6466919f, 0.65123546f, 0.6551796f, 0.65851927f, 0.66126895f, 0.6634558f,
    0.6651147f, 0.66628486f, 0.6193378f, 0.6188505f, 0.62262386f, 0.6270803f,
    0.6318347f, 0.6366133f, 0.64123297f, 0.6455745f, 0.649563f, 0.6531546f,
    0.65632725f, 0.6590744f, 0.66140056f, 0.6633176f, 0.66484284f, 0.6659969f,
    0.6274465f, 0.62677735f, 0.6302549f, 0.63406914f, 0.63797927f, 0.641821f,
    0.64548874f, 0.6489169f, 0.65206677f, 0.65491736f, 0.65745956f, 0.6596926f,
    0.6616209f, 0.66325295f, 0.6645996f, 0.6656734f, 0.6340334f, 0.6329635f,
    0.63614213f, 0.6394456f, 0.6427238f, 0.64587986f, 0.648855f, 0.6516155f,
    0.65414435f, 0.6564346f, 0.6584861f, 0.66030306f, 0.6618921f, 0.66326183f,
    0.66442144f, 0.6653807f, 0.63943106f, 0.637812f, 0.6407114f, 0.64360255f,
    0.64639497f, 0.64903516f, 0.65149444f, 0.65375936f, 0.65582603f, 0.6576961f,
    0.6593751f, 0.6608701f, 0.6621894f, 0.6633418f, 0.6643362f, 0.6651813f,
    0.64391625f, 0.641641f, 0.6442911f, 0.64684737f, 0.6492601f, 0.65150505f,
    0.6535734f, 0.65546507f, 0.6571848f, 0.65874f, 0.6601394f, 0.66139203f,
    0.66250724f, 0.66349363f, 0.66435987f, 0.6651137f, 0.64770955f, 0.644697f,
    0.6471312f, 0.64941543f, 0.65152913f, 0.6534684f, 0.65523773f, 0.6568463f,
    0.6583047f, 0.65962404f, 0.6608153f, 0.6618889f, 0.66285455f, 0.66372114f,
    0.6644969f, 0.66518927f, 0.668648f, 0.49262163f, 0.5154185f, 0.5681457f,
    0.61216235f, 0.6455391f, 0.6704363f, 0.6889766f, 0.7027676f, 0.7129721f,
    0.72042984f, 0.7257545f, 0.7294017f, 0.731715f, 0.7329578f, 0.7333343f,
    0.64846426f, 0.49361938f, 0.51594436f, 0.5683547f, 0.6122472f, 0.6455718f,
    0.67044574f, 0.6889755f, 0.7027621f, 0.7129652f, 0.7204231f, 0.7257487f,
    0.7293973f, 0.7317122f, 0.7329566f, 0.7333347f, 0.5742243f, 0.50048953f,
    0.52150536f, 0.57083017f, 0.6133399f, 0.6460408f, 0.6706205f, 0.68900895f,
    0.7027299f, 0.71290666f, 0.72035927f, 0.72569114f, 0.72935224f, 0.73168296f,
    0.7329446f, 0.7333406f, 0.55485207f, 0.5164185f, 0.5364353f, 0.578567f,
    0.6171159f, 0.6478283f, 0.6713993f, 0.68927085f, 0.7027318f, 0.7127878f,
    0.7201977f, 0.72553116f, 0.7292192f, 0.7315911f, 0.73290163f, 0.73335004f,
    0.56596845f, 0.5462543f, 0.56118196f, 0.59309715f, 0.6250114f, 0.6519616f,
    0.673445f, 0.69016284f, 0.70299613f, 0.7127271f, 0.7199896f, 0.7252817f,
    0.728992f, 0.7314242f, 0.7328173f, 0.7333606f, 0.58995634f, 0.5805096f,
    0.5911296f, 0.61307484f, 0.6370868f, 0.6589074f, 0.6772509f, 0.6920897f,
    0.7038206f, 0.71292853f, 0.7198662f, 0.7250208f, 0.728712f, 0.73119974f,
    0.7326949f, 0.7333687f, 0.6154794f, 0.6113885f, 0.6197077f, 0.6346274f,
    0.6516721f, 0.66814744f, 0.68279535f, 0.69520724f, 0.705399f, 0.7135671f,
    0.71996486f, 0.7248457f, 0.72844017f, 0.73094875f, 0.73254234f, 0.7333647f,
    0.63803285f, 0.63657594f, 0.64358395f, 0.65422004f, 0.66628593f, 0.6783165f,
    0.68946415f, 0.6993148f, 0.7077267f, 0.71471435f, 0.72037256f, 0.7248307f,
    0.7282271f, 0.73069537f, 0.7323584f, 0.73332584f, 0.65655434f, 0.65625316f,
    0.66232723f, 0.6703544f, 0.6791671f, 0.6879865f, 0.6963255f, 0.70390314f,
    0.7105798f, 0.7163097f, 0.72110593f, 0.72501665f, 0.7281084f, 0.73045576f,
    0.73213446f, 0.7332175f, 0.6713042f, 0.67134845f, 0.67666006f, 0.6829969f,
    0.6896934f, 0.69631994f, 0.70260864f, 0.7083981f, 0.7135982f, 0.7181675f,
    0.72209734f, 0.7254012f, 0.7281067f, 0.73024976f, 0.73187107f, 0.733013f,
    0.6829307f, 0.68286234f, 0.6875252f, 0.69268906f, 0.6979589f, 0.7030889f,
    0.7079315f, 0.7124008f, 0.71644944f, 0.72005564f, 0.7232142f, 0.7259312f,
    0.7282202f, 0.7300998f, 0.7315915f, 0.73271877f, 0.69210744f, 0.69165367f,
    0.6957629f, 0.70006806f, 0.7043321f, 0.7084137f, 0.7122328f, 0.71574664f,
    0.7189351f, 0.7217919f, 0.7243196f, 0.7265258f, 0.7284216f, 0.7300203f,
    0.7313358f, 0.7323832f, 0.69941425f, 0.6984004f, 0.7020401f, 0.7056933f,
    0.70922023f, 0.7125432f, 0.71562254f, 0.71844083f, 0.72099394f, 0.7232852f,
    0.7253223f, 0.7271154f, 0.72867584f, 0.73001534f, 0.7311458f, 0.7320787f,
    0.7053142f, 0.7036188f, 0.7068635f, 0.71000993f, 0.7129808f, 0.71573925f,
    0.7182712f, 0.72057533f, 0.722657f, 0.72452533f, 0.7261909f, 0.7276653f,
    0.7289599f, 0.7300858f, 0.73105365f, 0.7318735f, 0.7101646f, 0.70769644f,
    0.71061164f, 0.7133586f, 0.7159021f, 0.71823215f, 0.720352f, 0.72227055f,
    0.7239996f, 0.72555184f, 0.72693974f, 0.72817546f, 0.72927034f, 0.7302348f,
    0.73107857f, 0.7318104f, 0.71423656f, 0.71092343f, 0.71356624f, 0.71599615f,
    0.71820766f, 0.7202096f, 0.72201633f, 0.7236439f, 0.72510844f, 0.72642493f,
    0.72760725f, 0.72866786f, 0.7296181f, 0.730468f, 0.7312266f, 0.73190206f,
    0.60798365f, 0.4969573f, 0.56992143f, 0.6376214f, 0.6854309f, 0.718951f,
    0.742842f, 0.7601221f, 0.77272624f, 0.78192717f, 0.7885885f, 0.79331404f,
    0.796537f, 0.7985759f, 0.79966974f, 0.800001f, 0.5922552f, 0.4986381f,
    0.5704594f, 0.63780636f, 0.6854997f, 0.7189753f, 0.7428476f, 0.76011974f,
    0.77272075f, 0.7819208f, 0.78858256f, 0.79330903f, 0.7965332f, 0.79857355f,
    0.79966885f, 0.8000016f, 0.54669154f, 0.512943f, 0.576511f, 0.6401291f,
    0.6864483f, 0.7193596f, 0.7429808f, 0.7601374f, 0.77268684f, 0.78186774f,
    0.788527f, 0.7932601f, 0.7964958f, 0.7985501f, 0.79966056f, 0.8000088f,
    0.56483996f, 0.5458112f, 0.59376025f, 0.6477828f, 0.6899067f, 0.7209185f,
    0.7436328f, 0.7603434f, 0.7726757f, 0.7817591f, 0.7883863f, 0.7931238f,
    0.7963845f, 0.7984754f, 0.79962856f, 0.8000225f, 0.60270923f, 0.59247774f,
    0.6225939f, 0.66260606f, 0.69739f, 0.72466266f, 0.74542964f, 0.76110774f,
    0.77289456f, 0.7817021f, 0.78820753f, 0.79291344f, 0.796196f, 0.7983404f,
    0.7995658f, 0.80004245f, 0.64216447f, 0.6378251f, 0.65640146f, 0.68310255f,
    0.70902574f, 0.731078f, 0.74884427f, 0.7628018f, 0.7736096f, 0.78187615f,
    0.788105f, 0.79269683f, 0.7959664f, 0.7981612f, 0.7994763f, 0.8000669f,
    0.6760705f, 0.67486715f, 0.68747133f, 0.70501256f, 0.72314805f, 0.7397042f,
    0.7538792f, 0.7655741f, 0.7749911f, 0.7824291f, 0.7881916f, 0.79255205f,
    0.79574364f, 0.797961f, 0.7993645f, 0.8000862f, 0.70301807f, 0.7032016f,
    0.712592f, 0.7246453f, 0.7372615f, 0.74924344f, 0.7599884f, 0.7692629f,
    0.77704567f, 0.7834253f, 0.78853846f, 0.79253554f, 0.7955623f, 0.79775184f,
    0.7992219f, 0.8000744f, 0.7237713f, 0.7243711f, 0.731794f, 0.7405742f,
    0.7496225f, 0.7583148f, 0.76630425f, 0.77341694f, 0.77958876f, 0.78482395f,
    0.78916687f, 0.7926833f, 0.79544866f, 0.7975398f, 0.79903096f, 0.7999914f,
    0.7396039f, 0.7400932f, 0.746169f, 0.75288767f, 0.7596454f, 0.7661082f,
    0.7720931f, 0.77750427f, 0.78229916f, 0.7864692f, 0.7900277f, 0.7930016f,
    0.79542613f, 0.79734087f, 0.79878676f, 0.79980487f, 0.7517132f, 0.7517962f,
    0.7568825f, 0.7622171f, 0.76745355f, 0.77240837f, 0.7769877f, 0.7811465f,
    0.78486806f, 0.78815186f, 0.7910075f, 0.79345083f, 0.79550135f, 0.79718095f,
    0.7985124f, 0.79951894f, 0.76106775f, 0.7605648f, 0.7648942f, 0.76924855f,
    0.77342916f, 0.7773373f, 0.7809279f, 0.78418463f, 0.7871067f, 0.7897018f,
    0.79198205f, 0.79396176f, 0.79565626f, 0.79708123f, 0.7982522f, 0.7991844f,
    0.7684044f, 0.76719415f, 0.7709308f, 0.77456284f, 0.7779813f, 0.78113866f,
    0.7840189f, 0.7866218f, 0.78895575f, 0.79103285f, 0.7928669f, 0.7944723f,
    0.7958631f, 0.79705286f, 0.7980542f, 0.79887915f, 0.77426994f, 0.7722604f,
    0.7755266f, 0.77861136f, 0.7814629f, 0.78406626f, 0.7864236f, 0.78854525f,
    0.7904445f, 0.79213583f, 0.7936339f, 0.7949524f, 0.7961044f, 0.797102f,
    0.7979562f, 0.7986774f, 0.7790653f, 0.7761812f, 0.7790713f, 0.78173405f,
    0.7841555f, 0.7863419f, 0.78830785f, 0.7900701f, 0.7916456f, 0.7930503f,
    0.794299f, 0.795405f, 0.7963804f, 0.797236f, 0.7979817f, 0.798626f,
    0.7830843f, 0.7792611f, 0.7818501f, 0.7841843f, 0.7862756f, 0.7881454f,
    0.789816f, 0.79130876f, 0.792643f, 0.7938356f, 0.7949016f, 0.79585415f,
    0.79670477f, 0.79746336f, 0.798139f, 0.7987393f, 0.53863645f, 0.5411915f,
    0.6572286f, 0.7286801f, 0.77285826f, 0.8017938f, 0.8215921f, 0.835537f,
    0.84552664f, 0.8527282f, 0.85789645f, 0.86154056f, 0.8640158f, 0.86557776f,
    0.8664145f, 0.8666678f, 0.53123724f, 0.54329705f, 0.657688f, 0.7288184f,
    0.7729058f, 0.8018093f, 0.8215948f, 0.83553445f, 0.8455222f, 0.8527234f,
    0.85789216f, 0.86153704f, 0.86401325f, 0.8655762f, 0.86641407f, 0.86666846f,
    0.5339064f, 0.56329757f, 0.6632639f, 0.7306968f, 0.7736241f, 0.80208826f,
    0.8216877f, 0.83554476f, 0.84549695f, 0.8526855f, 0.8578534f, 0.8615037f,
    0.86398876f, 0.8655622f, 0.8664114f, 0.86667734f, 0.6006878f, 0.60924375f,
    0.6800211f, 0.73721385f, 0.7763846f, 0.8032884f, 0.82217973f, 0.8356999f,
    0.84549195f, 0.8526103f, 0.85775596f, 0.8614105f, 0.8639146f, 0.8655155f,
    0.8663966f, 0.8666968f, 0.6648772f, 0.6667736f, 0.7083204f, 0.75013936f,
    0.78250325f, 0.80624294f, 0.823573f, 0.8362929f, 0.8456715f, 0.8525838f,
    0.8576396f, 0.8612713f, 0.86379206f, 0.86543286f, 0.8663677f, 0.86672956f,
    0.715582f, 0.71697026f, 0.74086595f, 0.7681533f, 0.79212886f, 0.81134623f,
    0.82622653f, 0.837597f, 0.8462282f, 0.8527354f, 0.85758704f, 0.8611359f,
    0.86364836f, 0.8653275f, 0.86632967f, 0.8667768f, 0.753393f, 0.7549277f,
    0.7699202f, 0.78734887f, 0.8039087f, 0.81827617f, 0.83015853f, 0.8397191f,
    0.8472745f, 0.8531576f, 0.8576662f, 0.86104935f, 0.8635098f, 0.8652103f,
    0.86628175f, 0.8668292f, 0.78099525f, 0.7823985f, 0.79275095f, 0.80437124f,
    0.8157048f, 0.8260109f, 0.8349847f, 0.84256697f, 0.84882784f, 0.853896f,
    0.8579182f, 0.86103785f, 0.8633855f, 0.8650753f, 0.866205f, 0.86685735f,
    0.8010964f, 0.8021079f, 0.8097791f, 0.8180025f, 0.825997f, 0.83339435f,
    0.8400222f, 0.8458163f, 0.85077673f, 0.8549414f, 0.85836905f, 0.8611273f,
    0.8632861f, 0.8649127f, 0.86606985f, 0.8668143f, 0.81585157f, 0.8163069f,
    0.8222711f, 0.8284032f, 0.8342837f, 0.8397283f, 0.8446572f, 0.8490419f,
    0.8528817f, 0.8561927f, 0.85900044f, 0.8613365f, 0.8632354f, 0.8647326f,
    0.8658632f, 0.8666609f, 0.82683855f, 0.826631f, 0.83142805f, 0.8361896f,
    0.8406857f, 0.8448233f, 0.8485704f, 0.851923f, 0.8548902f, 0.8574876f,
    0.85973364f, 0.8616483f, 0.8632519f, 0.86456484f, 0.8656071f, 0.8663978f,
    0.8351748f, 0.83422375f, 0.83818203f, 0.84199584f, 0.84554327f, 0.848782f,
    0.8517044f, 0.85431874f, 0.8566396f, 0.8586842f, 0.86047f, 0.862014f,
    0.8633322f, 0.86443967f, 0.86535037f, 0.86607736f, 0.8416436f, 0.83987814f,
    0.8432123f, 0.84634304f, 0.84921354f, 0.8518121f, 0.85414565f, 0.85622823f,
    0.85807675f, 0.8597084f, 0.8611396f, 0.8623857f, 0.86346084f, 0.86437786f,
    0.8651483f, 0.86578256f, 0.84679365f, 0.8441465f, 0.84700507f, 0.8496285f,
    0.85200083f, 0.85413f, 0.85603213f, 0.8577252f, 0.8592269f, 0.8605536f,
    0.86172074f, 0.8627418f, 0.86362916f, 0.86439383f, 0.8650456f, 0.8655935f,
    0.8510118f, 0.8474177f, 0.8499085f, 0.8521477f, 0.8541461f, 0.85592467f,
    0.8575055f, 0.8589093f, 0.8601545f, 0.8612572f, 0.86223155f, 0.86308986f,
    0.8638431f, 0.8645006f, 0.8650708f, 0.86556107f, 0.85457283f, 0.8499696f,
    0.85217386f, 0.85411906f, 0.8558342f, 0.85734904f, 0.8586897f, 0.8598787f,
    0.86093503f, 0.86187464f, 0.86271113f, 0.8634562f, 0.86411977f, 0.8647104f,
    0.8652355f, 0.86570156f, 0.5038902f, 0.67145747f, 0.792669f, 0.84701735f,
    0.8767977f, 0.89510095f, 0.9071516f, 0.91542894f, 0.921258f, 0.9254102f,
    0.92836523f, 0.9304366f, 0.931838f, 0.93272f, 0.93319184f, 0.9333346f,
    0.5096806f, 0.673254f, 0.7929487f, 0.8470936f, 0.876823f, 0.8951093f,
    0.9071533f, 0.915428f, 0.92125607f, 0.9254081f, 0.9283634f, 0.9304352f,
    0.9318371f, 0.93271965f, 0.9331921f, 0.93333536f, 0.58350986f, 0.6923971f,
    0.7967377f, 0.848251f, 0.87725407f, 0.8952807f, 0.9072176f, 0.915445f,
    0.9212524f, 0.9253967f, 0.9283507f, 0.9304248f, 0.9318309f, 0.93271875f,
    0.93319696f, 0.9333463f, 0.69959337f, 0.7368633f, 0.8085051f, 0.8523773f,
    0.8789461f, 0.8960208f, 0.9075379f, 0.91556764f, 0.92128056f, 0.92538214f,
    0.928321f, 0.93039507f, 0.9318098f, 0.93271106f, 0.9332056f, 0.9333728f,
    0.7763783f, 0.7885257f, 0.828542f, 0.86050844f, 0.882614f, 0.8977794f,
    0.90838975f, 0.91596144f, 0.9214377f, 0.9254196f, 0.9283039f, 0.9303611f,
    0.93178123f, 0.9327014f, 0.9332229f, 0.93342197f, 0.82487154f, 0.8298386f,
    0.8515354f, 0.8719686f, 0.888333f, 0.90070397f, 0.9099004f, 0.9167251f,
    0.9217965f, 0.9255607f, 0.9283348f, 0.93034613f, 0.93176f, 0.9326989f,
    0.9332549f, 0.9334985f, 0.85647553f, 0.85887897f, 0.871669f, 0.8843279f,
    0.8954709f, 0.90470034f, 0.9120864f, 0.91788155f, 0.9223709f, 0.9258116f,
    0.9284149f, 0.9303482f, 0.931742f, 0.93269753f, 0.9332941f, 0.9335938f,
    0.8776492f, 0.8787561f, 0.8870782f, 0.8952657f, 0.9027382f, 0.9092786f,
    0.91483235f, 0.9194392f, 0.9231879f, 0.9261856f, 0.92854005f, 0.9303499f,
    0.93170124f, 0.9326669f, 0.9333078f, 0.9336745f, 0.8922164f, 0.8924308f,
    0.8982815f, 0.9039263f, 0.9090997f, 0.91372424f, 0.91778094f, 0.9212788f,
    0.9242447f, 0.92671716f, 0.9287407f, 0.9303618f, 0.9316261f, 0.93257624f,
    0.9332513f, 0.933686f, 0.9025171f, 0.90197027f, 0.9063207f, 0.910438f,
    0.91418904f, 0.9175513f, 0.92053366f, 0.9231536f, 0.9254308f, 0.9273863f,
    0.9290416f, 0.9304189f, 0.93154013f, 0.9324271f, 0.9331007f, 0.93358094f,
    0.91001636f, 0.90873194f, 0.9121034f, 0.91523945f, 0.9180754f, 0.92060906f,
    0.9228576f, 0.92484266f, 0.92658514f, 0.9281039f, 0.9294157f, 0.9305357f,
    0.9314774f, 0.9322536f, 0.9328759f, 0.9333552f, 0.91565126f, 0.91360164f,
    0.916299f, 0.91876745f, 0.92098284f, 0.9229538f, 0.9246984f, 0.9262372f,
    0.9275896f, 0.9287732f, 0.9298033f, 0.9306933f, 0.9314547f, 0.9320976f,
    0.9326308f, 0.93306196f, 0.9200336f, 0.91716427f, 0.9193783f, 0.92137235f,
    0.9231477f, 0.9247199f, 0.9261078f, 0.92732954f, 0.928402f, 0.9293401f,
    0.9301573f, 0.9308652f, 0.931474f, 0.93199253f, 0.9324286f, 0.9327889f,
    0.92357075f, 0.9198139f, 0.92167115f, 0.9233177f, 0.92477083f, 0.92605144f,
    0.9271789f, 0.9281704f, 0.9290405f, 0.929802f, 0.93046594f, 0.931042f,
    0.93153864f, 0.9319632f, 0.93232226f, 0.9326216f, 0.92653966f, 0.9218217f,
    0.9234108f, 0.9247982f, 0.9260117f, 0.9270756f, 0.92801046f, 0.92883277f,
    0.9295562f, 0.93019193f, 0.9307497f, 0.9312375f, 0.93166244f, 0.9320303f,
    0.9323465f, 0.93261546f, 0.9291315f, 0.92337906f, 0.92476505f, 0.92595816f,
    0.9269931f, 0.9278969f, 0.9286905f, 0.9293902f, 0.9300093f, 0.9305584f,
    0.9310463f, 0.9314803f, 0.9318667f, 0.9322107f, 0.93251675f, 0.93278867f,
    0.80288714f, 0.92985815f, 0.96306646f, 0.97450846f, 0.98017806f, 0.9834838f,
    0.9855923f, 0.98701096f, 0.98799604f, 0.9886909f, 0.989182f, 0.98952454f,
    0.9897555f, 0.9899005f, 0.98997796f, 0.9900013f, 0.8034146f, 0.930386f,
    0.96314615f, 0.9745338f, 0.9801891f, 0.98348963f, 0.9855957f, 0.9870132f,
    0.98799765f, 0.9886921f, 0.989183f, 0.9895255f, 0.9897564f, 0.9899014f,
    0.98997885f, 0.9900023f, 0.8589883f, 0.93558574f, 0.964269f, 0.9749344f,
    0.9803721f, 0.98358756f, 0.9856545f, 0.9870518f, 0.98802507f, 0.9887132f,
    0.9892003f, 0.98954046f, 0.9897701f, 0.9899145f, 0.9899917f, 0.99001515f,
    0.93118227f, 0.94578254f, 0.9664922f, 0.9759375f, 0.9809033f, 0.9838917f,
    0.98584056f, 0.98717296f, 0.98810905f, 0.98877513f, 0.98924905f,
    0.98958147f, 0.9898067f, 0.9899488f, 0.9900252f, 0.99004877f, 0.95958465f,
    0.9588294f, 0.969659f, 0.97705877f, 0.98154074f, 0.9843213f, 0.9861426f,
    0.9873902f, 0.98826987f, 0.9888986f, 0.9893482f, 0.98966503f, 0.98988074f,
    0.9900175f, 0.99009144f, 0.9901146f, 0.9719746f, 0.969007f, 0.974132f,
    0.97866756f, 0.982162f, 0.98464775f, 0.9863764f, 0.9875839f, 0.98843735f,
    0.9890459f, 0.98947984f, 0.9897852f, 0.98999316f, 0.9901254f, 0.99019754f,
    0.9902211f, 0.97866803f, 0.975638f, 0.9783914f, 0.9809038f, 0.9831614f,
    0.9850476f, 0.9865343f, 0.9876645f, 0.98850566f, 0.98912317f, 0.9895705f,
    0.9898884f, 0.9901071f, 0.99024856f, 0.990329f, 0.9903605f, 0.98278886f,
    0.9798763f, 0.9816119f, 0.9831057f, 0.9844767f, 0.9857271f, 0.98682535f,
    0.98775184f, 0.9885068f, 0.9891041f, 0.9895643f, 0.9899092f, 0.9901589f,
    0.99033076f, 0.9904392f, 0.9904958f, 0.9855667f, 0.98264384f, 0.983854f,
    0.98486084f, 0.98574567f, 0.98655427f, 0.98729706f, 0.98796856f, 0.9885609f,
    0.9890697f, 0.98949504f, 0.9898407f, 0.9901129f, 0.9903192f, 0.9904674f,
    0.9905651f, 0.98759335f, 0.9844951f, 0.985386f, 0.98612857f, 0.98675996f,
    0.9873164f, 0.9878206f, 0.9882833f, 0.9887071f, 0.98909146f, 0.9894345f,
    0.9897347f, 0.9899917f, 0.990206f, 0.990379f, 0.99051297f, 0.9891902f,
    0.9857565f, 0.9864337f, 0.98700637f, 0.98749065f, 0.98790616f, 0.98827046f,
    0.9885965f, 0.9888927f, 0.98916394f, 0.9894127f, 0.98963994f, 0.9898456f,
    0.99002945f, 0.99019116f, 0.99033046f, 0.99054617f, 0.98662704f,
    0.98715264f, 0.98760283f, 0.98798656f, 0.988314f, 0.9885949f, 0.9888386f,
    0.98905265f, 0.98924327f, 0.9894148f, 0.9895707f, 0.98971295f, 0.98984313f,
    0.9899622f, 0.9900705f, 0.9917789f, 0.9872343f, 0.9876486f, 0.98800564f,
    0.9883129f, 0.98857635f, 0.98880166f, 0.98899394f, 0.9891582f, 0.9892988f,
    0.9894195f, 0.98952377f, 0.98961437f, 0.9896935f, 0.9897631f, 0.98982465f,
    0.9929659f, 0.98766524f, 0.98799646f, 0.9882819f, 0.98852926f, 0.9887432f,
    0.98892754f, 0.9890854f, 0.98921955f, 0.98933244f, 0.98942655f, 0.98950386f,
    0.9895665f, 0.9896162f, 0.9896546f, 0.98968315f, 0.9941599f, 0.98798287f,
    0.9882524f, 0.98848444f, 0.9886867f, 0.9888635f, 0.98901844f, 0.9891536f,
    0.9892711f, 0.98937225f, 0.9894586f, 0.9895312f, 0.9895913f, 0.9896397f,
    0.98967737f, 0.9897051f, 0.99539804f, 0.9882349f, 0.98846066f, 0.98865557f,
    0.9888273f, 0.98898053f, 0.98911816f, 0.98924255f, 0.9893553f, 0.9894579f,
    0.9895513f, 0.98963654f, 0.98971444f, 0.9897857f, 0.9898508f, 0.9899104f};

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
  auto s = uvw.x * (ALBEDO_LUT_SIZE_3D - 1);
  auto t = uvw.y * (ALBEDO_LUT_SIZE_3D - 1);
  auto r = uvw.z * (ALBEDO_LUT_SIZE_3D - 1);

  // get image coordinates and residuals
  auto i = (int)s, j = (int)t, k = (int)r;
  auto ii = min(i + 1, ALBEDO_LUT_SIZE_3D - 1);
  auto jj = min(j + 1, ALBEDO_LUT_SIZE_3D - 1);
  auto kk = min(k + 1, ALBEDO_LUT_SIZE_3D - 1);
  auto u = s - i, v = t - j, w = r - k;

  // trilinear interpolation
  auto size2 = ALBEDO_LUT_SIZE_3D * ALBEDO_LUT_SIZE_3D;

  return lut[k * size2 + j * ALBEDO_LUT_SIZE_3D + i] * (1 - u) * (1 - v) *
             (1 - w) +
         lut[k * size2 + j * ALBEDO_LUT_SIZE_3D + ii] * u * (1 - v) * (1 - w) +
         lut[k * size2 + jj * ALBEDO_LUT_SIZE_3D + i] * (1 - u) * v * (1 - w) +
         lut[kk * size2 + j * ALBEDO_LUT_SIZE_3D + i] * (1 - u) * (1 - v) * w +
         lut[kk * size2 + jj * ALBEDO_LUT_SIZE_3D + i] * (1 - u) * v * w +
         lut[kk * size2 + j * ALBEDO_LUT_SIZE_3D + ii] * u * (1 - v) * w +
         lut[k * size2 + jj * ALBEDO_LUT_SIZE_3D + ii] * u * v * (1 - w) +
         lut[kk * size2 + jj * ALBEDO_LUT_SIZE_3D + ii] * u * v * w;
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
  const auto min_ior = 1.05f, max_ior = 3.0f;
  auto       w = (clamp(ior, min_ior, max_ior) - min_ior) / (max_ior - min_ior);
  auto       E = interpolate3d(
      E_lut, {abs(dot(normal, outgoing)), sqrt(roughness), w});
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
  auto C       = microfacet_compensation_dielectrics(
      my_albedo_lut, ior, roughness, normal, outgoing);
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
  const float coef[39] = {6.49168441e-01, 4.31646014e+02, 6.89941480e+01,
      -2.89728021e+00, -8.85863000e+02, -3.45067726e+02, -9.73426761e+02,
      -1.44659840e+02, 9.87532414e+01, 1.04788185e+02, 4.57721183e+02,
      2.49857079e+02, 1.24012597e+03, 7.98191971e+02, 8.90642220e+02,
      1.96431338e+03, 6.77446120e+01, -4.04008197e+00, -1.47140131e+02,
      -1.48844738e+02, 4.78297749e+02, 4.47409313e+01, -2.80271416e+01,
      -9.25693896e+02, -1.55392769e+02, -1.02689934e+01, 4.36875182e+02,
      7.01043799e+02, 1.04710280e+03, 4.51492795e+02, 9.15631578e+01,
      3.46607424e+02, 1.98820784e+02, 3.11167957e+02, 8.48050302e+02,
      8.31363562e+01, -1.33582515e+01, -1.33062918e+02, 1.47325862e+01};

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
  const float coef[19] = {1.01202782, -11.1084138, 13.68932726, 46.63441392,
      -56.78561075, 17.38577426, -29.2033844, 30.94339247, -5.38305905,
      -4.72530367, -10.45175028, 13.88865122, 43.49596666, -57.01339516,
      16.76996746, -21.80566626, 32.0408972, -5.48296756, -4.29104947};

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
  auto E_lut = dot(normal, outgoing) >= 0 ? entering_albedo_lut
                                          : leaving_albedo_lut;
  auto C     = microfacet_compensation_dielectrics(
      E_lut, ior, roughness, normal, outgoing);
  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_refractive_comp_fit(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const float coef_enter[69] = {9.83933590e-01, 3.90498414e+03, -8.72486343e+00,
      -8.44262322e+00, -1.02227351e+04, -5.74084045e+03, -5.58728703e+03,
      3.53169515e+01, 6.36685254e+01, 3.05724228e+02, 1.03934872e+04,
      8.64301527e+03, 1.12382275e+04, 6.51484898e+03, 2.76520619e+03,
      5.57537687e+03, -3.81696889e+01, -8.86135795e+01, -3.41716141e+02,
      -1.11254382e+02, -4.19835262e+03, -1.92017093e+03, 2.04553129e+03,
      1.80387958e+02, -1.81410546e+04, 5.81546816e+03, -2.90855411e+03,
      8.78662235e+02, -2.77202856e+03, -1.94763771e+03, 1.51079934e+01,
      7.62164777e+01, 2.86452109e+02, -3.53488397e+02, 3.53107703e+02,
      4.12462073e+03, -8.59256233e+00, -9.19455363e+00, -1.08156617e+04,
      -1.79736771e+03, 1.08403774e+03, 3.92656682e+01, 5.70913145e+01,
      3.35849668e+02, 1.22608827e+04, 5.37431377e+03, 1.15026652e+04,
      7.17584269e+03, -2.56728943e+04, 1.04247916e+04, -6.89194626e+01,
      -3.67546518e+00, -3.87852707e+02, -1.16083896e+02, -5.78353184e+03,
      -1.33923156e+03, -2.73370207e+03, -3.63193205e+03, 4.05972633e+03,
      -3.76604430e+03, 1.19532027e+03, 9.98271570e+03, 1.79717164e+03,
      2.78222779e+02, 8.44052147e+01, -1.63144883e+01, 3.04589439e+02,
      -3.85212794e+02, 3.84751122e+02};
  const float coef_leave[69] = {0.97320898, 53.06606246, -3.24631676,
      -3.59422002, 23.81745952, -10.51923984, -60.37521386, 4.57577625,
      9.31940084, 6.83328519, 50.55625703, -40.7239439, -69.67569833,
      11.61909832, 39.60714119, -41.93217033, -3.01484229, -6.36753263,
      -9.77073077, -7.65468753, -14.07442403, -12.6423237, -14.02383757,
      -0.37304838, 53.85715059, 21.47589888, -8.88317186, -2.446353,
      -32.45286435, 53.43699673, 0.74234777, 2.13806453, 3.13196967, 3.36283209,
      3.75950973, 57.26629594, -3.02341533, -3.91647943, 13.71047523,
      2.36806486, -70.34729715, 4.86827902, 8.40164435, 7.58322225, 36.17050003,
      -18.18005998, -42.35287195, -4.05443055, 7.62887172, -31.97718696,
      -2.53328742, -7.67882197, -8.34411194, -8.24246627, 8.60638127,
      -5.14515657, -44.84117084, 3.04177345, 21.37859141, 32.92156629,
      2.32949416, 0.68753223, -9.73938705, 43.47304807, 0.94767435, 1.47916557,
      4.13506016, 2.64199959, 3.8649428};

  auto coef = dot(normal, outgoing) >= 0 ? coef_enter : coef_leave;

  auto alpha        = sqrt(roughness);
  auto cos_theta    = abs(dot(normal, outgoing));
  auto reflectivity = eta_to_reflectivity(ior);

  auto C = 1 / eval_ratpoly3d_deg4(coef, reflectivity, alpha, cos_theta);

  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_refractive_comp_fit_slices(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const float coef_enter[114] = {9.07710791e-01, 1.42075408e+00,
      -1.45786209e+01, 4.69743195e+01, -6.19412155e+01, 2.79934959e+01,
      1.82980049e+00, 2.59453297e+00, -1.05149963e+02, 1.86190460e+02,
      -5.01489983e+01, -4.13181610e+01, 7.40284300e+00, -2.21080227e+01,
      -1.73042725e+02, 7.94921631e+02, -1.13264087e+03, 5.44225769e+02,
      3.48551903e+01, -7.26257248e+01, -5.53227173e+02, 3.28634619e+03,
      -5.20781982e+03, 2.56535229e+03, -1.79841785e+01, 7.41977310e+01,
      9.06797180e+01, -2.04787732e+03, 4.21196387e+03, -2.40200586e+03,
      1.57651062e+01, 5.61661415e+01, -8.77815552e+02, 3.89358984e+03,
      -5.66010840e+03, 2.60350635e+03, -3.11018906e+01, 7.85622482e+01,
      4.82495453e+02, -2.87746460e+03, 4.47520752e+03, -2.16676709e+03,
      3.63712463e+01, -1.07402122e+02, -5.92956482e+02, 3.77607690e+03,
      -5.95365137e+03, 2.89886108e+03, 4.79489756e+00, -2.50027603e+02,
      1.65404956e+03, -5.05982910e+03, 6.13045264e+03, -2.48851367e+03,
      1.99447842e+01, -2.07028687e+02, 5.91448853e+02, -7.36890137e+02,
      4.86685760e+02, -1.63509644e+02, 7.29156315e-01, 2.41908817e+01,
      -2.74124512e+02, 6.32102722e+02, -5.19485535e+02, 1.30387878e+02,
      6.86564970e+00, 2.68972416e+01, -5.24545410e+02, 1.86343286e+03,
      -2.44157617e+03, 1.08732410e+03, 5.68705063e+01, -7.72647934e+01,
      -5.80262695e+02, 3.71888062e+03, -6.23023047e+03, 3.16949927e+03,
      -1.75808029e+01, 8.56333618e+01, -5.00442841e+02, -1.51483582e+03,
      5.06542578e+03, -3.21542065e+03, 4.33624954e+01, 1.74508652e+02,
      -1.50696399e+03, 5.42438916e+03, -7.58501611e+03, 3.48620752e+03,
      -1.50731497e+01, 8.47822037e+01, -2.58671539e+02, -8.61152283e+02,
      2.57546240e+03, -1.56053137e+03, 4.47982101e+01, 1.22427345e+02,
      -1.79097510e+03, 7.34407861e+03, -1.05965908e+04, 4.93869189e+03,
      8.53486347e+00, -5.81029785e+02, 2.50499683e+03, -5.26210693e+03,
      5.17761084e+03, -1.85496191e+03, 3.45546112e+01, -5.16671448e+02,
      2.08265723e+03, -3.82417041e+03, 3.43424829e+03, -1.22252625e+03};
  const float coef_leave[133] = {9.50809777e-01, -8.67406559e+00,
      2.37412376e+01, -1.60442867e+01, -9.02987766e+00, 2.44810867e+01,
      -1.64771290e+01, -2.33038807e+00, 6.19740963e+00, -8.86026669e+00,
      -4.76516914e+00, 4.40512276e+01, -8.40391388e+01, 4.00811310e+01,
      -3.76153183e+00, 1.97191010e+01, -3.84058342e+01, 2.40418415e+01,
      1.96523023e+00, -8.14129448e+00, 5.36209154e+00, 3.02543819e-01,
      -5.55207551e-01, -2.81089768e-02, 3.11515242e-01, -3.27902031e+00,
      3.57679486e+00, -1.29692841e+00, 7.56731939e+00, -7.23894119e+01,
      2.02110275e+02, -1.50658768e+02, 1.85334721e+01, -3.49444847e+01,
      1.57683783e+01, 2.65407276e+00, -1.96176338e+01, 4.02963867e+01,
      -2.41231270e+01, -7.39397430e+00, 1.71820774e+01, -1.08469410e+01,
      -1.73411280e-01, 4.36204880e-01, -2.50476390e-01, -2.90653426e-02,
      -3.26540208e+00, 3.55536866e+00, -1.28923285e+00, -3.55664581e-01,
      1.34198368e+00, -1.68971574e+00, 7.06448555e-01, -3.18606687e+00,
      3.38475370e+00, -1.19853675e+00, -4.32364166e-01, 1.19771135e+00,
      -1.05822229e+00, 2.93167889e-01, -3.18336415e+00, 3.37971282e+00,
      -1.19645643e+00, -6.37053549e-01, 4.56544065e+00, -7.14346743e+00,
      2.24962354e+00, -7.22980404e+00, 1.63656235e+01, -9.94210339e+00,
      -2.15989470e+00, -8.18144650e+06, 8.52646720e+07, -1.27867752e+08,
      8.21938000e+06, 3.20046060e+07, -3.41680680e+07, -4.23290396e+00,
      3.06240978e+01, -6.73964005e+01, 4.43867226e+01, -5.17930698e+00,
      9.04582977e+00, -4.51415730e+00, 2.13509291e-01, -4.27287042e-01,
      4.37079631e-02, 1.95963532e-01, -3.26633620e+00, 3.55265188e+00,
      -1.28554463e+00, 6.07381725e+00, -7.57013016e+01, 1.93767471e+02,
      -1.32739639e+02, 8.93127441e+00, -1.64464188e+01, 6.74361467e+00,
      3.37666082e+00, -2.45707626e+01, 5.04679375e+01, -3.01210804e+01,
      -7.57279348e+00, 1.79949284e+01, -1.14825563e+01, 4.91610527e-01,
      -8.72743607e+00, 3.27090034e+01, -2.91253815e+01, 6.21183891e+01,
      -1.19599045e+02, 5.67490768e+01, -3.33121848e+00, 6.43610153e+01,
      -2.18928452e+02, 1.61815475e+02, 3.23672943e+02, -6.58291443e+02,
      3.33798889e+02, 1.44038334e-01, -7.60130703e-01, 1.14392579e+00,
      -5.27587652e-01, -3.19020438e+00, 3.39498067e+00, -1.20484161e+00,
      -1.05452967e+00, 7.22740269e+00, -1.20993109e+01, 4.87002659e+00,
      -7.45272541e+00, 1.73742752e+01, -1.07235098e+01};

  auto  x = eta_to_reflectivity(ior);
  float coef1[19];

  if (dot(normal, outgoing) >= 0) {
    for (auto i = 0; i < 19; i++) {
      auto j   = i * 6;
      coef1[i] = coef_enter[j] + coef_enter[j + 1] * x +
                 coef_enter[j + 2] * x * x + coef_enter[j + 3] * x * x * x +
                 coef_enter[j + 4] * x * x * x * x +
                 coef_enter[j + 5] * x * x * x * x * x;
    }

  } else {
    auto x2 = x * x, x3 = x * x * x;

    for (auto i = 0; i < 19; i++) {
      auto j   = i * 7;
      coef1[i] = (coef_leave[j] + coef_leave[j + 1] * x +
                     coef_leave[j + 2] * x2 + coef_leave[j + 3] * x3) /
                 (1 + coef_leave[j + 4] * x + coef_leave[j + 5] * x2 +
                     coef_leave[j + 6] * x3);
    }
  }

  auto C = microfacet_compensation_dielectrics_fit2(
      coef1, roughness, normal, outgoing);

  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
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
