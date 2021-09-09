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

inline float eval_ratpoly(const float coef[], float x, float y) {
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

inline vec3f microfacet_compensation_conductors_myfit(const vec3f& color,
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  const float coef[19] = {1.01202782, -11.1084138, 13.68932726, 46.63441392,
      -56.78561075, 17.38577426, -29.2033844, 30.94339247, -5.38305905,
      -4.72530367, -10.45175028, 13.88865122, 43.49596666, -57.01339516,
      16.76996746, -21.80566626, 32.0408972, -5.48296756, -4.29104947};

  auto alpha     = sqrt(roughness);
  auto cos_theta = abs(dot(normal, outgoing));

  auto E = eval_ratpoly(coef, alpha, cos_theta);

  return 1 + color * (1 - E) / E;
}

inline float microfacet_compensation_dielectrics_fit(const float coef[],
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  auto alpha     = sqrt(roughness);
  auto cos_theta = abs(dot(normal, outgoing));

  auto E = eval_ratpoly(coef, alpha, cos_theta);

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

static const auto entering_albedo_lut = vector<float>{0.93672705f, 0.904971f,
    0.8295644f, 0.76742125f, 0.71626127f, 0.6739944f, 0.6389545f, 0.6098208f,
    0.5855398f, 0.5652658f, 0.54831433f, 0.5341282f, 0.522251f, 0.5123071f,
    0.5039857f, 0.49702874f, 0.4912212f, 0.4863833f, 0.48236424f, 0.47903743f,
    0.4762962f, 0.4740505f, 0.47222424f, 0.470753f, 0.46958226f, 0.46866566f,
    0.4679639f, 0.46744353f, 0.46707606f, 0.46683732f, 0.4667067f, 0.46668163f,
    0.93672705f, 0.904971f, 0.8295644f, 0.76742125f, 0.71626127f, 0.6739944f,
    0.6389545f, 0.6098208f, 0.5855398f, 0.5652658f, 0.54831433f, 0.5341282f,
    0.522251f, 0.5123071f, 0.5039857f, 0.49702874f, 0.4912212f, 0.4863833f,
    0.48236424f, 0.47903743f, 0.4762962f, 0.4740505f, 0.47222424f, 0.470753f,
    0.46958226f, 0.46866566f, 0.4679639f, 0.46744353f, 0.46707606f, 0.46683732f,
    0.4667067f, 0.46668163f, 0.90872943f, 0.89318603f, 0.8263718f, 0.7659921f,
    0.71549624f, 0.67354876f, 0.63868314f, 0.6096518f, 0.58543396f, 0.5651999f,
    0.5482742f, 0.5341049f, 0.5222386f, 0.51230174f, 0.50398475f, 0.4970305f,
    0.4912245f, 0.48638728f, 0.48236847f, 0.47904158f, 0.4763f, 0.4740538f,
    0.472227f, 0.47075528f, 0.469584f, 0.4686669f, 0.46796468f, 0.46744385f,
    0.467076f, 0.46683693f, 0.46670598f, 0.46668082f, 0.8369103f, 0.8498626f,
    0.81313175f, 0.76014066f, 0.71238923f, 0.6717377f, 0.63757175f, 0.60895f,
    0.58498466f, 0.5649116f, 0.5480907f, 0.53399044f, 0.52216995f, 0.51226336f,
    0.50396615f, 0.49702457f, 0.49122632f, 0.48639363f, 0.48237705f,
    0.47905082f, 0.47630903f, 0.474062f, 0.47223398f, 0.47076082f, 0.46958807f,
    0.4686695f, 0.46796584f, 0.4674437f, 0.46707466f, 0.46683446f, 0.46670252f,
    0.46667698f, 0.77755356f, 0.7881133f, 0.78291655f, 0.745612f, 0.7045331f,
    0.6671232f, 0.6347165f, 0.6071256f, 0.58379674f, 0.56413114f, 0.5475775f,
    0.53365535f, 0.5219546f, 0.51212865f, 0.5038856f, 0.49698f, 0.49120522f,
    0.48638725f, 0.48237944f, 0.47905785f, 0.4763179f, 0.47407088f, 0.47224173f,
    0.47076666f, 0.46959162f, 0.4686706f, 0.46796447f, 0.46743992f, 0.4670686f,
    0.4668263f, 0.46669236f, 0.46666607f, 0.74003357f, 0.73782194f, 0.7397745f,
    0.719989f, 0.6895163f, 0.6579779f, 0.6289357f, 0.60336757f, 0.5813062f,
    0.5624606f, 0.54645026f, 0.53289413f, 0.5214427f, 0.5117876f, 0.5036618f,
    0.4968363f, 0.49111575f, 0.48633385f, 0.4823494f, 0.47904232f, 0.4763106f,
    0.47406757f, 0.4722395f, 0.47076365f, 0.46958667f, 0.46866298f, 0.46795383f,
    0.46742615f, 0.46705168f, 0.46680635f, 0.46666944f, 0.46664202f, 0.7096609f,
    0.70139325f, 0.6963484f, 0.6866365f, 0.6670623f, 0.6432241f, 0.619175f,
    0.59682035f, 0.57685626f, 0.5594045f, 0.54433626f, 0.5314258f, 0.5204214f,
    0.511078f, 0.50317025f, 0.49649733f, 0.4908831f, 0.48617482f, 0.48224056f,
    0.47896707f, 0.47625712f, 0.47402745f, 0.47220695f, 0.47073463f, 0.4695584f,
    0.46863365f, 0.46792236f, 0.46739188f, 0.46701428f, 0.46676558f,
    0.46662512f, 0.46659628f, 0.6808616f, 0.6718063f, 0.659837f, 0.652305f,
    0.6400819f, 0.62355876f, 0.6052259f, 0.5869925f, 0.5699223f, 0.5544923f,
    0.5408409f, 0.528929f, 0.5186326f, 0.5097936f, 0.5022464f, 0.4958315f,
    0.4904018f, 0.48582503f, 0.481984f, 0.4787759f, 0.4761112f, 0.4739123f,
    0.472112f, 0.47065246f, 0.46948367f, 0.46856275f, 0.46785265f, 0.4673217f,
    0.4669425f, 0.46669137f, 0.4665478f, 0.46651745f, 0.6530197f, 0.6451505f,
    0.63030666f, 0.6216131f, 0.61268413f, 0.60134625f, 0.588125f, 0.5741658f,
    0.5604185f, 0.54748523f, 0.5356808f, 0.5251262f, 0.5158255f, 0.5077166f,
    0.5007051f, 0.49468344f, 0.48954216f, 0.48517662f, 0.48148984f, 0.4783939f,
    0.47581032f, 0.47366947f, 0.47191027f, 0.47047946f, 0.46933028f,
    0.46842235f, 0.46772054f, 0.46719444f, 0.46681756f, 0.46656695f,
    0.46642214f, 0.46639073f, 0.6264247f, 0.6201608f, 0.6055717f, 0.59567046f,
    0.5876566f, 0.5792029f, 0.56968206f, 0.5593819f, 0.54884094f, 0.5385425f,
    0.5288265f, 0.51989317f, 0.5118357f, 0.50467336f, 0.49837947f, 0.4929001f,
    0.48816788f, 0.4841103f, 0.48065484f, 0.47773218f, 0.47527802f, 0.47323346f,
    0.4715456f, 0.4701673f, 0.46905655f, 0.4681765f, 0.4674947f, 0.46698263f,
    0.46661538f, 0.46637085f, 0.46622887f, 0.4661976f, 0.60121435f, 0.596478f,
    0.5837568f, 0.5736884f, 0.56586695f, 0.558795f, 0.55157113f, 0.5439577f,
    0.5360829f, 0.5281987f, 0.52055085f, 0.5133282f, 0.50665367f, 0.5005926f,
    0.49516648f, 0.49036613f, 0.48616245f, 0.48251465f, 0.47937593f, 0.4766975f,
    0.47443116f, 0.47253072f, 0.47095317f, 0.46965924f, 0.4686128f, 0.4677816f,
    0.46713686f, 0.46665272f, 0.4663063f, 0.4660768f, 0.46594453f, 0.46591535f,
    0.57728153f, 0.5738927f, 0.5636731f, 0.5544381f, 0.5469638f, 0.5406386f,
    0.53476554f, 0.52894574f, 0.5230589f, 0.5171498f, 0.51133156f, 0.50572634f,
    0.500436f, 0.49553296f, 0.49105987f, 0.48703456f, 0.48345563f, 0.4803081f,
    0.47756797f, 0.4752059f, 0.47318983f, 0.47148705f, 0.47006524f, 0.46889395f,
    0.4679439f, 0.46718848f, 0.46660322f, 0.46616572f, 0.46585554f, 0.4656537f,
    0.46554112f, 0.46551707f, 0.55436504f, 0.5521558f, 0.544585f, 0.5368478f,
    0.5301448f, 0.5244863f, 0.5195118f, 0.51488525f, 0.51040643f, 0.50600034f,
    0.50167274f, 0.49746957f, 0.49344808f, 0.48966083f, 0.4861483f, 0.48293656f,
    0.48003823f, 0.47745472f, 0.47517854f, 0.47319588f, 0.4714889f, 0.47003698f,
    0.46881828f, 0.4678113f, 0.46699393f, 0.46634567f, 0.46584684f, 0.46547896f,
    0.46522456f, 0.46506667f, 0.46498704f, 0.46497223f, 0.53214264f,
    0.53097683f, 0.525993f, 0.5201123f, 0.51458204f, 0.50975865f, 0.50557554f,
    0.5018439f, 0.49839294f, 0.49511242f, 0.491951f, 0.48889872f, 0.48596975f,
    0.48318815f, 0.4805791f, 0.4781637f, 0.4759568f, 0.47396636f, 0.47219414f,
    0.47063655f, 0.46928594f, 0.46813145f, 0.4671601f, 0.46635842f, 0.46571112f,
    0.4652035f, 0.46482083f, 0.4645487f, 0.46437284f, 0.46427846f, 0.46424797f,
    0.46424755f, 0.51029855f, 0.5100663f, 0.50752574f, 0.5036516f, 0.49958044f,
    0.4958194f, 0.49250168f, 0.48958588f, 0.4869771f, 0.48458573f, 0.4823483f,
    0.48022902f, 0.47821411f, 0.47630388f, 0.4745062f, 0.47283167f, 0.4712903f,
    0.46988988f, 0.4686349f, 0.46752685f, 0.46656403f, 0.46574217f, 0.46505475f,
    0.46449476f, 0.4640524f, 0.4637183f, 0.46348214f, 0.4633334f, 0.46326077f,
    0.46325162f, 0.46328872f, 0.46330866f, 0.48856825f, 0.48917562f,
    0.48890346f, 0.4870562f, 0.48461044f, 0.48210794f, 0.4797935f, 0.47773814f,
    0.47592807f, 0.474319f, 0.4728647f, 0.47152853f, 0.4702869f, 0.46912727f,
    0.46804526f, 0.4670418f, 0.46612033f, 0.4652851f, 0.46453977f, 0.4638868f,
    0.46332717f, 0.46286017f, 0.4624833f, 0.46219403f, 0.46198657f, 0.46185562f,
    0.46179473f, 0.4617965f, 0.46185222f, 0.46195102f, 0.46207544f, 0.46212217f,
    0.46676362f, 0.46812475f, 0.46993214f, 0.47005183f, 0.46929854f, 0.4681875f,
    0.46701068f, 0.46591002f, 0.4649373f, 0.46409687f, 0.46337238f, 0.46274224f,
    0.46218714f, 0.46169278f, 0.46125022f, 0.46085495f, 0.4605056f, 0.46020293f,
    0.45994845f, 0.459744f, 0.4595909f, 0.45948994f, 0.4594406f, 0.45944315f,
    0.45949432f, 0.45959175f, 0.4597316f, 0.45990896f, 0.46011716f, 0.46034628f,
    0.46057788f, 0.46065792f, 0.4447842f, 0.44681606f, 0.4505044f, 0.45247942f,
    0.4534092f, 0.45375475f, 0.4538136f, 0.45376325f, 0.45369953f, 0.4536667f,
    0.4536795f, 0.45373756f, 0.45383415f, 0.45396134f, 0.45411235f, 0.4542824f,
    0.45446873f, 0.45467034f, 0.4548874f, 0.45512074f, 0.45537165f, 0.4556411f,
    0.45592946f, 0.45623857f, 0.4565665f, 0.4569127f, 0.45727512f, 0.45765042f,
    0.45803314f, 0.45841375f, 0.4587717f, 0.45889145f, 0.42261586f, 0.4252354f,
    0.43059784f, 0.4342813f, 0.43682703f, 0.43863243f, 0.43997666f, 0.4410469f,
    0.44196242f, 0.44279498f, 0.44358456f, 0.44435138f, 0.44510412f, 0.4458454f,
    0.44657505f, 0.44729203f, 0.44799554f, 0.44868535f, 0.4493617f, 0.4500256f,
    0.4506782f, 0.45132086f, 0.4519542f, 0.4525811f, 0.45320022f, 0.45381218f,
    0.45441583f, 0.4550088f, 0.4555861f, 0.45613772f, 0.45663998f, 0.45680538f,
    0.4003197f, 0.40344363f, 0.41026694f, 0.41548717f, 0.4195388f, 0.4227551f,
    0.42538482f, 0.42760813f, 0.42955166f, 0.43130165f, 0.43291518f, 0.4344287f,
    0.43586498f, 0.43723777f, 0.43855542f, 0.43982306f, 0.44104397f,
    0.44222075f, 0.44335556f, 0.4444505f, 0.44550765f, 0.4465289f, 0.44751552f,
    0.44847125f, 0.44939503f, 0.45028812f, 0.4511501f, 0.45197883f, 0.4527693f,
    0.45351028f, 0.45417386f, 0.45439053f, 0.3780151f, 0.3815604f, 0.38962856f,
    0.39619792f, 0.401614f, 0.4061507f, 0.41002145f, 0.4133896f, 0.41637784f,
    0.41907662f, 0.42155153f, 0.4238495f, 0.42600366f, 0.42803735f, 0.42996693f,
    0.4318041f, 0.43355745f, 0.43523347f, 0.43683732f, 0.43837333f, 0.43984514f,
    0.44125605f, 0.44260806f, 0.44390607f, 0.4451494f, 0.44633994f, 0.44747746f,
    0.4485601f, 0.4495821f, 0.45053062f, 0.45137218f, 0.45164558f, 0.35585916f,
    0.35974467f, 0.36884347f, 0.37656602f, 0.3831845f, 0.38891992f, 0.39395022f,
    0.39841717f, 0.40243244f, 0.40608308f, 0.40943637f, 0.4125437f, 0.4154443f,
    0.41816804f, 0.4207375f, 0.42317015f, 0.42547944f, 0.42767605f, 0.42976847f,
    0.43176374f, 0.43366778f, 0.43548554f, 0.4372203f, 0.43887833f, 0.44045937f,
    0.4419661f, 0.44339854f, 0.4447546f, 0.44602787f, 0.44720298f, 0.4482401f,
    0.44857594f, 0.33402613f, 0.33817452f, 0.3480969f, 0.35677573f, 0.3644232f,
    0.37121472f, 0.37729502f, 0.38278294f, 0.38777545f, 0.39235142f,
    0.39657456f, 0.40049604f, 0.40415707f, 0.40759057f, 0.41082302f, 0.4138757f,
    0.4167659f, 0.41950777f, 0.42211285f, 0.42459092f, 0.4269501f, 0.42919716f,
    0.43133697f, 0.4333772f, 0.43531826f, 0.43716347f, 0.43891326f, 0.44056523f,
    0.44211185f, 0.44353497f, 0.44478723f, 0.44519204f, 0.31268975f,
    0.31702906f, 0.32757998f, 0.33702374f, 0.3455239f, 0.35321742f, 0.36021936f,
    0.36662638f, 0.37251964f, 0.37796718f, 0.38302594f, 0.38774347f,
    0.39215943f, 0.39630705f, 0.40021387f, 0.40390313f, 0.4073943f, 0.4107037f,
    0.4138452f, 0.41683066f, 0.4196701f, 0.422372f, 0.42494246f, 0.42739066f,
    0.42971766f, 0.43192738f, 0.43402046f, 0.43599415f, 0.4378396f, 0.43953544f,
    0.44102556f, 0.4415068f, 0.29200897f, 0.2964741f, 0.3074746f, 0.317503f,
    0.32668373f, 0.3351219f, 0.34290695f, 0.3501155f, 0.35681316f, 0.36305642f,
    0.36889374f, 0.3743669f, 0.37951186f, 0.38435957f, 0.3889368f, 0.3932665f,
    0.3973687f, 0.40126067f, 0.40495735f, 0.4084718f, 0.41181523f, 0.41499722f,
    0.41802493f, 0.4209086f, 0.4236497f, 0.42625266f, 0.42871812f, 0.43104273f,
    0.43321604f, 0.43521276f, 0.43696707f, 0.43753374f, 0.27211875f,
    0.27665234f, 0.28794253f, 0.29839048f, 0.30808938f, 0.31711835f,
    0.32554558f, 0.33343f, 0.34082323f, 0.34777036f, 0.35431117f, 0.3604805f,
    0.36630937f, 0.37182504f, 0.3770517f, 0.38201076f, 0.38672122f, 0.39119992f,
    0.39546177f, 0.39951992f, 0.4033861f, 0.4070702f, 0.4105798f, 0.41392568f,
    0.41710943f, 0.42013556f, 0.4230044f, 0.42571157f, 0.42824468f, 0.430574f,
    0.43262255f, 0.433285f, 0.253125f, 0.25767773f, 0.26911825f, 0.27983853f,
    0.28990698f, 0.29938197f, 0.30831403f, 0.3167475f, 0.32472163f, 0.33227134f,
    0.33942798f, 0.34621957f, 0.35267156f, 0.3588068f, 0.36464608f, 0.37020808f,
    0.37550983f, 0.3805667f, 0.38539246f, 0.38999966f, 0.3943994f, 0.39860132f,
    0.40261272f, 0.40644428f, 0.41009712f, 0.4135754f, 0.4168787f, 0.42000124f,
    0.422928f, 0.42562416f, 0.4280002f, 0.42877012f, 0.23510337f, 0.239634f,
    0.25110632f, 0.26197088f, 0.27227658f, 0.2820652f, 0.29137328f, 0.30023307f,
    0.30867353f, 0.31672084f, 0.32439882f, 0.33172926f, 0.3387321f, 0.34542572f,
    0.3518271f, 0.35795182f, 0.3638142f, 0.36942756f, 0.37480387f, 0.37995422f,
    0.38488853f, 0.38961545f, 0.39414135f, 0.39847618f, 0.40262017f, 0.4065765f,
    0.4103434f, 0.41391334f, 0.41726798f, 0.42036644f, 0.42310545f, 0.42399555f,
    0.21810046f, 0.2225754f, 0.23398122f, 0.24488138f, 0.2553094f, 0.26529413f,
    0.2748609f, 0.2840324f, 0.29282933f, 0.30127054f, 0.30937365f, 0.31715482f,
    0.32462925f, 0.33181116f, 0.33871377f, 0.34534952f, 0.35173f, 0.35786596f,
    0.36376742f, 0.36944348f, 0.3749025f, 0.38015157f, 0.38519576f, 0.3900437f,
    0.39469436f, 0.39914933f, 0.4034051f, 0.40745163f, 0.4112668f, 0.41480282f,
    0.41794088f, 0.41896448f, 0.20213711f, 0.2065297f, 0.21778962f, 0.22863527f,
    0.23908821f, 0.24916711f, 0.2588887f, 0.26826814f, 0.27731955f, 0.28605613f,
    0.2944903f, 0.30263385f, 0.31049782f, 0.3180929f, 0.32542914f, 0.332516f,
    0.33936247f, 0.34597695f, 0.35236734f, 0.35854074f, 0.36450368f,
    0.37026164f, 0.375818f, 0.38117984f, 0.38634437f, 0.39131147f, 0.39607534f,
    0.40062302f, 0.40492797f, 0.40893468f, 0.41250712f, 0.4136774f, 0.1872123f,
    0.19150218f, 0.20255373f, 0.213272f, 0.22366914f, 0.23375599f, 0.24354257f,
    0.25303835f, 0.26225242f, 0.27119356f, 0.2798703f, 0.2882908f, 0.29646316f,
    0.30439502f, 0.31209388f, 0.31956682f, 0.3268206f, 0.33386168f, 0.34069592f,
    0.3473287f, 0.35376486f, 0.36000824f, 0.36606058f, 0.37192723f, 0.37760374f,
    0.38308796f, 0.38837165f, 0.39343858f, 0.39825737f, 0.40276393f,
    0.40680355f, 0.40813315f, 0.17330758f, 0.17747974f, 0.18827543f, 0.1988089f,
    0.20908496f, 0.21910837f, 0.22888413f, 0.23841733f, 0.24771334f,
    0.25677767f, 0.2656159f, 0.27423373f, 0.28263682f, 0.29083073f, 0.29882103f,
    0.306613f, 0.3142118f, 0.32162222f, 0.32884875f, 0.33589536f, 0.34276554f,
    0.34946167f, 0.3559842f, 0.362337f, 0.36851397f, 0.37451103f, 0.3803176f,
    0.3859141f, 0.39126396f, 0.39629403f, 0.40082946f, 0.4023299f};

static const auto leaving_albedo_lut = vector<float>{0.99790466f, 0.99922127f,
    0.99980986f, 0.999916f, 0.99995303f, 0.99997026f, 0.9999797f, 0.9999854f,
    0.99998915f, 0.99999183f, 0.9999938f, 0.9999954f, 0.9999967f, 0.9999978f,
    0.9999989f, 1.0000001f, 1.0000014f, 1.0000031f, 1.0000055f, 1.0000094f,
    1.000017f, 1.000036f, 1.0001148f, 1.0059679f, 2.0062819f, 2.121196f,
    2.1636496f, 2.1831458f, 2.1927571f, 2.1974478f, 2.1994772f, 2.19981f,
    0.99790466f, 0.99922127f, 0.99980986f, 0.999916f, 0.99995303f, 0.99997026f,
    0.9999797f, 0.9999854f, 0.99998915f, 0.99999183f, 0.9999938f, 0.9999954f,
    0.9999967f, 0.9999978f, 0.9999989f, 1.0000001f, 1.0000014f, 1.0000031f,
    1.0000055f, 1.0000094f, 1.000017f, 1.000036f, 1.0001148f, 1.0059679f,
    2.0062819f, 2.121196f, 2.1636496f, 2.1831458f, 2.1927571f, 2.1974478f,
    2.1994772f, 2.19981f, 0.9740822f, 0.98994994f, 0.99763244f, 0.998979f,
    0.99943644f, 0.99964553f, 0.9997587f, 0.9998272f, 0.99987227f, 0.99990386f,
    0.9999273f, 0.9999456f, 0.9999607f, 0.9999742f, 0.999987f, 1.0000004f,
    1.0000156f, 1.000035f, 1.000063f, 1.0001084f, 1.0001957f, 1.0004127f,
    1.0013168f, 1.0533134f, 2.002226f, 2.1201572f, 2.163218f, 2.1829228f,
    2.1926265f, 2.1973634f, 2.1994185f, 2.1997583f, 0.9151202f, 0.953638f,
    0.9870774f, 0.99441767f, 0.99695224f, 0.99810094f, 0.99871624f, 0.99908537f,
    0.99932575f, 0.9994931f, 0.9996164f, 0.9997123f, 0.9997913f, 0.99986076f,
    0.9999265f, 0.99999446f, 1.000072f, 1.0001704f, 1.0003107f, 1.0005388f,
    1.0009772f, 1.0020609f, 1.0065192f, 1.1477371f, 1.9883618f, 2.116238f,
    2.1615157f, 2.1820166f, 2.1920793f, 2.1970024f, 2.1991627f, 2.19953f,
    0.88107437f, 0.90667444f, 0.9613529f, 0.9816899f, 0.9897871f, 0.99361646f,
    0.9956923f, 0.9969384f, 0.99774754f, 0.9983078f, 0.9987181f, 0.9990352f,
    0.99929446f, 0.9995205f, 0.99973255f, 0.9999498f, 1.0001954f, 1.0005052f,
    1.0009444f, 1.0016555f, 1.0030148f, 1.0063455f, 1.0195826f, 1.2344664f,
    1.9603385f, 2.1071508f, 2.157374f, 2.1797438f, 2.1906755f, 2.1960578f,
    2.1984816f, 2.198918f, 0.87651396f, 0.88029265f, 0.9260223f, 0.95826846f,
    0.97498405f, 0.98387706f, 0.988972f, 0.99211293f, 0.9941778f, 0.9956141f,
    0.9966661f, 0.99747664f, 0.9981358f, 0.9987058f, 0.9992358f, 0.9997731f,
    1.0003742f, 1.0011251f, 1.0021813f, 1.0038785f, 1.0070926f, 1.0148237f,
    1.0435967f, 1.3004992f, 1.9185156f, 2.090698f, 2.1494236f, 2.1752398f,
    2.1878307f, 2.1941092f, 2.1970563f, 2.1976323f, 0.87989175f, 0.8727413f,
    0.89560664f, 0.9287436f, 0.9524009f, 0.96742034f, 0.9769434f, 0.9831749f,
    0.98742026f, 0.99043626f, 0.99267167f, 0.9944035f, 0.9958126f, 0.9970266f,
    0.99814767f, 0.99927294f, 1.0005183f, 1.0020566f, 1.0041974f, 1.007599f,
    1.0139428f, 1.0287358f, 1.0785493f, 1.3494244f, 1.8686143f, 2.065489f,
    2.1362996f, 2.167526f, 2.1828425f, 2.1906338f, 2.1944802f, 2.195298f,
    0.8817869f, 0.8720885f, 0.87662786f, 0.9014121f, 0.92617995f, 0.9454363f,
    0.95936686f, 0.96931815f, 0.9765102f, 0.98182833f, 0.98587734f, 0.9890692f,
    0.9916923f, 0.9939609f, 0.9960524f, 0.9981391f, 1.000427f, 1.0032206f,
    1.0070591f, 1.0130664f, 1.0240198f, 1.0484279f, 1.1207298f, 1.3854628f,
    1.8179488f, 2.031527f, 2.1168866f, 2.1555989f, 2.174923f, 2.1850157f,
    2.1902611f, 2.1914601f, 0.88015664f, 0.87143815f, 0.866527f, 0.8809583f,
    0.9016999f, 0.92151695f, 0.9380563f, 0.9511582f, 0.96137327f, 0.969361f,
    0.9756985f, 0.98084646f, 0.98516667f, 0.98895305f, 0.9924658f, 0.99597025f,
    0.9997915f, 1.0044109f, 1.0106707f, 1.0202818f, 1.0372849f, 1.0729928f,
    1.1646347f, 1.4115102f, 1.771322f, 1.9903057f, 2.0905626f, 2.138542f,
    2.1632452f, 2.1765666f, 2.1838295f, 2.1855857f, 0.87506396f, 0.8682963f,
    0.8604949f, 0.86709416f, 0.88194424f, 0.8992996f, 0.9159683f, 0.93066126f,
    0.94311184f, 0.95350695f, 0.9621917f, 0.96953756f, 0.97589725f, 0.9815999f,
    0.9869706f, 0.99236834f, 0.99825394f, 1.0053191f, 1.0147613f, 1.0289378f,
    1.053096f, 1.1003836f, 1.2054003f, 1.4293896f, 1.730131f, 1.944244f,
    2.0573568f, 2.1156507f, 2.147007f, 2.1645532f, 2.1745505f, 2.1770759f,
    0.8669339f, 0.86209416f, 0.85493374f, 0.85722387f, 0.86692524f, 0.8806122f,
    0.8955636f, 0.9101435f, 0.9235606f, 0.9355545f, 0.94616026f, 0.9555637f,
    0.9640252f, 0.97184974f, 0.9793901f, 0.987081f, 0.9955176f, 1.0056131f,
    1.0189317f, 1.0384369f, 1.0702703f, 1.127968f, 1.2399074f, 1.4402184f,
    1.6936255f, 1.8957889f, 2.0179255f, 2.0865362f, 2.1255028f, 2.1482384f,
    2.1617444f, 2.1652792f, 0.8561458f, 0.8529029f, 0.84784836f, 0.84865963f,
    0.8550275f, 0.86536515f, 0.8778982f, 0.8912303f, 0.9044522f, 0.9170659f,
    0.92887086f, 0.9398674f, 0.9501931f, 0.9600905f, 0.96990675f, 0.98012805f,
    0.9914673f, 1.0050429f, 1.0227451f, 1.047994f, 1.0873127f, 1.1532036f,
    1.2666508f, 1.4446831f, 1.6602514f, 1.8467264f, 1.9733459f, 2.051176f,
    2.098197f, 2.1269312f, 2.1447167f, 2.1495173f, 0.8429806f, 0.8409515f,
    0.8383361f, 0.83945006f, 0.8443117f, 0.85237855f, 0.86273706f, 0.87446326f,
    0.8868143f, 0.8992804f, 0.91157025f, 0.9235761f, 0.93534374f, 0.94705737f,
    0.9590487f, 0.97183925f, 0.9862372f, 1.0035268f, 1.0258312f, 1.0567902f,
    1.1027306f, 1.1741309f, 1.2852144f, 1.4432418f, 1.628414f, 1.7979748f,
    1.9248224f, 2.009896f, 2.0647848f, 2.1000426f, 2.122796f, 2.129117f,
    0.82764316f, 0.8264778f, 0.8260952f, 0.82842505f, 0.83317333f, 0.84019077f,
    0.84912467f, 0.8595009f, 0.8708538f, 0.8828069f, 0.89510787f, 0.90763855f,
    0.92041737f, 0.93360734f, 0.94754076f, 0.9627726f, 0.9801842f, 1.001169f,
    1.0279545f, 1.0641251f, 1.1153165f, 1.18957f, 1.2958304f, 1.4362788f,
    1.5967993f, 1.7497563f, 1.8734398f, 1.963288f, 2.0252242f, 2.0671437f,
    2.0953815f, 2.1034493f, 0.810292f, 0.80969465f, 0.81112087f, 0.8149976f,
    0.82051814f, 0.82752293f, 0.835916f, 0.845535f, 0.8561764f, 0.8676479f,
    0.87981164f, 0.892614f, 0.9061092f, 0.9204854f, 0.9361025f, 0.9535557f,
    0.97377723f, 0.9982028f, 1.0290228f, 1.0695138f, 1.1243092f, 1.1990858f,
    1.2991105f, 1.4242096f, 1.5644732f, 1.7019038f, 1.8200417f, 1.9121021f,
    1.9797393f, 2.0280113f, 2.0619907f, 2.0719757f, 0.7910634f, 0.7907923f,
    0.7935477f, 0.79895854f, 0.80571586f, 0.8134506f, 0.8220993f, 0.8316546f,
    0.8420982f, 0.85340583f, 0.8655749f, 0.8786534f, 0.892772f, 0.9081785f,
    0.9252833f, 0.9447238f, 0.96745706f, 0.99489033f, 1.0290458f, 1.072712f,
    1.1294143f, 1.2028279f, 1.2958899f, 1.4075389f, 1.5308853f, 1.6541336f,
    1.765233f, 1.8571489f, 1.9287903f, 1.9826607f, 2.0223079f, 2.034293f,
    0.7700875f, 0.76994896f, 0.7735751f, 0.7803286f, 0.78849006f, 0.7974136f,
    0.8069345f, 0.8170715f, 0.82790077f, 0.8395166f, 0.85203314f, 0.8656032f,
    0.88044655f, 0.89688414f, 0.9153833f, 0.9366144f, 0.9615251f, 0.9914264f,
    1.0280732f, 1.0736774f, 1.1307182f, 1.2013321f, 1.2871206f, 1.3868685f,
    1.4958231f, 1.6062214f, 1.7094527f, 1.7992308f, 1.8730248f, 1.9313524f,
    1.9762214f, 1.9901762f, 0.7474991f, 0.7473409f, 0.75143445f, 0.75926495f,
    0.76881206f, 0.77915895f, 0.789976f, 0.8012219f, 0.81298816f, 0.82542837f,
    0.8387358f, 0.8531468f, 0.8689587f, 0.8865581f, 0.9064569f, 0.9293357f,
    0.95609254f, 0.98788536f, 1.0261447f, 1.0725033f, 1.1285541f, 1.1953349f,
    1.2737844f, 1.3628694f, 1.459338f, 1.5580782f, 1.6530546f, 1.7391096f,
    1.8132174f, 1.8745753f, 1.9238479f, 1.9396092f, 0.723445f, 0.7231503f,
    0.7273758f, 0.73600596f, 0.7468196f, 0.758668f, 0.77104014f, 0.7837865f,
    0.7969558f, 0.8107071f, 0.8252685f, 0.8409253f, 0.8580254f, 0.87699574f,
    0.89836603f, 0.922794f, 0.9510884f, 0.9842151f, 1.0232652f, 1.0693507f,
    1.123374f, 1.1856292f, 1.2568214f, 1.3362324f, 1.4216613f, 1.509757f,
    1.5963633f, 1.6774912f, 1.7502097f, 1.8130058f, 1.8655329f, 1.8827996f,
    0.69808704f, 0.6975695f, 0.7016622f, 0.7108384f, 0.7227578f, 0.7360897f,
    0.7501515f, 0.76466537f, 0.7796004f, 0.795078f, 0.81131893f, 0.82861835f,
    0.84733844f, 0.8679119f, 0.89085054f, 0.9167536f, 0.9463073f, 0.98026717f,
    1.0194048f, 1.0644004f, 1.1156515f, 1.1729658f, 1.2370775f, 1.3076179f,
    1.3831222f, 1.4614177f, 1.5396988f, 1.61502f, 1.6848549f, 1.7474543f,
    1.8018334f, 1.8201768f, 0.6716033f, 0.6708032f, 0.67456675f, 0.6840775f,
    0.6969388f, 0.7116858f, 0.7274871f, 0.74393386f, 0.7608954f, 0.7784244f,
    0.79669803f, 0.81598306f, 0.8366177f, 0.8590026f, 0.88359547f, 0.91090417f,
    0.9414716f, 0.97584367f, 1.0145133f, 1.0578257f, 1.1058248f, 1.1580015f,
    1.2152743f, 1.277617f, 1.3440785f, 1.4132799f, 1.4833726f, 1.5522723f,
    1.6179723f, 1.6788026f, 1.7334794f, 1.7523687f, 0.64418715f, 0.6430682f,
    0.64636964f, 0.65605366f, 0.6697122f, 0.6857876f, 0.70332545f, 0.7217932f,
    0.7409511f, 0.76076245f, 0.7813317f, 0.80286396f, 0.82563806f, 0.84998727f,
    0.87628376f, 0.90492076f, 0.93628895f, 0.9707411f, 1.0085397f, 1.049783f,
    1.0942717f, 1.1412798f, 1.1920007f, 1.2467276f, 1.3048677f, 1.3655753f,
    1.4276732f, 1.4897487f, 1.5503122f, 1.6079454f, 1.6613252f, 1.6801643f,
    0.6160447f, 0.6145907f, 0.6173546f, 0.6271017f, 0.6414436f, 0.65876245f,
    0.67800426f, 0.698524f, 0.71997094f, 0.7422053f, 0.7652377f, 0.7891846f,
    0.8142364f, 0.8406315f, 0.8686335f, 0.8985084f, 0.9304975f, 0.9647844f,
    1.0014505f, 1.0404131f, 1.0813057f, 1.1232337f, 1.1677185f, 1.2153487f,
    1.2657773f, 1.318516f, 1.3728495f, 1.427868f, 1.4825325f, 1.5357405f,
    1.5862939f, 1.6044663f, 0.5873906f, 0.58560306f, 0.5878041f, 0.59755236f,
    0.6124983f, 0.63099074f, 0.6518861f, 0.6744474f, 0.69821125f, 0.72292745f,
    0.74849784f, 0.77493155f, 0.8023101f, 0.8307585f, 0.86041963f, 0.89142966f,
    0.92389166f, 0.95784664f, 0.99323606f, 1.0298455f, 1.0671827f, 1.1042f,
    1.1427798f, 1.1837842f, 1.2270329f, 1.2722726f, 1.3190997f, 1.366964f,
    1.4151871f, 1.4629728f, 1.5093247f, 1.5262375f, 0.55844396f, 0.55633885f,
    0.55799437f, 0.5677241f, 0.58322805f, 0.6028305f, 0.6253319f, 0.6498934f,
    0.6759485f, 0.70313317f, 0.7312325f, 0.7601372f, 0.78980887f, 0.8202509f,
    0.851483f, 0.8835171f, 0.916334f, 0.9498571f, 0.98391855f, 1.0182034f,
    1.0521121f, 1.0844378f, 1.1174451f, 1.1522553f, 1.1887971f, 1.2269676f,
    1.2665662f, 1.307289f, 1.3487239f, 1.3903311f, 1.4313276f, 1.4464515f,
    0.5294225f, 0.52702796f, 0.52818996f, 0.53791684f, 0.5539611f, 0.5746347f,
    0.59868145f, 0.6251776f, 0.6534537f, 0.68303233f, 0.7135794f, 0.744864f,
    0.77672565f, 0.809047f, 0.84173006f, 0.8746742f, 0.90775466f, 0.9407982f,
    0.9735489f, 1.0056067f, 1.0362681f, 1.064146f, 1.0919044f, 1.1209158f,
    1.1511769f, 1.1826761f, 1.2153393f, 1.2490207f, 1.2834903f, 1.3184f,
    1.3531483f, 1.3660486f, 0.5005381f, 0.49789128f, 0.49863836f, 0.5084051f,
    0.52499443f, 0.5467089f, 0.5722406f, 0.6005858f, 0.6309761f, 0.66282386f,
    0.69567895f, 0.72919226f, 0.76308703f, 0.7971332f, 0.83112663f, 0.86486816f,
    0.8981436f, 0.93069875f, 0.962202f, 0.99217314f, 1.0197976f, 1.043478f,
    1.066294f, 1.0898671f, 1.1142336f, 1.139433f, 1.1654637f, 1.1922736f,
    1.2197461f, 1.2476598f, 1.2755456f, 1.285901f, 0.47199088f, 0.46913585f,
    0.4695656f, 0.479434f, 0.49658865f, 0.5193209f, 0.5462731f, 0.5763645f,
    0.60873246f, 0.64268523f, 0.67766404f, 0.7132113f, 0.7489449f, 0.78453565f,
    0.8196876f, 0.8541184f, 0.8875384f, 0.919623f, 0.9499707f, 0.97801876f,
    1.0028267f, 1.0225539f, 1.0407112f, 1.0591726f, 1.0779946f, 1.0972414f,
    1.1169479f, 1.1371112f, 1.1576761f, 1.178494f, 1.199178f, 1.2067889f,
    0.44396588f, 0.44095066f, 0.4411725f, 0.45121616f, 0.46896482f, 0.49269468f,
    0.5209971f, 0.5527169f, 0.58690286f, 0.62276757f, 0.65965486f, 0.69701236f,
    0.734368f, 0.77130973f, 0.80746555f, 0.8424841f, 0.87601155f, 0.9076601f,
    0.93695885f, 0.9632556f, 0.98546594f, 1.0014696f, 1.0152258f, 1.0288687f,
    1.0424631f, 1.0560825f, 1.0697753f, 1.0835595f, 1.0974064f, 1.1112012f,
    1.1245997f, 1.1293863f, 0.41662884f, 0.4135034f, 0.41363204f, 0.4239295f,
    0.44230378f, 0.46700937f, 0.49658477f, 0.5298026f, 0.56562984f, 0.60319424f,
    0.64175624f, 0.680685f, 0.7194361f, 0.7575314f, 0.7945391f, 0.8300512f,
    0.86365855f, 0.89491403f, 0.92327505f, 0.94799054f, 0.96781236f, 0.9803036f,
    0.9898883f, 0.99897414f, 1.007628f, 1.0159241f, 1.0239133f, 1.0316182f,
    1.0390178f, 1.0460081f, 1.0522631f, 1.0542551f, 0.390124f, 0.3869382f,
    0.38708803f, 0.39771724f, 0.41674662f, 0.4424014f, 0.47316462f, 0.50773996f,
    0.54502046f, 0.5840616f, 0.6240566f, 0.66431344f, 0.7042334f, 0.7432891f,
    0.7810029f, 0.81692153f, 0.8505872f, 0.88149524f, 0.90902764f, 0.9323238f,
    0.94995207f, 0.9591205f, 0.96473694f, 0.96949756f, 0.9734701f, 0.9767279f,
    0.97932094f, 0.9812713f, 0.9825587f, 0.9830817f, 0.9825247f, 0.9818455f,
    0.36457226f, 0.36137426f, 0.361655f, 0.37268886f, 0.39245152f, 0.4191091f,
    0.45082557f, 0.48661023f, 0.5251502f, 0.56544226f, 0.606629f, 0.64797527f,
    0.68884474f, 0.72867644f, 0.76695937f, 0.8032043f, 0.83690983f, 0.8675145f,
    0.8943207f, 0.91634786f, 0.9319608f, 0.9379756f, 0.93980134f, 0.94044167f,
    0.93996775f, 0.93845516f, 0.9359559f, 0.93249476f, 0.928055f, 0.9225413f,
    0.9156539f, 0.91250205f};

#define ALBEDO_LUT_SIZE_3D 16

static const auto glossy_albedo_lut = vector<float>{0.66503555f, 0.26555854f,
    0.0787219f, 0.02709442f, 0.010713238f, 0.004737806f, 0.0022866204f,
    0.001184202f, 0.00065283355f, 0.00038371317f, 0.00024307152f,
    0.00016865747f, 0.00012974475f, 0.00011041596f, 0.00010209243f,
    9.579773e-10f, 0.6311485f, 0.26337856f, 0.07870741f, 0.027144933f,
    0.010737951f, 0.0047485465f, 0.0022914112f, 0.0011864464f, 0.00065393525f,
    0.000384274f, 0.0002433637f, 0.00016881048f, 0.00012982321f,
    0.000110453475f, 0.00010210708f, 0.00010000208f, 0.44742906f, 0.23657532f,
    0.07715673f, 0.027348982f, 0.0109188445f, 0.00484111f, 0.002336362f,
    0.0012086927f, 0.00066529616f, 0.00039023944f, 0.00024655167f,
    0.00017051795f, 0.00013071789f, 0.00011089163f, 0.00010228492f,
    0.00010003055f, 0.26783898f, 0.17179039f, 0.068505526f, 0.026586888f,
    0.011060841f, 0.004997042f, 0.0024317903f, 0.0012620334f, 0.0006947457f,
    0.0004065875f, 0.00025567022f, 0.0001755763f, 0.00013345214f,
    0.000112273556f, 0.00010287098f, 0.0001001461f, 0.15164872f, 0.10750926f,
    0.05216495f, 0.023283659f, 0.01054346f, 0.0050059543f, 0.002508268f,
    0.0013241861f, 0.00073579623f, 0.0004320235f, 0.0002709832f, 0.00018457537f,
    0.00013855175f, 0.000114965595f, 0.00010407447f, 0.00010043039f,
    0.08676576f, 0.06431284f, 0.0355712f, 0.018108854f, 0.009093119f,
    0.004651504f, 0.0024559842f, 0.001344898f, 0.00076628744f, 0.00045703066f,
    0.00028857024f, 0.00019603247f, 0.00014556013f, 0.0001189091f,
    0.00010596002f, 0.00010095664f, 0.05114957f, 0.03871039f, 0.023180276f,
    0.013013828f, 0.007169316f, 0.003968187f, 0.0022335353f, 0.0012868572f,
    0.0007629368f, 0.00046876146f, 0.00030180643f, 0.00020673028f,
    0.00015303661f, 0.00012354946f, 0.000108387525f, 0.000101756625f,
    0.031270478f, 0.023924576f, 0.015045686f, 0.0090369405f, 0.005349862f,
    0.0031721464f, 0.0019004294f, 0.0011568152f, 0.00071917084f, 0.00045975237f,
    0.00030526458f, 0.00021333418f, 0.00015916064f, 0.00012801932f,
    0.000111039226f, 0.00010280099f, 0.019871898f, 0.015297756f, 0.009930996f,
    0.0062496117f, 0.0039025138f, 0.0024453008f, 0.0015466391f, 0.0009914367f,
    0.00064681075f, 0.000431952f, 0.00029774816f, 0.00021415425f, 0.0001626076f,
    0.00013151098f, 0.000113536895f, 0.00010400573f, 0.013131703f, 0.01014376f,
    0.00672951f, 0.0043782596f, 0.0028450969f, 0.00186181f, 0.001231905f,
    0.00082647894f, 0.0005639829f, 0.0003932326f, 0.0002819634f, 0.0002096427f,
    0.00016305404f, 0.00013359025f, 0.000115583265f, 0.00010526231f,
    0.009016916f, 0.0069755893f, 0.0046998695f, 0.0031348453f, 0.002100808f,
    0.0014233284f, 0.0009777073f, 0.00068222894f, 0.00048468987f, 0.0003517844f,
    0.0002620731f, 0.00020157415f, 0.00016103948f, 0.00013426684f,
    0.00011704293f, 0.000106472544f, 0.006424348f, 0.0049694953f, 0.0033868484f,
    0.002303764f, 0.0015828403f, 0.0011037146f, 0.0007826268f, 0.0005650414f,
    0.00041602054f, 0.00031309226f, 0.00024162044f, 0.00019192112f,
    0.00015748055f, 0.00013384716f, 0.00011793358f, 0.00010756982f,
    0.0047404673f, 0.0036616216f, 0.0025175377f, 0.0017404939f, 0.0012214391f,
    0.00087312f, 0.00063651294f, 0.00047354752f, 0.0003598553f, 0.00027970265f,
    0.00022277319f, 0.00018218094f, 0.00015324588f, 0.00013273173f,
    0.000118364216f, 0.000108521635f, 0.003614894f, 0.0027847963f,
    0.0019280947f, 0.0013519225f, 0.00096672866f, 0.00070653285f,
    0.00052799314f, 0.0004034708f, 0.00031532976f, 0.00025216796f,
    0.00020647942f, 0.00017322553f, 0.0001489582f, 0.00013127436f,
    0.00011847064f, 0.00010932105f, 0.0028420347f, 0.0021812462f, 0.0015189047f,
    0.0010786579f, 0.00078468584f, 0.0005852329f, 0.00044730966f,
    0.00035014874f, 0.00028056122f, 0.00023002268f, 0.00019290879f,
    0.00016542934f, 0.0001449778f, 0.00012972728f, 0.00011837476f,
    0.00010997603f, 0.0022979768f, 0.0017554038f, 0.0012283265f, 0.00088266993f,
    0.00065249257f, 0.0004958803f, 0.00038692015f, 0.0003095274f,
    0.00025354757f, 0.00021242771f, 0.00018183955f, 0.00015885878f,
    0.00014146687f, 0.00012824284f, 0.000118168355f, 0.0001105016f, 0.8931297f,
    0.6961296f, 0.49801707f, 0.3646168f, 0.27285516f, 0.20882602f, 0.16376905f,
    0.13196154f, 0.10955256f, 0.09388782f, 0.08310225f, 0.07586512f,
    0.07121549f, 0.06845292f, 0.067063175f, 0.06666678f, 0.8703438f,
    0.69324934f, 0.4971507f, 0.3643021f, 0.27274188f, 0.20879441f, 0.16377103f,
    0.13197613f, 0.10957028f, 0.09390447f, 0.08311614f, 0.07587577f,
    0.07122298f, 0.0684576f, 0.06706548f, 0.06666712f, 0.75975543f, 0.6605459f,
    0.486962f, 0.3603748f, 0.27119678f, 0.20826198f, 0.16368131f, 0.13207103f,
    0.109728254f, 0.09406822f, 0.08325999f, 0.07598994f, 0.07130565f,
    0.068510704f, 0.06709262f, 0.066672385f, 0.62532073f, 0.5746478f,
    0.45381767f, 0.34649155f, 0.26521575f, 0.20583072f, 0.1629031f, 0.13205066f,
    0.11002456f, 0.094460845f, 0.0836411f, 0.07631047f, 0.071547486f,
    0.06867178f, 0.067178756f, 0.066692844f, 0.4932404f, 0.46830496f,
    0.39597803f, 0.31769606f, 0.2512302f, 0.19931436f, 0.16017649f, 0.13122611f,
    0.110110156f, 0.09492936f, 0.08421415f, 0.07684604f, 0.07197903f,
    0.06897446f, 0.06735024f, 0.06674237f, 0.38917494f, 0.37364212f,
    0.33020684f, 0.27788657f, 0.22877339f, 0.18732287f, 0.15422082f,
    0.12864576f, 0.109343864f, 0.0950707f, 0.084743224f, 0.07747136f,
    0.07254425f, 0.0694029f, 0.06761173f, 0.06683313f, 0.31149372f, 0.29947045f,
    0.2711983f, 0.23651543f, 0.20203227f, 0.17110123f, 0.14499591f, 0.12383732f,
    0.107196406f, 0.09443823f, 0.08489894f, 0.077966616f, 0.07311128f,
    0.06988989f, 0.06794001f, 0.06696943f, 0.25432065f, 0.24410011f,
    0.22401407f, 0.20018722f, 0.17605567f, 0.15363325f, 0.13392755f,
    0.117289945f, 0.10367981f, 0.09284753f, 0.08445159f, 0.07812816f,
    0.07352915f, 0.070340574f, 0.06828941f, 0.06714391f, 0.21236557f,
    0.2034352f, 0.18819343f, 0.17102359f, 0.15373583f, 0.13745083f,
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
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  return 1 /
         interpolate2d(E_lut, {abs(dot(normal, outgoing)), sqrt(roughness)});
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
  auto up_normal    = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto reflectivity = eta_to_reflectivity({ior, ior, ior}).x;
  auto E1           = interpolate3d(glossy_albedo_lut,
      {abs(dot(up_normal, outgoing)), sqrt(roughness), reflectivity});
  auto halfway      = normalize(incoming + outgoing);
  auto C            = microfacet_compensation_dielectrics(
      my_albedo_lut, roughness, normal, outgoing);
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
  const float coef[19] = {1.01202782, -11.1084138, 13.68932726, 46.63441392,
      -56.78561075, 17.38577426, -29.2033844, 30.94339247, -5.38305905,
      -4.72530367, -10.45175028, 13.88865122, 43.49596666, -57.01339516,
      16.76996746, -21.80566626, 32.0408972, -5.48296756, -4.29104947};

  const float coef1[39] = {6.49168441e-01, 4.31646014e+02, 6.89941480e+01,
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
  auto up_normal    = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto reflectivity = eta_to_reflectivity({ior, ior, ior}).x;
  auto E1           = eval_ratpoly3d(
      coef1, reflectivity, sqrt(roughness), abs(dot(up_normal, outgoing)));
  auto halfway = normalize(incoming + outgoing);
  auto C       = microfacet_compensation_dielectrics_fit(
      coef, roughness, normal, outgoing);
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
  auto C = microfacet_compensation_dielectrics(
      my_albedo_lut, roughness, normal, outgoing);
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

  auto C = microfacet_compensation_dielectrics_fit(
      coef, roughness, normal, outgoing);
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
      E_lut, roughness, normal, outgoing);
  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

inline vec3f eval_refractive_comp_fit(const vec3f& color, float ior,
    float roughness, const vec3f& normal, const vec3f& outgoing,
    const vec3f& incoming) {
  const float coef_enter[19] = {0.93035746, 5.1926885, 9.905641, 60.261677,
      -25.489843, 27.794697, -53.568184, 60.657623, 6.1536565, 25.308775,
      3.9050202, 10.435972, 100.21542, -23.653866, 82.28039, -21.71348,
      90.937195, 3.4975662, 43.027836};
  const float coef_leave[19] = {0.97565085, -0.85566753, -2.6745534, 0.10666856,
      2.3761475, 1.8876839, -0.02559962, -0.04949585, -1.6205686, -0.06074347,
      -0.45433855, -3.0476403, 0.00326211, 1.2840537, 2.8514233, 0.02940547,
      0.026335, -0.88503194, -0.74383014};

  auto coef = dot(normal, outgoing) >= 0 ? coef_enter : coef_leave;
  auto C    = microfacet_compensation_dielectrics_fit(
      coef, roughness, normal, outgoing);
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
