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

static const auto entering_albedo_lut = vector<float>{0.9377883f, 0.9054122f,
    0.82969075f, 0.767479f, 0.7162924f, 0.6740125f, 0.6389655f, 0.6098274f,
    0.585544f, 0.5652683f, 0.54831576f, 0.53412896f, 0.52225137f, 0.51230717f,
    0.5039856f, 0.49702856f, 0.49122098f, 0.48638302f, 0.48236403f, 0.47903723f,
    0.476296f, 0.4740503f, 0.47222406f, 0.4707529f, 0.46958217f, 0.4686656f,
    0.46796384f, 0.46744347f, 0.46707606f, 0.46683732f, 0.4667067f,
    -2.0674025e-07f, 0.9374957f, 0.9052899f, 0.8296555f, 0.76746285f,
    0.71628374f, 0.6740075f, 0.6389624f, 0.6098256f, 0.5855428f, 0.56526756f,
    0.5483154f, 0.5341288f, 0.5222513f, 0.51230717f, 0.50398564f, 0.49702862f,
    0.49122104f, 0.4863831f, 0.4823641f, 0.47903728f, 0.47629607f, 0.47405037f,
    0.47222412f, 0.47075292f, 0.4695822f, 0.46866563f, 0.46796387f, 0.4674435f,
    0.46707606f, 0.46683735f, 0.46670672f, -2.3308213e-07f, 0.90872943f,
    0.89318603f, 0.8263718f, 0.7659921f, 0.71549624f, 0.67354876f, 0.63868314f,
    0.6096518f, 0.58543396f, 0.5651999f, 0.5482742f, 0.5341049f, 0.5222386f,
    0.51230174f, 0.50398475f, 0.4970305f, 0.4912245f, 0.48638728f, 0.48236847f,
    0.47904158f, 0.4763f, 0.4740538f, 0.472227f, 0.47075528f, 0.469584f,
    0.4686669f, 0.46796468f, 0.46744385f, 0.467076f, 0.46683693f, 0.46670598f,
    0.03999665f, 0.8369103f, 0.8498626f, 0.81313175f, 0.76014066f, 0.71238923f,
    0.6717377f, 0.63757175f, 0.60895f, 0.58498466f, 0.5649116f, 0.5480907f,
    0.53399044f, 0.52216995f, 0.51226336f, 0.50396615f, 0.49702457f,
    0.49122632f, 0.48639363f, 0.48237705f, 0.47905082f, 0.47630903f, 0.474062f,
    0.47223398f, 0.47076082f, 0.46958807f, 0.4686695f, 0.46796584f, 0.4674437f,
    0.46707466f, 0.46683446f, 0.46670252f, 0.4666611f, 0.77755356f, 0.7881133f,
    0.78291655f, 0.745612f, 0.7045331f, 0.6671232f, 0.6347165f, 0.6071256f,
    0.58379674f, 0.56413114f, 0.5475775f, 0.53365535f, 0.5219546f, 0.51212865f,
    0.5038856f, 0.49698f, 0.49120522f, 0.48638725f, 0.48237944f, 0.47905785f,
    0.4763179f, 0.47407088f, 0.47224173f, 0.47076666f, 0.46959162f, 0.4686706f,
    0.46796447f, 0.46743992f, 0.4670686f, 0.4668263f, 0.46669236f, 0.4666489f,
    0.74003357f, 0.73782194f, 0.7397745f, 0.719989f, 0.6895163f, 0.6579779f,
    0.6289357f, 0.60336757f, 0.5813062f, 0.5624606f, 0.54645026f, 0.53289413f,
    0.5214427f, 0.5117876f, 0.5036618f, 0.4968363f, 0.49111575f, 0.48633385f,
    0.4823494f, 0.47904232f, 0.4763106f, 0.47406757f, 0.4722395f, 0.47076365f,
    0.46958667f, 0.46866298f, 0.46795383f, 0.46742615f, 0.46705168f,
    0.46680635f, 0.46666944f, 0.46662277f, 0.7096609f, 0.70139325f, 0.6963484f,
    0.6866365f, 0.6670623f, 0.6432241f, 0.619175f, 0.59682035f, 0.57685626f,
    0.5594045f, 0.54433626f, 0.5314258f, 0.5204214f, 0.511078f, 0.50317025f,
    0.49649733f, 0.4908831f, 0.48617482f, 0.48224056f, 0.47896707f, 0.47625712f,
    0.47402745f, 0.47220695f, 0.47073463f, 0.4695584f, 0.46863365f, 0.46792236f,
    0.46739188f, 0.46701428f, 0.46676558f, 0.46662512f, 0.46657416f, 0.6808616f,
    0.6718063f, 0.659837f, 0.652305f, 0.6400819f, 0.62355876f, 0.6052259f,
    0.5869925f, 0.5699223f, 0.5544923f, 0.5408409f, 0.528929f, 0.5186326f,
    0.5097936f, 0.5022464f, 0.4958315f, 0.4904018f, 0.48582503f, 0.481984f,
    0.4787759f, 0.4761112f, 0.4739123f, 0.472112f, 0.47065246f, 0.46948367f,
    0.46856275f, 0.46785265f, 0.4673217f, 0.4669425f, 0.46669137f, 0.4665478f,
    0.466492f, 0.6530197f, 0.6451505f, 0.63030666f, 0.6216131f, 0.61268413f,
    0.60134625f, 0.588125f, 0.5741658f, 0.5604185f, 0.54748523f, 0.5356808f,
    0.5251262f, 0.5158255f, 0.5077166f, 0.5007051f, 0.49468344f, 0.48954216f,
    0.48517662f, 0.48148984f, 0.4783939f, 0.47581032f, 0.47366947f, 0.47191027f,
    0.47047946f, 0.46933028f, 0.46842235f, 0.46772054f, 0.46719444f,
    0.46681756f, 0.46656695f, 0.46642214f, 0.4663621f, 0.6264247f, 0.6201608f,
    0.6055717f, 0.59567046f, 0.5876566f, 0.5792029f, 0.56968206f, 0.5593819f,
    0.54884094f, 0.5385425f, 0.5288265f, 0.51989317f, 0.5118357f, 0.50467336f,
    0.49837947f, 0.4929001f, 0.48816788f, 0.4841103f, 0.48065484f, 0.47773218f,
    0.47527802f, 0.47323346f, 0.4715456f, 0.4701673f, 0.46905655f, 0.4681765f,
    0.4674947f, 0.46698263f, 0.46661538f, 0.46637085f, 0.46622887f, 0.46616668f,
    0.60121435f, 0.596478f, 0.5837568f, 0.5736884f, 0.56586695f, 0.558795f,
    0.55157113f, 0.5439577f, 0.5360829f, 0.5281987f, 0.52055085f, 0.5133282f,
    0.50665367f, 0.5005926f, 0.49516648f, 0.49036613f, 0.48616245f, 0.48251465f,
    0.47937593f, 0.4766975f, 0.47443116f, 0.47253072f, 0.47095317f, 0.46965924f,
    0.4686128f, 0.4677816f, 0.46713686f, 0.46665272f, 0.4663063f, 0.4660768f,
    0.46594453f, 0.4658842f, 0.57728153f, 0.5738927f, 0.5636731f, 0.5544381f,
    0.5469638f, 0.5406386f, 0.53476554f, 0.52894574f, 0.5230589f, 0.5171498f,
    0.51133156f, 0.50572634f, 0.500436f, 0.49553296f, 0.49105987f, 0.48703456f,
    0.48345563f, 0.4803081f, 0.47756797f, 0.4752059f, 0.47318983f, 0.47148705f,
    0.47006524f, 0.46889395f, 0.4679439f, 0.46718848f, 0.46660322f, 0.46616572f,
    0.46585554f, 0.4656537f, 0.46554112f, 0.46548924f, 0.55436504f, 0.5521558f,
    0.544585f, 0.5368478f, 0.5301448f, 0.5244863f, 0.5195118f, 0.51488525f,
    0.51040643f, 0.50600034f, 0.50167274f, 0.49746957f, 0.49344808f,
    0.48966083f, 0.4861483f, 0.48293656f, 0.48003823f, 0.47745472f, 0.47517854f,
    0.47319588f, 0.4714889f, 0.47003698f, 0.46881828f, 0.4678113f, 0.46699393f,
    0.46634567f, 0.46584684f, 0.46547896f, 0.46522456f, 0.46506667f,
    0.46498704f, 0.46495262f, 0.53214264f, 0.53097683f, 0.525993f, 0.5201123f,
    0.51458204f, 0.50975865f, 0.50557554f, 0.5018439f, 0.49839294f, 0.49511242f,
    0.491951f, 0.48889872f, 0.48596975f, 0.48318815f, 0.4805791f, 0.4781637f,
    0.4759568f, 0.47396636f, 0.47219414f, 0.47063655f, 0.46928594f, 0.46813145f,
    0.4671601f, 0.46635842f, 0.46571112f, 0.4652035f, 0.46482083f, 0.4645487f,
    0.46437284f, 0.46427846f, 0.46424797f, 0.46424246f, 0.51029855f, 0.5100663f,
    0.50752574f, 0.5036516f, 0.49958044f, 0.4958194f, 0.49250168f, 0.48958588f,
    0.4869771f, 0.48458573f, 0.4823483f, 0.48022902f, 0.47821411f, 0.47630388f,
    0.4745062f, 0.47283167f, 0.4712903f, 0.46988988f, 0.4686349f, 0.46752685f,
    0.46656403f, 0.46574217f, 0.46505475f, 0.46449476f, 0.4640524f, 0.4637183f,
    0.46348214f, 0.4633334f, 0.46326077f, 0.46325162f, 0.46328872f, 0.46332556f,
    0.48856825f, 0.48917562f, 0.48890346f, 0.4870562f, 0.48461044f, 0.48210794f,
    0.4797935f, 0.47773814f, 0.47592807f, 0.474319f, 0.4728647f, 0.47152853f,
    0.4702869f, 0.46912727f, 0.46804526f, 0.4670418f, 0.46612033f, 0.4652851f,
    0.46453977f, 0.4638868f, 0.46332717f, 0.46286017f, 0.4624833f, 0.46219403f,
    0.46198657f, 0.46185562f, 0.46179473f, 0.4617965f, 0.46185222f, 0.46195102f,
    0.46207544f, 0.462169f, 0.46676362f, 0.46812475f, 0.46993214f, 0.47005183f,
    0.46929854f, 0.4681875f, 0.46701068f, 0.46591002f, 0.4649373f, 0.46409687f,
    0.46337238f, 0.46274224f, 0.46218714f, 0.46169278f, 0.46125022f,
    0.46085495f, 0.4605056f, 0.46020293f, 0.45994845f, 0.459744f, 0.4595909f,
    0.45948994f, 0.4594406f, 0.45944315f, 0.45949432f, 0.45959175f, 0.4597316f,
    0.45990896f, 0.46011716f, 0.46034628f, 0.46057788f, 0.46074286f, 0.4447842f,
    0.44681606f, 0.4505044f, 0.45247942f, 0.4534092f, 0.45375475f, 0.4538136f,
    0.45376325f, 0.45369953f, 0.4536667f, 0.4536795f, 0.45373756f, 0.45383415f,
    0.45396134f, 0.45411235f, 0.4542824f, 0.45446873f, 0.45467034f, 0.4548874f,
    0.45512074f, 0.45537165f, 0.4556411f, 0.45592946f, 0.45623857f, 0.4565665f,
    0.4569127f, 0.45727512f, 0.45765042f, 0.45803314f, 0.45841375f, 0.4587717f,
    0.45902213f, 0.42261586f, 0.4252354f, 0.43059784f, 0.4342813f, 0.43682703f,
    0.43863243f, 0.43997666f, 0.4410469f, 0.44196242f, 0.44279498f, 0.44358456f,
    0.44435138f, 0.44510412f, 0.4458454f, 0.44657505f, 0.44729203f, 0.44799554f,
    0.44868535f, 0.4493617f, 0.4500256f, 0.4506782f, 0.45132086f, 0.4519542f,
    0.4525811f, 0.45320022f, 0.45381218f, 0.45441583f, 0.4550088f, 0.4555861f,
    0.45613772f, 0.45663998f, 0.4569888f, 0.4003197f, 0.40344363f, 0.41026694f,
    0.41548717f, 0.4195388f, 0.4227551f, 0.42538482f, 0.42760813f, 0.42955166f,
    0.43130165f, 0.43291518f, 0.4344287f, 0.43586498f, 0.43723777f, 0.43855542f,
    0.43982306f, 0.44104397f, 0.44222075f, 0.44335556f, 0.4444505f, 0.44550765f,
    0.4465289f, 0.44751552f, 0.44847125f, 0.44939503f, 0.45028812f, 0.4511501f,
    0.45197883f, 0.4527693f, 0.45351028f, 0.45417386f, 0.45463294f, 0.3780151f,
    0.3815604f, 0.38962856f, 0.39619792f, 0.401614f, 0.4061507f, 0.41002145f,
    0.4133896f, 0.41637784f, 0.41907662f, 0.42155153f, 0.4238495f, 0.42600366f,
    0.42803735f, 0.42996693f, 0.4318041f, 0.43355745f, 0.43523347f, 0.43683732f,
    0.43837333f, 0.43984514f, 0.44125605f, 0.44260806f, 0.44390607f, 0.4451494f,
    0.44633994f, 0.44747746f, 0.4485601f, 0.4495821f, 0.45053062f, 0.45137218f,
    0.45195273f, 0.35585916f, 0.35974467f, 0.36884347f, 0.37656602f, 0.3831845f,
    0.38891992f, 0.39395022f, 0.39841717f, 0.40243244f, 0.40608308f,
    0.40943637f, 0.4125437f, 0.4154443f, 0.41816804f, 0.4207375f, 0.42317015f,
    0.42547944f, 0.42767605f, 0.42976847f, 0.43176374f, 0.43366778f,
    0.43548554f, 0.4372203f, 0.43887833f, 0.44045937f, 0.4419661f, 0.44339854f,
    0.4447546f, 0.44602787f, 0.44720298f, 0.4482401f, 0.44895393f, 0.33402613f,
    0.33817452f, 0.3480969f, 0.35677573f, 0.3644232f, 0.37121472f, 0.37729502f,
    0.38278294f, 0.38777545f, 0.39235142f, 0.39657456f, 0.40049604f,
    0.40415707f, 0.40759057f, 0.41082302f, 0.4138757f, 0.4167659f, 0.41950777f,
    0.42211285f, 0.42459092f, 0.4269501f, 0.42919716f, 0.43133697f, 0.4333772f,
    0.43531826f, 0.43716347f, 0.43891326f, 0.44056523f, 0.44211185f,
    0.44353497f, 0.44478723f, 0.44564778f, 0.31268975f, 0.31702906f,
    0.32757998f, 0.33702374f, 0.3455239f, 0.35321742f, 0.36021936f, 0.36662638f,
    0.37251964f, 0.37796718f, 0.38302594f, 0.38774347f, 0.39215943f,
    0.39630705f, 0.40021387f, 0.40390313f, 0.4073943f, 0.4107037f, 0.4138452f,
    0.41683066f, 0.4196701f, 0.422372f, 0.42494246f, 0.42739066f, 0.42971766f,
    0.43192738f, 0.43402046f, 0.43599415f, 0.4378396f, 0.43953544f, 0.44102556f,
    0.44204894f, 0.29200897f, 0.2964741f, 0.3074746f, 0.317503f, 0.32668373f,
    0.3351219f, 0.34290695f, 0.3501155f, 0.35681316f, 0.36305642f, 0.36889374f,
    0.3743669f, 0.37951186f, 0.38435957f, 0.3889368f, 0.3932665f, 0.3973687f,
    0.40126067f, 0.40495735f, 0.4084718f, 0.41181523f, 0.41499722f, 0.41802493f,
    0.4209086f, 0.4236497f, 0.42625266f, 0.42871812f, 0.43104273f, 0.43321604f,
    0.43521276f, 0.43696707f, 0.43817303f, 0.27211875f, 0.27665234f,
    0.28794253f, 0.29839048f, 0.30808938f, 0.31711835f, 0.32554558f, 0.33343f,
    0.34082323f, 0.34777036f, 0.35431117f, 0.3604805f, 0.36630937f, 0.37182504f,
    0.3770517f, 0.38201076f, 0.38672122f, 0.39119992f, 0.39546177f, 0.39951992f,
    0.4033861f, 0.4070702f, 0.4105798f, 0.41392568f, 0.41710943f, 0.42013556f,
    0.4230044f, 0.42571157f, 0.42824468f, 0.430574f, 0.43262255f, 0.4340345f,
    0.253125f, 0.25767773f, 0.26911825f, 0.27983853f, 0.28990698f, 0.29938197f,
    0.30831403f, 0.3167475f, 0.32472163f, 0.33227134f, 0.33942798f, 0.34621957f,
    0.35267156f, 0.3588068f, 0.36464608f, 0.37020808f, 0.37550983f, 0.3805667f,
    0.38539246f, 0.38999966f, 0.3943994f, 0.39860132f, 0.40261272f, 0.40644428f,
    0.41009712f, 0.4135754f, 0.4168787f, 0.42000124f, 0.422928f, 0.42562416f,
    0.4280002f, 0.42964512f, 0.23510337f, 0.239634f, 0.25110632f, 0.26197088f,
    0.27227658f, 0.2820652f, 0.29137328f, 0.30023307f, 0.30867353f, 0.31672084f,
    0.32439882f, 0.33172926f, 0.3387321f, 0.34542572f, 0.3518271f, 0.35795182f,
    0.3638142f, 0.36942756f, 0.37480387f, 0.37995422f, 0.38488853f, 0.38961545f,
    0.39414135f, 0.39847618f, 0.40262017f, 0.4065765f, 0.4103434f, 0.41391334f,
    0.41726798f, 0.42036644f, 0.42310545f, 0.42501304f, 0.21810046f, 0.2225754f,
    0.23398122f, 0.24488138f, 0.2553094f, 0.26529413f, 0.2748609f, 0.2840324f,
    0.29282933f, 0.30127054f, 0.30937365f, 0.31715482f, 0.32462925f,
    0.33181116f, 0.33871377f, 0.34534952f, 0.35173f, 0.35786596f, 0.36376742f,
    0.36944348f, 0.3749025f, 0.38015157f, 0.38519576f, 0.3900437f, 0.39469436f,
    0.39914933f, 0.4034051f, 0.40745163f, 0.4112668f, 0.41480282f, 0.41794088f,
    0.4201426f, 0.20213711f, 0.2065297f, 0.21778962f, 0.22863527f, 0.23908821f,
    0.24916711f, 0.2588887f, 0.26826814f, 0.27731955f, 0.28605613f, 0.2944903f,
    0.30263385f, 0.31049782f, 0.3180929f, 0.32542914f, 0.332516f, 0.33936247f,
    0.34597695f, 0.35236734f, 0.35854074f, 0.36450368f, 0.37026164f, 0.375818f,
    0.38117984f, 0.38634437f, 0.39131147f, 0.39607534f, 0.40062302f,
    0.40492797f, 0.40893468f, 0.41250712f, 0.41503453f, 0.1872123f, 0.19150218f,
    0.20255373f, 0.213272f, 0.22366914f, 0.23375599f, 0.24354257f, 0.25303835f,
    0.26225242f, 0.27119356f, 0.2798703f, 0.2882908f, 0.29646316f, 0.30439502f,
    0.31209388f, 0.31956682f, 0.3268206f, 0.33386168f, 0.34069592f, 0.3473287f,
    0.35376486f, 0.36000824f, 0.36606058f, 0.37192723f, 0.37760374f,
    0.38308796f, 0.38837165f, 0.39343858f, 0.39825737f, 0.40276393f,
    0.40680355f, 0.4096871f, 0.17330758f, 0.17747974f, 0.18827543f, 0.1988089f,
    0.20908496f, 0.21910837f, 0.22888413f, 0.23841733f, 0.24771334f,
    0.25677767f, 0.2656159f, 0.27423373f, 0.28263682f, 0.29083073f, 0.29882103f,
    0.306613f, 0.3142118f, 0.32162222f, 0.32884875f, 0.33589536f, 0.34276554f,
    0.34946167f, 0.3559842f, 0.362337f, 0.36851397f, 0.37451103f, 0.3803176f,
    0.3859141f, 0.39126396f, 0.39629403f, 0.40082946f, 0.40409687f};

static const auto leaving_albedo_lut = vector<float>{0.9371709f, 1.0584852f,
    0.9998838f, 0.99994856f, 0.9999712f, 0.99998176f, 0.9999875f, 0.99999106f,
    0.9999934f, 0.999995f, 0.99999624f, 0.9999972f, 0.999998f, 0.9999987f,
    0.9999994f, 1.0f, 1.000001f, 1.0000019f, 1.0000035f, 1.0000058f, 1.0000105f,
    1.0000222f, 1.0000707f, 1.0037155f, 2.0064638f, 2.1212397f, 2.1636672f,
    2.1831546f, 2.1927624f, 2.1974509f, 2.1994793f, -1.3689473e-06f, 0.995158f,
    0.00057143765f, 0.9998635f, 0.99993956f, 0.9999662f, 0.99997854f,
    0.99998534f, 0.9999895f, 0.9999922f, 0.9999941f, 0.9999955f, 0.99999666f,
    0.9999976f, 0.99999845f, 0.9999992f, 1.0f, 1.000001f, 1.0000023f, 1.000004f,
    1.0000068f, 1.0000124f, 1.000026f, 1.0000829f, 1.0043423f, 2.0064127f,
    2.1212275f, 2.1636624f, 2.183152f, 2.1927607f, 2.1974502f, 2.1994789f,
    0.039998364f, 0.9740822f, 0.013569475f, 0.9987026f, 0.998979f, 0.99943644f,
    0.9996455f, 0.9997587f, 0.9998272f, 0.99987227f, 0.99990386f, 0.9999273f,
    0.9999456f, 0.9999607f, 0.9999742f, 0.999987f, 1.0000004f, 1.0000156f,
    1.000035f, 1.000063f, 1.0001084f, 1.0001957f, 1.0004127f, 1.0013168f,
    1.0533134f, 2.002226f, 2.1201572f, 2.163218f, 2.1829228f, 2.1926265f,
    2.1973634f, 2.1994185f, 2.1999526f, 0.9151202f, 0.953638f, 0.98707736f,
    0.99441767f, 0.99695224f, 0.99810094f, 0.99871624f, 0.99908537f,
    0.99932575f, 0.9994931f, 0.9996164f, 0.9997123f, 0.9997913f, 0.99986076f,
    0.9999265f, 0.99999446f, 1.000072f, 1.0001704f, 1.0003107f, 1.0005388f,
    1.0009772f, 1.0020609f, 1.0065192f, 1.1477371f, 1.9883618f, 2.116238f,
    2.1615157f, 2.1820166f, 2.1920793f, 2.1970024f, 2.1991627f, 2.19976f,
    0.88107437f, 0.90667444f, 0.9613529f, 0.9816899f, 0.9897871f, 0.99361646f,
    0.9956923f, 0.9969384f, 0.99774754f, 0.9983078f, 0.9987181f, 0.9990352f,
    0.99929446f, 0.9995205f, 0.99973255f, 0.9999498f, 1.0001954f, 1.0005052f,
    1.0009444f, 1.0016555f, 1.0030148f, 1.0063455f, 1.0195826f, 1.2344664f,
    1.9603385f, 2.1071508f, 2.157374f, 2.1797438f, 2.1906755f, 2.1960578f,
    2.1984816f, 2.1992385f, 0.87651396f, 0.88029265f, 0.9260223f, 0.95826846f,
    0.97498405f, 0.98387706f, 0.988972f, 0.99211293f, 0.9941778f, 0.9956141f,
    0.9966661f, 0.99747664f, 0.9981358f, 0.9987058f, 0.9992358f, 0.9997731f,
    1.0003742f, 1.0011251f, 1.0021813f, 1.0038785f, 1.0070926f, 1.0148237f,
    1.0435967f, 1.3004992f, 1.9185156f, 2.090698f, 2.1494236f, 2.1752398f,
    2.1878307f, 2.1941092f, 2.1970563f, 2.198134f, 0.87989175f, 0.8727413f,
    0.89560664f, 0.9287436f, 0.9524009f, 0.96742034f, 0.9769434f, 0.9831749f,
    0.98742026f, 0.99043626f, 0.99267167f, 0.9944035f, 0.9958126f, 0.9970266f,
    0.99814767f, 0.99927294f, 1.0005183f, 1.0020566f, 1.0041974f, 1.007599f,
    1.0139428f, 1.0287358f, 1.0785493f, 1.3494244f, 1.8686143f, 2.065489f,
    2.1362996f, 2.167526f, 2.1828425f, 2.1906338f, 2.1944802f, 2.1961148f,
    0.8817869f, 0.8720885f, 0.87662786f, 0.9014121f, 0.92617995f, 0.9454363f,
    0.95936686f, 0.96931815f, 0.9765102f, 0.98182833f, 0.98587734f, 0.9890692f,
    0.9916923f, 0.9939609f, 0.9960524f, 0.9981391f, 1.000427f, 1.0032206f,
    1.0070591f, 1.0130664f, 1.0240198f, 1.0484279f, 1.1207298f, 1.3854628f,
    1.8179488f, 2.031527f, 2.1168866f, 2.1555989f, 2.174923f, 2.1850157f,
    2.1902611f, 2.1927729f, 0.88015664f, 0.87143815f, 0.866527f, 0.8809583f,
    0.9016999f, 0.92151695f, 0.9380563f, 0.9511582f, 0.96137327f, 0.969361f,
    0.9756985f, 0.98084646f, 0.98516667f, 0.98895305f, 0.9924658f, 0.99597025f,
    0.9997915f, 1.0044109f, 1.0106707f, 1.0202818f, 1.0372849f, 1.0729928f,
    1.1646347f, 1.4115102f, 1.771322f, 1.9903057f, 2.0905626f, 2.138542f,
    2.1632452f, 2.1765666f, 2.1838295f, 2.1876264f, 0.87506396f, 0.8682963f,
    0.8604949f, 0.86709416f, 0.88194424f, 0.8992996f, 0.9159683f, 0.93066126f,
    0.94311184f, 0.95350695f, 0.9621917f, 0.96953756f, 0.97589725f, 0.9815999f,
    0.9869706f, 0.99236834f, 0.99825394f, 1.0053191f, 1.0147613f, 1.0289378f,
    1.053096f, 1.1003836f, 1.2054003f, 1.4293896f, 1.730131f, 1.944244f,
    2.0573568f, 2.1156507f, 2.147007f, 2.1645532f, 2.1745505f, 2.1801252f,
    0.8669339f, 0.86209416f, 0.85493374f, 0.85722387f, 0.86692524f, 0.8806122f,
    0.8955636f, 0.9101435f, 0.9235606f, 0.9355545f, 0.94616026f, 0.9555637f,
    0.9640252f, 0.97184974f, 0.9793901f, 0.987081f, 0.9955176f, 1.0056131f,
    1.0189317f, 1.0384369f, 1.0702703f, 1.127968f, 1.2399074f, 1.4402184f,
    1.6936255f, 1.8957889f, 2.0179255f, 2.0865362f, 2.1255028f, 2.1482384f,
    2.1617444f, 2.1696649f, 0.8561458f, 0.8529029f, 0.84784836f, 0.84865963f,
    0.8550275f, 0.86536515f, 0.8778982f, 0.8912303f, 0.9044522f, 0.9170659f,
    0.92887086f, 0.9398674f, 0.9501931f, 0.9600905f, 0.96990675f, 0.98012805f,
    0.9914673f, 1.0050429f, 1.0227451f, 1.047994f, 1.0873127f, 1.1532036f,
    1.2666508f, 1.4446831f, 1.6602514f, 1.8467264f, 1.9733459f, 2.051176f,
    2.098197f, 2.1269312f, 2.1447167f, 2.1556015f, 0.8429806f, 0.8409515f,
    0.8383361f, 0.83945006f, 0.8443117f, 0.85237855f, 0.86273706f, 0.87446326f,
    0.8868143f, 0.8992804f, 0.91157025f, 0.9235761f, 0.93534374f, 0.94705737f,
    0.9590487f, 0.97183925f, 0.9862372f, 1.0035268f, 1.0258312f, 1.0567902f,
    1.1027306f, 1.1741309f, 1.2852144f, 1.4432418f, 1.628414f, 1.7979748f,
    1.9248224f, 2.009896f, 2.0647848f, 2.1000426f, 2.122796f, 2.1372771f,
    0.82764316f, 0.8264778f, 0.8260952f, 0.82842505f, 0.83317333f, 0.84019077f,
    0.84912467f, 0.8595009f, 0.8708538f, 0.8828069f, 0.89510787f, 0.90763855f,
    0.92041737f, 0.93360734f, 0.94754076f, 0.9627726f, 0.9801842f, 1.001169f,
    1.0279545f, 1.0641251f, 1.1153165f, 1.18957f, 1.2958304f, 1.4362788f,
    1.5967993f, 1.7497563f, 1.8734398f, 1.963288f, 2.0252242f, 2.0671437f,
    2.0953815f, 2.11405f, 0.810292f, 0.80969465f, 0.81112087f, 0.8149976f,
    0.82051814f, 0.82752293f, 0.835916f, 0.845535f, 0.8561764f, 0.8676479f,
    0.87981164f, 0.892614f, 0.9061092f, 0.9204854f, 0.9361025f, 0.9535557f,
    0.97377723f, 0.9982028f, 1.0290228f, 1.0695138f, 1.1243092f, 1.1990858f,
    1.2991105f, 1.4242096f, 1.5644732f, 1.7019038f, 1.8200417f, 1.9121021f,
    1.9797393f, 2.0280113f, 2.0619907f, 2.0853288f, 0.7910634f, 0.7907923f,
    0.7935477f, 0.79895854f, 0.80571586f, 0.8134506f, 0.8220993f, 0.8316546f,
    0.8420982f, 0.85340583f, 0.8655749f, 0.8786534f, 0.892772f, 0.9081785f,
    0.9252833f, 0.9447238f, 0.96745706f, 0.99489033f, 1.0290458f, 1.072712f,
    1.1294143f, 1.2028279f, 1.2958899f, 1.4075389f, 1.5308853f, 1.6541336f,
    1.765233f, 1.8571489f, 1.9287903f, 1.9826607f, 2.0223079f, 2.050612f,
    0.7700875f, 0.76994896f, 0.7735751f, 0.7803286f, 0.78849006f, 0.7974136f,
    0.8069345f, 0.8170715f, 0.82790077f, 0.8395166f, 0.85203314f, 0.8656032f,
    0.88044655f, 0.89688414f, 0.9153833f, 0.9366144f, 0.9615251f, 0.9914264f,
    1.0280732f, 1.0736774f, 1.1307182f, 1.2013321f, 1.2871206f, 1.3868685f,
    1.4958231f, 1.6062214f, 1.7094527f, 1.7992308f, 1.8730248f, 1.9313524f,
    1.9762214f, 2.0095286f, 0.7474991f, 0.7473409f, 0.75143445f, 0.75926495f,
    0.76881206f, 0.77915895f, 0.789976f, 0.8012219f, 0.81298816f, 0.82542837f,
    0.8387358f, 0.8531468f, 0.8689587f, 0.8865581f, 0.9064569f, 0.9293357f,
    0.95609254f, 0.98788536f, 1.0261447f, 1.0725033f, 1.1285541f, 1.1953349f,
    1.2737844f, 1.3628694f, 1.459338f, 1.5580782f, 1.6530546f, 1.7391096f,
    1.8132174f, 1.8745753f, 1.9238479f, 1.9618737f, 0.723445f, 0.7231503f,
    0.7273758f, 0.73600596f, 0.7468196f, 0.758668f, 0.77104014f, 0.7837865f,
    0.7969558f, 0.8107071f, 0.8252685f, 0.8409253f, 0.8580254f, 0.87699574f,
    0.89836603f, 0.922794f, 0.9510884f, 0.9842151f, 1.0232652f, 1.0693507f,
    1.123374f, 1.1856292f, 1.2568214f, 1.3362324f, 1.4216613f, 1.509757f,
    1.5963633f, 1.6774912f, 1.7502097f, 1.8130058f, 1.8655329f, 1.9076396f,
    0.69808704f, 0.6975695f, 0.7016622f, 0.7108384f, 0.7227578f, 0.7360897f,
    0.7501515f, 0.76466537f, 0.7796004f, 0.795078f, 0.81131893f, 0.82861835f,
    0.84733844f, 0.8679119f, 0.89085054f, 0.9167536f, 0.9463073f, 0.98026717f,
    1.0194048f, 1.0644004f, 1.1156515f, 1.1729658f, 1.2370775f, 1.3076179f,
    1.3831222f, 1.4614177f, 1.5396988f, 1.61502f, 1.6848549f, 1.7474543f,
    1.8018334f, 1.8470342f, 0.6716033f, 0.6708032f, 0.67456675f, 0.6840775f,
    0.6969388f, 0.7116858f, 0.7274871f, 0.74393386f, 0.7608954f, 0.7784244f,
    0.79669803f, 0.81598306f, 0.8366177f, 0.8590026f, 0.88359547f, 0.91090417f,
    0.9414716f, 0.97584367f, 1.0145133f, 1.0578257f, 1.1058248f, 1.1580015f,
    1.2152743f, 1.277617f, 1.3440785f, 1.4132799f, 1.4833726f, 1.5522723f,
    1.6179723f, 1.6788026f, 1.7334794f, 1.780485f, 0.64418715f, 0.6430682f,
    0.64636964f, 0.65605366f, 0.6697122f, 0.6857876f, 0.70332545f, 0.7217932f,
    0.7409511f, 0.76076245f, 0.7813317f, 0.80286396f, 0.82563806f, 0.84998727f,
    0.87628376f, 0.90492076f, 0.93628895f, 0.9707411f, 1.0085397f, 1.049783f,
    1.0942717f, 1.1412798f, 1.1920007f, 1.2467276f, 1.3048677f, 1.3655753f,
    1.4276732f, 1.4897487f, 1.5503122f, 1.6079454f, 1.6613252f, 1.7086271f,
    0.6160447f, 0.6145907f, 0.6173546f, 0.6271017f, 0.6414436f, 0.65876245f,
    0.67800426f, 0.698524f, 0.71997094f, 0.7422053f, 0.7652377f, 0.7891846f,
    0.8142364f, 0.8406315f, 0.8686335f, 0.8985084f, 0.9304975f, 0.9647844f,
    1.0014505f, 1.0404131f, 1.0813057f, 1.1232337f, 1.1677185f, 1.2153487f,
    1.2657773f, 1.318516f, 1.3728495f, 1.427868f, 1.4825325f, 1.5357405f,
    1.5862939f, 1.6322747f, 0.5873906f, 0.58560306f, 0.5878041f, 0.59755236f,
    0.6124983f, 0.63099074f, 0.6518861f, 0.6744474f, 0.69821125f, 0.72292745f,
    0.74849784f, 0.77493155f, 0.8023101f, 0.8307585f, 0.86041963f, 0.89142966f,
    0.92389166f, 0.95784664f, 0.99323606f, 1.0298455f, 1.0671827f, 1.1042f,
    1.1427798f, 1.1837842f, 1.2270329f, 1.2722726f, 1.3190997f, 1.366964f,
    1.4151871f, 1.4629728f, 1.5093247f, 1.5523807f, 0.55844396f, 0.55633885f,
    0.55799437f, 0.5677241f, 0.58322805f, 0.6028305f, 0.6253319f, 0.6498934f,
    0.6759485f, 0.70313317f, 0.7312325f, 0.7601372f, 0.78980887f, 0.8202509f,
    0.851483f, 0.8835171f, 0.916334f, 0.9498571f, 0.98391855f, 1.0182034f,
    1.0521121f, 1.0844378f, 1.1174451f, 1.1522553f, 1.1887971f, 1.2269676f,
    1.2665662f, 1.307289f, 1.3487239f, 1.3903311f, 1.4313276f, 1.4699856f,
    0.5294225f, 0.52702796f, 0.52818996f, 0.53791684f, 0.5539611f, 0.5746347f,
    0.59868145f, 0.6251776f, 0.6534537f, 0.68303233f, 0.7135794f, 0.744864f,
    0.77672565f, 0.809047f, 0.84173006f, 0.8746742f, 0.90775466f, 0.9407982f,
    0.9735489f, 1.0056067f, 1.0362681f, 1.064146f, 1.0919044f, 1.1209158f,
    1.1511769f, 1.1826761f, 1.2153393f, 1.2490207f, 1.2834903f, 1.3184f,
    1.3531483f, 1.3861645f, 0.5005381f, 0.49789128f, 0.49863836f, 0.5084051f,
    0.52499443f, 0.5467089f, 0.5722406f, 0.6005858f, 0.6309761f, 0.66282386f,
    0.69567895f, 0.72919226f, 0.76308703f, 0.7971332f, 0.83112663f, 0.86486816f,
    0.8981436f, 0.93069875f, 0.962202f, 0.99217314f, 1.0197976f, 1.043478f,
    1.066294f, 1.0898671f, 1.1142336f, 1.139433f, 1.1654637f, 1.1922736f,
    1.2197461f, 1.2476598f, 1.2755456f, 1.3019748f, 0.47199088f, 0.46913585f,
    0.4695656f, 0.479434f, 0.49658865f, 0.5193209f, 0.5462731f, 0.5763645f,
    0.60873246f, 0.64268523f, 0.67766404f, 0.7132113f, 0.7489449f, 0.78453565f,
    0.8196876f, 0.8541184f, 0.8875384f, 0.919623f, 0.9499707f, 0.97801876f,
    1.0028267f, 1.0225539f, 1.0407112f, 1.0591726f, 1.0779946f, 1.0972414f,
    1.1169479f, 1.1371112f, 1.1576761f, 1.178494f, 1.199178f, 1.2184103f,
    0.44396588f, 0.44095066f, 0.4411725f, 0.45121616f, 0.46896482f, 0.49269468f,
    0.5209971f, 0.5527169f, 0.58690286f, 0.62276757f, 0.65965486f, 0.69701236f,
    0.734368f, 0.77130973f, 0.80746555f, 0.8424841f, 0.87601155f, 0.9076601f,
    0.93695885f, 0.9632556f, 0.98546594f, 1.0014696f, 1.0152258f, 1.0288687f,
    1.0424631f, 1.0560825f, 1.0697753f, 1.0835595f, 1.0974064f, 1.1112012f,
    1.1245997f, 1.1363652f, 0.41662884f, 0.4135034f, 0.41363204f, 0.4239295f,
    0.44230378f, 0.46700937f, 0.49658477f, 0.5298026f, 0.56562984f, 0.60319424f,
    0.64175624f, 0.680685f, 0.7194361f, 0.7575314f, 0.7945391f, 0.8300512f,
    0.86365855f, 0.89491403f, 0.92327505f, 0.94799054f, 0.96781236f, 0.9803036f,
    0.9898883f, 0.99897414f, 1.007628f, 1.0159241f, 1.0239133f, 1.0316182f,
    1.0390178f, 1.0460081f, 1.0522631f, 1.0566095f, 0.390124f, 0.3869382f,
    0.38708803f, 0.39771724f, 0.41674662f, 0.4424014f, 0.47316462f, 0.50773996f,
    0.54502046f, 0.5840616f, 0.6240566f, 0.66431344f, 0.7042334f, 0.7432891f,
    0.7810029f, 0.81692153f, 0.8505872f, 0.88149524f, 0.90902764f, 0.9323238f,
    0.94995207f, 0.9591205f, 0.96473694f, 0.96949756f, 0.9734701f, 0.9767279f,
    0.97932094f, 0.9812713f, 0.9825587f, 0.9830817f, 0.9825247f, 0.97977465f,
    0.36457226f, 0.36137426f, 0.361655f, 0.37268886f, 0.39245152f, 0.4191091f,
    0.45082557f, 0.48661023f, 0.5251502f, 0.56544226f, 0.606629f, 0.64797527f,
    0.68884474f, 0.72867644f, 0.76695937f, 0.8032043f, 0.83690983f, 0.8675145f,
    0.8943207f, 0.91634786f, 0.9319608f, 0.9379756f, 0.93980134f, 0.94044167f,
    0.93996775f, 0.93845516f, 0.9359559f, 0.93249476f, 0.928055f, 0.9225413f,
    0.9156539f, 0.9063508f};

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

// Microfacet energy compensation (E(cos(w)))
inline vec3f microfacet_compensation_conductors(const vector<float>& E_lut,
    const vec3f& color, float roughness, const vec3f& normal,
    const vec3f& outgoing) {
  auto E = interpolate2d(E_lut, {abs(dot(normal, outgoing)), sqrt(roughness)});
  return 1 + color * (1 - E) / E;
}

inline vec3f microfacet_compensation_dielectrics(const vector<float>& E_lut,
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  return vec3f{1, 1, 1} /
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
