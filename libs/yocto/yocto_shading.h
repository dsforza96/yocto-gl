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
static const auto conductors_albedo_lut = vector<float>{0.9633789f, 0.99560547f,
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

static const auto my_conductors_albedo_lut = vector<float>{0.9987483f,
    0.9995335f, 0.999886f, 0.99994946f, 0.9999715f, 0.99998164f, 0.9999871f,
    0.9999904f, 0.99999255f, 0.99999404f, 0.99999505f, 0.9999958f, 0.9999964f,
    0.9999969f, 0.99999726f, 0.99999756f, 0.9999978f, 0.999998f, 0.99999815f,
    0.99999833f, 0.99999845f, 0.9999985f, 0.9999986f, 0.9999987f, 0.99999875f,
    0.9999988f, 0.99999887f, 0.9999989f, 0.999999f, 0.999999f, 0.99999905f,
    0.99999905f, 0.99852264f, 0.9994506f, 0.99986607f, 0.9999407f, 0.99996656f,
    0.9999784f, 0.99998486f, 0.9999888f, 0.9999913f, 0.999993f, 0.9999942f,
    0.9999951f, 0.9999958f, 0.99999636f, 0.9999968f, 0.99999714f, 0.99999744f,
    0.9999977f, 0.99999785f, 0.99999803f, 0.99999815f, 0.9999983f, 0.9999984f,
    0.99999845f, 0.99999857f, 0.9999986f, 0.9999987f, 0.99999875f, 0.9999988f,
    0.99999887f, 0.99999887f, 0.9999989f, 0.97428286f, 0.9900636f, 0.9976743f,
    0.9989968f, 0.99944216f, 0.99964374f, 0.9997517f, 0.9998162f, 0.9998577f,
    0.9998861f, 0.99990624f, 0.99992114f, 0.99993247f, 0.99994123f, 0.99994814f,
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

static const auto dielectrics_albedo_lut_r = vector<float>{0.88894486f,
    0.83007085f, 0.69351447f, 0.5814902f, 0.4893404f, 0.4132307f, 0.35014302f,
    0.29769292f, 0.25398162f, 0.2174848f, 0.18696985f, 0.1614333f, 0.14005338f,
    0.12215367f, 0.10717471f, 0.09465193f, 0.0841982f, 0.07548986f,
    0.068255566f, 0.062267315f, 0.057333067f, 0.05329079f, 0.050003532f,
    0.047355372f, 0.045248058f, 0.043598223f, 0.042335067f, 0.04139839f,
    0.040736996f, 0.04030729f, 0.040072158f, -5.3355397e-08f, 0.8885852f,
    0.82990927f, 0.69346356f, 0.5814661f, 0.48932728f, 0.41322306f, 0.35013846f,
    0.2976902f, 0.25398f, 0.21748388f, 0.18696937f, 0.1614331f, 0.14005338f,
    0.12215377f, 0.10717486f, 0.09465212f, 0.08419839f, 0.07549005f,
    0.068255745f, 0.062267482f, 0.057333216f, 0.053290922f, 0.050003644f,
    0.047355466f, 0.045248136f, 0.043598287f, 0.04233511f, 0.041398425f,
    0.040737018f, 0.040307306f, 0.04007216f, -5.3251252e-08f, 0.8547721f,
    0.8147624f, 0.68892527f, 0.5793442f, 0.48817077f, 0.41254714f, 0.3497302f,
    0.297441f, 0.25382915f, 0.21739532f, 0.18692073f, 0.16141008f, 0.14004664f,
    0.12215717f, 0.10718429f, 0.09466483f, 0.084212564f, 0.07550448f,
    0.06826968f, 0.062280435f, 0.05734493f, 0.053301256f, 0.050012566f,
    0.04736299f, 0.045254327f, 0.04360323f, 0.042338908f, 0.041401174f,
    0.040738825f, 0.040308267f, 0.040072378f, 0.039999526f, 0.76843363f,
    0.76213986f, 0.67161864f, 0.5713211f, 0.48380762f, 0.40998358f, 0.34816405f,
    0.29646814f, 0.253225f, 0.21702653f, 0.18670453f, 0.16129337f, 0.13999459f,
    0.12214654f, 0.10719943f, 0.09469517f, 0.08425097f, 0.07554616f,
    0.06831147f, 0.062320307f, 0.05738163f, 0.05333408f, 0.050041173f,
    0.047387302f, 0.045274425f, 0.04361931f, 0.042351227f, 0.041410025f,
    0.040744517f, 0.04031111f, 0.040072672f, 0.03999756f, 0.6844415f,
    0.68299955f, 0.6330962f, 0.5521906f, 0.4732251f, 0.40370405f, 0.344284f,
    0.29402062f, 0.25167206f, 0.21604887f, 0.1861035f, 0.16094117f, 0.13980694f,
    0.12206689f, 0.1071894f, 0.09472852f, 0.08430973f, 0.07561806f,
    0.068388216f, 0.062396392f, 0.057453483f, 0.05339951f, 0.050098944f,
    0.047436837f, 0.04531561f, 0.04365232f, 0.042376444f, 0.04142795f,
    0.040755715f, 0.040316176f, 0.0400722f, 0.039992124f, 0.61617994f,
    0.60832196f, 0.57679725f, 0.51891696f, 0.4535404f, 0.39164367f, 0.33667344f,
    0.28912756f, 0.24850036f, 0.21399708f, 0.18479359f, 0.16012801f,
    0.13932769f, 0.12181145f, 0.10708236f, 0.09471779f, 0.08435923f,
    0.07570277f, 0.06849087f, 0.06250521f, 0.057560556f, 0.0534997f,
    0.05018906f, 0.047515072f, 0.04538112f, 0.043704942f, 0.042416453f,
    0.041455917f, 0.04077239f, 0.04032242f, 0.040068917f, 0.039980225f,
    0.553738f, 0.5437216f, 0.5158546f, 0.47472972f, 0.42423156f, 0.37249568f,
    0.32409328f, 0.28079405f, 0.24295293f, 0.21030675f, 0.18235748f,
    0.15854655f, 0.13833123f, 0.12121519f, 0.106758185f, 0.09457617f,
    0.08433725f, 0.07575616f, 0.068588555f, 0.06262548f, 0.05768845f,
    0.05362506f, 0.05030521f, 0.047617827f, 0.045468062f, 0.043774914f,
    0.042469174f, 0.04149171f, 0.04079198f, 0.040326778f, 0.04005916f,
    0.039957546f, 0.4945862f, 0.48551357f, 0.45890546f, 0.42680562f, 0.388269f,
    0.34689334f, 0.30623996f, 0.26843655f, 0.23442805f, 0.20444791f,
    0.17835765f, 0.15584704f, 0.1365428f, 0.120064564f, 0.10605213f,
    0.09417722f, 0.0841472f, 0.075704776f, 0.06862559f, 0.06271515f,
    0.057805426f, 0.05375164f, 0.050429232f, 0.047731176f, 0.045565553f,
    0.043853484f, 0.042527284f, 0.041528895f, 0.040808517f, 0.040323455f,
    0.04003709f, 0.03991804f, 0.43955615f, 0.4320794f, 0.40787798f, 0.38067025f,
    0.35014766f, 0.31734022f, 0.28417742f, 0.25231758f, 0.22280478f, 0.1961455f,
    0.17247969f, 0.15172835f, 0.13369569f, 0.11813302f, 0.10477644f, 0.0933678f,
    0.08366555f, 0.07544968f, 0.06852318f, 0.06271167f, 0.057861943f,
    0.053840186f, 0.050529964f, 0.047830254f, 0.045653578f, 0.043924324f,
    0.042577203f, 0.0415559f, 0.040811867f, 0.04030329f, 0.039994165f,
    0.03985351f, 0.3895176f, 0.38338712f, 0.36230904f, 0.3386006f, 0.3133133f,
    0.2867971f, 0.25988433f, 0.23355296f, 0.20860448f, 0.18556052f, 0.16468638f,
    0.14605603f, 0.12961698f, 0.11524142f, 0.102762245f, 0.09199666f,
    0.08276097f, 0.07487918f, 0.06818765f, 0.06253721f, 0.05779376f, 0.0538379f,
    0.050564077f, 0.04787948f, 0.04570282f, 0.04396313f, 0.042598598f,
    0.04155549f, 0.04078717f, 0.040253226f, 0.039918642f, 0.039753158f,
    0.34478897f, 0.33963022f, 0.3216022f, 0.30097649f, 0.2794473f, 0.25744197f,
    0.2353409f, 0.21363649f, 0.19281913f, 0.17328016f, 0.15527709f, 0.13894123f,
    0.12430333f, 0.11132231f, 0.09991053f, 0.08995337f, 0.08132342f,
    0.07389006f, 0.067525685f, 0.062109496f, 0.057529528f, 0.053683605f,
    0.050479524f, 0.04783478f, 0.045676026f, 0.043938387f, 0.042564712f,
    0.04150481f, 0.04071473f, 0.040156063f, 0.039795298f, 0.03960326f,
    0.30526534f, 0.3007888f, 0.28537437f, 0.26755062f, 0.24902989f, 0.23039916f,
    0.21193574f, 0.19389413f, 0.17653926f, 0.16011226f, 0.14480013f,
    0.13072367f, 0.117940225f, 0.106454134f, 0.09622964f, 0.08720326f,
    0.079294115f, 0.07241211f, 0.06646386f, 0.06135683f, 0.057002142f,
    0.053316217f, 0.05022174f, 0.04764805f, 0.04553115f, 0.043813538f,
    0.04244382f, 0.041376304f, 0.040570512f, 0.039990716f, 0.039605457f,
    0.0393871f, 0.2706032f, 0.26661935f, 0.2533261f, 0.23796865f, 0.2219958f,
    0.20604345f, 0.19039255f, 0.17521276f, 0.16064857f, 0.14683285f,
    0.13387814f, 0.121867485f, 0.110850476f, 0.10084444f, 0.09183891f,
    0.08380149f, 0.076683804f, 0.07042694f, 0.06496592f, 0.06023321f,
    0.056161396f, 0.052685007f, 0.049741793f, 0.047273472f, 0.04522617f,
    0.043550566f, 0.042201847f, 0.041139584f, 0.04032748f, 0.039733097f,
    0.039327566f, 0.039085265f, 0.24034837f, 0.23674001f, 0.2251428f,
    0.21188623f, 0.19809622f, 0.18435946f, 0.1709607f, 0.15804689f, 0.14571136f,
    0.13402663f, 0.12305243f, 0.112834774f, 0.10340366f, 0.094772026f,
    0.086936474f, 0.07987923f, 0.07357084f, 0.06797303f, 0.06304151f,
    0.05872835f, 0.054983992f, 0.05175885f, 0.049004477f, 0.04667441f,
    0.044724777f, 0.043114606f, 0.04180602f, 0.04076428f, 0.039957732f,
    0.039357714f, 0.038938373f, 0.038676508f, 0.21401255f, 0.21070725f,
    0.20047347f, 0.18897116f, 0.17703915f, 0.16516896f, 0.15362549f,
    0.14254652f, 0.13200639f, 0.12204974f, 0.11270651f, 0.10399712f,
    0.095933564f, 0.08851927f, 0.08174907f, 0.075609654f, 0.07008053f,
    0.065135255f, 0.060742803f, 0.056868948f, 0.053477544f, 0.0505316f,
    0.04799418f, 0.045829147f, 0.044001672f, 0.04247864f, 0.0412289f,
    0.040223416f, 0.039435323f, 0.038839933f, 0.0384147f, 0.038139123f,
    0.19111595f, 0.18806736f, 0.1789455f, 0.16889608f, 0.15853144f, 0.1482401f,
    0.13825001f, 0.12868641f, 0.11961482f, 0.11106823f, 0.103062406f,
    0.0956036f, 0.088691905f, 0.082322545f, 0.07648633f, 0.0711699f,
    0.06635607f, 0.062024277f, 0.05815115f, 0.054711174f, 0.051677372f,
    0.04902191f, 0.046716694f, 0.04473384f, 0.043046106f, 0.04162719f,
    0.040451996f, 0.03949677f, 0.038739234f, 0.038158637f, 0.03773576f,
    0.037452918f, 0.17120981f, 0.1683862f, 0.16018663f, 0.1513402f, 0.14228992f,
    0.133332f, 0.1246514f, 0.11635567f, 0.10850244f, 0.10111902f, 0.094215095f,
    0.08779038f, 0.08183876f, 0.07635049f, 0.07131324f, 0.066712566f,
    0.0625322f, 0.05875429f, 0.055359606f, 0.05232783f, 0.049637854f,
    0.047268074f, 0.045196705f, 0.04340206f, 0.04186281f, 0.04055817f,
    0.039468113f, 0.03857348f, 0.037856095f, 0.037298813f, 0.03688559f,
    0.036601465f, 0.15388685f, 0.15126458f, 0.14384243f, 0.13599648f,
    0.12804602f, 0.12021164f, 0.11263699f, 0.10540943f, 0.09857735f, 0.0921636f,
    0.08617498f, 0.0806086f, 0.07545587f, 0.070704915f, 0.06634189f,
    0.06235177f, 0.05871868f, 0.055426173f, 0.052457355f, 0.049795f,
    0.047421683f, 0.0453199f, 0.043472193f, 0.041861296f, 0.04047024f,
    0.039282482f, 0.03828201f, 0.037453417f, 0.03678197f, 0.036253672f,
    0.035855293f, 0.03557438f, 0.13878489f, 0.1363448f, 0.12958696f,
    0.122578636f, 0.11554942f, 0.10866067f, 0.10202015f, 0.0956953f,
    0.08972441f, 0.084125556f, 0.07890339f, 0.074054f, 0.06956819f, 0.06543367f,
    0.06163648f, 0.058161758f, 0.05499426f, 0.052118618f, 0.049519517f,
    0.047181763f, 0.045090348f, 0.043230496f, 0.041587707f, 0.0401478f,
    0.038896963f, 0.037821773f, 0.036909282f, 0.036147002f, 0.035522986f,
    0.03502582f, 0.034644656f, 0.034369234f, 0.12558633f, 0.123312004f,
    0.11712806f, 0.110826276f, 0.10457112f, 0.098478645f, 0.09262716f,
    0.087066196f, 0.08182402f, 0.07691362f, 0.07233741f, 0.06809075f,
    0.064164504f, 0.060546827f, 0.057224344f, 0.05418297f, 0.051408395f,
    0.048886385f, 0.046602957f, 0.04454447f, 0.042697683f, 0.041049767f,
    0.039588314f, 0.03830134f, 0.037177306f, 0.03620508f, 0.03537399f,
    0.034673788f, 0.03409468f, 0.03362731f, 0.03326278f, 0.032992646f,
    0.11401573f, 0.11189245f, 0.10620851f, 0.10050716f, 0.09490524f,
    0.089484565f, 0.08430001f, 0.07938597f, 0.074761555f, 0.070434645f,
    0.06640525f, 0.06266791f, 0.059213586f, 0.056031063f, 0.053107854f,
    0.050430905f, 0.047987025f, 0.04576318f, 0.04374666f, 0.04192518f,
    0.040286936f, 0.03882061f, 0.03751538f, 0.036360916f, 0.03534735f,
    0.034465272f, 0.033705708f, 0.033060107f, 0.032520328f, 0.032078627f,
    0.031727646f, 0.031460404f, 0.103836015f, 0.101850376f, 0.09660494f,
    0.09141744f, 0.08636958f, 0.08151736f, 0.07689742f, 0.072531804f,
    0.06843169f, 0.0646003f, 0.061035145f, 0.0577298f, 0.05467519f,
    0.051860623f, 0.04927448f, 0.046904758f, 0.044739403f, 0.042766567f,
    0.04097476f, 0.03935293f, 0.037890527f, 0.03657751f, 0.035404343f,
    0.034361996f, 0.033441912f, 0.032635998f, 0.03193659f, 0.031336438f,
    0.030828672f, 0.0304068f, 0.030064676f, 0.029796483f, 0.09484449f,
    0.0929842f, 0.08812532f, 0.08338039f, 0.07880497f, 0.07443557f, 0.07029475f,
    0.06639475f, 0.06274018f, 0.059330173f, 0.056159936f, 0.05322198f,
    0.05050703f, 0.048004754f, 0.045704253f, 0.04359444f, 0.041664302f,
    0.039903093f, 0.038300436f, 0.036846407f, 0.035531566f, 0.034346975f,
    0.03328419f, 0.032335266f, 0.03149271f, 0.030749481f, 0.030098952f,
    0.029534895f, 0.029051445f, 0.028643085f, 0.028304616f, 0.028031148f,
    0.086868696f, 0.08512246f, 0.08060579f, 0.07624434f, 0.07207394f,
    0.06811635f, 0.06438336f, 0.0608795f, 0.057604134f, 0.05455298f,
    0.051719207f, 0.049094304f, 0.046668712f, 0.0444323f, 0.042374708f,
    0.040485606f, 0.038754866f, 0.037172697f, 0.035729706f, 0.03441696f,
    0.03322599f, 0.032148838f, 0.031177992f, 0.030306421f, 0.029527532f,
    0.02883515f, 0.028223496f, 0.027687162f, 0.027221087f, 0.026820531f,
    0.026481051f, 0.026198486f, 0.07976248f, 0.07811998f, 0.0739074f,
    0.06988006f, 0.066058755f, 0.062454063f, 0.059069477f, 0.055903602f,
    0.052951667f, 0.050206635f, 0.04765999f, 0.04530232f, 0.04312376f,
    0.041114282f, 0.039263926f, 0.03756296f, 0.03600198f, 0.03457199f,
    0.03326444f, 0.03207124f, 0.03098478f, 0.029997913f, 0.029103952f,
    0.028296642f, 0.027570149f, 0.026919026f, 0.026338201f, 0.025822943f,
    0.025368843f, 0.02497179f, 0.02462795f, 0.024333745f, 0.07340242f,
    0.07185426f, 0.067912854f, 0.06417815f, 0.0606594f, 0.057358596f,
    0.054272912f, 0.051396396f, 0.04872109f, 0.0462378f, 0.043936685f,
    0.0418076f, 0.039840404f, 0.038025137f, 0.036352143f, 0.034812164f,
    0.033396386f, 0.032096468f, 0.030904552f, 0.029813275f, 0.028815733f,
    0.02790549f, 0.027076546f, 0.026323313f, 0.025640596f, 0.02502357f,
    0.024467751f, 0.02396897f, 0.023523362f, 0.023127332f, 0.022777539f,
    0.02247088f, 0.067684524f, 0.06622224f, 0.06252348f, 0.05904641f,
    0.055791333f, 0.052753605f, 0.049925573f, 0.047297813f, 0.04485994f,
    0.042601142f, 0.040510558f, 0.03857751f, 0.03679165f, 0.035143085f,
    0.033622418f, 0.032220777f, 0.030929832f, 0.029741788f, 0.028649371f,
    0.027645804f, 0.026724793f, 0.025880493f, 0.025107482f, 0.024400739f,
    0.023755614f, 0.023167796f, 0.0226333f, 0.022148432f, 0.021709776f,
    0.021314166f, 0.02095867f, 0.02064057f, 0.062521376f, 0.061137352f,
    0.057656504f, 0.054407462f, 0.05138347f, 0.04857476f, 0.04596999f,
    0.04355711f, 0.041323923f, 0.039258428f, 0.03734903f, 0.035584684f,
    0.03395495f, 0.03245001f, 0.031060712f, 0.029778529f, 0.02859554f,
    0.027504412f, 0.026498355f, 0.025571099f, 0.024716849f, 0.023930263f,
    0.023206409f, 0.02254074f, 0.02192906f, 0.0213675f, 0.02085249f,
    0.020380737f, 0.0199492f, 0.019555073f, 0.019195762f, 0.01886887f,
    0.05783958f, 0.05652701f, 0.053242646f, 0.050196487f, 0.047376215f,
    0.044768076f, 0.042357896f, 0.040131662f, 0.03807585f, 0.036177617f,
    0.034424882f, 0.032806385f, 0.031311672f, 0.029931068f, 0.028655658f,
    0.027477229f, 0.026388234f, 0.025381735f, 0.02445137f, 0.023591293f,
    0.022796145f, 0.022061002f, 0.021381348f, 0.020753037f, 0.02017226f,
    0.019635517f, 0.01913959f, 0.018681522f, 0.018258588f, 0.017868282f,
    0.017508296f, 0.017176498f, 0.053577583f, 0.05233042f, 0.049223945f,
    0.046359304f, 0.04371973f, 0.041288357f, 0.039048858f, 0.03698576f,
    0.035084587f, 0.03333193f, 0.031715434f, 0.030223748f, 0.028846487f,
    0.027574163f, 0.026398107f, 0.025310423f, 0.0243039f, 0.023371972f,
    0.022508644f, 0.02170845f, 0.020966401f, 0.020277945f, 0.019638918f,
    0.019045515f, 0.01849426f, 0.017981961f, 0.017505703f, 0.01706281f,
    0.016650826f, 0.016267499f, 0.015910756f, 0.015578695f, 0.04968374f,
    0.048496634f, 0.045551945f, 0.042850636f, 0.04037236f, 0.038097825f,
    0.036009055f, 0.034089517f, 0.032324087f, 0.030699011f, 0.029201793f,
    0.02782112f, 0.026546743f, 0.025369389f, 0.024280671f, 0.023272993f,
    0.022339476f, 0.02147389f, 0.020670585f, 0.019924434f, 0.019230781f,
    0.018585393f, 0.017984418f, 0.017424352f, 0.016901998f, 0.016414437f,
    0.01595901f, 0.015533281f, 0.015135022f, 0.014762195f, 0.01441293f,
    0.014085514f, 0.04611471f, 0.04498294f, 0.04218609f, 0.039632607f,
    0.037299268f, 0.035164855f, 0.033210166f, 0.031417903f, 0.02977253f,
    0.028260104f, 0.02686813f, 0.025585402f, 0.02440187f, 0.023308512f,
    0.022297224f, 0.021360723f, 0.02049245f, 0.019686494f, 0.018937526f,
    0.01824073f, 0.017591748f, 0.016986644f, 0.01642184f, 0.015894094f,
    0.01540046f, 0.014938259f, 0.014505048f, 0.014098604f, 0.013716895f,
    0.013358068f, 0.013020425f, 0.012702413f};

static const auto dielectrics_albedo_lut_t = vector<float>{0.111296214f,
    0.17022909f, 0.30654502f, 0.41846365f, 0.5105732f, 0.5866704f, 0.649758f,
    0.7022136f, 0.7459328f, 0.7824383f, 0.8129618f, 0.83850646f, 0.8598938f,
    0.8778002f, 0.8927852f, 0.90531325f, 0.91577166f, 0.92448413f, 0.931722f,
    0.9377134f, 0.9426504f, 0.946695f, 0.9499843f, 0.9526343f, 0.9547431f,
    0.95639426f, 0.9576586f, 0.9585962f, 0.9592585f, 0.9596889f, 0.95992464f,
    0.95999736f, 0.11165516f, 0.17042544f, 0.30660406f, 0.4184809f, 0.51057345f,
    0.5866632f, 0.64974755f, 0.70220214f, 0.7459214f, 0.7824274f, 0.81295174f,
    0.8384973f, 0.8598856f, 0.87779295f, 0.89277875f, 0.9053076f, 0.9157667f,
    0.92447984f, 0.9317183f, 0.93771017f, 0.9426476f, 0.9466926f, 0.9499822f,
    0.9526325f, 0.9547416f, 0.956393f, 0.95765746f, 0.9585953f, 0.9592577f,
    0.95968825f, 0.9599241f, 0.95999694f, 0.13748485f, 0.18528639f, 0.31145722f,
    0.42007947f, 0.51078534f, 0.5862327f, 0.64902306f, 0.7013606f, 0.74505633f,
    0.7815885f, 0.8121646f, 0.8377738f, 0.8592297f, 0.8772041f, 0.892254f,
    0.90484256f, 0.9153564f, 0.9241191f, 0.93140215f, 0.9374339f, 0.9424067f,
    0.9464831f, 0.9498004f, 0.9524749f, 0.95460534f, 0.95627546f, 0.95755625f,
    0.9585084f, 0.95918316f, 0.95962447f, 0.95986974f, 0.9599507f, 0.20445478f,
    0.22795731f, 0.3275091f, 0.4262126f, 0.5123517f, 0.585529f, 0.6471836f,
    0.6989896f, 0.7424892f, 0.7790142f, 0.80968845f, 0.83545184f, 0.8570887f,
    0.8752529f, 0.8904909f, 0.90325963f, 0.9139424f, 0.922861f, 0.9302863f,
    0.9364469f, 0.94153565f, 0.9457159f, 0.94912577f, 0.95188266f, 0.9540861f,
    0.95582074f, 0.9571585f, 0.95816076f, 0.9588797f, 0.9593598f, 0.9596391f,
    0.9597498f, 0.30382243f, 0.29982153f, 0.3596935f, 0.4404058f, 0.5173496f,
    0.5855734f, 0.6445333f, 0.6949216f, 0.7377512f, 0.7740512f, 0.8047655f,
    0.8307241f, 0.85264224f, 0.8711307f, 0.8867085f, 0.8998156f, 0.9108247f,
    0.9200515f, 0.9277638f, 0.9341886f, 0.93951875f, 0.94391793f, 0.9475256f,
    0.9504603f, 0.95282316f, 0.9547003f, 0.95616513f, 0.9572805f, 0.9580999f,
    0.9586691f, 0.95902723f, 0.9592078f, 0.41140005f, 0.38861263f, 0.40758866f,
    0.46469206f, 0.5279169f, 0.58802235f, 0.64217365f, 0.68977344f, 0.73107445f,
    0.7666416f, 0.7971269f, 0.82317275f, 0.84537244f, 0.8642563f, 0.88029015f,
    0.89387876f, 0.90537184f, 0.9150704f, 0.923233f, 0.9300811f, 0.93580496f,
    0.9405673f, 0.9445077f, 0.9477458f, 0.9503839f, 0.95251f, 0.9541994f,
    0.9555167f, 0.95651746f, 0.95724916f, 0.95775265f, 0.95806307f, 0.5082769f,
    0.47826555f, 0.4656997f, 0.4982523f, 0.54503137f, 0.59435576f, 0.6415341f,
    0.6847154f, 0.72331655f, 0.7573425f, 0.7870672f, 0.8128752f, 0.83518296f,
    0.8543983f, 0.8709024f, 0.8850412f, 0.8971237f, 0.9074231f, 0.91617906f,
    0.9236009f, 0.9298707f, 0.93514675f, 0.9395662f, 0.9432478f, 0.9462945f,
    0.9487951f, 0.9508266f, 0.9524554f, 0.953739f, 0.95472705f, 0.9554624f,
    0.9559821f, 0.58709335f, 0.55747324f, 0.526487f, 0.53781915f, 0.5679246f,
    0.6050981f, 0.6436968f, 0.6809706f, 0.71562314f, 0.7471205f, 0.7753375f,
    0.8003659f, 0.8224077f, 0.8417145f, 0.8585531f, 0.87318647f, 0.8858633f,
    0.89681315f, 0.90624446f, 0.9143443f, 0.92127943f, 0.9271976f, 0.93222904f,
    0.9364885f, 0.94007665f, 0.9430816f, 0.9455806f, 0.9476409f, 0.9493215f,
    0.9506737f, 0.9517423f, 0.9525663f, 0.647614f, 0.6216479f, 0.5833049f,
    0.579107f, 0.5944615f, 0.61960703f, 0.6489445f, 0.67932564f, 0.70899606f,
    0.73700875f, 0.7628935f, 0.7864646f, 0.80770516f, 0.8266966f, 0.8435742f,
    0.8585002f, 0.87164634f, 0.8831838f, 0.8932768f, 0.9020797f, 0.90973467f,
    0.91637146f, 0.92210764f, 0.9270488f, 0.9312894f, 0.9349137f, 0.93799675f,
    0.9406048f, 0.9427968f, 0.94462466f, 0.9461343f, 0.94736624f, 0.6922388f,
    0.670658f, 0.631945f, 0.61815274f, 0.62194157f, 0.63639545f, 0.65673697f,
    0.6799053f, 0.7039825f, 0.72778565f, 0.75060445f, 0.7720345f, 0.79186994f,
    0.8100333f, 0.8265296f, 0.84141487f, 0.8547757f, 0.8667153f, 0.8773443f,
    0.88677454f, 0.89511544f, 0.9024712f, 0.9089398f, 0.9146121f, 0.91957164f,
    0.9238946f, 0.9276502f, 0.93090117f, 0.9337038f, 0.9361087f, 0.9381615f,
    0.93990266f, 0.7238502f, 0.7063026f, 0.67068905f, 0.6521104f, 0.6478202f,
    0.65363055f, 0.66597444f, 0.68223536f, 0.7005915f, 0.71980345f, 0.7390424f,
    0.7577664f, 0.7756329f, 0.7924384f, 0.80807555f, 0.8225034f, 0.83572596f,
    0.8477771f, 0.85870993f, 0.86858886f, 0.8774845f, 0.88546985f, 0.8926175f,
    0.89899826f, 0.90467966f, 0.9097256f, 0.91419566f, 0.91814524f, 0.92162526f,
    0.92468256f, 0.92735994f, 0.9296963f, 0.7450496f, 0.73084825f, 0.6995553f,
    0.67944264f, 0.6701545f, 0.6695661f, 0.6753321f, 0.685456f, 0.69839036f,
    0.71298563f, 0.72840655f, 0.7440566f, 0.7595171f, 0.7745015f, 0.78882015f,
    0.80235493f, 0.8150396f, 0.8268457f, 0.8377717f, 0.8478351f, 0.85706633f,
    0.8655042f, 0.87319285f, 0.88017917f, 0.8865111f, 0.8922362f, 0.8974009f,
    0.90204996f, 0.9062259f, 0.9099688f, 0.9133163f, 0.9163034f, 0.7579677f,
    0.74640614f, 0.7194635f, 0.6996969f, 0.68775344f, 0.6828176f, 0.68354195f,
    0.6885581f, 0.69667584f, 0.70692885f, 0.7185602f, 0.7309897f, 0.7437793f,
    0.7566034f, 0.76922417f, 0.78147143f, 0.7932275f, 0.8044146f, 0.8149855f,
    0.82491636f, 0.83420026f, 0.84284323f, 0.85086066f, 0.8582744f, 0.8651108f,
    0.87139904f, 0.87716997f, 0.882455f, 0.88728553f, 0.89169264f, 0.8957062f,
    0.8993551f, 0.7642785f, 0.7547403f, 0.731649f, 0.71312976f, 0.7001055f,
    0.69247067f, 0.68957514f, 0.69057953f, 0.6946498f, 0.70104194f, 0.7091277f,
    0.7183929f, 0.728425f, 0.738898f, 0.7495573f, 0.76020706f, 0.7706984f,
    0.7809207f, 0.79079336f, 0.80026007f, 0.8092835f, 0.8178414f, 0.8259232f,
    0.83352727f, 0.8406589f, 0.84732866f, 0.8535509f, 0.8593425f, 0.8647225f,
    0.86971086f, 0.8743282f, 0.87859535f, 0.76527375f, 0.75725543f, 0.7373439f,
    0.72036695f, 0.7072011f, 0.6980607f, 0.6927215f, 0.69073206f, 0.69156086f,
    0.6946802f, 0.6996079f, 0.7059227f, 0.71326697f, 0.72134244f, 0.729904f,
    0.73875225f, 0.74772686f, 0.7567002f, 0.765572f, 0.77426434f, 0.7827181f,
    0.7908895f, 0.7987471f, 0.80626965f, 0.8134441f, 0.8202639f, 0.82672805f,
    0.83283925f, 0.8386038f, 0.8440302f, 0.849129f, 0.85391206f, 0.76194525f,
    0.7550482f, 0.7376428f, 0.7221682f, 0.7093431f, 0.6994822f, 0.69258755f,
    0.6884604f, 0.68679595f, 0.68724966f, 0.6894775f, 0.6931578f, 0.69800144f,
    0.7037551f, 0.71020055f, 0.7171521f, 0.7244536f, 0.7319748f, 0.73960775f,
    0.747264f, 0.7548718f, 0.76237327f, 0.7697226f, 0.776884f, 0.7838302f,
    0.790541f, 0.7970019f, 0.8032036f, 0.8091407f, 0.8148111f, 0.8202154f,
    0.8253565f, 0.75505805f, 0.74897116f, 0.7334674f, 0.7192934f, 0.7069948f,
    0.69687927f, 0.68904746f, 0.6834464f, 0.67992437f, 0.67827517f, 0.6782702f,
    0.67967945f, 0.68228346f, 0.6858801f, 0.6902875f, 0.69534445f, 0.70090985f,
    0.7068615f, 0.7130941f, 0.7195179f, 0.7260565f, 0.73264575f, 0.7392317f,
    0.7457697f, 0.7522229f, 0.7585614f, 0.76476103f, 0.7708028f, 0.77667195f,
    0.78235734f, 0.78785115f, 0.7931481f, 0.74520963f, 0.73969173f, 0.7255762f,
    0.7124407f, 0.700679f, 0.6905483f, 0.68217546f, 0.67558f, 0.67070246f,
    0.66743153f, 0.6656259f, 0.66513073f, 0.66578877f, 0.66744757f, 0.6699638f,
    0.6732054f, 0.67705256f, 0.681398f, 0.686146f, 0.69121236f, 0.6965231f,
    0.7020136f, 0.7076279f, 0.7133176f, 0.71904093f, 0.72476256f, 0.7304521f,
    0.7360842f, 0.7416375f, 0.74709433f, 0.75244015f, 0.75766325f, 0.7328761f,
    0.727741f, 0.7145914f, 0.7022267f, 0.6909194f, 0.68086535f, 0.6721816f,
    0.6649147f, 0.6590542f, 0.65454733f, 0.65131277f, 0.64925206f, 0.64825803f,
    0.64822125f, 0.6490341f, 0.6505936f, 0.6528032f, 0.6555735f, 0.65882266f,
    0.66247624f, 0.6664672f, 0.67073536f, 0.6752269f, 0.67989415f, 0.68469465f,
    0.6895911f, 0.69455075f, 0.6995447f, 0.704548f, 0.7095389f, 0.7144985f,
    0.7194108f, 0.7184469f, 0.71355134f, 0.7010275f, 0.6891876f, 0.67821145f,
    0.6682375f, 0.6593583f, 0.6516223f, 0.64503956f, 0.63958937f, 0.6352279f,
    0.6318952f, 0.629521f, 0.62802976f, 0.6273438f, 0.627386f, 0.6280818f,
    0.62936f, 0.63115364f, 0.6334003f, 0.6360424f, 0.63902694f, 0.6423054f,
    0.6458337f, 0.64957184f, 0.65348375f, 0.65753686f, 0.66170186f, 0.66595256f,
    0.67026556f, 0.6746199f, 0.67899716f, 0.70224965f, 0.69748414f, 0.68531704f,
    0.6737891f, 0.66301155f, 0.65307426f, 0.64404166f, 0.6359521f, 0.62881976f,
    0.6226382f, 0.617384f, 0.61302114f, 0.60950434f, 0.6067824f, 0.60480064f,
    0.6035031f, 0.60283375f, 0.60273784f, 0.60316265f, 0.6040578f, 0.6053758f,
    0.60707206f, 0.60910505f, 0.6114361f, 0.61402965f, 0.6168527f, 0.6198751f,
    0.6230691f, 0.6264095f, 0.62987304f, 0.6334388f, 0.63708764f, 0.6845687f,
    0.6798502f, 0.6678307f, 0.65643865f, 0.6457349f, 0.6357708f, 0.62658435f,
    0.61819947f, 0.61062574f, 0.60385996f, 0.59788805f, 0.5926868f, 0.5882262f,
    0.5844711f, 0.58138317f, 0.5789219f, 0.577046f, 0.57571405f, 0.5748856f,
    0.57452106f, 0.57458246f, 0.5750336f, 0.57584006f, 0.57696944f, 0.5783913f,
    0.58007705f, 0.5820001f, 0.5841356f, 0.58646053f, 0.5889535f, 0.5915948f,
    0.5943661f, 0.66565686f, 0.6609236f, 0.6488924f, 0.63749576f, 0.626758f,
    0.61670035f, 0.6073386f, 0.5986815f, 0.5907305f, 0.5834799f, 0.5769175f,
    0.571026f, 0.56578314f, 0.5611634f, 0.5571387f, 0.55367917f, 0.550754f,
    0.5483321f, 0.54638225f, 0.54487383f, 0.54377705f, 0.54306304f, 0.542704f,
    0.54267347f, 0.5429462f, 0.5434983f, 0.5443071f, 0.5453513f, 0.5466109f,
    0.548067f, 0.549702f, 0.5514993f, 0.6457439f, 0.6409508f, 0.62878996f,
    0.617281f, 0.60642153f, 0.5962103f, 0.58664465f, 0.57771987f, 0.56942815f,
    0.56175876f, 0.5546979f, 0.54822874f, 0.5423326f, 0.53698856f, 0.5321743f,
    0.5278665f, 0.5240411f, 0.52067405f, 0.51774096f, 0.51521784f, 0.51308113f,
    0.51130795f, 0.50987613f, 0.5087641f, 0.50795144f, 0.5074184f, 0.50714636f,
    0.50711733f, 0.5073145f, 0.5077217f, 0.5083239f, 0.5091067f, 0.62504107f,
    0.6201566f, 0.6077822f, 0.596082f, 0.58503425f, 0.57462054f, 0.56482404f,
    0.5556284f, 0.5470174f, 0.53897417f, 0.5314814f, 0.524521f, 0.5180741f,
    0.5121215f, 0.5066435f, 0.5016201f, 0.49703133f, 0.49285716f, 0.48907778f,
    0.48567373f, 0.48262593f, 0.47991577f, 0.47752526f, 0.475437f, 0.47363412f,
    0.4721006f, 0.47082093f, 0.4697804f, 0.46896487f, 0.46836093f, 0.46795583f,
    0.46773735f, 0.603744f, 0.5987474f, 0.58610326f, 0.5741578f, 0.5628749f,
    0.55222344f, 0.54217535f, 0.53270525f, 0.5237893f, 0.5154051f, 0.50753117f,
    0.5001467f, 0.49323156f, 0.48676622f, 0.48073146f, 0.4751086f, 0.4698794f,
    0.4650261f, 0.46053147f, 0.45637873f, 0.45255157f, 0.4490344f, 0.44581196f,
    0.44286975f, 0.44019374f, 0.4377705f, 0.43558705f, 0.43363115f, 0.431891f,
    0.4303553f, 0.42901334f, 0.4278549f, 0.58203363f, 0.57691306f, 0.5639656f,
    0.5517416f, 0.54019445f, 0.5292833f, 0.51897174f, 0.50922704f, 0.50001925f,
    0.4913211f, 0.48310706f, 0.47535357f, 0.46803838f, 0.46114063f, 0.45464054f,
    0.4485194f, 0.44275942f, 0.4373437f, 0.43225598f, 0.42748094f, 0.42300385f,
    0.4188107f, 0.4148881f, 0.41122317f, 0.40780383f, 0.40461835f, 0.40165564f,
    0.3989051f, 0.3963566f, 0.39400056f, 0.3918278f, 0.38982952f, 0.560077f,
    0.5548272f, 0.54156107f, 0.52904254f, 0.51721716f, 0.50603706f, 0.49545926f,
    0.48544547f, 0.47596097f, 0.46697426f, 0.45845658f, 0.4503815f, 0.4427247f,
    0.4354637f, 0.4285776f, 0.42204696f, 0.41585365f, 0.40998065f, 0.4044121f,
    0.39913294f, 0.39412916f, 0.38938746f, 0.38489532f, 0.3806409f, 0.37661296f,
    0.37280098f, 0.3691949f, 0.36578518f, 0.36256287f, 0.35951933f, 0.35664654f,
    0.3539367f, 0.5380265f, 0.5326478f, 0.51906174f, 0.50624657f, 0.49414197f,
    0.48269454f, 0.47185668f, 0.4615857f, 0.45184314f, 0.44259432f, 0.43380767f,
    0.42545447f, 0.41750842f, 0.4099455f, 0.4027435f, 0.39588204f, 0.3893423f,
    0.3831068f, 0.37715933f, 0.3714848f, 0.36606914f, 0.36089927f, 0.3559629f,
    0.3512486f, 0.34674555f, 0.34244367f, 0.33833346f, 0.33440593f, 0.33065268f,
    0.32706577f, 0.3236377f, 0.32036138f, 0.5160199f, 0.5105166f, 0.49662063f,
    0.48351777f, 0.47114322f, 0.45943958f, 0.44835562f, 0.43784547f,
    0.42786786f, 0.41838557f, 0.4093649f, 0.40077522f, 0.39258868f, 0.3847798f,
    0.37732536f, 0.37020388f, 0.3633958f, 0.35688302f, 0.3506488f, 0.3446777f,
    0.33895537f, 0.3334686f, 0.32820496f, 0.323153f, 0.31830198f, 0.3136419f,
    0.3091634f, 0.3048577f, 0.3007166f, 0.2967324f, 0.29289785f, 0.28920618f,
    0.49417984f, 0.48855942f, 0.474372f, 0.46099883f, 0.4483719f, 0.4364307f,
    0.42512137f, 0.4143956f, 0.4042101f, 0.3945257f, 0.38530707f, 0.37652218f,
    0.36814186f, 0.36013958f, 0.3524911f, 0.34517422f, 0.33816853f, 0.33145535f,
    0.32501745f, 0.3188389f, 0.31290504f, 0.30720228f, 0.301718f, 0.2964405f,
    0.29135892f, 0.2864631f, 0.28174365f, 0.27719173f, 0.2727991f, 0.26855808f,
    0.26446146f, 0.26050246f, 0.47261402f, 0.46688628f, 0.452432f, 0.4388124f,
    0.42595685f, 0.4138027f, 0.4022941f, 0.391381f, 0.3810184f, 0.37116587f,
    0.36178684f, 0.35284814f, 0.3443197f, 0.33617413f, 0.32838637f, 0.32093358f,
    0.31379482f, 0.30695087f, 0.30038396f, 0.29407784f, 0.28801745f,
    0.28218886f, 0.2765792f, 0.27117652f, 0.26596972f, 0.26094848f, 0.25610322f,
    0.25142494f, 0.2469053f, 0.24253647f, 0.23831117f, 0.23422255f};

// Microfacet energy compensation (E(cos(w)))
inline vec3f microfacet_compensation_conductors(const vector<float>& E_lut,
    const vec3f& color, float roughness, const vec3f& normal,
    const vec3f& outgoing) {
  auto s = abs(dot(normal, outgoing)) * (ALBEDO_LUT_SIZE - 1);
  auto t = sqrt(roughness) * (ALBEDO_LUT_SIZE - 1);

  // get image coordinates and residuals
  auto i = (int)s, j = (int)t;
  auto ii = min(i + 1, ALBEDO_LUT_SIZE - 1);
  auto jj = min(j + 1, ALBEDO_LUT_SIZE - 1);
  auto u = s - i, v = t - j;

  // handle interpolation
  auto E = E_lut[j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) +
           E_lut[jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v +
           E_lut[j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) +
           E_lut[jj * ALBEDO_LUT_SIZE + ii] * u * v;

  return 1 + color * (1 - E) / E;
}

// Microfacet energy compensation (E(cos(w)))
inline vec3f microfacet_compensation_dielectrics(const vec3f& color,
    float roughness, const vec3f& normal, const vec3f& outgoing) {
  auto E1_lut = dielectrics_albedo_lut_r;
  auto E2_lut = dielectrics_albedo_lut_t;

  auto s = abs(dot(normal, outgoing)) * (ALBEDO_LUT_SIZE - 1);
  auto t = sqrt(roughness) * (ALBEDO_LUT_SIZE - 1);

  // get image coordinates and residuals
  auto i = (int)s, j = (int)t;
  auto ii = min(i + 1, ALBEDO_LUT_SIZE - 1);
  auto jj = min(j + 1, ALBEDO_LUT_SIZE - 1);
  auto u = s - i, v = t - j;

  // handle interpolation
  auto E1 = E1_lut[j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) +
            E1_lut[jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v +
            E1_lut[j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) +
            E1_lut[jj * ALBEDO_LUT_SIZE + ii] * u * v;
  auto E2 = E2_lut[j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) +
            E2_lut[jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v +
            E2_lut[j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) +
            E2_lut[jj * ALBEDO_LUT_SIZE + ii] * u * v;

  return 1 + color * (1 - (E1 + E2)) / (E1 + E2);
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

// Sample a specular BRDF lobe.
inline vec3f sample_specular(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  if (rnl < fresnel_dielectric(ior, up_normal, outgoing)) {
    auto halfway = sample_microfacet(roughness, up_normal, rn);
    return reflect(outgoing, halfway);
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
      conductors_albedo_lut, color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

inline vec3f eval_metallic_comp_mytab(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  auto C = microfacet_compensation_conductors(
      my_conductors_albedo_lut, color, roughness, normal, outgoing);
  return C * eval_metallic(color, roughness, normal, outgoing, incoming);
}

// Sample a metal BRDF lobe.
inline vec3f sample_metallic(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  return reflect(outgoing, halfway);
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
    auto halfway = sample_microfacet(roughness, up_normal, rn);
    return reflect(outgoing, halfway);
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
    auto F       = fresnel_dielectric(ior, halfway, incoming);
    auto D       = microfacet_distribution(roughness, up_normal, halfway);
    auto G       = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, incoming);
    return vec3f{1, 1, 1} * F * D * G /
           (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
           abs(dot(up_normal, incoming));
  } else {
    auto reflected = reflect(-incoming, up_normal);
    auto halfway   = normalize(reflected + outgoing);
    auto F         = fresnel_dielectric(ior, halfway, incoming);
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
      color, roughness, normal, outgoing);
  return C *
         eval_transparent(color, ior, roughness, normal, outgoing, incoming);
}

// Sample a transmission BRDF lobe.
inline vec3f sample_transparent(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
  auto up_normal = dot(normal, outgoing) <= 0 ? -normal : normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  if (rnl < fresnel_dielectric(ior, halfway, outgoing)) {
    return reflect(outgoing, halfway);
  } else {
    auto reflected = reflect(outgoing, halfway);
    return -reflect(reflected, up_normal);
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
  auto C = microfacet_compensation_dielectrics(
      color, roughness, normal, outgoing);
  return C * eval_refractive(color, ior, roughness, normal, outgoing, incoming);
}

// Sample a refraction BRDF lobe.
inline vec3f sample_refractive(const vec3f& color, float ior, float roughness,
    const vec3f& normal, const vec3f& outgoing, float rnl, const vec2f& rn) {
  auto entering  = dot(normal, outgoing) >= 0;
  auto up_normal = entering ? normal : -normal;
  auto halfway   = sample_microfacet(roughness, up_normal, rn);
  if (rnl < fresnel_dielectric(entering ? ior : (1 / ior), halfway, outgoing)) {
    return reflect(outgoing, halfway);
  } else {
    return refract(outgoing, halfway, entering ? (1 / ior) : ior);
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
           abs(dot(halfway, outgoing)) /
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
