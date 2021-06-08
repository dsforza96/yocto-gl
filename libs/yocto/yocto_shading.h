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

// Microfacet energy compensation (E(cos(w)))
inline vec3f microfacet_compensation_tab(const vec3f& color, float roughness,
    const vec3f& normal, const vec3f& outgoing) {
  auto s = abs(dot(normal, outgoing)) * ALBEDO_LUT_SIZE;
  auto t = sqrt(roughness) * ALBEDO_LUT_SIZE;

  // get image coordinates and residuals
  auto i  = clamp((int)s, 0, ALBEDO_LUT_SIZE - 1);
  auto j  = clamp((int)t, 0, ALBEDO_LUT_SIZE - 1);
  auto ii = min(i + 1, ALBEDO_LUT_SIZE - 1);
  auto jj = min(j + 1, ALBEDO_LUT_SIZE - 1);
  auto u = s - i, v = t - j;

  // handle interpolation
  auto E = albedo_lut[j * ALBEDO_LUT_SIZE + i] * (1 - u) * (1 - v) +
           albedo_lut[jj * ALBEDO_LUT_SIZE + i] * (1 - u) * v +
           albedo_lut[j * ALBEDO_LUT_SIZE + ii] * u * (1 - v) +
           albedo_lut[jj * ALBEDO_LUT_SIZE + ii] * u * v;

  return 1 + color * (1 - E) / E;
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
  auto C = microfacet_compensation_tab(color, roughness, normal, outgoing);
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
