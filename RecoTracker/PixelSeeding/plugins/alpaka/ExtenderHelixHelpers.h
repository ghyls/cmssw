#ifndef RecoTracker_PixelSeeding_plugins_alpaka_ExtenderHelixHelpers_h
#define RecoTracker_PixelSeeding_plugins_alpaka_ExtenderHelixHelpers_h

// Helix-propagation helpers for the OT hit-attach walk (sole consumer: CAExtension.dev.cc).
// Conventions match BrokenLine.h so arc-length signs / PCA position / curvature direction agree.

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct HelixState {
    float phi0;
    float tip;
    float cotTheta;
    float zip;
    float rho;
    float xc, yc;
    float alphaOrigin;
  };

  // Build a HelixState from raw BL state parameters (phi0, tip, invPt, cotTheta, zip).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixState
  makeHelixStateFromParams(TAcc const& acc, float phi0, float tip, float invPt, float cotTheta, float zip, float bf) {
    HelixState h;
    h.phi0 = phi0;
    h.tip = tip;
    h.cotTheta = cotTheta;
    h.zip = zip;
    h.rho = (invPt != 0.f) ? 1.f / (invPt * bf) : 1e9f;
    const float sp = alpaka::math::sin(acc, h.phi0);
    const float cp = alpaka::math::cos(acc, h.phi0);
    const float x0 = tip * sp;
    const float y0 = -tip * cp;
    h.xc = x0 + h.rho * sp;
    h.yc = y0 - h.rho * cp;
    h.alphaOrigin = alpaka::math::atan2(acc, -h.yc, -h.xc);
    return h;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixState
  makeHelixState(TAcc const& acc, const ::reco::TrackSoAConstView tracks, int i, float bf) {
    return makeHelixStateFromParams(acc,
                                    tracks[i].state()(0),
                                    tracks[i].state()(1),
                                    tracks[i].state()(2),
                                    tracks[i].state()(3),
                                    tracks[i].state()(4),
                                    bf);
  }

  struct Prediction {
    float phi;        // azimuth of the intersection (rad in [-pi, pi])
    float secondary;  // z for barrel prediction, r for endcap prediction
    float arcS;       // signed forward arc length from origin direction
    float branch;     // barrel: +1/-1, which of the two cylinder roots was taken (endcap: 0)
    bool valid;
  };

  static constexpr float kExtenderPi = 3.14159265f;
  static constexpr float kExtenderMaxArcLengthCm = 250.f;

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float foldPi(float a) {
    if (a > kExtenderPi)
      a -= 2.f * kExtenderPi;
    if (a < -kExtenderPi)
      a += 2.f * kExtenderPi;
    return a;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Prediction predictOnBarrel(TAcc const& acc, HelixState const& h, float R) {
    Prediction out{0.f, 0.f, 0.f, 0.f, false};
    const float dc2 = h.xc * h.xc + h.yc * h.yc;
    const float absRho = alpaka::math::abs(acc, h.rho);
    const float dc = alpaka::math::sqrt(acc, dc2);
    if (R < dc - absRho - 1e-3f || R > dc + absRho + 1e-3f)
      return out;
    const float K = R * R + dc2 - h.rho * h.rho;
    float c = K / (2.f * dc * R);
    if (c > 1.f)
      c = 1.f;
    if (c < -1.f)
      c = -1.f;
    const float phi_c = alpaka::math::atan2(acc, h.yc, h.xc);
    const float dphi = alpaka::math::acos(acc, c);
    const float thetaA = phi_c + dphi;
    const float thetaB = phi_c - dphi;
    auto arcAt = [&](float theta) {
      const float x = R * alpaka::math::cos(acc, theta);
      const float y = R * alpaka::math::sin(acc, theta);
      const float alphaH = alpaka::math::atan2(acc, y - h.yc, x - h.xc);
      return h.rho * foldPi(h.alphaOrigin - alphaH);
    };
    const float sA = arcAt(thetaA);
    const float sB = arcAt(thetaB);
    float chosenS, chosenTheta, chosenBranch;
    const bool aOk = sA > 0.f;
    const bool bOk = sB > 0.f;
    if (aOk && bOk) {
      if (sA <= sB) {
        chosenS = sA;
        chosenTheta = thetaA;
        chosenBranch = 1.f;
      } else {
        chosenS = sB;
        chosenTheta = thetaB;
        chosenBranch = -1.f;
      }
    } else if (aOk) {
      chosenS = sA;
      chosenTheta = thetaA;
      chosenBranch = 1.f;
    } else if (bOk) {
      chosenS = sB;
      chosenTheta = thetaB;
      chosenBranch = -1.f;
    } else {
      return out;
    }
    if (chosenS > kExtenderMaxArcLengthCm)
      return out;
    out.phi = foldPi(chosenTheta);
    out.secondary = h.zip + chosenS * h.cotTheta;
    out.arcS = chosenS;
    out.branch = chosenBranch;
    out.valid = true;
    return out;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Prediction predictOnEndcap(TAcc const& acc, HelixState const& h, float zLayer) {
    Prediction out{0.f, 0.f, 0.f, 0.f, false};
    if (alpaka::math::abs(acc, h.cotTheta) < 1e-4f)
      return out;
    const float arcS = (zLayer - h.zip) / h.cotTheta;
    if (arcS <= 0.f || arcS > kExtenderMaxArcLengthCm)
      return out;
    const float alphaH = h.alphaOrigin - arcS / h.rho;
    const float absRho = alpaka::math::abs(acc, h.rho);
    const float x = h.xc + absRho * alpaka::math::cos(acc, alphaH);
    const float y = h.yc + absRho * alpaka::math::sin(acc, alphaH);
    const float r = alpaka::math::sqrt(acc, x * x + y * y);
    out.phi = alpaka::math::atan2(acc, y, x);
    out.secondary = r;
    out.arcS = arcS;
    out.valid = true;
    return out;
  }

  // dphi/dr of the fitted helix at the layer surface (the track side of the stub-bend row), the
  // closed-form derivative of the crossing solve in predictOnBarrel/predictOnEndcap, kept separate to
  // preserve that path's floating-point contraction. Returns 0 when the crossing is degenerate.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extBendPredDPhiDr(TAcc const& acc,
                                                         HelixState const& h,
                                                         bool isBarrel,
                                                         float surf) {
    constexpr float kEps = 1.0e-6f;
    const Prediction p = isBarrel ? predictOnBarrel(acc, h, surf) : predictOnEndcap(acc, h, surf);
    if (!p.valid)
      return 0.f;
    const float absRho = alpaka::math::abs(acc, h.rho);
    if (isBarrel) {
      const float dc2 = h.xc * h.xc + h.yc * h.yc;
      const float dc = alpaka::math::sqrt(acc, dc2);
      if (!(dc > kEps) || !(surf > kEps))
        return 0.f;
      const float Aa = dc2 - h.rho * h.rho;
      float c0 = (surf * surf + Aa) / (2.f * dc * surf);
      c0 = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, -1.f, c0));
      const float sq2 = 1.f - c0 * c0;
      if (!(sq2 > 1.0e-8f))
        return 0.f;
      const float sq = alpaka::math::sqrt(acc, sq2);
      const float cp1 = 1.f / (2.f * dc) - Aa / (2.f * dc * surf * surf);
      const float phic = alpaka::math::atan2(acc, h.yc, h.xc);
      const float sBranch = (foldPi(p.phi - phic) >= 0.f) ? 1.f : -1.f;
      return -sBranch * cp1 / sq;  // dphi/dR
    }
    const float cot = h.cotTheta;
    if (!(alpaka::math::abs(acc, cot) > 1.0e-4f) || !(absRho > kEps))
      return 0.f;
    const float a1 = -1.f / (h.rho * cot);  // dalphaH/dz, constant in z
    const float alphaH = h.alphaOrigin - p.arcS / h.rho;
    const float sA = alpaka::math::sin(acc, alphaH);
    const float cA = alpaka::math::cos(acc, alphaH);
    const float x = h.xc + absRho * cA;
    const float y = h.yc + absRho * sA;
    const float r2 = x * x + y * y;
    if (!(r2 > 1.0e-6f))
      return 0.f;
    const float rr = alpaka::math::sqrt(acc, r2);
    const float xp = -absRho * sA * a1;
    const float yp = absRho * cA * a1;
    const float dphidz = (x * yp - y * xp) / r2;
    const float drdz = (x * xp + y * yp) / rr;
    return (alpaka::math::abs(acc, drdz) > 1.0e-9f) ? dphidz / drdz : 0.f;
  }

  // Analytic state Jacobian of the bend prediction, H_b = d(dphi/dr)/d(d0, 1/pT, cot, z0); the phi0
  // partial is zero by rotation invariance. `g` is the value extBendPredDPhiDr returned (carries the
  // branch sign). Returns false if degenerate.
  // rho = 1/(invPt*bf), T = d0 + rho, A = |T|, P = |rho|, rho_i = -rho^2 bf, dA/dd0 = sgn(T),
  // dA/d(invPt) = sgn(T) rho_i.
  // Barrel (radius R): Aa = A^2 - rho^2, c = (R^2 + Aa)/(2 A R), s = sqrt(1 - c^2),
  // c' = 1/(2A) - Aa/(2 A R^2), g = -sigma c'/s:
  //     dg/du = g ( (dc'/du)/c' + c (dc/du)/s^2 )
  //     dc/du = (dAa/du)/(2 A R) - c (dA/du)/A,   dc'/du = -(dA/du / A) c' - (dAa/du)/(2 A R^2),
  //     dAa = 2 A dA - 2 rho drho;  dH_b/dcot == dH_b/dz0 == 0.
  // Endcap (surface z): arc = (z - z0)/cot, beta = arc/rho, N = P - A cos(beta), D = -A sin(beta),
  // r^2 = A^2 + P^2 - 2 A P cos(beta), g = N / (D r):
  //     dN = dP - cos(beta) dA + A sin(beta) dbeta,   dD = -sin(beta) dA - A cos(beta) dbeta,
  //     dr = [A dA + P dP - cos(beta)(A dP + P dA) + A P sin(beta) dbeta] / r,
  //     dg/du = g ( dN/N - dD/D - dr/r ),
  //     dP/d(invPt) = sgn(rho) rho_i, dbeta/d(invPt) = arc bf, dbeta/dcot = -beta/cot, dbeta/dz0 = -1/(rho cot).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool extBendPredDPhiDrGrad(TAcc const& acc,
                                                            HelixState const& h,
                                                            bool isBarrel,
                                                            float surf,
                                                            float bf,
                                                            float g,
                                                            float& hTip,
                                                            float& hInvPt,
                                                            float& hCot,
                                                            float& hZip) {
    constexpr float kEpsG = 1.0e-6f;
    hTip = 0.f;
    hInvPt = 0.f;
    hCot = 0.f;
    hZip = 0.f;
    if (g == 0.f || !alpaka::math::isfinite(acc, g))
      return false;
    const float rho = h.rho;
    const float T = h.tip + rho;
    const float sT = (T >= 0.f) ? 1.f : -1.f;
    const float A = alpaka::math::sqrt(acc, h.xc * h.xc + h.yc * h.yc);
    if (!(A > kEpsG))
      return false;
    const float rhoI = -rho * rho * bf;  // d rho / d(1/pT)
    const float dATip = sT;
    const float dAPt = sT * rhoI;
    if (isBarrel) {
      const float R = surf;
      if (!(R > kEpsG))
        return false;
      const float Aa = A * A - rho * rho;
      float c = (R * R + Aa) / (2.f * A * R);
      c = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, -1.f, c));
      const float s2 = 1.f - c * c;
      if (!(s2 > 1.0e-8f))
        return false;
      const float cp = 1.f / (2.f * A) - Aa / (2.f * A * R * R);
      if (!(alpaka::math::abs(acc, cp) > 0.f))
        return false;
      // eq (1), once per non-trivial parameter. drho is zero for d0 and rhoI for 1/pT.
      const float dAaTip = 2.f * A * dATip;
      const float dcTip = dAaTip / (2.f * A * R) - c * dATip / A;
      const float dcpTip = -(dATip / A) * cp - dAaTip / (2.f * A * R * R);
      hTip = g * (dcpTip / cp + c * dcTip / s2);
      const float dAaPt = 2.f * A * dAPt - 2.f * rho * rhoI;
      const float dcPt = dAaPt / (2.f * A * R) - c * dAPt / A;
      const float dcpPt = -(dAPt / A) * cp - dAaPt / (2.f * A * R * R);
      hInvPt = g * (dcpPt / cp + c * dcPt / s2);
      return true;  // eq (2): the cot and z0 partials stay at the zero they were set to
    }
    const float cot = h.cotTheta;
    if (!(alpaka::math::abs(acc, cot) > 1.0e-4f))
      return false;
    const float P = alpaka::math::abs(acc, rho);
    if (!(P > kEpsG))
      return false;
    const float arc = (surf - h.zip) / cot;
    const float beta = arc / rho;
    const float cb = alpaka::math::cos(acc, beta);
    const float sb = alpaka::math::sin(acc, beta);
    const float Nn = P - A * cb;
    const float Dd = -A * sb;
    const float r2 = A * A + P * P - 2.f * A * P * cb;
    if (!(r2 > 1.0e-6f) || Nn == 0.f || Dd == 0.f)
      return false;
    const float r = alpaka::math::sqrt(acc, r2);
    const float sRho = (rho >= 0.f) ? 1.f : -1.f;
    // eq (4), evaluated four times with the (dA, dP, dbeta) triple of each parameter.
    const float invN = 1.f / Nn, invD = 1.f / Dd, invR = 1.f / r;
    auto row = [&](float dA, float dP, float dB) {
      const float dN = dP - cb * dA + A * sb * dB;
      const float dD = -sb * dA - A * cb * dB;
      const float dr = (A * dA + P * dP - cb * (A * dP + P * dA) + A * P * sb * dB) * invR;
      return g * (dN * invN - dD * invD - dr * invR);
    };
    hTip = row(dATip, 0.f, 0.f);
    hInvPt = row(dAPt, sRho * rhoI, arc * bf);
    hCot = row(0.f, 0.f, -beta / cot);
    hZip = row(0.f, 0.f, -1.f / (rho * cot));
    return true;
  }

  // VecDual: forward-mode AD carrying a value plus a 5-component gradient in one pass, so the endcap
  // r(params) evaluates each transcendental once; a finite-difference gradient would amplify
  // ULP-level roundoff of the crossing solve into visible chi2 shifts.

  struct VecDual {
    float v;     // value
    float d[5];  // partials w.r.t. {phi0, tip, invPt, cotTheta, zip}
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator+(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v + b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] + b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v - b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] - b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator+(VecDual a, float b) {
    VecDual r;
    r.v = a.v + b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator+(float a, VecDual b) {
    VecDual r;
    r.v = a + b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(VecDual a, float b) {
    VecDual r;
    r.v = a.v - b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(float a, VecDual b) {
    VecDual r;
    r.v = a - b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = -b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(VecDual a) {  // unary minus
    VecDual r;
    r.v = -a.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = -a.d[k];
    return r;
  }
  // d(a*b) = a.d*b.v + a.v*b.d
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator*(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v * b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] * b.v + a.v * b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator*(VecDual a, float b) {
    VecDual r;
    r.v = a.v * b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] * b;
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator*(float a, VecDual b) {
    VecDual r;
    r.v = a * b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a * b.d[k];
    return r;
  }
  // d(a/b) = (a.d*b.v - a.v*b.d) / (b.v*b.v)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator/(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v / b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = (a.d[k] * b.v - a.v * b.d[k]) / (b.v * b.v);
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator/(VecDual a, float b) {
    VecDual r;
    r.v = a.v / b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] / b;
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator/(float a, VecDual b) {
    VecDual r;
    r.v = a / b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = (-a * b.d[k]) / (b.v * b.v);
    return r;
  }

  // Value via alpaka::math::*, derivative by the chain rule; each transcendental is evaluated once.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dsin(TAcc const& acc, VecDual u) {
    // d(sin u) = cos(u.v) * u.d
    const float c = alpaka::math::cos(acc, u.v);
    VecDual r;
    r.v = alpaka::math::sin(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = c * u.d[k];
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dcos(TAcc const& acc, VecDual u) {
    // d(cos u) = -sin(u.v) * u.d
    const float s = alpaka::math::sin(acc, u.v);
    VecDual r;
    r.v = alpaka::math::cos(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = -s * u.d[k];
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dsqrt(TAcc const& acc, VecDual u) {
    // d(sqrt u) = u.d / (2*sqrt(u.v))
    const float s = alpaka::math::sqrt(acc, u.v);
    VecDual r;
    r.v = s;
    for (int k = 0; k < 5; ++k)
      r.d[k] = u.d[k] / (2.f * s);
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dabs(TAcc const& acc, VecDual u) {
    // d(|u|) = sign(u.v) * u.d
    const float sgn = (u.v >= 0.f) ? 1.f : -1.f;
    VecDual r;
    r.v = alpaka::math::abs(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = sgn * u.d[k];
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual datan2(TAcc const& acc, VecDual y, VecDual x) {
    // d(atan2(y,x)) = (x.v*y.d - y.v*x.d) / (x.v*x.v + y.v*y.v)
    const float den = x.v * x.v + y.v * y.v;
    VecDual r;
    r.v = alpaka::math::atan2(acc, y.v, x.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = (x.v * y.d[k] - y.v * x.d[k]) / den;
    return r;
  }

  // d(acos u) = -u.d / sqrt(1 - u.v^2); the caller guards |u.v| away from 1 (a grazing crossing).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dacos(TAcc const& acc, VecDual u) {
    const float w = alpaka::math::sqrt(acc, alpaka::math::max(acc, 1.f - u.v * u.v, 1e-12f));
    VecDual r;
    r.v = alpaka::math::acos(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = -u.d[k] / w;
    return r;
  }

  // 2pi folding of the value only: the branch shift is a constant, so the partials are untouched.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dfoldPi(VecDual u) {
    u.v = foldPi(u.v);
    return u;
  }

  // The five measurement rows of one layer crossing, from the closed-form crossing code in VecDual arithmetic
  // (no hand-derived Jacobian or sign), written for the surface the walk predicts on (barrel: fixed R;
  // endcap: fixed z) at the current state and used for the whole visit:
  //   Hphi[5] = d(r * phi)/d{phi0, tip, invPt, cot, zip}   [cm per unit parameter]
  //   Hsec[5] = d(sec)/d{...}, sec = z in the barrel and r in the endcap
  // `branch` is predictOnBarrel's root sign (ignored in the endcap). Returns false when degenerate.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool crossWithGrad5(TAcc const& acc,
                                                     float phi0,
                                                     float tip,
                                                     float invPt,
                                                     float cotTheta,
                                                     float zip,
                                                     bool isBarrel,
                                                     float surf,  // R (barrel) or z (endcap)
                                                     float bf,
                                                     float branch,
                                                     float Hphi[5],
                                                     float Hsec[5]) {
    const VecDual dPhi0{phi0, {1.f, 0.f, 0.f, 0.f, 0.f}};
    const VecDual dTip{tip, {0.f, 1.f, 0.f, 0.f, 0.f}};
    const VecDual dInvPt{invPt, {0.f, 0.f, 1.f, 0.f, 0.f}};
    const VecDual dCot{cotTheta, {0.f, 0.f, 0.f, 1.f, 0.f}};
    const VecDual dZip{zip, {0.f, 0.f, 0.f, 0.f, 1.f}};

    // makeHelixStateFromParams: rho = 1/(invPt*bf); xc=(tip+rho)*sin(phi0); yc=-(tip+rho)*cos(phi0).
    if (!(alpaka::math::abs(acc, invPt) > 1e-9f))
      return false;
    const VecDual rho = 1.f / (dInvPt * bf);
    const VecDual sp = dsin(acc, dPhi0);
    const VecDual cp = dcos(acc, dPhi0);
    const VecDual tr = dTip + rho;
    const VecDual xc = tr * sp;
    const VecDual yc = -tr * cp;
    const VecDual alphaOrigin = datan2(acc, -yc, -xc);
    const VecDual absRho = dabs(acc, rho);

    VecDual phi, sec;
    if (isBarrel) {
      // predictOnBarrel: c = (R^2 + |C|^2 - rho^2)/(2 |C| R); theta = atan2(yc,xc) + branch*acos(c).
      const VecDual dc2 = xc * xc + yc * yc;
      const VecDual dc = dsqrt(acc, dc2);
      if (!(dc.v > 1e-6f) || !(surf > 1e-6f))
        return false;
      const VecDual cc = (surf * surf + dc2 - rho * rho) / (2.f * dc * surf);
      if (!(alpaka::math::abs(acc, cc.v) < 1.f - 1e-5f))
        return false;  // grazing: the acos derivative blows up, and the crossing is not trusted anyway
      const VecDual theta = datan2(acc, yc, xc) + branch * dacos(acc, cc);
      const VecDual x = surf * dcos(acc, theta);
      const VecDual y = surf * dsin(acc, theta);
      const VecDual alphaH = datan2(acc, y - yc, x - xc);
      const VecDual arcS = rho * dfoldPi(alphaOrigin - alphaH);
      phi = theta;
      sec = dZip + arcS * dCot;  // z at fixed R, circle terms included through arcS
    } else {
      // predictOnEndcap: arcS = (z - zip)/cot; alphaH = alphaOrigin - arcS/rho; (x,y) on the disc.
      if (!(alpaka::math::abs(acc, cotTheta) > 1e-4f))
        return false;
      const VecDual arcS = (surf - dZip) / dCot;
      const VecDual alphaH = alphaOrigin - arcS / rho;
      const VecDual x = xc + absRho * dcos(acc, alphaH);
      const VecDual y = yc + absRho * dsin(acc, alphaH);
      sec = dsqrt(acc, x * x + y * y);
      phi = datan2(acc, y, x);
    }
    const float rSurf = isBarrel ? surf : sec.v;
    if (!(rSurf > 1e-6f))
      return false;
    for (int k = 0; k < 5; ++k) {
      Hphi[k] = rSurf * phi.d[k];
      Hsec[k] = sec.d[k];
      if (!alpaka::math::isfinite(acc, Hphi[k]) || !alpaka::math::isfinite(acc, Hsec[k]))
        return false;
    }
    return true;
  }

  // RunningHelix: the walk's perigee state (phi0, tip, 1/pT, cot, zip) and its FULL 5x5 covariance,
  // packed symmetric in the reco::TrackSoA layout (off[5] below). One filter: every measurement --
  // barrel or endcap, with or without the stub bend row -- enters through updateState5 with its own
  // rows, and the material of each traversed gap enters through addKinkNoise, so the r-phi and the
  // r-z halves of the state keep their correlations throughout the walk.
  struct RunningHelix {
    float phi0, tip, invPt, cotTheta, zip;
    float C[15];  // packed symmetric 5x5, index cIdx(a,b)
    // Derived helix geometry, refreshed by recomputeHelix().
    float rho;
    float xc, yc;
    float alphaOrigin;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE static int cIdx(int a, int b) {
      constexpr int off[5] = {0, 4, 7, 9, 10};
      const int lo = a < b ? a : b;
      const int hi = a < b ? b : a;
      return off[lo] + hi;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void recomputeHelix(TAcc const& acc, float bf) {
      rho = (invPt != 0.f) ? 1.f / (invPt * bf) : 1e9f;
      const float sp = alpaka::math::sin(acc, phi0);
      const float cp = alpaka::math::cos(acc, phi0);
      const float x0 = tip * sp;
      const float y0 = -tip * cp;
      xc = x0 + rho * sp;
      yc = y0 - rho * cp;
      alphaOrigin = alpaka::math::atan2(acc, -yc, -xc);
    }

    // Read-only HelixState view, so predictOn{Barrel,Endcap} need not be templated on RunningHelix.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixState helix() const {
      HelixState h;
      h.phi0 = phi0;
      h.tip = tip;
      h.cotTheta = cotTheta;
      h.zip = zip;
      h.rho = rho;
      h.xc = xc;
      h.yc = yc;
      h.alphaOrigin = alphaOrigin;
      return h;
    }

    // H C H^T for one row (the prediction variance of that measurement).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float predVar(const float H[5]) const {
      float v = 0.f;
      for (int a = 0; a < 5; ++a) {
        v += C[cIdx(a, a)] * H[a] * H[a];
        for (int b = a + 1; b < 5; ++b)
          v += 2.f * C[cIdx(a, b)] * H[a] * H[b];
      }
      return v;
    }

    // Rank-structured multiple-scattering process noise of one traversed gap, in the perigee frame. A kink of
    // variance q at perigee arc s_k enters the state as (dphi0, dtip) = (angle, +angle*s_k) and
    // (dcot, dzip) = (a, -a*s_k), so summed over the gap's material
    //   Q(phi0,tip) = cPhi * [[W, S1],[S1, S2]],   Q(cot,zip) = cCot * [[W, -S1],[-S1, S2]],
    // with W, S1, S2 the moments of rho*dl in the perigee arc s_k (not about the arrival end: the caller
    // converts). Curvature is untouched.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void addKinkNoise(float cPhi, float cCot, float W, float S1, float S2) {
      C[cIdx(0, 0)] += cPhi * W;
      C[cIdx(0, 1)] += cPhi * S1;
      C[cIdx(1, 1)] += cPhi * S2;
      C[cIdx(3, 3)] += cCot * W;
      C[cIdx(3, 4)] -= cCot * S1;
      C[cIdx(4, 4)] += cCot * S2;
    }

    // Joint Kalman update from n <= 3 measurement rows with a full n x n noise R. `d` is the residual
    // pred - meas, so the mean update is x -= K d. Returns the innovation chi2 against the pre-update
    // state and, in `sInvOut` (packed n(n+1)/2), the inverse innovation covariance; nRows<=0 or a
    // singular S leaves the state untouched and returns -1.
    template <typename TAcc, int NMAX = 3>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float updateState5(
        TAcc const& acc, int n, const float H[NMAX][5], const float d[NMAX], const float R[NMAX][NMAX], bool apply) {
      if (n < 1 || n > NMAX)
        return -1.f;
      float PH[NMAX][5];  // (C H^T)^T, row m = C H_m
      for (int m = 0; m < n; ++m)
        for (int a = 0; a < 5; ++a) {
          float v = 0.f;
          for (int b = 0; b < 5; ++b)
            v += C[cIdx(a, b)] * H[m][b];
          PH[m][a] = v;
        }
      float S[3][3];
      for (int m = 0; m < n; ++m)
        for (int l = 0; l < n; ++l) {
          float v = R[m][l];
          for (int a = 0; a < 5; ++a)
            v += H[m][a] * PH[l][a];
          S[m][l] = v;
        }
      float Si[3][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
      if (n == 1) {
        if (!(S[0][0] > 0.f))
          return -1.f;
        Si[0][0] = 1.f / S[0][0];
      } else if (n == 2) {
        const float det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        if (!(det > 0.f) || !alpaka::math::isfinite(acc, det))
          return -1.f;
        const float invDet = 1.f / det;
        Si[0][0] = S[1][1] * invDet;
        Si[1][1] = S[0][0] * invDet;
        Si[0][1] = Si[1][0] = -S[0][1] * invDet;
      } else {
        const float a0 = S[1][1] * S[2][2] - S[1][2] * S[2][1];
        const float a1 = S[1][2] * S[2][0] - S[1][0] * S[2][2];
        const float a2 = S[1][0] * S[2][1] - S[1][1] * S[2][0];
        const float det = S[0][0] * a0 + S[0][1] * a1 + S[0][2] * a2;
        if (!(det > 0.f) || !alpaka::math::isfinite(acc, det))
          return -1.f;
        const float invDet = 1.f / det;
        Si[0][0] = a0 * invDet;
        Si[0][1] = Si[1][0] = a1 * invDet;
        Si[0][2] = Si[2][0] = a2 * invDet;
        Si[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * invDet;
        Si[1][2] = Si[2][1] = (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * invDet;
        Si[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * invDet;
      }
      float chi2 = 0.f;
      for (int m = 0; m < n; ++m)
        for (int l = 0; l < n; ++l)
          chi2 += d[m] * Si[m][l] * d[l];
      if (!alpaka::math::isfinite(acc, chi2) || chi2 < 0.f)
        return -1.f;
      if (!apply)
        return chi2;
      // K = C H^T S^-1 ; x -= K d ; C -= (C H^T) S^-1 (C H^T)^T
      float K[3][5];
      for (int m = 0; m < n; ++m)
        for (int a = 0; a < 5; ++a) {
          float v = 0.f;
          for (int l = 0; l < n; ++l)
            v += Si[m][l] * PH[l][a];
          K[m][a] = v;
        }
      float dx[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
      for (int m = 0; m < n; ++m)
        for (int a = 0; a < 5; ++a)
          dx[a] += K[m][a] * d[m];
      phi0 -= dx[0];
      tip -= dx[1];
      invPt -= dx[2];
      cotTheta -= dx[3];
      zip -= dx[4];
      for (int a = 0; a < 5; ++a)
        for (int b = a; b < 5; ++b) {
          float v = 0.f;
          for (int m = 0; m < n; ++m)
            v += PH[m][a] * K[m][b];
          C[cIdx(a, b)] -= v;
        }
      return chi2;
    }
  };

  // Initialize a RunningHelix from a track's BL state and its full published covariance.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE RunningHelix
  makeRunningHelix(TAcc const& acc, const ::reco::TrackSoAConstView tracks, int i, float bf) {
    RunningHelix h;
    h.phi0 = tracks[i].state()(0);
    h.tip = tracks[i].state()(1);
    h.invPt = tracks[i].state()(2);
    h.cotTheta = tracks[i].state()(3);
    h.zip = tracks[i].state()(4);
    for (int k = 0; k < 15; ++k)
      h.C[k] = tracks[i].covariance()(k);
    h.recomputeHelix(acc, bf);
    return h;
  }

  // Project a packed symmetric 5x5 (layout off[5]={0,4,7,9,10}) onto a measurement row H: H^T C H.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float projectCov(const float* C, const float H[5]) {
    constexpr int off[5] = {0, 4, 7, 9, 10};
    float acc2 = 0.f;
    for (int a = 0; a < 5; ++a) {
      acc2 += C[off[a] + a] * H[a] * H[a];
      for (int b = a + 1; b < 5; ++b)
        acc2 += 2.f * C[off[a] + b] * H[a] * H[b];
    }
    return acc2;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_ExtenderHelixHelpers_h
