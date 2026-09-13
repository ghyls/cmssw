#ifndef RecoTracker_PixelSeeding_plugins_alpaka_ExtDerivedTables_h
#define RecoTracker_PixelSeeding_plugins_alpaka_ExtDerivedTables_h

#include <array>
#include <cmath>
#include <cstddef>

// Inputs of the extension walk's hole hypothesis -- which OT layers carry a usable hit and how
// densely hits sit -- plus the analytic chi2 quantiles its gate is cut at. The first group are
// measured properties of the Phase-2 tracker, the second is pure statistics; neither is a tuning
// constant. The one operating point, kExtGateEps below, is spent against them.

namespace extDerivedTables {

  // The acceptance of the walk: the probability mass of the innovation chi2 a correct hit is required
  // to fall inside. Everything the walk cuts on follows from it -- the per-dof accept threshold, the
  // phi window that bounds the same chi2 ball, the reachability slack, the ranking and the hole prior.
  // Measured on TTbar PU200 at 0.55, 0.90 and 0.99: the wider acceptances attach more outer-tracker
  // hits per track but lose merged efficiency (-0.0015, -0.0025) while adding fake (+0.0020, +0.0031),
  // because past 0.55 the hits they admit are predominantly wrong ones.
  inline constexpr float kExtGateEps = 0.55f;

  // Row geometry of the measured detector properties below; mirrors caExtension::kExtOTLayers,
  // static_asserted against it in PixelTracksSoAMerger.cc.
  inline constexpr int kOTLayers = 26;  // CA layers 28..53

  // kEtaL: measured per-layer stub availability, eta_L = P(particle left a usable stub on this
  // layer | it crossed it), CA layers 28..53: the detection probability of the hole hypothesis.
  inline constexpr std::array<double, 26> kEtaL = {
      0.87764, 0.86782, 0.85745, 0.79163, 0.77172, 0.75899, 0.69925, 0.78985, 0.69350,
      0.79261, 0.66750, 0.78358, 0.64763, 0.77362, 0.63455, 0.76000, 0.69331, 0.79013,
      0.69044, 0.79073, 0.66446, 0.78392, 0.64815, 0.77152, 0.63464, 0.75438,
  };

  // kRho: measured per-layer stub areal density [cm^-2], CA layers 28..53. The background a window
  // winner competes against; the hole cost moves as 2 ln rho.
  inline constexpr std::array<double, 26> kRho = {
      0.822121, 0.214107, 0.091798, 0.101623, 0.071336, 0.049993, 0.624300, 0.099876, 0.701091,
      0.122131, 0.388552, 0.113048, 0.457734, 0.136658, 0.542857, 0.147108, 0.626862, 0.100555,
      0.703274, 0.122321, 0.389612, 0.112945, 0.459801, 0.135974, 0.543302, 0.147127,
  };

  // kRho3: measured stub density in the 3-dof (stub-bend) space [cm^-1 rad^-1],
  // rho_3 = rho_A / (2 b99), where b99 is the per-layer 99th-percentile bend half-range.
  inline constexpr std::array<double, 26> kRho3 = {
      6.0943,  3.68515, 2.38435, 6.03535, 4.30563, 2.41512, 4.29521, 3.13091, 4.70765,
      3.27429, 3.18225, 3.16661, 3.37372, 3.26823, 3.76722, 3.68451, 4.31628, 3.15219,
      4.72948, 3.27938, 3.22794, 3.18156, 3.40846, 3.2452,  3.76791, 3.68721,
  };

  // kEtaLRaw: measured raw-round conditional availability, eta_cond = P(raw OT cluster | no stub),
  // CA layers 28..53.
  inline constexpr std::array<double, 26> kEtaLRaw = {
      0.37887, 0.32535, 0.25681, 0.22707, 0.18528, 0.09702, 0.14246, 0.23299, 0.12979,
      0.19146, 0.0932,  0.1648,  0.08115, 0.14615, 0.07105, 0.13146, 0.14062, 0.23313,
      0.12687, 0.20057, 0.09368, 0.15754, 0.07846, 0.1424,  0.07032, 0.13163,
  };

  // ---------------------------------------------------------------------------------------------
  // Analytic chi2 quantiles. The walk's gate is chi2_d < Q_d(eps): nothing measured, just the
  // inverse cumulative of a chi2 with d degrees of freedom. d = 2 has the closed form
  // Q_2(eps) = -2 ln(1 - eps); d = 1 and d = 3 have none, so they are tabulated on the eps grid
  // below and interpolated linearly (host-side, once per job -- the kernel receives plain floats).
  // dof 5 is the duplicate removal's, tabulated the same way.
  inline constexpr int kNQuantEps = 21;
  inline constexpr std::array<double, kNQuantEps> kQuantEps = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30,  0.35,
                                                               0.40, 0.45, 0.50, 0.55, 0.60, 0.65,  0.70,
                                                               0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99};
  inline constexpr std::array<double, kNQuantEps> kChi2Quant1 = {
      0.00393, 0.01579, 0.03577, 0.06418, 0.10153, 0.14847, 0.20590, 0.27500, 0.35732, 0.45494, 0.57065,
      0.70833, 0.87346, 1.07419, 1.32330, 1.64237, 2.07225, 2.70554, 3.84146, 5.02389, 6.63490};
  inline constexpr std::array<double, kNQuantEps> kChi2Quant3 = {
      0.35185, 0.58437, 0.79777, 1.00517, 1.21253, 1.42365, 1.64158, 1.86917, 2.10947, 2.36597, 2.64301,
      2.94617, 3.28311, 3.66487, 4.10834, 4.64163, 5.31705, 6.25139, 7.81473, 9.34840, 11.34487};
  // dof 5: the duplicate-removal compatibility test on the full (phi, tip, q/pT, cot, zip) difference.
  inline constexpr std::array<double, kNQuantEps> kChi2Quant5 = {
      1.14548, 1.61031, 1.99382, 2.34253, 2.67460, 2.99991, 3.32511, 3.65550,  3.99594,  4.35146, 4.72776,
      5.13187, 5.57307, 6.06443, 6.62568, 7.28928, 8.11520, 9.23636, 11.07050, 12.83250, 15.08627};

  // The duplicate removal does not gate at eps: the walk accepts a hit inside the probability mass eps of a
  // correct hit, while the duplicate test keeps two tracks apart only when their fits disagree, so its
  // threshold is a rejection significance: one-sided 5 sigma (p = 5.733e-7), taken jointly so the parameter
  // correlations are used. A tighter threshold splits real duplicates, the more expensive error.
  inline constexpr double kDedupRejectChi2_1 = 25.0;     // 1 dof: one coordinate, (5 sigma)^2
  inline constexpr double kDedupRejectChi2_3 = 31.8121;  // 3 dof: (phi, q/pT, cot), pre-refit
  inline constexpr double kDedupRejectChi2_5 = 37.0948;  // 5 dof: tip and zip too, post-refit

  // Q_d(eps) for d = 1, 2, 3, 5. Host-side; eps is clamped to the tabulated range.
  inline double chi2Quantile(int dof, double eps) {
    if (eps <= kQuantEps.front())
      eps = kQuantEps.front();
    if (eps >= kQuantEps.back())
      eps = kQuantEps.back();
    if (dof == 2)
      return -2. * std::log(1. - eps);
    auto const& tab = (dof == 1) ? kChi2Quant1 : ((dof == 3) ? kChi2Quant3 : kChi2Quant5);
    int k = 0;
    while (k + 2 < kNQuantEps && kQuantEps[k + 1] < eps)
      ++k;
    const double t = (eps - kQuantEps[k]) / (kQuantEps[k + 1] - kQuantEps[k]);
    return tab[k] + t * (tab[k + 1] - tab[k]);
  }

}  // namespace extDerivedTables

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_ExtDerivedTables_h
