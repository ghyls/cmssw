// The wrong-stub test of the General Broken Lines refit. A stub carries a measurement the outlier stage does not
// use: its bend dPhiDr, i.e. the track's stereo angle at that radius and therefore its curvature. The test
// compares that bend with the one the fitted circle has at the stub's radius and lets a stub past the 99.9 %
// chi2(1) bound be dropped like a position outlier. Measured: the pull of the bend model over clean stubs per
// module class (centred at 0, unit width); the false-drop rate at the bound (~0.1 %), also with the declared
// error 2x too loose and 2x too sharp; the drop efficiency on a deliberately wrong stub (a 2 GeV bend on a
// 1 GeV track); that a right stub with a 3-sigma bend fluctuation is kept; that the refit pulls do not move on
// tracks without a wrong stub; device vs host twin. Generation as in testBLStubCovDevice (RK4 in a constant
// field, Highland kinks per material lump, PS/2S readout uniform over the cell), the bend evaluated on the true
// local direction and smeared with the class bend precision.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"         // host twin
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"         // material walk + prep
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // the fit and the test under test

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace gblh = ::generalBrokenLine;                              // host twin
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;  // device
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr int kNOt = 6;                              // hits of the outer-tracker-only layout
  constexpr int kNPix = 8;                             // hits of the pixel-seeded control layout
  constexpr int kNVar = 5;                             // see the file head
  constexpr double kBz = 3.8112;                       // T
  constexpr double kBField = 0.29979246 * kBz / 100.;  // GeV/cm
  constexpr int kRep = 1500;                           // replicas per configuration
  //!< the two 99.9 % bounds the outlier stage uses: chi2(2) on the position rows, chi2(1) on the bend.
  constexpr double kOutlierChi2Cut = 13.8;
  constexpr double kStubBendChi2Cut = 10.83;

  // ------------------------------------------------------------------------------- sensors and surfaces
  enum Sensor { kPixel, kPS, kSS };
  struct SensorGeom {
    double pitchX, lenY, spacing;  // measured pitch, unmeasured length, stack spacing (smallest of the class)
  };
  constexpr SensorGeom kGeom[3] = {{0.0100, 0.0150, 0.}, {0.0100, 0.1467, 0.16}, {0.0090, 5.0250, 0.18}};

  struct Barrel {
    double r, halfZ;
    Sensor sensor;
    bool tilted;
  };
  struct Disc {
    double z, rMin, rMax;
    Sensor sensor;
    double rSwitch;
  };
  const std::vector<Barrel> kBarrels = {{3.0, 25., kPixel, false},
                                        {6.8, 25., kPixel, false},
                                        {10.2, 25., kPixel, false},
                                        {16.0, 25., kPixel, false},
                                        {23.0, 120., kPS, true},
                                        {36.0, 120., kPS, true},
                                        {51.0, 120., kPS, true},
                                        {68.0, 120., kSS, false},
                                        {86.0, 120., kSS, false},
                                        {108.0, 120., kSS, false}};
  const std::vector<Disc> kDiscs = {{33., 4.5, 25., kPixel, 1e9},
                                    {40., 4.5, 25., kPixel, 1e9},
                                    {48., 4.5, 25., kPixel, 1e9},
                                    {58., 4.5, 25., kPixel, 1e9},
                                    {70., 4.5, 25., kPixel, 1e9},
                                    {85., 4.5, 25., kPixel, 1e9},
                                    {103., 4.5, 25., kPixel, 1e9},
                                    {126., 4.5, 25., kPixel, 1e9},
                                    {131., 22., 112., kSS, 60.},
                                    {155., 22., 112., kSS, 60.},
                                    {178., 22., 112., kSS, 60.},
                                    {201., 22., 112., kSS, 60.},
                                    {225., 22., 112., kSS, 60.},
                                    {250., 22., 112., kSS, 60.}};

  //!< Stub class of a crossing, for the layout label.
  enum StubClass { kFlatPS, kFlat2S, kTiltedPS, kEndcapPS, kEndcap2S, kNoStub };

  //!< One recorded crossing: the true point and direction, the module's measurement frame and stack
  //!< direction, and its sensor.
  struct Hit {
    double p[3];
    double t[3];          // true direction at the crossing (unit)
    double ex[3], ey[3];  // measured and unmeasured directions, global unit vectors
    double nR, nZ;        // stack direction (r, z components); (0, 0) for a pixel
    Sensor sensor;
    StubClass cls;
    bool isStub;
  };
  struct Track {
    Hit hit[kNPix];
    bool ok = false;
  };

  void frameAt(double x, double y, double z, bool barrel, bool tilted, Hit& h) {
    const double r = std::hypot(x, y);
    const double cs = (r > 0.) ? x / r : 1., sn = (r > 0.) ? y / r : 0.;
    h.ex[0] = -sn;
    h.ex[1] = cs;
    h.ex[2] = 0.;
    if (!barrel) {  // disc: the strips run radially, the stack is along z
      h.ey[0] = cs;
      h.ey[1] = sn;
      h.ey[2] = 0.;
      h.nR = 0.;
      h.nZ = 1.;
      return;
    }
    if (!tilted || std::abs(z) < 15.) {  // flat barrel module: along the beam, stack radial
      h.ey[0] = h.ey[1] = 0.;
      h.ey[2] = 1.;
      h.nR = 1.;
      h.nZ = 0.;
      return;
    }
    const double n = std::hypot(r, z);  // pointing module: normal along (r, z)
    h.ey[0] = -z * cs / n;
    h.ey[1] = -z * sn / n;
    h.ey[2] = r / n;
    h.nR = r / n;
    h.nZ = z / n;
  }

  double theta0Of(double x, double p) {
    if (!(x > 0.))
      return 0.;
    constexpr double kMassPion = 0.13957;
    const double beta = p / std::sqrt(p * p + kMassPion * kMassPion);
    return 13.6e-3 / (beta * p) * std::sqrt(x) * (1. + 0.038 * std::log(x));
  }

  //!< RK4 in a constant solenoid field from the origin (phi0 = 0), Highland kinks per material lump of
  //!< the map when rng != nullptr, logarithm at xLogTotal when > 0.
  Track makeTrack(const float* rho,
                  double pT,
                  double eta,
                  int q,
                  std::mt19937_64* rng,
                  int nWanted,
                  bool otOnly,
                  double xLogTotal = 0.,
                  double* xTot = nullptr) {
    Track tk;
    const double tanl = std::sinh(eta);
    const double p = pT * std::cosh(eta);
    const double invn = 1. / std::sqrt(1. + tanl * tanl);
    std::array<double, 3> pos = {0., 0., 0.};
    std::array<double, 3> dir = {invn, 0., tanl * invn};
    const double ds = 0.2;
    int nHit = 0;
    double xCluster = 0., xSum = 0.;
    std::normal_distribution<double> gauss(0., 1.);
    for (int step = 0; step < 6000 && nHit < nWanted; ++step) {
      const auto prev = pos;
      auto deriv = [&](const std::array<double, 3>& t) {
        return std::array<double, 3>{(q / p) * t[1] * kBField, -(q / p) * t[0] * kBField, 0.};
      };
      std::array<double, 3> k1p = dir, k1t = deriv(dir), p2{}, t2{}, p3{}, t3{}, p4{}, t4{};
      for (int i = 0; i < 3; ++i) {
        p2[i] = pos[i] + 0.5 * ds * k1p[i];
        t2[i] = dir[i] + 0.5 * ds * k1t[i];
      }
      auto k2p = t2, k2t = deriv(t2);
      for (int i = 0; i < 3; ++i) {
        p3[i] = pos[i] + 0.5 * ds * k2p[i];
        t3[i] = dir[i] + 0.5 * ds * k2t[i];
      }
      auto k3p = t3, k3t = deriv(t3);
      for (int i = 0; i < 3; ++i) {
        p4[i] = pos[i] + ds * k3p[i];
        t4[i] = dir[i] + ds * k3t[i];
      }
      auto k4p = t4, k4t = deriv(t4);
      for (int i = 0; i < 3; ++i) {
        pos[i] += ds / 6. * (k1p[i] + 2. * k2p[i] + 2. * k3p[i] + k4p[i]);
        dir[i] += ds / 6. * (k1t[i] + 2. * k2t[i] + 2. * k3t[i] + k4t[i]);
      }
      double dn = 0.;
      for (int i = 0; i < 3; ++i)
        dn += dir[i] * dir[i];
      dn = std::sqrt(dn);
      for (int i = 0; i < 3; ++i)
        dir[i] /= dn;

      {
        const double rr = std::hypot(pos[0], pos[1]);
        const double dens = blMaterialMap::rhoAt(rho, float(rr), float(pos[2]));
        xCluster += dens * ds;
        xSum += dens * ds;
        constexpr double kDense = 1.e-3, kLump = 2.e-3;
        if (rng != nullptr && xCluster > 0. && (dens < kDense || xCluster > kLump)) {
          const double th0 = (xLogTotal > 0.) ? theta0Of(xCluster, p) * (1. + 0.038 * std::log(xLogTotal)) /
                                                    (1. + 0.038 * std::log(xCluster))
                                              : theta0Of(xCluster, p);
          xCluster = 0.;
          std::array<double, 3> u = {-dir[1], dir[0], 0.};
          const double un = std::hypot(u[0], u[1]);
          for (int i = 0; i < 3; ++i)
            u[i] /= un;
          const std::array<double, 3> v = {-dir[2] * u[1], dir[2] * u[0], dir[0] * u[1] - dir[1] * u[0]};
          const double a = th0 * gauss(*rng), b = th0 * gauss(*rng);
          double nn = 0.;
          for (int i = 0; i < 3; ++i) {
            dir[i] += a * u[i] + b * v[i];
            nn += dir[i] * dir[i];
          }
          nn = std::sqrt(nn);
          for (int i = 0; i < 3; ++i)
            dir[i] /= nn;
        }
      }

      auto record = [&](double x, double y, double z, bool barrel, bool tilted, Sensor s) {
        Hit& h = tk.hit[nHit];
        h.p[0] = x;
        h.p[1] = y;
        h.p[2] = z;
        for (int i = 0; i < 3; ++i)
          h.t[i] = dir[i];
        h.sensor = s;
        h.isStub = (s != kPixel);
        frameAt(x, y, z, barrel, tilted, h);
        if (!h.isStub)
          h.cls = kNoStub;
        else if (!barrel)
          h.cls = (s == kPS) ? kEndcapPS : kEndcap2S;
        else if (tilted && std::abs(z) >= 15.)
          h.cls = kTiltedPS;
        else
          h.cls = (s == kPS) ? kFlatPS : kFlat2S;
        ++nHit;
      };
      const double r0 = std::hypot(prev[0], prev[1]), r1 = std::hypot(pos[0], pos[1]);
      for (auto const& bl : kBarrels) {
        if (nHit >= nWanted)
          break;
        if (otOnly && bl.sensor == kPixel)
          continue;
        if ((r0 - bl.r) * (r1 - bl.r) < 0.) {
          const double f = (bl.r - r0) / (r1 - r0);
          const double z = prev[2] + f * (pos[2] - prev[2]);
          if (std::abs(z) < bl.halfZ)
            record(prev[0] + f * (pos[0] - prev[0]), prev[1] + f * (pos[1] - prev[1]), z, true, bl.tilted, bl.sensor);
        }
      }
      for (auto const& dc : kDiscs) {
        if (nHit >= nWanted)
          break;
        if (otOnly && dc.sensor == kPixel)
          continue;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          const double zt = sgn * dc.z;
          if ((prev[2] - zt) * (pos[2] - zt) < 0.) {
            const double f = (zt - prev[2]) / (pos[2] - prev[2]);
            const double x = prev[0] + f * (pos[0] - prev[0]), y = prev[1] + f * (pos[1] - prev[1]);
            const double r = std::hypot(x, y);
            if (r > dc.rMin && r < dc.rMax)
              record(x, y, zt, false, false, (r > dc.rSwitch) ? kSS : kPS);
          }
        }
      }
      if (r1 > 112. || std::abs(pos[2]) > 265.)
        break;
    }
    tk.ok = (nHit == nWanted);
    if (xTot != nullptr)
      *xTot = xSum;
    return tk;
  }

  // ------------------------------------------------------------------------------ the stub bend, on host
  //!< Declared bend precision (dPhiDrErrorPrec units, rad/cm) per stub class, set so that the bend of a
  //!< 2 GeV track placed on a 1 GeV one gives the per-class chi2 the stub census measures (141 flat 2S,
  //!< 32 endcap 2S, 16 flat PS, 1.8 endcap PS). The tilted-PS entry is a stand-in until the PS bend fix
  //!< lands; the declared/true error sweep below covers it.
  constexpr double kSigBend[5] = {7.12e-4, 2.40e-4, 2.14e-3, 2.12e-3, 5.04e-4};

  //!< The producer's bend model at a crossing, from a transverse direction (cos(beta), sin(beta)) and the
  //!< dip, for a module whose sensors are stacked along (nR, nZ) in the (r, z) plane. This is the same
  //!< expression the device function under test evaluates on the fitted circle.
  double bendOf(double r, double z, double cosBeta, double sinBeta, double cot, double nR, double nZ) {
    const double den = nR * cosBeta + nZ * cot;
    if (!(std::abs(den) > 1.e-6) || !(r > 1.e-4))
      return 0.;
    return sinBeta * (nR + nZ * z / r) / (r * den);
  }

  //!< (cos(beta), sin(beta), cot) of a recorded crossing, from its true direction.
  void betaOf(const Hit& h, double& cosBeta, double& sinBeta, double& cot) {
    const double r = std::hypot(h.p[0], h.p[1]);
    const double tT = std::hypot(h.t[0], h.t[1]);
    cosBeta = (h.t[0] * h.p[0] + h.t[1] * h.p[1]) / (r * tT);
    sinBeta = (h.p[0] * h.t[1] - h.p[1] * h.t[0]) / (r * tT);
    cot = h.t[2] / tT;
  }

  // --------------------------------------------------------------------------------- the fit, on device
  // Output layout per (track, variant).
  template <int N>
  struct Out {
    static constexpr int kPar = 0;           // helix at the PCA with the test on (after any drop)
    static constexpr int kVar = kPar + 5;    // its declared variances
    static constexpr int kParNo = kVar + 5;  // the same with the test off
    static constexpr int kVarNo = kParNo + 5;
    static constexpr int kQCharge = kVarNo + 5;
    static constexpr int kDropped = kQCharge + 1;  // fit-hit index dropped by the bend test, -1 if none
    static constexpr int kDropPos = kDropped + 1;  // fit-hit index dropped by the position rows, -1 if none
    static constexpr int kNextRef = kDropPos + 1;  // the fitted circle (cx, cy, R, -q/cot)
    static constexpr int kChi2B = kNextRef + 4;    // per fit hit, the bend chi2 (-1: no bend measurement)
    static constexpr int kPred = kChi2B + N;       // per fit hit, the predicted bend
    static constexpr int kSize = kPred + N;
  };

  template <int N>
  struct StubTestKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* bendIn,  // 4 per hit: dPhiDr, sigma, nR, nZ
                                  const float* rho,
                                  int nTracks,
                                  gbld::GblNodeData* nodesAll,
                                  double* scratchAll,
                                  double* out) const {
      using O = Out<N>;
      constexpr int kNodesMax = 2 * N + 1;
      for (auto j : cms::alpakatools::uniform_elements(acc, nTracks * kNVar)) {
        const std::size_t base = std::size_t(j);
        Eigen::Matrix<double, 3, N> hits;
        Eigen::Matrix<float, 6, N> hits_ge;
        for (int c = 0; c < N; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * N * base + 3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * N * base + 6 * c + r];
        }
        gbld::GblNodeData* nodes = nodesAll + base * kNodesMax;
        double* scratch = scratchAll + base * std::size_t(gbld::kGblScratchDoubles<2 * N>);
        double* o = out + base * O::kSize;
        for (int a = 0; a < O::kSize; ++a)
          o[a] = 0.;
        o[O::kDropped] = -1.;
        o[O::kDropPos] = -1.;

        Eigen::Vector4d ff;
        bld::fastFit(acc, hits, ff);
        bld::PreparedGblData<N> data;
        double gapD1[N], gapW1[N];
        bld::prepareGblFitData(acc, hits, ff, kBField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        const bool usedSplit = gbld::prepareGblDataSplit<Acc1D, N>(acc,
                                                                   hits,
                                                                   hits_ge,
                                                                   ff,
                                                                   kBField,
                                                                   data.qCharge,
                                                                   data.sTransverse,
                                                                   data.sTotal,
                                                                   data.matXX0,
                                                                   gapD1,
                                                                   gapW1,
                                                                   data.innerXX0,
                                                                   data.innerD1,
                                                                   data.innerW1,
                                                                   nodes,
                                                                   /*applyELoss=*/false,
                                                                   /*bMap=*/nullptr,
                                                                   /*bFieldOrigin=*/0.,
                                                                   /*trajectoryCorrections=*/false,
                                                                   /*scatteringLogAtTotal=*/false,
                                                                   /*elossCumulative=*/false);
        if (!usedSplit)
          continue;  // the split layout is what the refit runs on these multiplicities
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        gbld::Matrix5d cov = gbld::Matrix5d::Zero();
        double chi2 = 0.;
        double fullDelta[2 * kNodesMax + 1];
        double nodeVar[3 * kNodesMax];
        cov = gbld::gblFitPca<Acc1D, 2 * N>(acc, nodes, scratch, &corr, fullDelta, &chi2, nodeVar);
        gbld::Vector5d hp = gbld::Vector5d::Zero();
        gbld::Matrix5d hc = gbld::Matrix5d::Zero();
        Eigen::Vector4d nextRef;
        gbld::gblHelixAtPca(
            acc, ff, data.qCharge, kBField, double(data.sTransverse(0)), hits(2, 0), corr, cov, hp, hc, &nextRef);
        for (int a = 0; a < 5; ++a) {
          o[O::kParNo + a] = hp(a);
          o[O::kVarNo + a] = hc(a, a);
          o[O::kPar + a] = hp(a);
          o[O::kVar + a] = hc(a, a);
        }
        o[O::kQCharge] = double(data.qCharge);
        for (int a = 0; a < 4; ++a)
          o[O::kNextRef + a] = nextRef(a);

        // The outlier scan, as Kernel_BLFitPhaseOutlier runs it: every measured node scored by the larger
        // of its two ratios to its own cut, the worst above 1 dropped and the fit re-solved once. Run
        // twice -- position rows only, then with the bend test -- so that the A/B isolates this change.
        const double fitCot = (nextRef(3) != 0.) ? -double(data.qCharge) / nextRef(3) : 0.;
        for (int pass = 0; pass < 2; ++pass) {
          const bool bendOn = (pass == 1);
          double worst = 0.;
          int worstNode = -1, worstHit = -1;
          bool worstIsBend = false;
          int di = 0;
          for (int k = 0; k < kNodesMax; ++k) {
            if (!nodes[k].hasMeas)
              continue;
            const int i = di++;
            const double ru = nodes[k].measResidual(0) - fullDelta[1 + 2 * k];
            const double rv = nodes[k].measResidual(1) - fullDelta[1 + 2 * k + 1];
            const gbld::Matrix2d V = gbld::inv2(nodes[k].measPrec);
            const double s00 = V(0, 0) - nodeVar[3 * k];
            const double s01 = V(0, 1) - nodeVar[3 * k + 1];
            const double s11 = V(1, 1) - nodeVar[3 * k + 2];
            const double det = s00 * s11 - s01 * s01;
            double score = 0.;
            if (s00 > 0. && s11 > 0. && det > 0.)
              score = (ru * (s11 * ru - s01 * rv) + rv * (s00 * rv - s01 * ru)) / det / kOutlierChi2Cut;
            bool isBend = false;
            const double* sb = bendIn + 4 * (std::size_t(N) * base + std::size_t(i));
            double pred = 0.;
            if (pass == 0)
              o[O::kChi2B + i] = -1.;
            if (sb[1] > 0. && gbld::gblStubBendPrediction(acc,
                                                          nextRef(0),
                                                          nextRef(1),
                                                          nextRef(2),
                                                          data.qCharge,
                                                          fitCot,
                                                          hits(0, i),
                                                          hits(1, i),
                                                          hits(2, i),
                                                          sb[2],
                                                          sb[3],
                                                          pred)) {
              const double d = (sb[0] - pred) / sb[1];
              if (pass == 0) {
                o[O::kChi2B + i] = d * d;
                o[O::kPred + i] = pred;
              }
              if (bendOn && d * d / kStubBendChi2Cut > score) {
                score = d * d / kStubBendChi2Cut;
                isBend = true;
              }
            }
            if (score > worst) {
              worst = score;
              worstNode = k;
              worstHit = i;
              worstIsBend = isBend;
            }
          }
          if (worstNode < 0 || !(worst > 1.))
            continue;  // nothing dropped: this pass keeps the undropped fit already written out
          if (bendOn && worstIsBend)
            o[O::kDropped] = double(worstHit);
          else if (!bendOn)
            o[O::kDropPos] = double(worstHit);
          nodes[worstNode].hasMeas = false;
          gbld::Vector5d corr2 = gbld::Vector5d::Zero();
          double chi2b = 0.;
          const gbld::Matrix5d cov2 = gbld::gblFitPca<Acc1D, 2 * N>(acc, nodes, scratch, &corr2, nullptr, &chi2b);
          gbld::Vector5d hp2 = gbld::Vector5d::Zero();
          gbld::Matrix5d hc2 = gbld::Matrix5d::Zero();
          gbld::gblHelixAtPca(
              acc, ff, data.qCharge, kBField, double(data.sTransverse(0)), hits(2, 0), corr2, cov2, hp2, hc2);
          nodes[worstNode].hasMeas = true;  // restore for the second pass
          const int off = bendOn ? O::kPar : O::kParNo;
          const int offV = bendOn ? O::kVar : O::kVarNo;
          for (int a = 0; a < 5; ++a) {
            o[off + a] = hp2(a);
            o[offV + a] = hc2(a, a);
          }
        }
      }
    }
  };

  struct Config {
    double pT, eta;
    int q;
  };

  //!< +-2 sigma truncated r.m.s., iterated and rescaled by the Gaussian truncation factor; returns the
  //!< core sigma and the core mean.
  void coreOf(const std::vector<double>& v, double& mu, double& sg) {
    mu = 0.;
    for (double x : v)
      mu += x;
    mu /= double(v.size());
    sg = 0.;
    for (double x : v)
      sg += (x - mu) * (x - mu);
    sg = std::sqrt(sg / double(v.size()));
    for (int it = 0; it < 6 && sg > 0.; ++it) {
      double s = 0., ss = 0.;
      int n = 0;
      for (double x : v)
        if (std::abs(x - mu) < 2. * sg) {
          s += x;
          ss += x * x;
          ++n;
        }
      if (n < 10)
        break;
      mu = s / n;
      sg = std::sqrt(std::max(0., ss / n - mu * mu) / 0.77374);
    }
  }

  //!< Everything the assertions at the end need.
  struct Summary {
    long nCleanStub = 0, nFalseDrop = 0;                       // clean stubs seen / dropped at the bound
    long nCleanStubLoose = 0, nFalseDropLoose = 0;             // declared error 2x too loose
    long nCleanStubSharp = 0, nFalseDropSharp = 0;             // declared error 2x too sharp
    long nWrong[3] = {0, 0, 0}, nWrongDropped[3] = {0, 0, 0};  // the wrong stub, per host-track pT cell
    long nFluct = 0, nFluctDropped = 0;                        // the right stub with a 3-sigma fluctuation
    double worstParShift = 0.;                                 // largest increase of |pull width - 1|, clean tracks
    double worstBias = 0.;                                     // |mean| of the bend pull, per class
    double worstWidth = 0.;                                    // |core width - 1| of the bend pull, per class
    double worstTwin = 0.;                                     // device vs host twin of the prediction
    long nClass[5] = {0, 0, 0, 0, 0};
    long nDropClass[5] = {0, 0, 0, 0, 0};
    // the wrong stub on a 1 GeV host, per module class
    long nWrongClass[5] = {0, 0, 0, 0, 0};
    long nWrongDropClass[5] = {0, 0, 0, 0, 0};
  };

  template <int NH>
  void runLayout(Queue& queue, const float* rho, bool otOnly, int nRep, const char* label, Summary& sum) {
    using O = Out<NH>;
    std::vector<Config> cfgs;
    for (int q : {+1, -1})
      for (double pT : {1., 3., 10.})
        for (double aeta : {0.3, 0.8, 1.2, 1.7, 2.2})
          cfgs.push_back({pT, aeta, q});

    std::mt19937_64 rng(20260914);
    std::uniform_real_distribution<double> flat(-0.5, 0.5);
    std::normal_distribution<double> gauss(0., 1.);

    std::vector<Track> tracks;
    std::vector<double> xTotOf(cfgs.size(), 0.);
    std::vector<std::vector<int>> index(cfgs.size());
    for (std::size_t c = 0; c < cfgs.size(); ++c) {
      makeTrack(rho, cfgs[c].pT, cfgs[c].eta, cfgs[c].q, nullptr, NH, otOnly, 0., &xTotOf[c]);
      for (int r = 0; r < nRep; ++r) {
        Track tk;
        for (int try_ = 0; try_ < 8 && !tk.ok; ++try_)
          tk = makeTrack(rho, cfgs[c].pT, cfgs[c].eta, cfgs[c].q, &rng, NH, otOnly, xTotOf[c]);
        if (!tk.ok)
          continue;
        index[c].push_back(int(tracks.size()));
        tracks.push_back(tk);
      }
      REQUIRE(index[c].size() > std::size_t(nRep / 2));
    }
    const int nTracks = int(tracks.size());

    // Variants, in this order: clean; clean with the declared error 2x too loose (true noise halved);
    // clean with it 2x too sharp (true noise doubled); one stub replaced by the bend a 2 GeV track would
    // have at that point; the same stub given a 3-sigma bend fluctuation.
    const int kVarWrong = 3, kVarFluct = 4;
    const double kWrongPt = 2.;
    std::vector<double> hitsHost(std::size_t(nTracks) * kNVar * 3 * NH);
    std::vector<float> geHost(std::size_t(nTracks) * kNVar * 6 * NH, 0.f);
    std::vector<double> bendHost(std::size_t(nTracks) * kNVar * 4 * NH, 0.);
    std::vector<int> clsHost(std::size_t(nTracks) * NH, int(kNoStub));
    std::vector<int> spikedHit(std::size_t(nTracks), -1);
    std::vector<int> ptCellOf(std::size_t(nTracks), 0);
    for (std::size_t c = 0; c < cfgs.size(); ++c)
      for (int t : index[c])
        ptCellOf[t] = (cfgs[c].pT < 2.) ? 0 : ((cfgs[c].pT < 5.) ? 1 : 2);
    for (int t = 0; t < nTracks; ++t) {
      // the stub the wrong-stub and fluctuation variants act on: rotated over the track's stubs, so
      // that every module class is spiked
      std::vector<int> stubs;
      for (int i = 0; i < NH; ++i)
        if (tracks[t].hit[i].isStub)
          stubs.push_back(i);
      spikedHit[t] = stubs.empty() ? -1 : stubs[std::size_t(t) % stubs.size()];
      for (int i = 0; i < NH; ++i) {
        const Hit& h = tracks[t].hit[i];
        clsHost[std::size_t(t) * NH + i] = int(h.cls);
        const SensorGeom g = kGeom[h.sensor];
        const double dx = g.pitchX * flat(rng), dy = g.lenY * flat(rng);
        const double varX = g.pitchX * g.pitchX / 12., varY = g.lenY * g.lenY / 12.;
        double f = 1.;  // BrokenLineFitKernels' stub transverse-error factors
        if (h.isStub) {
          const bool barrelHit = std::abs(h.p[2]) < 118.;
          f = (h.sensor == kSS) ? (barrelHit ? 0.4624 : 0.8464) : (barrelHit ? 0.64 : 0.9025);
        }
        const double vx = varX * f;
        const double c3[2][3] = {{h.ex[0], h.ex[1], h.ex[2]}, {h.ey[0], h.ey[1], h.ey[2]}};
        auto comp = [&](int a, int b) { return vx * c3[0][a] * c3[0][b] + varY * c3[1][a] * c3[1][b]; };
        // the true bend of this crossing, and the bend a kWrongPt track through the origin would have
        double cosB, sinB, cot;
        betaOf(h, cosB, sinB, cot);
        const double r = std::hypot(h.p[0], h.p[1]), z = h.p[2];
        const double bendTrue = h.isStub ? bendOf(r, z, cosB, sinB, cot, h.nR, h.nZ) : 0.;
        const double sinW = std::copysign(std::min(0.99, std::abs(kBField / kWrongPt * r / 2.)), sinB);
        const double bendWrong = h.isStub ? bendOf(r, z, std::sqrt(1. - sinW * sinW), sinW, cot, h.nR, h.nZ) : 0.;
        const double sig = h.isStub ? kSigBend[h.cls] : -1.;
        for (int v = 0; v < kNVar; ++v) {
          const std::size_t b = std::size_t(t) * kNVar + v;
          for (int r3 = 0; r3 < 3; ++r3)
            hitsHost[3 * NH * b + 3 * i + r3] = h.p[r3] + dx * h.ex[r3] + dy * h.ey[r3];
          float* ge = geHost.data() + 6 * NH * b + 6 * i;
          ge[0] = float(comp(0, 0));
          ge[1] = float(comp(0, 1));
          ge[2] = float(comp(1, 1));
          ge[3] = float(comp(0, 2));
          ge[4] = float(comp(1, 2));
          ge[5] = float(comp(2, 2));
          double* sb = bendHost.data() + 4 * NH * b + 4 * i;
          sb[1] = sig;
          sb[2] = h.nR;
          sb[3] = h.nZ;
          if (!h.isStub)
            continue;
          const double noiseScale = (v == 1) ? 0.5 : ((v == 2) ? 2. : 1.);
          double meas = bendTrue + sig * noiseScale * gauss(rng);
          if (i == spikedHit[t]) {
            if (v == kVarWrong)
              meas = bendWrong + sig * gauss(rng);
            else if (v == kVarFluct)
              meas = bendTrue + 3. * sig;
          }
          sb[0] = meas;
        }
      }
    }

    const std::size_t nLane = std::size_t(nTracks) * kNVar;
    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(hitsHost.size());
    std::copy(hitsHost.begin(), hitsHost.end(), hits_h.data());
    auto ge_h = cms::alpakatools::make_host_buffer<float[], Platform>(geHost.size());
    std::copy(geHost.begin(), geHost.end(), ge_h.data());
    auto bend_h = cms::alpakatools::make_host_buffer<double[], Platform>(bendHost.size());
    std::copy(bendHost.begin(), bendHost.end(), bend_h.data());
    auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kBufferFloats);
    std::copy_n(rho, blMaterialMap::kBufferFloats, rho_h.data());
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(nLane * O::kSize);
    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, hitsHost.size());
    auto ge_d = cms::alpakatools::make_device_buffer<float[]>(queue, geHost.size());
    auto bend_d = cms::alpakatools::make_device_buffer<double[]>(queue, bendHost.size());
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kBufferFloats);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, nLane * O::kSize);
    auto nodes_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, nLane * (2 * NH + 1));
    auto scr_d =
        cms::alpakatools::make_device_buffer<double[]>(queue, nLane * std::size_t(gbld::kGblScratchDoubles<2 * NH>));
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ge_d, ge_h);
    alpaka::memcpy(queue, bend_d, bend_h);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>((nLane + 63) / 64, 64),
                        StubTestKernel<NH>{},
                        hits_d.data(),
                        ge_d.data(),
                        bend_d.data(),
                        rho_d.data(),
                        nTracks,
                        nodes_d.data(),
                        scr_d.data(),
                        out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);
    const double* out = out_h.data();

    // ---------------------------------------------------------------------------------------- read out
    const char* cname[5] = {"flat PS", "flat 2S", "tilted PS", "endcap PS", "endcap 2S"};
    std::vector<std::vector<double>> bendPull(5);
    for (int t = 0; t < nTracks; ++t) {
      for (int v = 0; v < kNVar; ++v) {
        const double* o = out + (std::size_t(t) * kNVar + v) * O::kSize;
        if (o[O::kVarNo + 2] <= 0.)
          continue;  // the lane did not fit
        const int dropped = int(o[O::kDropped]);
        for (int i = 0; i < NH; ++i) {
          const int cls = clsHost[std::size_t(t) * NH + i];
          if (cls == int(kNoStub) || o[O::kChi2B + i] < 0.)
            continue;
          const bool spiked = (i == spikedHit[t]) && (v == kVarWrong || v == kVarFluct);
          const bool drop = (dropped == i);
          if (v == 0 && !spiked) {
            ++sum.nCleanStub;
            ++sum.nClass[cls];
            if (drop) {
              ++sum.nFalseDrop;
              ++sum.nDropClass[cls];
            }
            const double* sb = bendHost.data() + 4 * NH * (std::size_t(t) * kNVar + v) + 4 * i;
            bendPull[cls].push_back((sb[0] - o[O::kPred + i]) / sb[1]);
            // device vs host twin on the same fitted circle and the same node
            double predHost = 0.;
            const double* hp3 = hitsHost.data() + 3 * NH * (std::size_t(t) * kNVar + v) + 3 * i;
            const double cotHost = (o[O::kNextRef + 3] != 0.) ? -o[O::kQCharge] / o[O::kNextRef + 3] : 0.;
            if (gblh::gblStubBendPrediction(o[O::kNextRef + 0],
                                            o[O::kNextRef + 1],
                                            o[O::kNextRef + 2],
                                            int(o[O::kQCharge]),
                                            cotHost,
                                            hp3[0],
                                            hp3[1],
                                            hp3[2],
                                            sb[2],
                                            sb[3],
                                            predHost)) {
              const double scale = std::max(1.e-12, std::abs(predHost));
              sum.worstTwin = std::max(sum.worstTwin, std::abs(predHost - o[O::kPred + i]) / scale);
            }
          } else if (v == 1 && !spiked) {
            ++sum.nCleanStubLoose;
            if (drop)
              ++sum.nFalseDropLoose;
          } else if (v == 2 && !spiked) {
            ++sum.nCleanStubSharp;
            if (drop)
              ++sum.nFalseDropSharp;
          } else if (spiked && v == kVarWrong) {
            ++sum.nWrong[ptCellOf[t]];
            if (drop)
              ++sum.nWrongDropped[ptCellOf[t]];
            if (ptCellOf[t] == 0) {
              ++sum.nWrongClass[cls];
              if (drop)
                ++sum.nWrongDropClass[cls];
            }
          } else if (spiked && v == kVarFluct) {
            ++sum.nFluct;
            if (drop)
              ++sum.nFluctDropped;
          }
        }
      }
    }

    printf("\n== GBL wrong-stub test, %s, %d hits, %d replicas/cell, %d tracks ==\n", label, NH, nRep, nTracks);
    printf("  bend pull (measured - predicted)/sigma over clean stubs, and the drops at chi2(1) > %.2f\n",
           kStubBendChi2Cut);
    printf("  class      |      n |   mean |  core |  false drops | a 2 GeV stub on a 1 GeV host, dropped\n");
    for (int c = 0; c < 5; ++c) {
      if (bendPull[c].size() < 50)
        continue;
      double mu, sg;
      coreOf(bendPull[c], mu, sg);
      printf("  %-10s | %6zu | %+6.3f | %5.3f | %6ld (%.3f %%) | %6.1f %% of %ld\n",
             cname[c],
             bendPull[c].size(),
             mu,
             sg,
             sum.nDropClass[c],
             100. * double(sum.nDropClass[c]) / double(std::max(1L, sum.nClass[c])),
             100. * double(sum.nWrongDropClass[c]) / double(std::max(1L, sum.nWrongClass[c])),
             sum.nWrongClass[c]);
      sum.worstBias = std::max(sum.worstBias, std::abs(mu));
      sum.worstWidth = std::max(sum.worstWidth, std::abs(sg - 1.));
    }

    // The refit's own pulls on the clean variant, with the test on and off: tracks with no wrong stub
    // must not move.
    const char* pname[5] = {"phi", "d0", "|k|", "cot", "z0"};
    printf("\n  refit pulls on clean tracks, test off -> on (core width, core mean)\n");
    for (std::size_t c = 0; c < cfgs.size(); ++c) {
      const Config& cf = cfgs[c];
      std::vector<double> pOn[5], pOff[5];
      double cotSum = 0.;
      int nCot = 0;
      for (int t : index[c]) {
        const double* o = out + (std::size_t(t) * kNVar + 0) * O::kSize;
        if (o[O::kVarNo + 2] <= 0.)
          continue;
        cotSum += o[O::kParNo + 3];
        ++nCot;
      }
      if (nCot < 20)
        continue;
      const double cotTrue = (cotSum >= 0. ? 1. : -1.) * std::sinh(cf.eta);
      const double truth[5] = {0., 0., kBField / cf.pT, cotTrue, 0.};
      for (int t : index[c]) {
        const double* o = out + (std::size_t(t) * kNVar + 0) * O::kSize;
        if (o[O::kVarNo + 2] <= 0.)
          continue;
        for (int a = 0; a < 5; ++a) {
          const double v = (a == 2) ? std::abs(o[O::kPar + a]) : o[O::kPar + a];
          const double vNo = (a == 2) ? std::abs(o[O::kParNo + a]) : o[O::kParNo + a];
          if (o[O::kVar + a] > 0.)
            pOn[a].push_back((v - truth[a]) / std::sqrt(o[O::kVar + a]));
          if (o[O::kVarNo + a] > 0.)
            pOff[a].push_back((vNo - truth[a]) / std::sqrt(o[O::kVarNo + a]));
        }
      }
      printf("  q %+d pT %4.0f eta %3.1f |", cf.q, cf.pT, cf.eta);
      for (int a = 0; a < 5; ++a) {
        if (pOn[a].size() < 50 || pOff[a].size() < 50) {
          printf("  %s      -      |", pname[a]);
          continue;
        }
        double mOn, sOn, mOff, sOff;
        coreOf(pOn[a], mOn, sOn);
        coreOf(pOff[a], mOff, sOff);
        printf(" %s %5.3f->%5.3f |", pname[a], sOff, sOn);
        sum.worstParShift = std::max(sum.worstParShift, std::abs(sOn - 1.) - std::abs(sOff - 1.));
      }
      printf("\n");
    }
  }

}  // namespace

TEST_CASE("GBL wrong-stub outlier test on the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  const float* rho = blMaterialMap::blMaterialMapData();

  for (auto const& device : devices) {
    auto queue = Queue(device);
    printf("\n%s\n", alpaka::getName(device).c_str());
    Summary ot, pix;
    runLayout<kNOt>(queue, rho, /*otOnly=*/true, kRep, "outer-tracker only (the displaced layout)", ot);
    runLayout<kNPix>(queue, rho, /*otOnly=*/false, kRep / 4, "pixel-seeded (control)", pix);

    auto rate = [](long a, long b) { return 100. * double(a) / double(std::max(1L, b)); };
    printf(
        "\n  false-drop rate at the bound: %.3f %% (outer tracker, %ld stubs) / %.3f %% (pixel-seeded, %ld)\n"
        "    declared error 2x too loose: %.3f %% / %.3f %%;  2x too sharp: %.3f %% / %.3f %%\n"
        "  a 2 GeV stub placed on a host of pT 1 / 3 / 10 GeV, dropped: %.1f / %.1f / %.1f %% (outer tracker,"
        " %ld at 1 GeV), %.1f / %.1f / %.1f %% (pixel-seeded)\n"
        "  right stub with a 3-sigma bend fluctuation dropped: %.2f %% / %.2f %%\n"
        "  largest bend-pull |mean| %.3f / %.3f, largest |core width - 1| %.3f / %.3f\n"
        "  largest increase of |pull width - 1| on clean tracks %+.4f / %+.4f; device vs host twin %.2e / %.2e\n",
        rate(ot.nFalseDrop, ot.nCleanStub),
        ot.nCleanStub,
        rate(pix.nFalseDrop, pix.nCleanStub),
        pix.nCleanStub,
        rate(ot.nFalseDropLoose, ot.nCleanStubLoose),
        rate(pix.nFalseDropLoose, pix.nCleanStubLoose),
        rate(ot.nFalseDropSharp, ot.nCleanStubSharp),
        rate(pix.nFalseDropSharp, pix.nCleanStubSharp),
        rate(ot.nWrongDropped[0], ot.nWrong[0]),
        rate(ot.nWrongDropped[1], ot.nWrong[1]),
        rate(ot.nWrongDropped[2], ot.nWrong[2]),
        ot.nWrong[0],
        rate(pix.nWrongDropped[0], pix.nWrong[0]),
        rate(pix.nWrongDropped[1], pix.nWrong[1]),
        rate(pix.nWrongDropped[2], pix.nWrong[2]),
        rate(ot.nFluctDropped, ot.nFluct),
        rate(pix.nFluctDropped, pix.nFluct),
        ot.worstBias,
        pix.worstBias,
        ot.worstWidth,
        pix.worstWidth,
        ot.worstParShift,
        pix.worstParShift,
        ot.worstTwin,
        pix.worstTwin);

    // The model has to be unbiased with unit pull, the bound has to cost ~0.1 % of the clean stubs, a
    // stub from a 2 GeV track on a 1 GeV one has to go, a 3-sigma fluctuation has to stay, and tracks
    // with no wrong stub must not move.
    CHECK(std::max(ot.worstBias, pix.worstBias) < 0.1);
    CHECK(std::max(ot.worstWidth, pix.worstWidth) < 0.1);
    CHECK(rate(ot.nFalseDrop, ot.nCleanStub) < 0.3);
    CHECK(rate(pix.nFalseDrop, pix.nCleanStub) < 0.3);
    // The bound's reach is the per-class census: a 2 GeV stub on a 1 GeV track is chi2 141 on flat 2S,
    // 32 on endcap 2S, 16 on flat PS and only 1.8 on endcap PS, so the endcap-PS class is blind to it
    // by construction of the bend precision, not by the test.
    CHECK(rate(ot.nWrongDropClass[kFlat2S], ot.nWrongClass[kFlat2S]) > 95.);
    CHECK(rate(ot.nWrongDropClass[kEndcap2S], ot.nWrongClass[kEndcap2S]) > 90.);
    CHECK(rate(ot.nWrongDropClass[kFlatPS], ot.nWrongClass[kFlatPS]) > 80.);
    CHECK(rate(ot.nFluctDropped, ot.nFluct) < 5.);
    CHECK(std::max(ot.worstParShift, pix.worstParShift) < 0.05);
    CHECK(std::max(ot.worstTwin, pix.worstTwin) < 1.e-5);
  }
}
