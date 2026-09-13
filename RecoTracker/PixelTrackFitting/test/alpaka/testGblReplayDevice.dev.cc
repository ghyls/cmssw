// Checks that the host material walk of test/gblTestMaterial.h reproduces the device one: for each
// fixture the device prep (brokenline::prepareGblFitData, material walk and upstream split included) is
// run and gblTestMaterial::fillMatData must match every number on the host: matXX0 per gap, the per-gap
// two-thin split (gapD1, gapW1), innerXX0 and its own split.
//
// Tolerance: the two walks are the same arithmetic in the same order, differing only in
// alpaka::math::sqrt vs std::sqrt and in contraction (~1e-14 accumulated). One benign failure mode: a
// hit sitting exactly on a cell boundary, where a last-bit difference moves one cell piece between two
// neighbours; with equal densities on both sides even that is invisible.

#include <algorithm>
#include <cmath>
#include <cstdio>
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
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"         // upstream host fastFit
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"  // the device march under test
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"
#include "RecoTracker/PixelTrackFitting/test/gblTestMaterial.h"  // the host copy gblReplay uses

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace blh = ::brokenline;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  //!< same arithmetic in the same order, ~1e-14 accumulated.
  constexpr double kTolMarch = 1e-9;

  template <int N>
  struct MatKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  double const* ffIn,
                                  float const* rho,
                                  double bField,
                                  double* out) const {
      for ([[maybe_unused]] auto lane : cms::alpakatools::uniform_elements(acc, 1)) {
        Eigen::Matrix<double, 3, N> hits;
        for (int c = 0; c < N; ++c)
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);

        bld::PreparedGblData<N> data;
        double gapD1[N], gapW1[N];
        blMaterialMap::ElossColumn matCol[N];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1, matCol);
        // the upstream (PCA -> hit0) segment's own two-thin split, from the walk prepareGblFitData ran
        const double innerD1 = data.innerD1, innerW1 = data.innerW1;

        for (int i = 0; i < N; ++i) {
          out[i] = data.matXX0(i);
          out[N + i] = gapD1[i];
          out[2 * N + i] = gapW1[i];
          out[3 * N + 3 * i] = matCol[i].e;
          out[3 * N + 3 * i + 1] = matCol[i].eLnI;
          out[3 * N + 3 * i + 2] = matCol[i].eLnRho;
        }
        out[6 * N] = data.innerXX0;
        out[6 * N + 1] = innerD1;
        out[6 * N + 2] = innerW1;
        out[6 * N + 3] = data.innerCol.e;
        out[6 * N + 4] = data.innerCol.eLnI;
        out[6 * N + 5] = data.innerCol.eLnRho;
      }
    }
  };

  struct MaxRel {
    double v = 0.;
    bool any = false;
    void add(double got, double ref) {
      if (std::abs(ref) > 1e-30) {
        any = true;
        v = std::max(v, std::abs(got - ref) / std::abs(ref));
      }
    }
  };

  template <int N>
  void runFixture(Queue& queue, const char* devName, const char* label, const double D[N][9], const float* rhoDev) {
    constexpr int kOut = 6 * N + 6;
    Eigen::Matrix<double, 3, N> hits;
    for (int k = 0; k < N; ++k) {
      hits(0, k) = D[k][0];
      hits(1, k) = D[k][1];
      hits(2, k) = D[k][2];
    }
    Eigen::Vector4d ff;
    blh::fastFit(hits, ff);

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * N);
    auto ff_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < N; ++c)
      for (int r = 0; r < 3; ++r)
        hits_h[3 * c + r] = hits(r, c);
    for (int j = 0; j < 4; ++j)
      ff_h[j] = ff(j);

    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * N);
    auto ff_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, kOut);
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(kOut);
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ff_d, ff_h);
    alpaka::memset(queue, out_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        MatKernel<N>{},
                        hits_d.data(),
                        ff_d.data(),
                        rhoDev,
                        gblTestFixtures::kB,
                        out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);

    // the arc lengths the walk needs (3-D path per gap, PCA of the upstream segment), host twin
    blh::PreparedBrokenLineData<N> hdata;
    blh::prepareBrokenLineData(hits, ff, gblTestFixtures::kB, hdata);
    gblTestMaterial::MatData<N> md;
    gblTestMaterial::fillMatData<N>(hits, hdata.sTransverse, hdata.sTotal, -hdata.qCharge / ff(3), md);

    MaxRel mat, gd1, gw1, inner, col;
    for (int i = 0; i < N; ++i) {
      mat.add(md.matXX0[i], out_h[i]);
      gd1.add(md.gapD1[i], out_h[N + i]);
      gw1.add(md.gapW1[i], out_h[2 * N + i]);
      col.add(md.matCol[i].e, out_h[3 * N + 3 * i]);
      col.add(md.matCol[i].eLnI, out_h[3 * N + 3 * i + 1]);
      col.add(md.matCol[i].eLnRho, out_h[3 * N + 3 * i + 2]);
    }
    inner.add(md.innerXX0, out_h[6 * N]);
    inner.add(md.innerD1, out_h[6 * N + 1]);
    inner.add(md.innerW1, out_h[6 * N + 2]);
    col.add(md.innerCol.e, out_h[6 * N + 3]);
    col.add(md.innerCol.eLnI, out_h[6 * N + 4]);
    col.add(md.innerCol.eLnRho, out_h[6 * N + 5]);

    std::printf("  %-28s N=%2d | matXX0=%.2e gapD1=%.2e gapW1=%.2e inner=%.2e dedx=%.2e   (host innerXX0=%.6g)  [%s]\n",
                label,
                N,
                mat.v,
                gd1.v,
                gw1.v,
                inner.v,
                col.v,
                md.innerXX0,
                devName);

    // positive controls: an all-zero march would agree with anything.
    REQUIRE(mat.any);
    REQUIRE(gd1.any);
    REQUIRE(gw1.any);
    REQUIRE(inner.any);
    REQUIRE(col.any);
    REQUIRE(md.innerXX0 > 0.);

    REQUIRE(mat.v < kTolMarch);
    REQUIRE(gd1.v < kTolMarch);
    REQUIRE(gw1.v < kTolMarch);
    REQUIRE(inner.v < kTolMarch);
    REQUIRE(col.v < kTolMarch);
  }

  // Brute-force integral of the same table along the same chord, midpoint rule at `step` cm: an
  // independent reference for the cell walk, with no closed form and no cell bookkeeping. Its own
  // discretization error at the cell boundaries is what the tolerance below leaves room for.
  void bruteForce(double r0, double z0, double r1, double z1, double step, double& W, double& S1, double& S2) {
    const float* rho = blMaterialMap::blMaterialMapData();
    const double dr = r1 - r0, dz = z1 - z0;
    const double L = std::sqrt(dr * dr + dz * dz);
    const int n = std::max(2, int(L / step));
    const double dl = L / n;
    W = S1 = S2 = 0.;
    for (int k = 0; k < n; ++k) {
      const double t = (k + 0.5) / n;
      const double q = blMaterialMap::rhoAt(rho, float(r0 + t * dr), float(z0 + t * dz)) * dl;
      const double d = (1. - t) * L;
      W += q;
      S1 += q * d;
      S2 += q * d * d;
    }
  }

  //!< the walk is exact by construction; this leaves room for the reference's own step error.
  constexpr double kTolBrute = 1.e-3;

  template <int N>
  void runBrute(const char* label, const double D[N][9]) {
    double maxrel = 0.;
    for (int g = -1; g + 1 < N; ++g) {
      const double r0 = (g < 0) ? 0. : std::hypot(D[g][0], D[g][1]);
      const double z0 = (g < 0) ? 0. : D[g][2];
      const double r1 = std::hypot(D[g + 1][0], D[g + 1][1]);
      const double z1 = D[g + 1][2];
      double L, W, S1, S2, bW, bS1, bS2;
      gblTestMaterial::segmentWalk(r0, z0, r1, z1, /*path3D=*/0., L, W, S1, S2);
      bruteForce(r0, z0, r1, z1, 1.e-4, bW, bS1, bS2);
      if (!(bW > 0.))
        continue;
      const double rel[3] = {std::abs(W / bW - 1.), std::abs(S1 / bS1 - 1.), std::abs(S2 / bS2 - 1.)};
      for (double x : rel)
        maxrel = std::max(maxrel, x);
      REQUIRE(rel[0] < kTolBrute);
      REQUIRE(rel[1] < kTolBrute);
      REQUIRE(rel[2] < kTolBrute);
    }
    std::printf("  %-28s worst |walk/brute-1| over W,S1,S2 = %.2e\n", label, maxrel);
  }

}  // namespace

TEST_CASE("gblReplay host material march vs the device for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kBufferFloats);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kBufferFloats, rho_h.data());

  using namespace gblTestFixtures;
  for (auto const& device : devices) {
    auto queue = Queue(device);
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kBufferFloats);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::wait(queue);
    const std::string dn = alpaka::getName(device);
    std::printf("\n=== %s : gblTestMaterial (host) vs brokenline::prepareGblFitData (device) ===\n", dn.c_str());
    runFixture<4>(queue, dn.c_str(), "barrel eta~0 (IT only)", BARREL0, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "central eta~0.36 IT+OT", CENTRAL, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "forward eta~3.2 IT only", FWD, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "tilted eta~1.0 OT barrel", TILT, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "REAL barrel eta~0.27", BARRELR027, rho_d.data());
    runFixture<4>(queue, dn.c_str(), "displaced OT only", DISPLACED1, rho_d.data());
  }
}

TEST_CASE("material cell walk vs a fine-step integral of the same table", "[gblTestMaterial]") {
  using namespace gblTestFixtures;
  std::printf("\n=== exact cell walk vs brute force (1e-4 cm steps) on the same chords ===\n");
  runBrute<4>("barrel eta~0 (IT only)", BARREL0);
  runBrute<10>("central eta~0.36 IT+OT", CENTRAL);
  runBrute<10>("forward eta~3.2 IT only", FWD);
  runBrute<10>("tilted eta~1.0 OT barrel", TILT);
  runBrute<10>("REAL barrel eta~0.27", BARRELR027);
  runBrute<4>("displaced OT only", DISPLACED1);
}
