// The normalized (Bz, Br) solenoid map (interface/BLBFieldMap.h) on the device, without EventSetup: a synthetic
// analytic lattice is uploaded, read through the same blBFieldMap::bBendAt / bBendAndBrAt the fit kernels
// call, and checked against the analytic oracle
//     Bz(r,z)/Bz(0,0) = 1 - 7.5e-2 * (z/kZMax)^2      quadratic in z  -> bilinear leaves a known residual
//     Br(r,z)/Bz(0,0) = 1.8e-2 * (r/kRMax) * (z/kZMax) bilinear       -> reproduced exactly
// (CMS magnitudes: Bz falls ~7.5 % at |z| = 2.7 m, |Br/Bz| reaches ~1.8 % in the outer-tracker endcap).
// Asserted: exact values at lattice nodes (indexing, block order, sign); device == host to 1e-14; the
// interpolation at cell centres within 5e-5 of the profile; clamping outside the box; bBendAndBrAt consistent
// with bBendAt; the node builder differs with and without the map; the B_r row flips with the sign of B_r;
// and a snapshot guard on how much the field rows depend on the measured node positions (6.2 % on the
// reference fixture when the outermost node moves 1 cm off the circle).

#include <algorithm>
#include <array>
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

#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace blh = ::brokenline;
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr double kBzCurv = 7.5e-2;   // Bz/Bz00 = 1 - kBzCurv * (z/kZMax)^2
  constexpr double kBrScale = 1.8e-2;  // Br/Bz00 = kBrScale * (r/kRMax) * (z/kZMax)
  constexpr double kTlca = 0.7;        // a representative tanLambda * cos(alpha)

  //!< device vs host for the same constexpr function on the same buffer.
  constexpr double kTolExact = 1e-14;
  //!< bilinear-vs-continuous bound at cell centres, a*h^2/4 = 2.4e-5.
  constexpr double kTolBilinear = 5e-5;

  double bzAnalytic(double r, double z) {
    (void)r;
    const double t = z / double(blBFieldMap::kZMax);
    return 1. - kBzCurv * t * t;
  }
  double brAnalytic(double r, double z) {
    return kBrScale * (r / double(blBFieldMap::kRMax)) * (z / double(blBFieldMap::kZMax));
  }

  struct FieldKernel {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, float const* map, double const* pts, double* out, int nPts, double tlca) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, nPts)) {
        const double r = pts[2 * int(i)], z = pts[2 * int(i) + 1];
        double br = 0.;
        out[3 * int(i) + 0] = blBFieldMap::bBendAt(map, r, z, tlca);
        out[3 * int(i) + 1] = blBFieldMap::bBendAndBrAt(map, r, z, tlca, br);
        out[3 * int(i) + 2] = br;
      }
    }
  };

  //!< the FWD fixture through the production node builder, with and without the map.
  constexpr int kNf = 10;
  constexpr int kNodesF = 2 * kNf + 1;
  // Node-builder variants (see the header): 0 = no map, 1 = map, 2 = map + trajectoryCorrections (the B_r
  // lambda row live), 3 = the same with B_r zeroed, 4 = the same with B_r negated. `bMap` holds the three
  // lattices back to back.
  constexpr int kNVar = 5;

  struct WiringKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* ffIn,
                                  float const* rho,
                                  float const* bMap,
                                  double bField,
                                  gbld::GblNodeData* nodes,
                                  double* scratch,
                                  double* out) const {
      for (auto v : cms::alpakatools::uniform_elements(acc, kNVar)) {
        Eigen::Matrix<double, 3, kNf> hits;
        Eigen::Matrix<float, 6, kNf> hits_ge;
        for (int c = 0; c < kNf; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * c + r];
        }
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);
        bld::PreparedGblData<kNf> data;
        double gapD1[kNf], gapW1[kNf];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        // the upstream (PCA -> hit0) segment's own two-thin split, from the walk prepareGblFitData ran
        const double innerD1 = data.innerD1, innerW1 = data.innerW1;

        // variant wiring (no lookup table: a device lambda cannot read a host namespace array)
        const int lattice = (int(v) <= 2) ? 0 : (int(v) == 3 ? 1 : 2);  // as sampled / B_r = 0 / B_r negated
        const bool trajCorr = (int(v) >= 2);
        const bool noMap = (int(v) == 0);
        gbld::GblNodeData* myNodes = nodes + int(v) * kNodesF;
        const bool ok =
            gbld::prepareGblDataSplit<Acc1D, kNf>(acc,
                                                  hits,
                                                  hits_ge,
                                                  ff,
                                                  bField,
                                                  data.qCharge,
                                                  data.sTransverse,
                                                  data.sTotal,
                                                  data.matXX0,
                                                  gapD1,
                                                  gapW1,
                                                  data.innerXX0,
                                                  innerD1,
                                                  innerW1,
                                                  myNodes,
                                                  /*applyELoss=*/false,
                                                  /*bMap=*/noMap ? nullptr : (bMap + lattice * blBFieldMap::kNValues),
                                                  /*bFieldOrigin=*/bField,
                                                  /*trajectoryCorrections=*/trajCorr,
                                                  /*scatteringLogAtTotal=*/false,
                                                  /*elossCumulative=*/false);
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        double chi2 = 0.;
        const auto cov = gbld::gblFitPca<Acc1D, 2 * kNf>(
            acc, myNodes, scratch + int(v) * gbld::kGblScratchDoubles<2 * kNf>, &corr, nullptr, &chi2);
        double* o = out + int(v) * 32;
        o[0] = ok ? 1. : 0.;
        for (int a = 0; a < 5; ++a) {
          o[1 + a] = corr(a);
          o[6 + a] = cov(a, a);
        }
        o[11] = chi2;
      }
    }
  };

  //!< assertion 8: a 1 GeV fixture (radius comparable to the outer radius) refitted four ways -- map off /
  //!< map on, times nominal / one node moved off the reference circle along its own radius from the fitted
  //!< centre. The move keeps that node's azimuth about the centre, so its arc length is unchanged and the
  //!< only thing it changes is how far off the circle the node sits.
  constexpr int kNp = 9;
  constexpr int kNodesP = 2 * kNp + 1;
  constexpr int kNVarP = 4;
  constexpr double kRadialNudge = 1.0;  // [cm]

  struct NudgeKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* ffIn,
                                  float const* rho,
                                  float const* bMap,
                                  double bField,
                                  gbld::GblNodeData* nodes,
                                  double* scratch,
                                  double* out) const {
      for (auto v : cms::alpakatools::uniform_elements(acc, kNVarP)) {
        Eigen::Matrix<double, 3, kNp> hits;
        Eigen::Matrix<float, 6, kNp> hits_ge;
        for (int c = 0; c < kNp; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * c + r];
        }
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);
        if (int(v) >= 2) {  // move the outermost node off the circle, at fixed azimuth about the centre
          const int c = kNp - 1;
          const double vx = hits(0, c) - ff(0), vy = hits(1, c) - ff(1);
          const double rr = alpaka::math::sqrt(acc, vx * vx + vy * vy);
          const double f = (rr > 0.) ? (rr + kRadialNudge) / rr : 1.;
          hits(0, c) = ff(0) + vx * f;
          hits(1, c) = ff(1) + vy * f;
        }
        bld::PreparedGblData<kNp> data;
        double gapD1[kNp], gapW1[kNp];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        // the upstream (PCA -> hit0) segment's own two-thin split, from the walk prepareGblFitData ran
        const double innerD1 = data.innerD1, innerW1 = data.innerW1;
        gbld::GblNodeData* myNodes = nodes + int(v) * kNodesP;
        const bool ok = gbld::prepareGblDataSplit<Acc1D, kNp>(acc,
                                                              hits,
                                                              hits_ge,
                                                              ff,
                                                              bField,
                                                              data.qCharge,
                                                              data.sTransverse,
                                                              data.sTotal,
                                                              data.matXX0,
                                                              gapD1,
                                                              gapW1,
                                                              data.innerXX0,
                                                              innerD1,
                                                              innerW1,
                                                              myNodes,
                                                              /*applyELoss=*/false,
                                                              /*bMap=*/(int(v) % 2 == 0) ? nullptr : bMap,
                                                              /*bFieldOrigin=*/bField,
                                                              /*trajectoryCorrections=*/true,
                                                              /*scatteringLogAtTotal=*/false,
                                                              /*elossCumulative=*/false);
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        double chi2 = 0.;
        gbld::gblFitPca<Acc1D, 2 * kNp>(
            acc, myNodes, scratch + int(v) * gbld::kGblScratchDoubles<2 * kNp>, &corr, nullptr, &chi2);
        double* o = out + int(v) * 8;
        o[0] = ok ? 1. : 0.;
        for (int a = 0; a < 5; ++a)
          o[1 + a] = corr(a);
      }
    }
  };

}  // namespace

TEST_CASE("blBFieldMap on the device for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  using blBFieldMap::kNNodes;
  using blBFieldMap::kNR;
  using blBFieldMap::kNValues;
  using blBFieldMap::kNZ;
  const double dr = double(blBFieldMap::kRMax) / double(kNR - 1);
  const double dz = 2. * double(blBFieldMap::kZMax) / double(kNZ - 1);

  // the synthetic lattice
  std::vector<float> mapHost(kNValues);
  for (int ir = 0; ir < kNR; ++ir)
    for (int iz = 0; iz < kNZ; ++iz) {
      const double r = ir * dr, z = -double(blBFieldMap::kZMax) + iz * dz;
      mapHost[ir * kNZ + iz] = float(bzAnalytic(r, z));
      mapHost[kNNodes + ir * kNZ + iz] = float(brAnalytic(r, z));
    }

  // query points: every lattice node, every cell centre, and four out-of-box probes
  std::vector<double> pts;
  std::vector<int> nodeIr, nodeIz;  // lattice-node bookkeeping for assertion 1
  for (int ir = 0; ir < kNR; ++ir)
    for (int iz = 0; iz < kNZ; ++iz) {
      pts.push_back(ir * dr);
      pts.push_back(-double(blBFieldMap::kZMax) + iz * dz);
      nodeIr.push_back(ir);
      nodeIz.push_back(iz);
    }
  const int nNodesPts = int(nodeIr.size());
  for (int ir = 0; ir + 1 < kNR; ++ir)
    for (int iz = 0; iz + 1 < kNZ; ++iz) {
      pts.push_back((ir + 0.5) * dr);
      pts.push_back(-double(blBFieldMap::kZMax) + (iz + 0.5) * dz);
    }
  const int nCentres = (kNR - 1) * (kNZ - 1);
  const double rMax = double(blBFieldMap::kRMax), zMax = double(blBFieldMap::kZMax);
  const double clampProbes[4][2] = {{2. * rMax, 0.}, {-5., 0.}, {10., 3. * zMax}, {10., -3. * zMax}};
  const double clampRefs[4][2] = {{rMax, 0.}, {0., 0.}, {10., zMax}, {10., -zMax}};
  for (int i = 0; i < 4; ++i) {
    pts.push_back(clampProbes[i][0]);
    pts.push_back(clampProbes[i][1]);
  }
  for (int i = 0; i < 4; ++i) {
    pts.push_back(clampRefs[i][0]);
    pts.push_back(clampRefs[i][1]);
  }
  const int nPts = int(pts.size()) / 2;
  const int iClamp = nNodesPts + nCentres;

  // the forward fixture, for the wiring check
  Eigen::Matrix<double, 3, kNf> fhits;
  Eigen::Matrix<float, 6, kNf> fge = Eigen::Matrix<float, 6, kNf>::Zero();
  for (int k = 0; k < kNf; ++k) {
    fhits(0, k) = gblTestFixtures::FWD[k][0];
    fhits(1, k) = gblTestFixtures::FWD[k][1];
    fhits(2, k) = gblTestFixtures::FWD[k][2];
    const double zz = (gblTestFixtures::FWD[k][8] > 0.) ? gblTestFixtures::FWD[k][8] : 1.0;
    fge.col(k) << gblTestFixtures::FWD[k][3], gblTestFixtures::FWD[k][4], gblTestFixtures::FWD[k][5],
        gblTestFixtures::FWD[k][6], gblTestFixtures::FWD[k][7], zz;
  }
  Eigen::Vector4d fff;
  blh::fastFit(fhits, fff);

  // the 1 GeV fixture for assertion 8
  Eigen::Matrix<double, 3, kNp> phits;
  Eigen::Matrix<float, 6, kNp> pge = Eigen::Matrix<float, 6, kNp>::Zero();
  for (int k = 0; k < kNp; ++k) {
    phits(0, k) = gblTestFixtures::PT1[k][0];
    phits(1, k) = gblTestFixtures::PT1[k][1];
    phits(2, k) = gblTestFixtures::PT1[k][2];
    const double zz = (gblTestFixtures::PT1[k][8] > 0.) ? gblTestFixtures::PT1[k][8] : 1.0;
    pge.col(k) << gblTestFixtures::PT1[k][3], gblTestFixtures::PT1[k][4], gblTestFixtures::PT1[k][5],
        gblTestFixtures::PT1[k][6], gblTestFixtures::PT1[k][7], zz;
  }
  Eigen::Vector4d pff;
  blh::fastFit(phits, pff);

  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kBufferFloats);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kBufferFloats, rho_h.data());

  for (auto const& device : devices) {
    auto queue = Queue(device);
    const std::string dn = alpaka::getName(device);

    auto map_h = cms::alpakatools::make_host_buffer<float[], Platform>(kNValues);
    std::copy_n(mapHost.data(), kNValues, map_h.data());
    auto pts_h = cms::alpakatools::make_host_buffer<double[], Platform>(2 * nPts);
    std::copy_n(pts.data(), 2 * nPts, pts_h.data());
    auto map_d = cms::alpakatools::make_device_buffer<float[]>(queue, kNValues);
    auto pts_d = cms::alpakatools::make_device_buffer<double[]>(queue, 2 * nPts);
    auto fout_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * nPts);
    auto fout_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * nPts);
    alpaka::memcpy(queue, map_d, map_h);
    alpaka::memcpy(queue, pts_d, pts_h);
    // nPts is a few thousand, more than one block can hold, so grid-stride it
    constexpr int kThreads = 256;
    const int kBlocks = (nPts + kThreads - 1) / kThreads;
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(kBlocks, kThreads),
                        FieldKernel{},
                        map_d.data(),
                        pts_d.data(),
                        fout_d.data(),
                        nPts,
                        kTlca);
    alpaka::memcpy(queue, fout_h, fout_d);
    alpaka::wait(queue);

    // 1. lattice nodes: exactly the two stored floats combined by the bending law.
    double worstNode = 0.;
    for (int i = 0; i < nNodesPts; ++i) {
      const double bz = double(mapHost[nodeIr[i] * kNZ + nodeIz[i]]);
      const double br = double(mapHost[kNNodes + nodeIr[i] * kNZ + nodeIz[i]]);
      const double want = bz - br * kTlca;
      worstNode = std::max(worstNode, std::abs(fout_h[3 * i] - want) / std::max(1e-30, std::abs(want)));
    }

    // 2. device == host for the same constexpr function, and 5. the two entry points stay in sync.
    double worstHost = 0., worstSync = 0.;
    for (int i = 0; i < nPts; ++i) {
      const double r = pts[2 * i], z = pts[2 * i + 1];
      const double hostVal = blBFieldMap::bBendAt(mapHost.data(), r, z, kTlca);
      worstHost = std::max(worstHost, std::abs(fout_h[3 * i] - hostVal) / std::max(1e-30, std::abs(hostVal)));
      const double pureBz = blBFieldMap::bBendAt(mapHost.data(), r, z, 0.);
      const double rebuilt = fout_h[3 * i + 1] + fout_h[3 * i + 2] * kTlca;
      worstSync = std::max(worstSync, std::abs(rebuilt - pureBz) / std::max(1e-30, std::abs(pureBz)));
    }

    // 3. cell centres vs the continuous profile.
    double worstBilinear = 0.;
    for (int c = 0; c < nCentres; ++c) {
      const int i = nNodesPts + c;
      const double r = pts[2 * i], z = pts[2 * i + 1];
      const double want = bzAnalytic(r, z) - brAnalytic(r, z) * kTlca;
      worstBilinear = std::max(worstBilinear, std::abs(fout_h[3 * i] - want));
    }

    // 4. clamping.
    double worstClamp = 0.;
    for (int i = 0; i < 4; ++i)
      worstClamp = std::max(worstClamp, std::abs(fout_h[3 * (iClamp + i)] - fout_h[3 * (iClamp + 4 + i)]));

    std::printf(
        "\n=== %s : blBFieldMap synthetic lattice (%d nodes, %d centres) ===\n", dn.c_str(), nNodesPts, nCentres);
    std::printf(
        "  node exactness=%.2e  device-vs-host=%.2e  bBendAndBrAt sync=%.2e  bilinear-vs-analytic=%.2e  "
        "clamp=%.2e\n",
        worstNode,
        worstHost,
        worstSync,
        worstBilinear,
        worstClamp);
    REQUIRE(worstNode < kTolExact);
    REQUIRE(worstHost < kTolExact);
    REQUIRE(worstSync < kTolExact);
    REQUIRE(worstBilinear < kTolBilinear);
    REQUIRE(worstClamp < 1e-15);

    // 6. wiring: the node builder must see the map
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kBufferFloats);
    auto fh_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * kNf);
    auto fg_h = cms::alpakatools::make_host_buffer<float[], Platform>(6 * kNf);
    auto fq_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < kNf; ++c) {
      for (int r = 0; r < 3; ++r)
        fh_h[3 * c + r] = fhits(r, c);
      for (int r = 0; r < 6; ++r)
        fg_h[6 * c + r] = fge(r, c);
    }
    for (int j = 0; j < 4; ++j)
      fq_h[j] = fff(j);
    auto fh_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * kNf);
    auto fg_d = cms::alpakatools::make_device_buffer<float[]>(queue, 6 * kNf);
    auto fq_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto wn_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kNVar * kNodesF);
    auto ws_d = cms::alpakatools::make_device_buffer<double[]>(queue, kNVar * gbld::kGblScratchDoubles<2 * kNf>);
    auto wo_d = cms::alpakatools::make_device_buffer<double[]>(queue, kNVar * 32);
    auto wo_h = cms::alpakatools::make_host_buffer<double[], Platform>(kNVar * 32);
    // three lattices back to back: B_r as sampled, B_r = 0, B_r negated. Bz is the same in all three, so the
    // only thing that changes between variants 2, 3 and 4 is the B_r lambda row.
    auto map3_h = cms::alpakatools::make_host_buffer<float[], Platform>(3 * kNValues);
    for (int b = 0; b < 3; ++b)
      for (int i = 0; i < kNValues; ++i)
        map3_h[b * kNValues + i] = (i < kNNodes) ? mapHost[i] : (b == 0 ? mapHost[i] : (b == 1 ? 0.f : -mapHost[i]));
    auto map3_d = cms::alpakatools::make_device_buffer<float[]>(queue, 3 * kNValues);
    alpaka::memcpy(queue, map3_d, map3_h);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::memcpy(queue, fh_d, fh_h);
    alpaka::memcpy(queue, fg_d, fg_h);
    alpaka::memcpy(queue, fq_d, fq_h);
    alpaka::memset(queue, wo_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, kNVar),
                        WiringKernel{},
                        fh_d.data(),
                        fg_d.data(),
                        fq_d.data(),
                        rho_d.data(),
                        map3_d.data(),
                        gblTestFixtures::kB,
                        wn_d.data(),
                        ws_d.data(),
                        wo_d.data());
    alpaka::memcpy(queue, wo_h, wo_d);
    alpaka::wait(queue);

    REQUIRE(wo_h[0] > 0.5);   // the split layout was built without the map
    REQUIRE(wo_h[32] > 0.5);  // and with it
    double maxCorrRel = 0.;
    for (int a = 0; a < 5; ++a) {
      const double ref = wo_h[1 + a];
      if (std::abs(ref) > 1e-30)
        maxCorrRel = std::max(maxCorrRel, std::abs(wo_h[32 + 1 + a] - ref) / std::abs(ref));
    }
    std::printf(
        "  wiring (FWD fixture): max relative change of the corrections with the map = %.3e  chi2 %.6g -> %.6g\n",
        maxCorrRel,
        wo_h[11],
        wo_h[32 + 11]);
    // only that the map reaches the node builder; the size of the shift is a physics number
    REQUIRE(maxCorrRel > 1e-9);

    // 7. the B_r lambda row of the field-profile offset (trajectoryCorrections): live, linear in B_r and
    //    odd in its sign. Variants 2/3/4 differ only in the B_r block of the lattice, so the difference of
    //    the corrections is that row alone; B_r(r,-z) = -B_r(r,z) (asserted above) then makes it odd in z.
    for (int v = 2; v < kNVar; ++v)
      REQUIRE(wo_h[v * 32] > 0.5);
    double rowMax = 0., oddMax = 0., oddScale = 0.;
    for (int a = 0; a < 5; ++a) {
      const double withBr = wo_h[2 * 32 + 1 + a] - wo_h[3 * 32 + 1 + a];
      const double negBr = wo_h[4 * 32 + 1 + a] - wo_h[3 * 32 + 1 + a];
      rowMax = std::max(rowMax, std::abs(withBr));
      oddMax = std::max(oddMax, std::abs(withBr + negBr));
      oddScale = std::max(oddScale, std::abs(withBr));
    }
    std::printf(
        "  B_r lambda row: max |corr(B_r) - corr(0)| = %.3e, max antisymmetry residual = %.3e\n", rowMax, oddMax);
    REQUIRE(rowMax > 1e-9);                    // the row is not silently dead
    REQUIRE(oddMax < 1e-6 * (oddScale + 1.));  // and it flips with the sign of B_r

    // 8. how far the field rows' contribution moves with the measurement residuals. Four fits of the 1 GeV
    //    fixture on ONE reference: map off / on, times nominal / the outermost node moved 1 cm off the
    //    circle at fixed azimuth about the centre (its arc length is therefore unchanged). The measured
    //    drift is 6.2 % of the shift; see the header note for why it is not zero.
    auto ph_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * kNp);
    auto pg_h = cms::alpakatools::make_host_buffer<float[], Platform>(6 * kNp);
    auto pq_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < kNp; ++c) {
      for (int r = 0; r < 3; ++r)
        ph_h[3 * c + r] = phits(r, c);
      for (int r = 0; r < 6; ++r)
        pg_h[6 * c + r] = pge(r, c);
    }
    for (int j = 0; j < 4; ++j)
      pq_h[j] = pff(j);
    auto ph_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * kNp);
    auto pg_d = cms::alpakatools::make_device_buffer<float[]>(queue, 6 * kNp);
    auto pq_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto pn_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kNVarP * kNodesP);
    auto ps_d = cms::alpakatools::make_device_buffer<double[]>(queue, kNVarP * gbld::kGblScratchDoubles<2 * kNp>);
    auto po_d = cms::alpakatools::make_device_buffer<double[]>(queue, kNVarP * 8);
    auto po_h = cms::alpakatools::make_host_buffer<double[], Platform>(kNVarP * 8);
    alpaka::memcpy(queue, ph_d, ph_h);
    alpaka::memcpy(queue, pg_d, pg_h);
    alpaka::memcpy(queue, pq_d, pq_h);
    alpaka::memset(queue, po_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, kNVarP),
                        NudgeKernel{},
                        ph_d.data(),
                        pg_d.data(),
                        pq_d.data(),
                        rho_d.data(),
                        map3_d.data(),
                        gblTestFixtures::kB,
                        pn_d.data(),
                        ps_d.data(),
                        po_d.data());
    alpaka::memcpy(queue, po_h, po_d);
    alpaka::wait(queue);
    for (int v = 0; v < kNVarP; ++v)
      REQUIRE(po_h[v * 8] > 0.5);
    double shiftScale = 0., shiftDrift = 0.;
    for (int a = 0; a < 5; ++a) {
      const double nom = po_h[1 * 8 + 1 + a] - po_h[0 * 8 + 1 + a];
      const double dis = po_h[3 * 8 + 1 + a] - po_h[2 * 8 + 1 + a];
      shiftScale = std::max(shiftScale, std::abs(nom));
      shiftDrift = std::max(shiftDrift, std::abs(dis - nom));
    }
    std::printf(
        "  residual independence of the field rows (1 GeV fixture): |shift| = %.3e, drift over a %.1f cm nudge = %.3e "
        "(%.2f %%)\n",
        shiftScale,
        kRadialNudge,
        shiftDrift,
        100. * shiftDrift / std::max(1e-30, shiftScale));
    REQUIRE(shiftScale > 1e-9);
    REQUIRE(shiftDrift < 0.12 * shiftScale);
  }
}
