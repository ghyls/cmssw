// Split-build TU: extended-N refit scans runRefitScanBin<N, Phase2OTStubs>, N=9..12 (stride kRefitStride).
// Each N instantiated once (ODR).
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_REFIT_SIG(9);
  BLFIT_REFIT_SIG(10);
  BLFIT_REFIT_SIG(11);
  BLFIT_REFIT_SIG(12);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
