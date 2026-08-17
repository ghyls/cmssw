// Split-build TU: extended-N refit runRefitBin<N, Phase2OTStubs> for N=3..8
// (lane stride kRefitStride=2048). Each N instantiated in exactly one TU (ODR).
#include "BrokenLineFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  BLFIT_REFIT_SIG(3);
  BLFIT_REFIT_SIG(4);
  BLFIT_REFIT_SIG(5);
  BLFIT_REFIT_SIG(6);
  BLFIT_REFIT_SIG(7);
  BLFIT_REFIT_SIG(8);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
