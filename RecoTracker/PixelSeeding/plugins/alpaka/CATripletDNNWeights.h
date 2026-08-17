#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h

// Dispatcher for the per-iteration triplet-DNN weight banks. Each bank lives in its own
// auto-generated header, so separate bake runs never clobber each other:
//   - CATripletDNNWeights_prompt.h : namespace caTripletDNN_prompt
// The bank actually evaluated is chosen at runtime (DnnBank, see CADnnBank.h) from the
// producer's pixelTrack::Iteration. caTripletDNN_eval::score<BANK>() in CATripletDNN.h is
// compile-time specialised per bank. Only the prompt bank exists so far, so every bank name
// resolves to it. `caTripletDNN` is the symbol path used by the kernels for the bank-free
// constants (kNFeat / kDefaultThreshold, e.g. the in-kernel feature-vector sizing in
// CATripletCuts.h).

#include "CATripletDNNWeights_prompt.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caTripletDNN_displaced = caTripletDNN_prompt;
  namespace caTripletDNN = caTripletDNN_prompt;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_h
