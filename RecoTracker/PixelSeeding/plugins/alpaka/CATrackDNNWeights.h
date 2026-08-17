#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_h

// Dispatcher for the per-iteration loose->tight (Stage-1) DNN weight banks. Each bank lives in
// its own auto-generated header, so separate bake runs never clobber each other:
//   - CATrackDNNWeights_prompt.h : namespace caTrackDNN_prompt
// The bank is chosen at runtime (DnnBank, see CADnnBank.h) from the producer's
// pixelTrack::Iteration; caTrackDNN_eval::score<BANK>() in CATrackDNN.h is compile-time
// specialised per bank. Only the prompt bank exists so far, so every bank name resolves to it.
// `caTrackDNN` is the symbol path used by the kernels for the bank-free constants (kNFeat /
// kDefaultThreshold, e.g. the caTrackFeatures::kNFeat ABI static_assert in Kernel_classifyTracks).

#include "CATrackDNNWeights_prompt.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace caTrackDNN_displaced = caTrackDNN_prompt;
  namespace caTrackDNN = caTrackDNN_prompt;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_h
