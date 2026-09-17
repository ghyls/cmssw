#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskSoA_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskSoA_h

#include <cstdint>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // Per-hit veto mask of a multi-iteration masked chain: recHitMask[i] != 0 means "hit i is already
  // used, skip it". It is indexed in lockstep with whatever hit collection it was built for, and it
  // is deliberately kept out of that collection's own layout so that the hit SoA and the mask can be
  // produced, consumed and released independently.
  GENERATE_SOA_LAYOUT(TrackingRecHitsMaskingLayout, SOA_COLUMN(uint32_t, recHitMask));

  using TrackingRecHitsMaskingSoA = TrackingRecHitsMaskingLayout<>;
  using TrackingRecHitsMaskingView = TrackingRecHitsMaskingSoA::View;
  using TrackingRecHitsMaskingConstView = TrackingRecHitsMaskingSoA::ConstView;

  using TrackingRecHitsMaskingHost = PortableHostCollection<reco::TrackingRecHitsMaskingSoA>;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskSoA_h
