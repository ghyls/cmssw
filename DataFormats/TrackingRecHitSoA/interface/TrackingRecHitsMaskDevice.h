#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskDevice_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskDevice_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskSoA.h"

namespace reco {

  template <typename TDev>
  using TrackingRecHitsMaskingDevice = PortableDeviceCollection<TDev, reco::TrackingRecHitsMaskingSoA>;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskDevice_h
