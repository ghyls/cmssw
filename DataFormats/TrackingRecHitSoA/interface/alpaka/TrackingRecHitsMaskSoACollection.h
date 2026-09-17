#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsMaskSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsMaskSoACollection_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using TrackingRecHitsMaskingCollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                              ::reco::TrackingRecHitsMaskingHost,
                                                              ::reco::TrackingRecHitsMaskingDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

namespace cms::alpakatools {

  template <typename TDevice>
  struct CopyToHost<::reco::TrackingRecHitsMaskingDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::reco::TrackingRecHitsMaskingDevice<TDevice> const& deviceData) {
      auto nHits = deviceData.view().metadata().size();

      ::reco::TrackingRecHitsMaskingHost hostData(queue, nHits);

      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());

      return hostData;
    }

    // No postCopy needed - host data is already synchronized
    static void postCopy(::reco::TrackingRecHitsMaskingHost&) {}
  };

}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::TrackingRecHitsMaskingCollection, reco::TrackingRecHitsMaskingHost);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsMaskSoACollection_h
