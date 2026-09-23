#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::OTRecHitsHost, reco::OTRecHitsSoACollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::StubsHost, reco::StubsSoACollection);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::TrackingRecHitHost, reco::TrackingRecHitsSoACollection);
