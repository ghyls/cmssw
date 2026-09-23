#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::OTRecHitsHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::StubsHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(reco::TrackingRecHitHost);
