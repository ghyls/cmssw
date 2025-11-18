#include "TrivialSerialisation/Common/interface/SerialiserFactory.h"
#include "TrivialSerialisation/Common/interface/Serialiser.h"

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

#include "Eigen/Core"

#include "DataFormats/EcalDigi/interface/EcalDigiSoA.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/VertexSoA/interface/ZVertexSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"


// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollection<EcalDigiSoALayout<128,false> >);

// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollection<EcalUncalibratedRecHitSoALayout<128,false> >);

// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollection<hcal::HcalRecHitSoALayout<128,false> >);

// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollection<reco::PFRecHitSoALayout<128,false> >);

// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollection<reco::PFClusterSoALayout<128,false> >);

// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollection<reco::PFRecHitFractionSoALayout<128,false> >);



using PortableHostCollectionEcalDigi = PortableHostCollection<EcalDigiSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionEcalDigi);

using PortableHostCollectionEcalUncalibratedRecHit = PortableHostCollection<EcalUncalibratedRecHitSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionEcalUncalibratedRecHit);

using PortableHostCollectionHcalRecHit = PortableHostCollection<hcal::HcalRecHitSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionHcalRecHit);

using PortableHostCollectionPFRecHit = PortableHostCollection<reco::PFRecHitSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionPFRecHit);

using PortableHostCollectionPFCluster = PortableHostCollection<reco::PFClusterSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionPFCluster);

using PortableHostCollectionPFRecHitFraction = PortableHostCollection<reco::PFRecHitFractionSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionPFRecHitFraction);

using PortableHostObjectBeamSpotPOD = PortableHostObject<BeamSpotPOD>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostObjectBeamSpotPOD);

using PortableHostCollectionSiPixelClusters = PortableHostCollection<SiPixelClustersSoA>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionSiPixelClusters);

// ?
// using PortableHostCollectionTrackingRecHit = reco::TrackingRecHitSoA;
// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionTrackingRecHit);

using PortableHostCollectionTracks = PortableHostCollection<reco::TrackLayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionTracks);

// SiPixelDigisHost
DEFINE_TRIVIAL_SERIALISER_PLUGIN(SiPixelDigisHost);


// PortableHostMultiCollection<TrackSoA, TrackHitSoA>
using PortableHostMultiCollectionTracks = PortableHostMultiCollection<reco::TrackSoA, reco::TrackHitSoA>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostMultiCollectionTracks);

// using PortableHostMultiCollectionTrackingRecHits = PortableHostMultiCollection<reco::TrackingHitsLayout<128>, reco::HitModulesLayout<128>>;
// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostMultiCollectionTrackingRecHits);

// using PortableHostMultiCollectionTracks = PortableHostMultiCollection<reco::TrackLayout<128>, reco::TrackHitsLayout<128>>;
// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostMultiCollectionTracks);

// using PortableHostMultiCollectionZVertex = PortableHostMultiCollection<reco::ZVertexLayout<128>, reco::ZVertexTracksLayout<128>>;
// DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostMultiCollectionZVertex);
