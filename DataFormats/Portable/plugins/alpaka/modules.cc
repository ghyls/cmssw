

#include <alpaka/alpaka.hpp>
#include <Eigen/Core>

#include "DataFormats/Portable/interface/PortableCollection.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
#include "DataFormats/EcalDigi/interface/EcalDigiSoA.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceFactory.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSource.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using PortableCollectionTestSoALayout = PortableCollection<portabletest::TestSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionTestSoALayout>,
                  typeid(PortableCollectionTestSoALayout).name());

using PortableCollectionHcalRecHitSoALayout = PortableCollection<hcal::HcalRecHitSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionHcalRecHitSoALayout>,
                  typeid(PortableCollectionHcalRecHitSoALayout).name());

using PortableCollectionPFRecHitSoALayout = PortableCollection<reco::PFRecHitSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionPFRecHitSoALayout>,
                  typeid(PortableCollectionPFRecHitSoALayout).name());

using PortableCollectionPFClusterSoALayout = PortableCollection<reco::PFClusterSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionPFClusterSoALayout>,
                  typeid(PortableCollectionPFClusterSoALayout).name());

using PortableCollectionPFRecHitFractionSoALayout =
    PortableCollection<reco::PFRecHitFractionSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionPFRecHitFractionSoALayout>,
                  typeid(PortableCollectionPFRecHitFractionSoALayout).name());

using PortableCollectionEcalDigiSoALayout = PortableCollection<EcalDigiSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionEcalDigiSoALayout>,
                  typeid(PortableCollectionEcalDigiSoALayout).name());

using PortableCollectionEcalUncalibratedRecHitSoALayout =
    PortableCollection<EcalUncalibratedRecHitSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionEcalUncalibratedRecHitSoALayout>,
                  typeid(PortableCollectionEcalUncalibratedRecHitSoALayout).name());
