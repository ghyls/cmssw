

#include <alpaka/alpaka.hpp>
#include <Eigen/Core>

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
#include "DataFormats/EcalDigi/interface/EcalDigiSoA.h"

// #include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceObject.h"
#include "DataFormats/Portable/interface/PortableDeviceObject.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceFactory.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSource.h"
#include "TrivialSerialisation/Common/interface/SerialiserFactory.h"
#include "TrivialSerialisation/Common/interface/Serialiser.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

using DeviceProductPortableObjectTestStruct =
    edm::DeviceProduct<PortableObject<portabletest::TestStruct, Device, void>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(DeviceProductPortableObjectTestStruct);

using PortableCollectionTestSoALayout = PortableCollection<portabletest::TestSoALayout<128, false>, Device>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableCollectionTestSoALayout>,
                  typeid(PortableCollectionTestSoALayout).name());

using DeviceProductPortableCollectionTestSoALayout = edm::DeviceProduct<PortableCollectionTestSoALayout>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(DeviceProductPortableCollectionTestSoALayout);

using PortableMultiCollection2 = edm::DeviceProduct<
    PortableMultiCollection<Device, portabletest::TestSoALayout<128, false>, portabletest::TestSoALayout2<128, false>>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableMultiCollection2);

using PortableMultiCollection3 = edm::DeviceProduct<PortableMultiCollection<Device,
                                                                            portabletest::TestSoALayout<128, false>,
                                                                            portabletest::TestSoALayout2<128, false>,
                                                                            portabletest::TestSoALayout3<128, false>>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableMultiCollection3);

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
