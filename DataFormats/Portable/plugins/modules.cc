

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceFactory.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSource.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableHostObject<portabletest::TestStruct>>,
                  typeid(PortableHostObject<portabletest::TestStruct>).name());

DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<edm::EventID>,
                  typeid(edm::EventID).name());

// using PortableHostCollectionTestSoALayout = PortableHostCollection<portabletest::TestSoALayout<128, false>>;
// DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
//                   ngt::TrivialSerialiserSource<PortableHostCollectionTestSoALayout>,
//                   typeid(PortableHostCollectionTestSoALayout).name());

using PortableHostMultiCollectionTestSoALayout2 =
    PortableHostMultiCollection<portabletest::TestSoALayout<128, false>, portabletest::TestSoALayout2<128, false>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableHostMultiCollectionTestSoALayout2>,
                  typeid(PortableHostMultiCollectionTestSoALayout2).name());

using PortableHostMultiCollectionTestSoALayout3 = PortableHostMultiCollection<portabletest::TestSoALayout<128, false>,
                                                                              portabletest::TestSoALayout2<128, false>,
                                                                              portabletest::TestSoALayout3<128, false>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<PortableHostMultiCollectionTestSoALayout3>,
                  typeid(PortableHostMultiCollectionTestSoALayout3).name());
