#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "TrivialSerialisation/Common/interface/SerialiserFactory.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/TestStruct.h"

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/Portable/interface/PortableDeviceObject.h"

using PortableHostObjectTestStruct = PortableHostObject<portabletest::TestStruct>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostObjectTestStruct);

using PortableHostCollectionTestSoALayout = PortableHostCollection<portabletest::TestSoALayout<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableHostCollectionTestSoALayout);

using PortableMultiCollection2 =
    PortableHostMultiCollection<portabletest::TestSoALayout<128, false>, portabletest::TestSoALayout2<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableMultiCollection2);

using PortableMultiCollection3 = PortableHostMultiCollection<portabletest::TestSoALayout<128, false>,
                                                             portabletest::TestSoALayout2<128, false>,
                                                             portabletest::TestSoALayout3<128, false>>;
DEFINE_TRIVIAL_SERIALISER_PLUGIN(PortableMultiCollection3);
