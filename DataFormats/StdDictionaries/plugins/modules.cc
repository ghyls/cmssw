
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceFactory.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiserSource.h"
#include "TrivialSerialisation/Common/interface/TrivialSerialiser.h"

DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory, ngt::TrivialSerialiserSource<int>, typeid(int).name());

DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<unsigned short>,
                  typeid(unsigned short).name());

using basic_string = std::basic_string<char, std::char_traits<char>>;
DEFINE_EDM_PLUGIN(ngt::TrivialSerialiserSourceFactory,
                  ngt::TrivialSerialiserSource<basic_string>,
                  typeid(basic_string).name());
