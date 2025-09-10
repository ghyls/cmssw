#ifndef TrivialSerialisation_Common_interface_TrivialSerialiserBase_h
#define TrivialSerialisation_Common_interface_TrivialSerialiserBase_h

#include "DataFormats/Common/interface/AnyBuffer.h"
#include "DataFormats/Common/interface/WrapperBase.h"

#include <memory>
#include <span>
#include <vector>

namespace ngt {

  class TrivialSerialiserBase {
  public:
    TrivialSerialiserBase() {};

	virtual bool hasTrivialCopyTraits() const = 0;
    virtual bool hasTrivialCopyProperties() const = 0;
    virtual void trivialCopyInitialize(edm::AnyBuffer const& args) = 0;
    virtual edm::AnyBuffer trivialCopyParameters() const = 0;
    virtual std::vector<std::span<const std::byte>> trivialCopyRegions() const = 0;
    virtual std::vector<std::span<std::byte>> trivialCopyRegions() = 0;
    virtual void trivialCopyFinalize() = 0;

    virtual ~TrivialSerialiserBase() = default;
  };

}  // namespace ngt

#endif  // TrivialSerialisation_Common_interface_TrivialSerialiserBase_h