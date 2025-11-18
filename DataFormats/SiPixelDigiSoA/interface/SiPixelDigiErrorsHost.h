#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h

#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/TrivialCopyTraits.h"
#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

class SiPixelDigiErrorsHost : public PortableHostCollection<SiPixelDigiErrorsSoA> {
public:
  SiPixelDigiErrorsHost(edm::Uninitialized) : PortableHostCollection<SiPixelDigiErrorsSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(int maxFedWords, TQueue queue)
      : PortableHostCollection<SiPixelDigiErrorsSoA>(maxFedWords, queue), maxFedWords_(maxFedWords) {}

  int maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_ = 0;
};


// Specialize the TrivialCopyTraits for SiPixelDigisHost
namespace edm {

  template <>
  struct TrivialCopyTraits<SiPixelDigiErrorsHost> {
    using value_type = SiPixelDigiErrorsHost;
    struct Properties {
      int32_t size;
      int maxFedWords_;
    };

    static Properties properties(value_type const& object) {
      return {static_cast<int32_t>(object->metadata().size()), object.maxFedWords()};
    }

    static void initialize(value_type& object, Properties const& props) {
      // replace the default-constructed empty object with one where the buffer has been allocated in pageable system memory
      object = value_type(props.size, cms::alpakatools::host());
    }

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      std::byte* address = reinterpret_cast<std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}, {reinterpret_cast<std::byte*>(object.maxFedWords()), sizeof(int)}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      const std::byte* address = reinterpret_cast<const std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}, {reinterpret_cast<const std::byte*>(object.maxFedWords()), sizeof(int)}};
    }
  };

}  // namespace edm


#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
