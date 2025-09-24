#ifndef TrivialSerialisation_Common_TrivialSerialiser_h
#define TrivialSerialisation_Common_TrivialSerialiser_h

#include <cstdio>
#include <concepts>
#include "TrivialSerialisation/Common/interface/TrivialSerialiserBase.h"
#include "TrivialSerialisation/Common/interface/TrivialCopyTraits.h"
#include "DataFormats/Common/interface/Uninitialized.h"

#include "FWCore/Framework/interface/WrapperBaseHandle.h"

// defines all methods of TrivialSerialiserBase

namespace ngt {
  template <typename T>
  class TrivialSerialiser : public TrivialSerialiserBase {
  public:
    TrivialSerialiser(int tag) : TrivialSerialiserBase(), present(false) {printf("%d\n", tag);}
    // TrivialSerialiser() : TrivialSerialiserBase(), present(false) {}
    // TODO: Where is present set to true?

    bool hasTrivialCopyTraits() const override;
    bool hasTrivialCopyProperties() const override;
    void trivialCopyInitialize(edm::WrapperBase& wrapper, edm::AnyBuffer const& args) override;
    edm::AnyBuffer trivialCopyParameters(edm::WrapperBase const& wrapper) const override;
    std::vector<std::span<const std::byte>> trivialCopyRegions(edm::WrapperBase const& wrapper) const override;
    std::vector<std::span<std::byte>> trivialCopyRegions(edm::WrapperBase& wrapper) override;
    void trivialCopyFinalize(edm::WrapperBase& wrapper) override;

  private:
    constexpr T construct_() {
      if constexpr (requires { T(); }) {
        return T();
      } else {
        return T(edm::kUninitialized);
      }
    }

    const T& getWrappedObj_(edm::WrapperBase const& wrapper) const {
      auto& w = static_cast<const edm::Wrapper<T>&>(wrapper);
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      return *w.product();
    }

    T& getWrappedObj_(edm::WrapperBase& wrapper) {
      auto& w = static_cast<edm::Wrapper<T>&>(wrapper);
      if (not w.isPresent()) {
        throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
      }
      return *w.bareProduct();
    }



  private:
    // TODO: KEEP IT CONST DO CONST_CAST ONLY ON THE NON CONST VERSION OF TCR
    // T* obj = nullptr;
    bool present;
    // edm::WrapperBase *wrapper;
  };

  template <typename T>
  inline bool TrivialSerialiser<T>::hasTrivialCopyTraits() const {
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline bool TrivialSerialiser<T>::hasTrivialCopyProperties() const {
    if constexpr (requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      printf("TrivialCopyTraits<%s>::Properties is defined\n", edm::typeDemangle(typeid(T).name()).c_str());
      return true;
    }
    return false;
  }

  template <typename T>
  // requires (!std::is_const_v<T>)
  // void TrivialSerialiser<T>::trivialCopyInitialize([[maybe_unused]] edm::AnyBuffer const& args) {
  void TrivialSerialiser<T>::trivialCopyInitialize(edm::WrapperBase& wrapper, edm::AnyBuffer const& args) {
    printf("In TrivialSerialiser::trivialCopyInitialize. Heloosa\n");
    auto& w = dynamic_cast<edm::Wrapper<T>&>(wrapper);
    T& obj = w.bareProduct();
    // const T* obj = w.product();
    present = true;

    if (not wrapper.isPresent()) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty TrivialSerialiser66";
    }
    // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call initialize()
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return;
    } else
      // if edm::TrivialCopyTraits<T>::Properties is void, call initialize() without any additional arguments
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        edm::TrivialCopyTraits<T>::initialize(obj);
      } else
      // if edm::TrivialCopyTraits<T>::Properties is not void, cast args to Properties and pass it as an additional argument to initialize()
      {
        edm::TrivialCopyTraits<T>::initialize(obj, args.cast_to<typename edm::TrivialCopyTraits<T>::Properties>());
      }
  }


  template <typename T>
  inline edm::AnyBuffer TrivialSerialiser<T>::trivialCopyParameters(edm::WrapperBase const& wrapper) const {

    // auto& w = dynamic_cast<edm::Wrapper<T> const&>(wrapper);
    // if (not w.isPresent()) {
    //   throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    // }
    const T& obj = getWrappedObj_(wrapper);
    // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call properties()
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return {};
    } else
      // if edm::TrivialCopyTraits<T>::Properties is void, do not call properties()
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        return {};
      } else
      // if edm::TrivialCopyTraits<T>::Properties is not void, call properties() and wrap the result in an edm::AnyBuffer
      {
        typename edm::TrivialCopyTraits<T>::Properties p = edm::TrivialCopyTraits<T>::properties(obj);
        return edm::AnyBuffer(p);
      }
  }

  template <typename T>
  inline std::vector<std::span<const std::byte>> TrivialSerialiser<T>::trivialCopyRegions(edm::WrapperBase const& wrapper) const {

    const T& obj = getWrappedObj_(wrapper);

    if constexpr (requires(T const& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline std::vector<std::span<std::byte>> TrivialSerialiser<T>::trivialCopyRegions(edm::WrapperBase& wrapper) {

    auto& w = dynamic_cast<edm::Wrapper<T>&>(wrapper);
    T& obj = w.bareProduct();

    if (not w.isPresent()) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty TrivialSerialiser";
    }
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {

      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline void TrivialSerialiser<T>::trivialCopyFinalize(edm::WrapperBase& wrapper) {
    if (not wrapper.isPresent()) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty TrivialSerialiser23";
    }
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::finalize(t); }) {
      auto& w = dynamic_cast<edm::Wrapper<T>&>(wrapper);
      edm::TrivialCopyTraits<T>::finalize(*w.product());
    }
  }

}  // namespace ngt

#endif
