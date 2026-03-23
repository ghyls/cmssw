#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Serialiser.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {
  using SerialiserFactoryDevice = edmplugin::PluginFactory<SerialiserBase*()>;

  namespace detail {
    // Returns the type_info of the key used to look up a device serialiser from
    // the configured type string. For collection-like products (with ::Layout) this
    // is the Layout type; for object-like products (with ::Product) it is the Product
    // type; otherwise TDevice itself is used.
    template <typename TDevice>
    const std::type_info& serialiserLookupKey() {
      if constexpr (requires { typename TDevice::Layout; })
        return typeid(typename TDevice::Layout);
      else if constexpr (requires { typename TDevice::Product; })
        return typeid(typename TDevice::Product);
      else
        return typeid(TDevice);
    }
  }  // namespace detail
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

// Helper macro to define Serialiser plugins.
//
// TYPE_DEVICE is the inner product type (e.g. PortableDeviceCollection<...>),
// not wrapped in DeviceProduct, and without ALPAKA_ACCELERATOR_NAMESPACE::
// attached to it (it is attached here).
//
// TYPE_HOST is the type that was passed to DEFINE_TRIVIAL_SERIALISER_PLUGIN
// when registering this type in the non-alpaka TrivialSerialisation factory. It
// is required to match a host type with a device serialiser. The H to D product
// transformation can then be registered through this device serialiser.
//
// The plugin is registered under two keys:
//   1. mangled typeid name of TYPE_HOST: used to look up the device serialiser
//      from the host type, enabling automatic H->D transform registration.
//   2. mangled typeid name of a lookup key derived from TYPE_DEVICE:
//      - TYPE_DEVICE::Layout for collection-like products,
//      - TYPE_DEVICE::Product for object-like products,
//      - TYPE_DEVICE itself as final fallback.
//      This supports PortableDeviceCollection<Layout> type strings in configuration.
#define DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE(TYPE_HOST, TYPE_DEVICE)                                   \
  DEFINE_EDM_PLUGIN2(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     typeid(TYPE_HOST).name());                                                                 \
  DEFINE_EDM_PLUGIN3(ALPAKA_ACCELERATOR_NAMESPACE::ngt::SerialiserFactoryDevice,                               \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::Serialiser<ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>, \
                     ALPAKA_ACCELERATOR_NAMESPACE::ngt::detail::serialiserLookupKey<                           \
                         ALPAKA_ACCELERATOR_NAMESPACE::TYPE_DEVICE>().name())
#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserFactoryDevice_h
