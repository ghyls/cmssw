// -*- C++ -*-
#include <algorithm>
#include <mutex>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPIServices/interface/MPIConsistencyChecker.h"

MPIConsistencyChecker::MPIConsistencyChecker(edm::ParameterSet const&) {}

void MPIConsistencyChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("MPIConsistencyChecker", desc);
  descriptions.setComment("This service records and validates MPI sender and receiver module information.");
}

void MPIConsistencyChecker::required() {
  edm::Service<MPIConsistencyChecker> service;
  if (not service.isAvailable()) {
    throw cms::Exception("Configuration") << R"(The MPIConsistencyChecker is required by this module.
Please add it to the configuration, for example via

process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")
)";
  }
}

void MPIConsistencyChecker::recordMPIModuleInfo(bool is_sender,
                                                std::string const& module_label,
                                                std::string const& upstream_label,
                                                int instance,
                                                std::vector<std::string> const& product_types) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  modules_info_.push_back(MPIModuleInfo{is_sender, instance, module_label, product_types});
  module_upstream_labels_.push_back(upstream_label);
}

void MPIConsistencyChecker::getSerializedMPIModuleInfo(std::vector<char>& buffer, std::string const& origin_name) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  auto const& modules = mpi_paths_mappings_[origin_name];
  TBufferFile buf(TBuffer::kWrite);

  TClass* cls = TClass::GetClass<std::vector<MPIModuleInfo>>();

  if (!cls) {
    throw cms::Exception("MPIConsistencyChecker") << "Failed to get TClass for std::vector<MPIModuleInfo>";
  }

  cls->Streamer(const_cast<std::vector<MPIModuleInfo>*>(&modules), buf);

  buffer.resize(buf.Length());
  std::memcpy(buffer.data(), buf.Buffer(), buf.Length());
}

void MPIConsistencyChecker::deserializeMPIModuleInfo(std::vector<char> const& buffer,
                                                     std::vector<MPIModuleInfo>& info) {
  TBufferFile buf(TBuffer::kRead, buffer.size(), const_cast<char*>(buffer.data()), false);
  TClass* cls = TClass::GetClass<std::vector<MPIModuleInfo>>();

  if (!cls) {
    throw cms::Exception("MPIConsistencyChecker") << "Failed to get TClass for std::vector<MPIModuleInfo>";
  }

  cls->Streamer(&info, buf);
}

void MPIConsistencyChecker::registerMPIPathOrigin(std::string const& origin_name) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  mpi_paths_mappings_.try_emplace(origin_name);
}

void MPIConsistencyChecker::reconstructMPIPaths() {
  std::call_once(paths_reconstructed_flag_, [&]() {
    std::lock_guard<std::mutex> lock(modules_info_mutex_);
    std::vector<bool> module_assigned(modules_info_.size(), false);
    for (auto& origin : mpi_paths_mappings_) {
      for (size_t i = 0; i < modules_info_.size(); ++i) {
        if (module_upstream_labels_[i] == origin.first) {
          origin.second.push_back(modules_info_[i]);
          module_assigned[i] = true;
        }
      }
    }
    bool new_module_assigned = true;
    while (new_module_assigned) {
      new_module_assigned = false;
      for (size_t i = 0; i < modules_info_.size(); ++i) {
        if (!module_assigned[i]) {
          for (auto& path : mpi_paths_mappings_) {
            if (std::find_if(path.second.begin(), path.second.end(), [&](auto const& module) {
                  return module.module_label == module_upstream_labels_[i];
                }) != path.second.end()) {
              path.second.push_back(modules_info_[i]);
              module_assigned[i] = true;
              new_module_assigned = true;
              break;
            }
          }
        }
      }
    }
    for (size_t i = 0; i < modules_info_.size(); ++i) {
      if (!module_assigned[i]) {
        throw cms::Exception("MPIConsistencyChecker")
            << "Could not assign MPI module " << modules_info_[i].module_label << " to any MPI path";
      }
    }
  });
}

void MPIConsistencyChecker::compareMPIModules(std::vector<MPIModuleInfo> const& other,
                                              std::string const& origin_name,
                                              std::string const& other_process_name,
                                              std::string const& this_process_name) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  auto const& local_modules = mpi_paths_mappings_[origin_name];

  // print the local and remote modules info for debugging
  LogDebug("MPIConsistencyChecker") << "Local MPI modules info: ";
  for (auto const& module : local_modules) {
    LogDebug("MPIConsistencyChecker") << "  is_sender: " << module.is_sender << ", instance: " << module.instance
                                      << ", product_types: ";
    for (auto const& type : module.product_types) {
      LogDebug("MPIConsistencyChecker") << "    " << type;
    }
  }
  LogDebug("MPIConsistencyChecker") << "Remote MPI modules info: ";
  for (auto const& module : other) {
    LogDebug("MPIConsistencyChecker") << "  is_sender: " << module.is_sender << ", instance: " << module.instance
                                      << ", product_types: ";
    for (auto const& type : module.product_types) {
      LogDebug("MPIConsistencyChecker") << "    " << type;
    }
  }

  if (local_modules.size() != other.size()) {
    throw cms::Exception("MPIConsistencyChecker")
        << "Mismatch in number of MPI modules between process " << this_process_name << " and process "
        << other_process_name << ": " << local_modules.size() << " vs " << other.size();
  }
  for (auto const& local_module : local_modules) {
    auto it = std::find_if(other.begin(), other.end(), [&](MPIModuleInfo const& remote_module) {
      return remote_module.is_sender != local_module.is_sender && remote_module.instance == local_module.instance;
    });
    if (it == other.end()) {
      throw cms::Exception("MPIConsistencyChecker")
          << "No matching sender/receiver found in process " << other_process_name << " for MPI module instance "
          << local_module.instance << " in process " << this_process_name;
    }
    if (local_module.product_types.size() != it->product_types.size()) {
      throw cms::Exception("MPIConsistencyChecker")
          << "Mismatch in number of product types between sender module " << local_module.instance << " in process "
          << this_process_name << " and receiver module " << it->instance << " in process " << other_process_name
          << ": " << local_module.product_types.size() << " vs " << it->product_types.size();
    }
    for (size_t i = 0; i < local_module.product_types.size(); ++i) {
      if (local_module.product_types[i] != it->product_types[i]) {
        throw cms::Exception("MPIConsistencyChecker")
            << "Mismatch in product type at index " << i << " between sender module " << local_module.instance
            << " in process " << this_process_name << " (" << local_module.product_types[i] << ") and receiver module "
            << it->instance << " in process " << other_process_name << " (" << it->product_types[i] << ")";
      }
    }
  }
}
