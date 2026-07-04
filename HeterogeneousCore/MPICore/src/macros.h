#ifndef HeterogeneousCore_MPICore_interface_macros_h
#define HeterogeneousCore_MPICore_interface_macros_h

// MPI headers
#include <mpi.h>

// clang-format off

// Build a named MPI_Datatype (TYPE) for the given C++ struct (STRUCT), listing its fields (for
// documentation only) as the remaining, unused arguments.
//
// This intentionally does *not* describe the struct to MPI field-by-field via
// MPI_Type_create_struct(), even though the fields are listed here for readability. On at least
// one tested setup (OpenMPI 5.0.10rc2 over the "cm" PML / libfabric "ofi" MTL on a RoCE fabric),
// a multi-field MPI_Type_create_struct()-based datatype silently fails to transfer any data at
// all -- the receiving MPI_Mrecv() call reports success but leaves the destination buffer
// completely untouched -- whenever another thread concurrently calls MPI_Comm_dup() on a related
// communicator while a matched probe/receive (MPI_Mprobe()/MPI_Mrecv()) using such a datatype is
// in flight. This is exactly the pattern used elsewhere in this package (MPIController::produce()
// duplicates a fresh per-slot communicator for every new in-flight event, concurrently with
// MPISource's matched-probe receive loop for Run/LuminosityBlock/Event transitions), so it is not
// an exotic corner case. A single-field struct (e.g. EDM_MPI_Empty) or a type built with
// MPI_Type_contiguous() (as below), describing the exact same bytes as one opaque block instead of
// several individually-typed ones, are both unaffected by this issue in the same stress test.
//
// Since all of these types are only ever used to exchange values between two instances of the
// very same CMSSW binary (never between different architectures, compilers, or bit-widths), there
// is no practical benefit to describing the struct's internal layout to MPI in the first place:
// both ends already agree on it by construction. Treating it as sizeof(STRUCT) opaque bytes is
// therefore both safe and simpler, and sidesteps whatever is going wrong in the multi-field
// pack/unpack path on the affected setup.
#define DECLARE_MPI_TYPE(TYPE, STRUCT, ...)                                                         \
  {                                                                                                 \
    MPI_Type_contiguous(sizeof(STRUCT), MPI_BYTE, &TYPE);                                            \
    MPI_Type_commit(&TYPE);                                                                         \
  }

// clang-format on

#endif  // HeterogeneousCore_MPICore_interface_macros_h
