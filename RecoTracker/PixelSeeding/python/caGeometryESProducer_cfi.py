import FWCore.ParameterSet.Config as cms

# Generic entry point for the CA geometry ESProducer. The producer is a class template on
# TrackerTraits, registered per topology (mirroring CAHitNtupletAlpaka); see
# caGeometryESProducerPhase2OTStubs_cfi / caGeometryESProducerPhase2OT_cfi. The OT-stubs chain
# (phase2CAStubs on) uses Phase2OTStubs, so the generic `caGeometryESProducer` name aliases that
# instantiation. Load a per-topology cfi directly when you need a specific topology.
#
# `geometry` (pairGraph/startingPairs/skipsLayers/fishboneCuts) MUST match the CA producer that shares
# this geometry; copy it from that producer's geometry PSet when instantiating.
from RecoTracker.PixelSeeding.caGeometryESProducerPhase2OTStubs_cfi import (
    caGeometryESProducerPhase2OTStubs as caGeometryESProducer,
)
