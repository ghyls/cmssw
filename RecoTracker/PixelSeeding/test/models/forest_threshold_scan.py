# In-situ threshold scan of the three high-purity selectors, inside one validation job.
#
# The threshold a trainer prints is an offline starting point. The deployed value is the one that
# holds up in the full reconstruction, so it is chosen here: for every scan point a clone of the
# selector runs at a scaled threshold, is converted to reco tracks and validated as its own
# collection. The prompt and displaced points also get their own merger, merged selector and
# converter, so the effect of that iteration's threshold on the MERGED collection is measured too.
# The nominal chain is untouched -- it is the 1.0 point of every curve. The factors multiply the
# thresholds the configuration carries.
#
# Use it from any validation configuration that runs the whole pixel chain and the track
# validator:
#
#   from forest_threshold_scan import add_scan
#   add_scan(process)                                  # the default factors
#   add_scan(process, factors=(0.8, 1.0, 1.25))        # a finer scan around the nominal point
#
# The directory of this file must be on PYTHONPATH (it is when the configuration lives next to
# it; otherwise add it, for example with PYTHONPATH=$CMSSW_BASE/src/RecoTracker/PixelSeeding/
# test/models). README.md, "Working points", explains how to read the curves out of the MTV
# output.
import FWCore.ParameterSet.Config as cms

FACTORS = (0.5, 0.7, 1.4, 2.0)


def tag(f):
    """The factor as it appears in a collection name: three digits, the factor times 100."""
    return "%03d" % int(round(f * 100))


def add_scan(process, factors=FACTORS):
    """Add one validated collection per selector and per factor. Returns the collection names."""
    selP = process.hltPhase2PixelTrackTorchHighPuritySelector
    selD = process.hltPhase2PixelTrackHighPuritySelectorDisplaced
    selM = process.hltPhase2PixelTrackHighPuritySelectorMerged
    merger = process.hltPhase2PixelTracksSoAMerger
    convP = process.hltPhase2PixelTracksPromptOnly
    convD = process.hltPhase2PixelTracksDisplacedOnly
    convM = process.hltPhase2PixelTracks
    P, D = selP.scoreThreshold.value(), selD.scoreThreshold.value()
    Mhi, Mlo = selM.scoreThreshold.value(), selM.scoreThresholdLowDxy.value()
    print("[scan] nominal thresholds: prompt %g displaced %g merged %g / lowDxy %g; factors %s"
          % (P, D, Mhi, Mlo, factors))
    task = cms.Task()
    labels = []

    def add(name, mod):
        setattr(process, name, mod)
        task.add(getattr(process, name))
        return name

    for f in factors:
        t = tag(f)
        # prompt selector point, alone and through the merger
        s = add("scanPromptSel" + t, selP.clone(scoreThreshold=P * f))
        labels.append(add("scanPromptOnly" + t, convP.clone(trackSrc=s)))
        m = add("scanMergerP" + t,
                merger.clone(inputTkSoAs=[s, "hltPhase2PixelTrackHighPuritySelectorDisplaced"]))
        sm = add("scanMergedSelP" + t, selM.clone(pixelTrackSrc=m))
        labels.append(add("scanMergedP" + t, convM.clone(trackSrc=sm)))
        # displaced selector point, alone and through the merger
        s = add("scanDisplacedSel" + t, selD.clone(scoreThreshold=D * f))
        labels.append(add("scanDisplacedOnly" + t, convD.clone(trackSrc=s)))
        m = add("scanMergerD" + t,
                merger.clone(inputTkSoAs=["hltPhase2PixelTrackTorchHighPuritySelector", s]))
        sm = add("scanMergedSelD" + t, selM.clone(pixelTrackSrc=m))
        labels.append(add("scanMergedD" + t, convM.clone(trackSrc=sm)))
        # merged selector point (both ramp values scaled together)
        sm = add("scanMergedSelM" + t, selM.clone(scoreThreshold=Mhi * f, scoreThresholdLowDxy=Mlo * f))
        labels.append(add("scanMergedM" + t, convM.clone(trackSrc=sm)))

    process.scanTask = task
    process.prevalidation_step.associate(process.scanTask)
    v = process.hltTrackValidator
    v.label = cms.VInputTag(list(v.label) + [cms.InputTag(l) for l in labels])
    print("[scan] %d scan collections added; validator labels = %d" % (len(labels), len(v.label)))
    return labels
