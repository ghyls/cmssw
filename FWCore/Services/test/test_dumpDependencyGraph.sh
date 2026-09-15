#!/bin/bash


# Files the test generates in the working directory. They are removed once every
# check has passed, and left behind on failure so that it can be looked at.
OUTPUTS="test_dumpDependencyGraph.json test_dumpDependencyGraph.log"

# Print what failed and its status
function die { echo Failure $1: status $2 ; exit $2 ; }

# Generate the dependency graph
cmsRun ${SCRAM_TEST_PATH}/test_dumpDependencyGraph_cfg.py &> test_dumpDependencyGraph.log || die "cmsRun test_dumpDependencyGraph_cfg.py" $?

# Do various checks on the generated dependency graph
python3 ${SCRAM_TEST_PATH}/test_dumpDependencyGraph_check.py test_dumpDependencyGraph.json || die "Check of the dependency graph JSON" $?

rm -f ${OUTPUTS}

exit 0
