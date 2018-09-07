module ReinforcementLearningBase

include("spaces/space.jl")
include("abstractenv.jl")
include("tests.jl")

export interact!, reset!, getstate, plotenv, actionspace, sample,
test_envinterface
end # module
