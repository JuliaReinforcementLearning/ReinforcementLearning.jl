module ReinforcementLearningBase

export RLBase

"""
[ReinforcementLearningBase.jl](@ref)
(**RLBase**) provides common constants, traits, abstractions and interfaces
in developing reinforcement learning algorithms in Julia. 

Foundational types and utilities for two main concepts of reinforcement learning are provided:

    - [`AbstractPolicy`](@ref)
    - [`AbstractEnv`](@ref)
"""

const RLBase = ReinforcementLearningBase

using Random
using Reexport

include("inline_export.jl")
include("interface.jl")
include("CommonRLInterface.jl")
include("base.jl")
include("space.jl")

end # module
