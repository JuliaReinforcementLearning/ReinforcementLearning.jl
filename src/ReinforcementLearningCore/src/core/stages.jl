export AbstractStage,
    PreExperimentStage,
    PostExperimentStage,
    PreEpisodeStage,
    PostEpisodeStage,
    PreActStage,
    PostActStage

import Base.push!
abstract type AbstractStage end

struct PreExperimentStage <: AbstractStage end
struct PostExperimentStage <: AbstractStage end
struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end
struct PreActStage <: AbstractStage end
struct PostActStage <: AbstractStage end

Base.push!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv) = nothing
Base.push!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv, ::Symbol) = nothing

RLBase.optimise!(::AbstractPolicy) = nothing

RLBase.optimise!(policy, ::PreEpisodeStage) = nothing
RLBase.optimise!(policy::AbstractPolicy, ::PostEpisodeStage) = RLBase.optimise!(policy)
RLBase.optimise!(policy, ::PreActStage) = nothing
RLBase.optimise!(policy, ::PostActStage) = nothing
