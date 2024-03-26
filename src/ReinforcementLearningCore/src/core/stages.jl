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
Base.push!(p::AbstractPolicy, ::PostActStage, ::AbstractEnv, action) = nothing
Base.push!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv, ::Player) = nothing
Base.push!(p::AbstractPolicy, ::PostActStage, ::AbstractEnv, action, ::Player) = nothing

RLBase.optimise!(policy::P, ::S) where {P<:AbstractPolicy,S<:AbstractStage} = nothing

RLBase.optimise!(policy::P, ::S, batch) where {P<:AbstractPolicy, S<:AbstractStage} = nothing
