export AbstractStage,
    PreExperimentStage,
    PostExperimentStage,
    PreEpisodeStage,
    PostEpisodeStage,
    PreActStage,
    PostActStage

import Base.push!
abstract type AbstractStage end

"Stage that is executed before the `Experiment` starts."
struct PreExperimentStage <: AbstractStage end

"Stage that is executed after the `Experiment` is over."
struct PostExperimentStage <: AbstractStage end

"Stage that is executed before the `Episode` starts."
struct PreEpisodeStage <: AbstractStage end

"Stage that is executed after the `Episode` is over."
struct PostEpisodeStage <: AbstractStage end

"Stage that is executed before the `Agent` acts."
struct PreActStage <: AbstractStage end

"Stage that is executed after the `Agent` acts."
struct PostActStage <: AbstractStage end

Base.push!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv) = nothing
Base.push!(p::AbstractPolicy, ::PostActStage, ::AbstractEnv, action) = nothing
Base.push!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv, ::Player) = nothing
Base.push!(p::AbstractPolicy, ::PostActStage, ::AbstractEnv, action, ::Player) = nothing

RLBase.optimise!(policy::P, ::S) where {P<:AbstractPolicy,S<:AbstractStage} = nothing

RLBase.optimise!(policy::P, ::S, batch) where {P<:AbstractPolicy, S<:AbstractStage} = nothing
