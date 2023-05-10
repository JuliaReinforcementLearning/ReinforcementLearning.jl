export AbstractStage,
    PreExperimentStage,
    PostExperimentStage,
    PreEpisodeStage,
    PostEpisodeStage,
    PreActStage,
    PostActStage

abstract type AbstractStage end

struct PreExperimentStage <: AbstractStage end
struct PostExperimentStage <: AbstractStage end
struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end
struct PreActStage <: AbstractStage end
struct PostActStage <: AbstractStage end

update!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv) = nothing
update!(p::AbstractPolicy, ::AbstractStage, ::AbstractEnv, ::Symbol) = nothing

RLBase.optimise!(::AbstractPolicy) = nothing
