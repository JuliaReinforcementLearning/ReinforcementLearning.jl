export AbstractStage,
    PreExperimentStage,
    PostExperimentStage,
    PreEpisodeStage,
    PostEpisodeStage,
    PreActStage,
    PostActStage,
    PRE_EXPERIMENT_STAGE,
    POST_EXPERIMENT_STAGE,
    PRE_EPISODE_STAGE,
    POST_EPISODE_STAGE,
    PRE_ACT_STAGE,
    POST_ACT_STAGE

#####
# Stage
#####

abstract type AbstractStage end

struct PreExperimentStage <: AbstractStage end
const PRE_EXPERIMENT_STAGE = PreExperimentStage()

struct PostExperimentStage <: AbstractStage end
const POST_EXPERIMENT_STAGE = PostExperimentStage()

struct PreEpisodeStage <: AbstractStage end
const PRE_EPISODE_STAGE = PreEpisodeStage()

struct PostEpisodeStage <: AbstractStage end
const POST_EPISODE_STAGE = PostEpisodeStage()

struct PreActStage <: AbstractStage end
const PRE_ACT_STAGE = PreActStage()

struct PostActStage <: AbstractStage end
const POST_ACT_STAGE = PostActStage()

(p::AbstractPolicy)(::AbstractStage, ::AbstractEnv) = nothing
(p::AbstractPolicy)(::AbstractStage, ::AbstractEnv, action) = nothing
