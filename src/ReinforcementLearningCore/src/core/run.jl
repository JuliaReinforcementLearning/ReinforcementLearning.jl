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

import Base: run

function run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)
    check(policy, env)
    _run(policy, env, stop_condition, hook)
end

"Inject some customized checkings here by overwriting this function"
function check(policy, env) end

function _run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook)

    hook(PRE_EXPERIMENT_STAGE, policy, env)
    policy(PRE_EXPERIMENT_STAGE, env)
    is_stop = false
    while !is_stop
        reset!(env)
        policy(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, policy, env)

        while !is_terminated(env) # one episode
            action = policy(env)

            policy(PRE_ACT_STAGE, env, action)
            hook(PRE_ACT_STAGE, policy, env, action)

            optimise!(policy)
            env(action)

            policy(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, policy, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)  # let the policy see the last observation
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
    hook
end

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