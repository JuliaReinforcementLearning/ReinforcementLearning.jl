export WrappedEnv,
    MultiThreadEnv, AbstractPreprocessor, CloneStatePreprocessor, ComposedPreprocessor

using MacroTools: @forward
using Random

import Base.Threads.@spawn

#####
# WrappedEnv
#####

"""
    WrappedEnv(;preprocessor=identity, env, postprocessor=identity)

The observation of the inner `env` is first transformed by the `preprocessor`.
And the action is transformed by `postprocessor` and then send to the inner `env`.
"""
Base.@kwdef struct WrappedEnv{P,E<:AbstractEnv,T} <: AbstractEnv
    preprocessor::P = identity
    env::E
    postprocessor::T = identity
end

"TODO: Deprecate"
WrappedEnv(p, env) = WrappedEnv(preprocessor = p, env = env)

(env::WrappedEnv)(args...) = env.env(env.postprocessor(args)...)

@forward WrappedEnv.env DynamicStyle,
ChanceStyle,
InformationStyle,
RewardStyle,
UtilityStyle,
ActionStyle,
get_action_space,
get_observation_space,
get_current_player,
get_player_id,
get_num_players,
get_history,
render,
reset!,
Random.seed!,
Base.copy

observe(env::WrappedEnv, player) = env.preprocessor(observe(env.env, player))
observe(env::WrappedEnv) = env.preprocessor(observe(env.env))

#####
## Preprocessors
#####

abstract type AbstractPreprocessor end

"""
    (p::AbstractPreprocessor)(obs)

By default a [`StateOverriddenObs`](@ref) is returned to avoid modifying original observation.
"""
(p::AbstractPreprocessor)(obs) = StateOverriddenObs(obs = obs, state = p(get_state(obs)))

"""
    ComposedPreprocessor(p::AbstractPreprocessor...)

Compose multiple preprocessors.
"""
struct ComposedPreprocessor{T} <: AbstractPreprocessor
    preprocessors::T
end

ComposedPreprocessor(p::AbstractPreprocessor...) = ComposedPreprocessor(p)
(p::ComposedPreprocessor)(obs) = reduce((x, f) -> f(x), p.preprocessors, init = obs)

#####
# CloneStatePreprocessor
#####

"""
    CloneStatePreprocessor()

Do `deepcopy` for the state in an observation.
"""
struct CloneStatePreprocessor <: AbstractPreprocessor end

(p::CloneStatePreprocessor)(obs) = StateOverriddenObs(obs, deepcopy(get_state(obs)))

#####
# MultiThreadEnv
#####

"""
    MultiThreadEnv(envs::Vector{<:AbstractEnv})

Wrap multiple environments in one environment.
Each environment will run in parallel by leveraging `Threads.@spawn`.
"""
struct MultiThreadEnv{O,E} <: AbstractEnv
    obs::BatchObs{O}
    envs::Vector{E}
end

MultiThreadEnv(envs) = MultiThreadEnv(BatchObs([observe(env) for env in envs]), envs)

get_action_space(env::MultiThreadEnv) = get_action_space(env.envs[1])
get_observation_space(env::MultiThreadEnv) = get_observation_space(env.envs[1])

function (env::MultiThreadEnv)(actions)
    @sync for i in 1:length(env)
        @spawn begin
            env[i](actions[i])
            env.obs[i] = observe(env.envs[i])
        end
    end
end

observe(env::MultiThreadEnv) = env.obs

function reset!(env::MultiThreadEnv; is_force = false)
    if is_force
        for i in 1:length(env)
            reset!(env.envs[i])
        end
    else
        @sync for i in 1:length(env)
            if get_terminal(env.obs[i])
                @spawn begin
                    reset!(env.envs[i])
                    env.obs[i] = observe(env.envs[i])
                end
            end
        end
    end
end

@forward MultiThreadEnv.envs Base.getindex, Base.length, Base.setindex!

# TODO general APIs for MultiThreadEnv are missing
