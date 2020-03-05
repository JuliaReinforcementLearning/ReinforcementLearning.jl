export WrappedEnv, MultiThreadEnv

using MacroTools: @forward
using Random

import Base.Threads.@spawn

#####
# WrappedEnv
#####

"""
    WrappedEnv(;preprocessor, env)

Wrap the `env` with a `preprocessor`
"""
Base.@kwdef struct WrappedEnv{P<:AbstractPreprocessor,E<:AbstractEnv} <: AbstractEnv
    preprocessor::P
    env::E
end

(env::WrappedEnv)(args...; kwargs...) = env.env(args..., kwargs...)

@forward WrappedEnv.env DynamicStyle,
get_current_player,
get_action_space,
get_observation_space,
render,
reset!,
Random.seed!

observe(env::WrappedEnv, player) = env.preprocessor(observe(env.env, player))
observe(env::WrappedEnv) = env.preprocessor(observe(env.env))

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

@forward MultiThreadEnv.envs Base.getindex, Base.length, Base.setindex!

RLBase.get_action_space(env::MultiThreadEnv) = get_action_space(env.envs[1])
RLBase.get_observation_space(env::MultiThreadEnv) = get_observation_space(env.envs[1])

function (env::MultiThreadEnv)(actions)
    @sync for i in 1:length(env)
        @spawn begin
            env[i](actions[i])
            env.obs[i] = observe(env.envs[i])
        end
    end
end

RLBase.observe(env::MultiThreadEnv) = env.obs

function RLBase.reset!(env::MultiThreadEnv; is_force = false)
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
