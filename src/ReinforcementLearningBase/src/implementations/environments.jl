export WrappedEnv

using MacroTools: @forward
using Random

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
