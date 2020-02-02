export WrappedEnv

using MacroTools: @forward
using Random

Base.@kwdef struct WrappedEnv{P<:AbstractPreprocessor,E<:AbstractEnv} <: AbstractEnv
    preprocessor::P
    env::E
end

(env::WrappedEnv)(args...; kwargs...) = env.env(args..., kwargs...)

@forward WrappedEnv.env RLBase.DynamicStyle,
RLBase.get_current_player,
RLBase.get_action_space,
RLBase.get_observation_space,
RLBase.render,
RLBase.reset!,
Random.seed!

RLBase.observe(env::WrappedEnv, player) = env.preprocessor(observe(env.env, player))
RLBase.observe(env::WrappedEnv) = env.preprocessor(observe(env.env))
