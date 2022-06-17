export RewardOverriddenEnv

"""
    RewardOverriddenEnv(env, f)

Apply `f` on `env` to generate a custom reward.
"""
struct RewardOverriddenEnv{F,E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    f::F
end

RLBase.reward(env::RewardOverriddenEnv, args...; kwargs...) =
    env.f(env.env, args...; kwargs...)
