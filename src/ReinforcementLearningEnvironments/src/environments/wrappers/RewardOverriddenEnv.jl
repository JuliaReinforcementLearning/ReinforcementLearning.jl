export RewardOverriddenEnv

"""
    RewardOverriddenEnv(env, f)

Apply `f` on `reward(env)`.
"""
struct RewardOverriddenEnv{F,E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    f::F
end

RLBase.reward(env::RewardOverriddenEnv, args...; kwargs...) =
    env.f(reward(env.env, args...; kwargs...))
