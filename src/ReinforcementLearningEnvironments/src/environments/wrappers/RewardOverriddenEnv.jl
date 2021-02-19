export RewardOverriddenEnv

struct RewardOverriddenEnv{F,E <: AbstractEnv} <: AbstractEnvWrapper
    env::E
    f::F
end

RewardOverriddenEnv(f) = env -> RewardOverriddenEnv(f, env)

RLBase.reward(env::RewardOverriddenEnv, args...; kwargs...) =
    env.f(reward(env.env, args...; kwargs...))
