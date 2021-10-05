export RewardTransformedEnv

"""
    RewardTransformedEnv(env, f)

Apply `f` on `reward(env)`.
"""
struct RewardTransformedEnv{F,E<:AbstractEnv} <: AbstractEnvWrapper
    env::E
    reward_mapping::F
end

RewardTransformedEnv(env; reward_mapping=identity) = 
    RewardTransformedEnv(env, reward_mapping)

RLBase.reward(env::RewardTransformedEnv, args...; kwargs...) =
    env.reward_mapping(reward(env.env, args...; kwargs...))
