export RewardNormalizer
"""
    RewardNormalizer()

Use as the `f` reward mapping function in a `RewardTransformedEnv` to create an environment that returns rewards with mean 0 and variance 1.
A running estimate of the mean and standard deviation is kept in memory.
#example
```
normalized_env = RewardTransformedEnv(env, RewardNormalizer())   
```
"""
mutable struct RewardNormalizer{T}
    mean::T
    moment2::T
    std::T
    step_count::Int
end

RewardNormalizer() = RewardNormalizer(0f0, 1f0, 1f0, 0)

function (rn::RewardNormalizer)(rewards)
    N = length(rewards)
    rn.step_count += N
    tmp_mean = rn.mean
    rn.mean = (rn.step_count-N)/rn.step_count * rn.mean + sum(rewards)/rn.step_count
    rn.moment2 += sum((rewards .- tmp_mean) .* (rewards .- rn.mean))
    rn.std = max(sqrt(rn.moment2/(max(1,rn.step_count-1))), eps(rn.std))
    return (rewards .- rn.std)./rn.std
end

#= it is pretty useless to track the evolution of a reward that is normalized to be 0. So we track the actual rewards.
function (hook::RewardsPerEpisode)(::PostActStage, agent, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    push!(hook.rewards[end], reward(env.env))
end

function (hook::RewardsPerEpisode)(::PostActStage, agent::NamedPolicy, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    push!(hook.rewards[end], reward(env.env, nameof(agent)))
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    hook.reward += reward(env.env)
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent::NamedPolicy, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    hook.reward += reward(env.env, nameof(agent))
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    hook.reward += reward(env.env)
end

function (hook::TotalRewardPerEpisode)(::PostActStage, agent::NamedPolicy, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    hook.reward += reward(env.env, nameof(agent))
end

function (hook::TotalBatchRewardPerEpisode)(::PostActStage, agent, env::RewardTransformedEnv{<:AbstractEnv, <:RewardNormalizer})
    R = agent isa NamedPolicy ? reward(env.env, nameof(agent)) : reward(env.env)
    for (i, (t, r)) in enumerate(zip(is_terminated(env), R))
        hook.reward[i] += r
        if t
            push!(hook.rewards[i], hook.reward[i])
            hook.reward[i] = 0.0
        end
    end
end=#