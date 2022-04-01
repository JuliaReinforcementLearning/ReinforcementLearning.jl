export RewardNormalizer, ExpRewardNormalizer, AbstractRewardNormalizer
abstract type AbstractRewardNormalizer end

"""
    RewardNormalizer()

Use as the `f` reward mapping function in a `AbstractEnv` to create an environment that returns rewards with mean 0 and variance 1.
It can also directly be used in an algorithm to normalize sampled rewards without affecting the stored rewards. To do so, use
`rewardnormalizer(rewards)` to update the estimates and obtain normalized inputs, normalize without update with `rewardnormalizer(rewards, update = false)`.
A running estimate of the mean and standard deviation is kept in memory. `RewardNormalizer` gives the same weight to each sample to compute its estimate.
For a version that uses an exponential estimate, a more suited approach for non-stationary estimates, see `ExpRewardNormalizer`.
#example
```
normalized_env = AbstractEnv(env, RewardNormalizer())   
```
"""
mutable struct RewardNormalizer{T} <: AbstractRewardNormalizer
    mean::T
    moment2::T
    std::T
    step_count::Int
end

RewardNormalizer() = RewardNormalizer(0f0, 1f0, 1f0, 0)

function (rn::RewardNormalizer)(rewards; update = true)
    if update
        N = length(rewards)
        rn.step_count += N
        tmp_mean = rn.mean
        rn.mean = (rn.step_count-N)/rn.step_count * rn.mean + sum(rewards)/rn.step_count
        rn.moment2 += sum((rewards .- tmp_mean) .* (rewards .- rn.mean))
        rn.std = max(sqrt(rn.moment2/(max(1,rn.step_count-1))), eps(rn.std))
    end
    return (rewards .- rn.std) ./ rn.std
end

"""
    ExpRewardNormalizer()

Use as the `f` reward mapping function in a `AbstractEnv` to create an environment that returns rewards with mean 0 and variance 1.
It can also directly be used in an algorithm to normalize sampled rewards without affecting the stored rewards. To do so, use
`exprewardnormalizer(rewards)` to update the estimates and obtain normalized inputs, normalize without update with `rewardnormalizer(rewards, update = false)`.
A running estimate of the mean and standard deviation is kept in memory. `RewardNormalizer` gives the same weight to each sample to compute its estimate.
For a version that does not use an exponential estimate, see `RewardNormalizer`.
#example
```
normalized_env = AbstractEnv(env, RewardNormalizer())   
```
"""
mutable struct ExpRewardNormalizer{T} <: AbstractRewardNormalizer
    mean::T
    var::T
    std::T
    factor::T
    first::Bool
end

ExpRewardNormalizer(factor = 0.2f0) = ExpRewardNormalizer(0f0, 0f0, 0f0, factor, true)

function (rn::ExpRewardNormalizer)(rewards; update = true)
    if update
        if rn.first
            rn.first = false
            rn.mean = mean(rewards)
        else
            rn.mean = (1 - rn.factor) * rn.mean + rn.factor * sum(rewards)
        end
        rn.var = (1 - rn.factor) * (rn.var + rn.factor * sum(rewards .^ 2))
        rn.std = sqrt(rn.var)
    end
    return (rewards .- rn.std)./rn.std
end