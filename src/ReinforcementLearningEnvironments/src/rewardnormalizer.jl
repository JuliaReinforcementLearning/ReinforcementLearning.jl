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
    rn.std = max(sqrt(rn.moment2/(rn.step_count-1)), eps(rn.std))
    return (rewards .- rn.std)./rn.std
end