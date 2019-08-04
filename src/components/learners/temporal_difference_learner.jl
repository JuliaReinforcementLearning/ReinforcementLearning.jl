export TDLearner, update!

using .Utils: discount_reward

"""
    TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0) where {Tapp<:VApproximator}
    TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0, method::Symbol=:SARSA) where {Tapp<:QApproximator} 

The `TDLearner`(Temporal Difference Learner) use the latest `n` step experiences
to update the `approximator`. `γ` is the discount rate of experience.
`α` is the learning rate.

For [`VApproximator`](@ref), the only supported update method is `:SRS`, which means
only **S**tates, **R**ewards and next_**S**ates are used to update the `approximator`.

For [`QApproximator`](@ref), the following methods are supported:

- `:SARSA`
- `:ExpectedSARSA`
"""
struct TDLearner{Tapp <: AbstractApproximator, method}
    approximator::Tapp
    γ::Float64
    α::Float64
    n::Int

    TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0) where {Tapp<:VApproximator} = new{Tapp, :SRS}(approximator, π, γ, α, n)

    function TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0, method::Symbol=:SARSA) where {Tapp<:QApproximator} 
        supported_methods = [:SARSA, :ExpectedSARSA]
        !in(method, supported_methods) && throw(ArgumentError("Supported methods are $supported_methods , your input is $method"))
        new{Tapp, method}(approximator, γ, α, n)
    end
end

(learner::TDLearner)(s) = learner.approximator(s)
(learner::TDLearner{<:QApproximator})(s, a) = learner.approximator(s, a)

function update!(learner::TDLearner{<:VApproximator, :SRS}, states, rewards, terminals, next_states)
    n, γ, V, α, N = learner.n, learner.γ, learner.approximator, learner.α, length(states)
    discounted_rewards = discount_reward(rewards, γ)
    # only use the latest `n` steps to update `learner`
    # note that `n` starts with `0`, which means looking forward for the next `n` steps
    @views for i in max(1, N - n):N
        G = discounted_rewards[i] + terminals[i] * γ^n * V(next_states[end])
        s = states[i]
        update!(V, s => α * (G - V(s)))
    end
end

function update!(learner::TDLearner{<:QApproximator, :SARSA}, states, actions, rewards, terminals, next_states, next_actions)
    n, γ, Q, α = learner.n, learner.γ, learner.approximator, learner.α
end