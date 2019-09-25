export ReinforcePolicy

using Flux: softmax
using LinearAlgebra: dot
using StatsBase

Base.@kwdef struct ReinforcePolicy{A<:AbstractQApproximator} <: AbstractPolicy
    approximator::A
    α::Float64
    γ::Float64
end

(π::ReinforcePolicy)(obs::Observation) =
    obs |> get_state |> π.approximator |> softmax |> x -> Weights(x, 1.0) |> sample

get_prob(π::ReinforcePolicy, s) = s |> π.approximator |> softmax
get_prob(π::ReinforcePolicy, s, a) = get_prob(π, s)[a]

# TODO: handle neural network q approximator
function update!(π::ReinforcePolicy{<:LinearQApproximator}, buffer::EpisodeTurnBuffer)
    if isfull(buffer)
        states, actions, rewards = state(buffer)[1:end-1],
            action(buffer)[1:end-1],
            reward(buffer)[2:end]
        Q, α, γ = π.approximator, π.α, π.γ
        gains = discount_rewards(rewards, γ)
        γₜ = 1.0

        for (i, (s, a, g)) in enumerate(zip(states, actions, gains))
            # !!! we will multiply `Q.feature_func(s, a)` in the `LinearQApproximator` again!
            update!(
                Q,
                (s, a) => α * γₜ * g *
                          (Q.feature_func(s, a) .-
                           sum(x -> get_prob(π, s, x) .* Q.feature_func(s, x), Q.actions)),
            )
            γₜ *= γ
        end
    end
end