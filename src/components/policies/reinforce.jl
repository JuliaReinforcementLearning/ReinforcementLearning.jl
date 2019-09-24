export ReinforcePolicy

using Flux:softmax
using LinearAlgebra:dot
using StatsBase

Base.@kwdef struct ReinforcePolicy{A<:AbstractQApproximator} <: AbstractPolicy
    approximator::A
    α::Float64
    γ::Float64
end

(π::ReinforcePolicy)(obs::Observation) = obs |> get_state |> π.approximator |> softmax |> x -> Weights(x, 1.0) |> sample

get_prob(π::ReinforcePolicy, s) = s |> π.approximator |> softmax

function update!(
    π::ReinforcePolicy{<:LinearQApproximator},
    buffer::EpisodeTurnBuffer,
)
    if isfull(buffer)
        states, actions, rewards = state(buffer)[1:end-1], action(buffer)[1:end-1], reward(buffer)[2:end]
        Q, α, γ = π.approximator, π.α, π.γ
        gains = discount_rewards(rewards, γ)
        γₜ = 1.0

        for (i, (s, a, g)) in enumerate(zip(states, actions, gains))
            update!(Q, s => α * γₜ * g * (Q(s, a) - dot(get_prob(π, s), π.approximator(s))))
            γₜ *= γ
        end
    end
end