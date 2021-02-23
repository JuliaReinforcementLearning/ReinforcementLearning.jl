export policy_evaluation!, policy_improvement!, policy_iteration!

using Distributions: probs

"""
    policy_evaluation!(;V, π, model, γ, θ)

# Keyword arguments

- `V`, an [`AbstractApproximator`](@ref).
- `π`, an `AbstractPolicy`.
- `model`, a distribution based environment model(given a state and action
  pair, return all possible reward, next state, termination info and
  corresponding probability).
- `γ::Float64`, discount rate.
- `θ::Float64`, threshold to stop evaluation.
"""
function policy_evaluation!(;
    V::AbstractApproximator,
    π::AbstractPolicy,
    model::AbstractEnvironmentModel,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
)
    states, actions = state_space(model), action_space(model)
    while true
        Δ = 0.0
        for s in states
            v = sum(
                prob(π, s, a) *
                sum(p * (r + (1 - t) * γ * V(s′)) for ((r, t, s′), p) in model(s, a))
                for a in actions
            )
            δ = V(s) - v
            update!(V, s => δ)
            Δ = max(Δ, abs(δ))
        end
        Δ < θ && break
    end
    V
end

function policy_improvement!(;
    V::AbstractApproximator,
    π::AbstractPolicy,
    model::AbstractEnvironmentModel,
    γ::Float64 = 0.9,
)
    states, actions = state_space(model), action_space(model)
    is_policy_stable = true
    for s in states
        old_a = π(s)
        best_action_inds = find_all_max([
            sum(p * (r + (1 - t) * γ * V(s′)) for ((r, t, s′), p) in model(s, a)) for
            a in actions
        ])[2]
        new_a = rand(best_action_inds)  # break tie
        if new_a != old_a
            update!(π, s => new_a)
            is_policy_stable = false
        end
    end
    is_policy_stable
end

function policy_iteration!(;
    V::AbstractApproximator,
    π::AbstractPolicy,
    model::AbstractEnvironmentModel,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
    max_iter = typemax(Int),
)
    for i in 1:max_iter
        policy_evaluation!(; V = V, π = π, model = model, γ = γ, θ = θ)
        policy_improvement!(; V = V, π = π, model = model, γ = γ) && return i
    end
    return max_iter
end

function value_iteration!(;
    V::AbstractApproximator,
    model::AbstractEnvironmentModel,
    γ::Float64 = 0.9,
    θ::Float64 = 1e-4,
    max_iter = typemax(Int),
)
    states, actions = state_space(model), action_space(model)
    for i in 1:max_iter
        Δ = 0.0
        for s in states
            v = maximum(
                sum(p * (r + (1 - t) * γ * V(s′)) for ((r, t, s′), p) in model(s, a))
                for a in actions
            )
            δ = V(s) - v
            update!(V, s => δ)
            Δ = max(Δ, abs(δ))
        end
        Δ < θ && break
    end
end
