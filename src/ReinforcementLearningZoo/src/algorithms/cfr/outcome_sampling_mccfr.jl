export OutcomeSamplingMCCFRPolicy

"""
    OutcomeSamplingMCCFRPolicy

This implementation uses stochasticaly-weighted averaging.

Ref:

- [MONTE CARLO SAMPLING AND REGRET MINIMIZATION FOR EQUILIBRIUM COMPUTATION AND DECISION-MAKING IN LARGE EXTENSIVE FORM GAMES](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf)
- [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf)
"""
struct OutcomeSamplingMCCFRPolicy{S,B,R<:AbstractRNG} <: AbstractCFRPolicy
    nodes::Dict{S,InfoStateNode}
    behavior_policy::B
    ϵ::Float64
    rng::R
end

(p::OutcomeSamplingMCCFRPolicy)(env::AbstractEnv) = p.behavior_policy(env)

RLBase.prob(p::OutcomeSamplingMCCFRPolicy, env::AbstractEnv) =
    prob(p.behavior_policy, env)

function OutcomeSamplingMCCFRPolicy(; state_type = String, rng = Random.GLOBAL_RNG, ϵ = 0.6)
    OutcomeSamplingMCCFRPolicy(
        Dict{state_type,InfoStateNode}(),
        TabularRandomPolicy(;
            rng = rng,
            table = Dict{state_type,Vector{Float64}}(),
            is_normalized = true,
        ),
        ϵ,
        rng,
    )
end

"Run one interation"
function RLBase.update!(p::OutcomeSamplingMCCFRPolicy, env::AbstractEnv)
    for x in get_players(env)
        if x != chance_player(env)
            outcome_sampling(copy(env), x, p.nodes, p.ϵ, 1.0, 1.0, 1.0, p.rng)
        end
    end
end

function RLBase.update!(p::OutcomeSamplingMCCFRPolicy)
    for (k, v) in p.nodes
        s = sum(v.cumulative_strategy)
        if s != 0
            update!(p.behavior_policy, k => v.cumulative_strategy ./ s)
        else
            # The TabularLearner will return uniform distribution by default. 
            # So we do nothing here.
        end
    end
end

function outcome_sampling(env, i, nodes, ϵ, πᵢ, π₋ᵢ, s, rng)
    current_player = current_player(env)

    if is_terminated(env)
        reward(env, i) / s, 1.0
    elseif current_player == chance_player(env)
        env(rand(rng, action_space(env)))
        outcome_sampling(env, i, nodes, ϵ, πᵢ, π₋ᵢ, s, rng)
    else
        I = state(env)
        legal_actions = legal_action_space(env)
        n = length(legal_actions)
        node = get!(nodes, I, InfoStateNode(n))
        regret_matching!(node; is_reset_neg_regrets = false)
        σ, rI, sI = node.strategy, node.cumulative_regret, node.cumulative_strategy

        if i == current_player
            aᵢ = rand(rng) >= ϵ ? sample(rng, Weights(σ, 1.0)) : rand(rng, 1:n)
            pᵢ = σ[aᵢ] * (1 - ϵ) + ϵ / n
            πᵢ′, π₋ᵢ′, s′ = πᵢ * pᵢ, π₋ᵢ, s * pᵢ
        else
            aᵢ = sample(rng, Weights(σ, 1.0))
            pᵢ = σ[aᵢ]
            πᵢ′, π₋ᵢ′, s′ = πᵢ, π₋ᵢ * pᵢ, s * pᵢ
        end

        env(legal_action_space(env)[aᵢ])
        u, πₜₐᵢₗ = outcome_sampling(env, i, nodes, ϵ, πᵢ′, π₋ᵢ′, s′, rng)

        if i == current_player
            w = u * π₋ᵢ
            rI .+= w * πₜₐᵢₗ .* ((1:n .== aᵢ) .- σ[aᵢ])
        else
            sI .+= π₋ᵢ / s .* σ
        end

        u, πₜₐᵢₗ * σ[aᵢ]
    end
end
