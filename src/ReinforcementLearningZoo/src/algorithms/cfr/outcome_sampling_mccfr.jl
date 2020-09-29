export OutcomeSamplingMCCFRPolicy

using Random
using StatsBase: sample, Weights

"""
    OutcomeSamplingMCCFRPolicy

This implementation uses stochasticaly-weighted averaging.

Ref:

- [MONTE CARLO SAMPLING AND REGRET MINIMIZATION FOR EQUILIBRIUM COMPUTATION AND DECISION-MAKING IN LARGE EXTENSIVE FORM GAMES](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf)
- [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf)
"""
struct OutcomeSamplingMCCFRPolicy{S,T,R<:AbstractRNG} <: AbstractPolicy
    nodes::Dict{S,InfoStateNode}
    behavior_policy::QBasedPolicy{TabularLearner{S,T},WeightedExplorer{true,R}}
end

(p::OutcomeSamplingMCCFRPolicy)(env::AbstractEnv) = p.behavior_policy(env)

RLBase.get_prob(p::OutcomeSamplingMCCFRPolicy, env::AbstractEnv) =
    get_prob(p.behavior_policy, env)

function OutcomeSamplingMCCFRPolicy(;
    env::AbstractEnv,
    n_iter::Int,
    rng = Random.GLOBAL_RNG,
    ϵ = 0.6,
)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DynamicStyle(env) === SEQUENTIAL
    @assert RewardStyle(env) === TERMINAL_REWARD
    @assert ChanceStyle(env) === EXPLICIT_STOCHASTIC
    @assert DefaultStateStyle(env) === Information{String}()

    nodes = init_info_state_nodes(env)

    for i in 1:n_iter
        for p in get_players(env)
            if p != get_chance_player(env)
                outcome_sampling(copy(env), p, nodes, ϵ, 1.0, 1.0, 1.0, rng)
            end
        end
    end

    behavior_policy = QBasedPolicy(;
        learner = TabularLearner{String}(),
        explorer = WeightedExplorer(; is_normalized = true, rng = rng),
    )

    for (k, v) in nodes
        s = sum(v.cumulative_strategy)
        if s != 0
            update!(behavior_policy, k => v.cumulative_strategy ./ s)
        end
    end

    OutcomeSamplingMCCFRPolicy(nodes, behavior_policy)
end

function outcome_sampling(env, i, nodes, ϵ, πᵢ, π₋ᵢ, s, rng)
    current_player = get_current_player(env)

    if get_terminal(env)
        get_reward(env, i) / s, 1.0
    elseif current_player == get_chance_player(env)
        env(rand(rng, get_actions(env)))
        outcome_sampling(env, i, nodes, ϵ, πᵢ, π₋ᵢ, s, rng)
    else
        I = get_state(env)
        node = nodes[I]
        regret_matching!(node)
        σ, rI, sI = node.strategy, node.cumulative_regret, node.cumulative_strategy
        n = length(node.strategy)

        if i == current_player
            aᵢ = rand(rng) >= ϵ ? sample(rng, Weights(σ, 1.0)) : rand(rng, 1:n)
            pᵢ = σ[aᵢ] * (1 - ϵ) + ϵ / n
            πᵢ′, π₋ᵢ′, s′ = πᵢ * pᵢ, π₋ᵢ, s * pᵢ
        else
            aᵢ = sample(rng, Weights(σ, 1.0))
            pᵢ = σ[aᵢ]
            πᵢ′, π₋ᵢ′, s′ = πᵢ, π₋ᵢ * pᵢ, s * pᵢ
        end

        env(get_legal_actions(env)[aᵢ])
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
