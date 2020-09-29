export ExternalSamplingMCCFRPolicy

using Random
using StatsBase: sample, Weights

"""
    ExternalSamplingMCCFRPolicy

This implementation uses stochasticaly-weighted averaging.

Ref:

- [MONTE CARLO SAMPLING AND REGRET MINIMIZATION FOR EQUILIBRIUM COMPUTATION AND DECISION-MAKING IN LARGE EXTENSIVE FORM GAMES](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf)
- [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf)
"""
struct ExternalSamplingMCCFRPolicy{S,T,R<:AbstractRNG} <: AbstractPolicy
    nodes::Dict{S,InfoStateNode}
    behavior_policy::QBasedPolicy{TabularLearner{S,T},WeightedExplorer{true,R}}
end

(p::ExternalSamplingMCCFRPolicy)(env::AbstractEnv) = p.behavior_policy(env)

RLBase.get_prob(p::ExternalSamplingMCCFRPolicy, env::AbstractEnv) =
    get_prob(p.behavior_policy, env)

function ExternalSamplingMCCFRPolicy(;
    env::AbstractEnv,
    n_iter::Int,
    rng = Random.GLOBAL_RNG,
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
                external_sampling(copy(env), p, nodes, rng)
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

    ExternalSamplingMCCFRPolicy(nodes, behavior_policy)
end

function external_sampling(env, i, nodes, rng)
    current_player = get_current_player(env)

    if get_terminal(env)
        get_reward(env, i)
    elseif current_player == get_chance_player(env)
        env(rand(rng, get_actions(env)))
        external_sampling(env, i, nodes, rng)
    else
        I = get_state(env)
        node = nodes[I]
        regret_matching!(node)
        σ, rI, sI = node.strategy, node.cumulative_regret, node.cumulative_strategy
        n = length(node.strategy)

        if i == current_player
            u = zeros(n)
            uσ = 0
            for (aᵢ, a) in enumerate(get_legal_actions(env))
                u[aᵢ] = external_sampling(child(env, a), i, nodes, rng)
                uσ += σ[aᵢ] * u[aᵢ]
            end
            rI .+= u .- uσ
            uσ
        else
            a′ = sample(rng, Weights(σ, 1.0))
            env(get_legal_actions(env)[a′])
            u = external_sampling(env, i, nodes, rng)
            sI .+= σ
            u
        end
    end
end
