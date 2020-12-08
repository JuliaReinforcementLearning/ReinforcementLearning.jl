export ExternalSamplingMCCFRPolicy

"""
    ExternalSamplingMCCFRPolicy

This implementation uses stochasticaly-weighted averaging.

Ref:

- [MONTE CARLO SAMPLING AND REGRET MINIMIZATION FOR EQUILIBRIUM COMPUTATION AND DECISION-MAKING IN LARGE EXTENSIVE FORM GAMES](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf)
- [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf)
"""
struct ExternalSamplingMCCFRPolicy{S,B,R<:AbstractRNG} <: AbstractCFRPolicy
    nodes::Dict{S,InfoStateNode}
    behavior_policy::B
    rng::R
end

(p::ExternalSamplingMCCFRPolicy)(env::AbstractEnv) = p.behavior_policy(env)

RLBase.get_prob(p::ExternalSamplingMCCFRPolicy, env::AbstractEnv) =
    get_prob(p.behavior_policy, env)

function ExternalSamplingMCCFRPolicy(; state_type = String, rng = Random.GLOBAL_RNG)
    ExternalSamplingMCCFRPolicy(
        Dict{state_type,InfoStateNode}(),
        TabularRandomPolicy(;
            rng = rng,
            table = Dict{state_type,Vector{Float64}}(),
            is_normalized = true,
        ),
        rng,
    )
end

function RLBase.update!(p::ExternalSamplingMCCFRPolicy)
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

"Run one interation"
function RLBase.update!(p::ExternalSamplingMCCFRPolicy, env::AbstractEnv)
    for x in get_players(env)
        if x != get_chance_player(env)
            external_sampling(copy(env), x, p.nodes, p.rng)
        end
    end
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
        legal_actions = get_legal_actions(env)
        n = length(legal_actions)
        node = get!(nodes, I, InfoStateNode(n))
        regret_matching!(node; is_reset_neg_regrets = false)
        σ, rI, sI = node.strategy, node.cumulative_regret, node.cumulative_strategy

        if i == current_player
            u = zeros(n)
            uσ = 0
            for (aᵢ, a) in enumerate(legal_actions)
                u[aᵢ] = external_sampling(child(env, a), i, nodes, rng)
                uσ += σ[aᵢ] * u[aᵢ]
            end
            rI .+= u .- uσ
            uσ
        else
            a′ = sample(rng, Weights(σ, 1.0))
            env(legal_actions[a′])
            u = external_sampling(env, i, nodes, rng)
            sI .+= σ
            u
        end
    end
end
