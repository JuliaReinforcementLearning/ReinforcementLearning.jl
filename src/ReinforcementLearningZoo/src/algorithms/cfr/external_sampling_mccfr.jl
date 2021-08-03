export ExternalSamplingMCCFRPolicy

"""
    ExternalSamplingMCCFRPolicy

This implementation uses stochastically-weighted averaging.

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

RLBase.prob(p::ExternalSamplingMCCFRPolicy, env::AbstractEnv) = prob(p.behavior_policy, env)

RLBase.prob(p::ExternalSamplingMCCFRPolicy, env::AbstractEnv, action) =
    prob(p.behavior_policy, env, action)

function ExternalSamplingMCCFRPolicy(; state_type = String, rng = Random.GLOBAL_RNG)
    ExternalSamplingMCCFRPolicy(
        Dict{state_type,InfoStateNode}(),
        TabularRandomPolicy(; rng = rng, table = Dict{state_type,Vector{Float64}}()),
        rng,
    )
end

function RLBase.update!(p::ExternalSamplingMCCFRPolicy)
    for (k, v) in p.nodes
        s = sum(v.cumulative_strategy)
        if s != 0
            m = v.mask
            strategy = zeros(length(m))
            strategy[m] .= v.cumulative_strategy ./ s
            update!(p.behavior_policy, k => strategy)
        else
            # The TabularRandomPolicy will return uniform distribution by default. 
            # So we do nothing here.
        end
    end
end

"Run one interation"
function RLBase.update!(p::ExternalSamplingMCCFRPolicy, env::AbstractEnv)
    for x in players(env)
        if x != chance_player(env)
            external_sampling(copy(env), x, p.nodes, p.rng)
        end
    end
end

function external_sampling(env, i, nodes, rng)
    player = current_player(env)

    if is_terminated(env)
        reward(env, i)
    elseif player == chance_player(env)
        env(sample(rng, action_space(env), Weights(prob(env), 1.0)))
        external_sampling(env, i, nodes, rng)
    else
        I = state(env)
        legal_actions = legal_action_space(env)
        M = legal_action_space_mask(env)
        n = length(legal_actions)
        node = get!(nodes, I, InfoStateNode(M))
        regret_matching!(node; is_reset_neg_regrets = false)
        σ, rI, sI = node.strategy, node.cumulative_regret, node.cumulative_strategy

        if i == player
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
