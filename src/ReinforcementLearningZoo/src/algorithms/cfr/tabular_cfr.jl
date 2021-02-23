export TabularCFRPolicy

struct InfoStateNode{M<:AbstractVector{Bool}}
    strategy::Vector{Float64}
    cumulative_regret::Vector{Float64}
    cumulative_strategy::Vector{Float64}
    mask::M
end

function InfoStateNode(mask)
    n = sum(mask)
    InfoStateNode(fill(1 / n, n), zeros(n), zeros(n), mask)
end

#####
# TabularCFRPolicy
#####

mutable struct TabularCFRPolicy{S,T,R<:AbstractRNG} <: AbstractCFRPolicy
    nodes::Dict{S,InfoStateNode}
    behavior_policy::TabularRandomPolicy{S,T,R}
    is_reset_neg_regrets::Bool
    is_linear_averaging::Bool
    weighted_averaging_delay::Int
    is_alternating_update::Bool
    rng::R
    n_iteration::Int
end

(p::TabularCFRPolicy)(env::AbstractEnv) = p.behavior_policy(env)

RLBase.prob(p::TabularCFRPolicy, env::AbstractEnv) = prob(p.behavior_policy, env)

"""
    TabularCFRPolicy(;kwargs...)

Some useful papers while implementing this algorithm:

- [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
- [MONTE CARLO SAMPLING AND REGRET MINIMIZATION FOR EQUILIBRIUM COMPUTATION AND DECISION-MAKING IN LARGE EXTENSIVE FORM GAMES](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf)
- [Solving Large Imperfect Information Games Using CFR⁺](https://arxiv.org/pdf/1407.5042.pdf)
- [Revisiting CFR⁺ and Alternating Updates](https://arxiv.org/pdf/1810.11542v1.pdf)
- [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/pdf/1809.04040.pdf)

# Keyword Arguments

- `is_alternating_update=true`: If `true`, we update the players alternatively.
- `is_reset_neg_regrets=true`: Whether to use **regret matching⁺**.
- `is_linear_averaging=true`
- `weighted_averaging_delay=0`. The averaging delay in number of iterations. Only valid when `is_linear_averaging` is set to `true`.
- `state_type=String`, the data type of information set.
- `rng=Random.GLOBAL_RNG`
"""
function TabularCFRPolicy(;
    is_reset_neg_regrets = true,
    is_linear_averaging = true,
    weighted_averaging_delay = 0,
    is_alternating_update = true,
    state_type = String,
    rng = Random.GLOBAL_RNG,
    n_iteration = 1,
)
    TabularCFRPolicy(
        Dict{state_type,InfoStateNode}(),
        TabularRandomPolicy(; rng = rng, table = Dict{state_type,Vector{Float64}}()),
        is_reset_neg_regrets,
        is_linear_averaging,
        weighted_averaging_delay,
        is_alternating_update,
        rng,
        n_iteration,
    )
end

"Update the `behavior_policy`"
function RLBase.update!(p::TabularCFRPolicy)
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
function RLBase.update!(p::TabularCFRPolicy, env::AbstractEnv)
    w = p.is_linear_averaging ? max(p.n_iteration - p.weighted_averaging_delay, 0) : 1
    if p.is_alternating_update
        for x in players(env)
            if x != chance_player(env)
                cfr!(p.nodes, env, x, w)
                regret_matching!(p)
            end
        end
    else
        cfr!(p.nodes, env, nothing, w)
        regret_matching!(p)
    end
    p.n_iteration += 1
end

"""
Symbol meanings:

π: reach prob
π′: new reach prob
π₋ᵢ: opponents' reach prob
p: player to update. `nothing` means simultaneous update.
w: weight
v: counterfactual value **before weighted by opponent's reaching probability**
V: a vector containing the `v` after taking each action with current information set. Used to calculate the **regret value**
"""
function cfr!(nodes, env, p, w, π = Dict(x => 1.0 for x in players(env)))
    if is_terminated(env)
        reward(env, p)
    else
        if current_player(env) == chance_player(env)
            v = 0.0
            for (a, pₐ) in zip(action_space(env), prob(env))
                if pₐ > 0
                    π′ = copy(π)
                    π′[current_player(env)] *= pₐ
                    v += pₐ * cfr!(nodes, child(env, a), p, w, π′)
                end
            end
            v
        else
            v = 0.0
            legal_actions = legal_action_space(env)
            M = legal_action_space_mask(env)
            node = get!(nodes, state(env), InfoStateNode(M))

            is_update = isnothing(p) || p == current_player(env)
            V = is_update ? Vector{Float64}(undef, length(legal_actions)) : nothing

            for (i, a) in enumerate(legal_actions)
                σᵢ = node.strategy[i]
                π′ = copy(π)
                π′[current_player(env)] *= σᵢ

                vₐ = cfr!(nodes, child(env, a), p, w, π′)
                is_update && (V[i] = vₐ)
                v += σᵢ * vₐ
            end

            if is_update
                πᵢ = π[current_player(env)]
                π₋ᵢ = reduce(*, (prob for (p, prob) in π if p != current_player(env)))
                node.cumulative_regret .+= π₋ᵢ .* (V .- v)
                node.cumulative_strategy .+= w .* πᵢ .* node.strategy
            end
            v
        end
    end
end

function regret_matching!(p::TabularCFRPolicy)
    for node in values(p.nodes)
        regret_matching!(node; is_reset_neg_regrets = p.is_reset_neg_regrets)
    end
end

regret_matching!(node::InfoStateNode; kwargs...) =
    regret_matching!(node.strategy, node.cumulative_regret; kwargs...)

function regret_matching!(strategy, cumulative_regret; is_reset_neg_regrets = true)
    if is_reset_neg_regrets
        for i in 1:length(cumulative_regret)
            if cumulative_regret[i] < 0
                cumulative_regret[i] = 0
            end
        end
    end
    s = mapreduce(x -> max(0, x), +, cumulative_regret)
    if s > 0
        strategy .= max.(0.0, cumulative_regret) ./ s
    else
        fill!(strategy, 1 / length(strategy))
    end
end
