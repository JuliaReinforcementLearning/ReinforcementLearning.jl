export TabularCFRPolicy

struct InfoStateNode
    strategy::Vector{Float64}
    cumulative_regret::Vector{Float64}
    cumulative_strategy::Vector{Float64}
end

InfoStateNode(n) = InfoStateNode(fill(1/n,n), zeros(n), zeros(n))

function init_info_state_nodes(env::AbstractEnv)
    nodes = Dict{String, InfoStateNode}()
    walk(env) do x
        if !get_terminal(x) && get_current_player(x) != get_chance_player(x)
            get!(nodes, get_state(x), InfoStateNode(length(get_legal_actions(x))))
        end
    end
    nodes
end

"""
    TabularCFRPolicy

See more details: [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
"""
struct TabularCFRPolicy{S,T,R<:AbstractRNG} <: AbstractPolicy
    nodes::Dict{S, InfoStateNode}
    behavior_policy::QBasedPolicy{TabularLearner{S,T}, WeightedExplorer{true,R}}
end

(p::TabularCFRPolicy)(env::AbstractEnv) = p.behavior_policy(env)

RLBase.get_prob(p::TabularCFRPolicy, env::AbstractEnv) = get_prob(p.behavior_policy, env)

"""
    TabularCFRPolicy(;n_iter::Int, env::AbstractEnv)
"""
function TabularCFRPolicy(;n_iter::Int, env::AbstractEnv, rng=Random.GLOBAL_RNG, is_reset_neg_regrets=false, is_linear_averaging=false)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DynamicStyle(env) === SEQUENTIAL
    @assert RewardStyle(env) === TERMINAL_REWARD
    @assert ChanceStyle(env) === EXPLICIT_STOCHASTIC
    @assert DefaultStateStyle(env) === Information{String}()

    nodes = init_info_state_nodes(env)

    for i in 1:n_iter
        for p in get_players(env)
            if p != get_chance_player(env)
                init_reach_prob = Dict(x=>1.0 for x in get_players(env) if x != get_chance_player(env))
                cfr!(nodes, env, p, init_reach_prob, 1.0, is_linear_averaging ? i : 1)
                update_strategy!(nodes)

                if is_reset_neg_regrets
                    for node in values(nodes)
                        node.cumulative_regret .= max.(node.cumulative_regret, 0)
                    end
                end
            end
        end
    end

    behavior_policy = QBasedPolicy(;learner=TabularLearner{String}(), explorer=WeightedExplorer(;is_normalized=true, rng=rng))

    for (k,v) in nodes
        s = sum(v.cumulative_strategy)
        if s != 0
            update!(behavior_policy, k => v.cumulative_strategy ./ s)
        end
    end

    TabularCFRPolicy(nodes, behavior_policy)
end

function cfr!(nodes, env, player, reach_probs, chance_player_reach_prob, ratio)
    if get_terminal(env)
        get_reward(env, player)
    else
        if get_current_player(env) == get_chance_player(env)
            v = 0.
            for a::ActionProbPair in get_legal_actions(env)
                v += a.prob * cfr!(nodes, child(env, a), player, reach_probs, chance_player_reach_prob * a.prob, ratio)
            end
            v
        else
            v = 0.
            node = nodes[get_state(env)]
            legal_actions = get_legal_actions(env)
            U = player == get_current_player(env) ? Vector{Float64}(undef, length(legal_actions)) : nothing

            for (i, action) in enumerate(legal_actions)
                prob = node.strategy[i]
                new_reach_probs = copy(reach_probs)
                new_reach_probs[get_current_player(env)] *= prob

                u = cfr!(nodes, child(env, action), player, new_reach_probs, chance_player_reach_prob, ratio)
                isnothing(U) || (U[i] = u)
                v += prob * u
            end

            if player == get_current_player(env)
                reach_prob = reach_probs[player]
                counterfactual_reach_prob = reduce(
                    *,
                    (reach_probs[p] for p in get_players(env) if p != player && p != get_chance_player(env));
                    init=chance_player_reach_prob)
                node.cumulative_regret .+= counterfactual_reach_prob .* (U .- v)
                node.cumulative_strategy .+= ratio .* reach_prob .* node.strategy
            end
            v
        end
    end
end

function regret_matching!(strategy, cumulative_regret)
    s = mapreduce(x->max(0,x), +,cumulative_regret)
    if s > 0
        strategy .= max.(0., cumulative_regret) ./ s
    else
        fill!(strategy, 1/length(strategy))
    end
end

function update_strategy!(nodes)
    for node in values(nodes)
        regret_matching!(node.strategy, node.cumulative_regret)
    end
end