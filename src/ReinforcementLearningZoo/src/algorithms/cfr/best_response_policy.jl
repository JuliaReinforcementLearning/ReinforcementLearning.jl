export BestResponsePolicy

using Flux:onehot

struct BestResponsePolicy{E, S, A, X, P<:AbstractPolicy} <: AbstractCFRPolicy
    cfr_reach_prob::Dict{S, Vector{Pair{E, Float64}}}
    best_response_action_cache::Dict{S,A}
    best_response_value_cache::Dict{E,Float64}
    best_responder::X
    policy::P
end

"""
    BestResponsePolicy(policy, env, best_responder)

- `policy`, the original policy to be wrapped in the best response policy.
- `env`, the environment to handle.
- `best_responder`, the player to choose best response action.
"""
function BestResponsePolicy(policy, env, best_responder; state_type=String, action_type=Int)
    # S = typeof(get_state(env))  # TODO: currently it will break the OpenSpielEnv. Can not get information set for chance player
    # A = eltype(get_actions(env))  # TODO: for chance players it will return ActionProbPair
    S = state_type
    A = action_type
    E = typeof(env)

    p = BestResponsePolicy(
        Dict{S, Vector{Pair{E, Float64}}}(),
        Dict{S, A}(),
        Dict{E, Float64}(),
        best_responder,
        policy
    )

    e = copy(env)
    @assert e == env  "The copy method doesn't seem to be implemented for environment: $env"
    @assert hash(e) == hash(env) "The hash method doesn't seem to be implemented for environment: $env"
    RLBase.reset!(e)  # start from the root!
    init_cfr_reach_prob!(p, e)
    p
end

function (p::BestResponsePolicy)(env::AbstractEnv)
    if get_current_player(env) == p.best_responder
        best_response_action(p, env)
    else
        p.policy(env)
    end
end

function init_cfr_reach_prob!(p, env, reach_prob=1.0)
    if !get_terminal(env)
        if get_current_player(env) == p.best_responder
            push!(get!(p.cfr_reach_prob, get_state(env), []), env => reach_prob)

            for a in get_legal_actions(env)
                init_cfr_reach_prob!(p, child(env, a), reach_prob)
            end
        elseif get_current_player(env) == get_chance_player(env)
            for a::ActionProbPair in get_actions(env)
                init_cfr_reach_prob!(p, child(env, a), reach_prob * a.prob)
            end
        else  # opponents
            for a in get_legal_actions(env)
                init_cfr_reach_prob!(p, child(env, a), reach_prob * get_prob(p.policy, env, a))
            end
        end
    end
end

function best_response_value(p, env)
    get!(p.best_response_value_cache, env) do
        if get_terminal(env)
            get_reward(env, p.best_responder)
        elseif get_current_player(env) == p.best_responder
                a = best_response_action(p, env)
                best_response_value(p, child(env, a))
        elseif get_current_player(env) == get_chance_player(env)
            v = 0.
            for a::ActionProbPair in get_actions(env)
                v += a.prob * best_response_value(p, child(env, a))
            end
            v
        else
            v = 0.
            for a in get_legal_actions(env)
                v += get_prob(p.policy, env, a) * best_response_value(p, child(env, a))
            end
            v
        end
    end
end

function best_response_action(p, env)
    get!(p.best_response_action_cache, get_state(env)) do
        best_action, best_action_value = nothing, typemin(Float64)
        for a in get_legal_actions(env)
            # for each information set (`get_state(env)` here), we may have several paths to reach it
            # here we sum the cfr reach prob weighted value to find out the best action
            v = sum(p.cfr_reach_prob[get_state(env)]) do (e, reach_prob)
                reach_prob * best_response_value(p, child(e, a))
            end
            if v > best_action_value
                best_action, best_action_value = a, v
            end
        end
        best_action
    end
end

RLBase.update!(p::BestResponsePolicy, args...) = nothing

function RLBase.get_prob(p::BestResponsePolicy, env::AbstractEnv)
    if get_current_player(env) == p.best_responder
        onehot(p(env), get_actions(env))
    else
        get_prob(p.policy, env)
    end
end
