export BestResponsePolicy

struct BestResponsePolicy{E,S,A,X,P<:AbstractPolicy} <: AbstractCFRPolicy
    cfr_reach_prob::Dict{S,Vector{Pair{E,Float64}}}
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
function BestResponsePolicy(policy, env, best_responder;)
    S = eltype(state_space(env))
    A = eltype(action_space(env))
    E = typeof(env)

    p = BestResponsePolicy(
        Dict{S,Vector{Pair{E,Float64}}}(),
        Dict{S,A}(),
        Dict{E,Float64}(),
        best_responder,
        policy,
    )

    e = copy(env)
    @assert e == env "The copy method doesn't seem to be implemented for environment: $env"
    @assert hash(e) == hash(env) "The hash method doesn't seem to be implemented for environment: $env"
    RLBase.reset!(e)  # start from the root!
    init_cfr_reach_prob!(p, e)
    p
end

function (p::BestResponsePolicy)(env::AbstractEnv)
    if current_player(env) == p.best_responder
        best_response_action(p, env)
    else
        p.policy(env)
    end
end

function init_cfr_reach_prob!(p, env, reach_prob = 1.0)
    if !is_terminated(env)
        if current_player(env) == p.best_responder
            push!(get!(p.cfr_reach_prob, state(env), []), env => reach_prob)

            for a in legal_action_space(env)
                init_cfr_reach_prob!(p, child(env, a), reach_prob)
            end
        elseif current_player(env) == chance_player(env)
            for (a, pₐ) in zip(action_space(env), prob(env))
                if pₐ > 0
                    init_cfr_reach_prob!(p, child(env, a), reach_prob * pₐ)
                end
            end
        else  # opponents
            for a in legal_action_space(env)
                init_cfr_reach_prob!(p, child(env, a), reach_prob * prob(p.policy, env, a))
            end
        end
    end
end

function best_response_value(p, env)
    get!(p.best_response_value_cache, env) do
        if is_terminated(env)
            reward(env, p.best_responder)
        elseif current_player(env) == p.best_responder
            a = best_response_action(p, env)
            best_response_value(p, child(env, a))
        elseif current_player(env) == chance_player(env)
            v = 0.0
            A, P = action_space(env), prob(env)
            @assert length(A) == length(P)
            for (a, pₐ) in zip(A, P)
                if pₐ > 0
                    v += pₐ * best_response_value(p, child(env, a))
                end
            end
            v
        else
            v = 0.0
            for a in legal_action_space(env)
                v += prob(p.policy, env, a) * best_response_value(p, child(env, a))
            end
            v
        end
    end
end

function best_response_action(p, env)
    get!(p.best_response_action_cache, state(env)) do
        best_action, best_action_value = nothing, typemin(Float64)
        for a in legal_action_space(env)
            # for each information set (`state(env)` here), we may have several paths to reach it
            # here we sum the cfr reach prob weighted value to find out the best action
            v = sum(p.cfr_reach_prob[state(env)]) do (e, reach_prob)
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

function RLBase.prob(p::BestResponsePolicy, env::AbstractEnv)
    if current_player(env) == p.best_responder
        onehot(p(env), action_space(env))
    else
        prob(p.policy, env)
    end
end

function RLBase.prob(p::BestResponsePolicy, env::AbstractEnv, action)
    if current_player(env) == p.best_responder
        action == best_response_action(p, env)
    else
        prob(p.policy, env, action)
    end
end
