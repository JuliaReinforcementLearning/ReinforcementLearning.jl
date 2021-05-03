export MinimaxPolicy

"""
    MinimaxPolicy(;value_function, depth::Int)
The minimax algorithm with [Alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha-beta_pruning)
## Keyword Arguments
- `maximum_depth::Int=30`, the maximum depth of search.
- `value_function=nothing`, estimate the value of `env`. `value_function(env) -> Number`. It is only called after searching for `maximum_depth` and the `env` is not terminated yet.
"""
Base.@kwdef mutable struct MinimaxPolicy{F} <: AbstractPolicy
    maximum_depth::Int = 30
    value_function::F = nothing
    v::Float64 = 0.0
end

(p::MinimaxPolicy)(env::AbstractEnv) = p(env, DynamicStyle(env), NumAgentStyle(env))

function (p::MinimaxPolicy)(env::AbstractEnv, ::Sequential, ::MultiAgent{2})
    if is_terminated(env)
        rand(action_space(env))  # just a dummy action
    else
        a, v = α_β_search(
            env,
            p.value_function,
            p.maximum_depth,
            -Inf,
            Inf,
            current_player(env),
        )
        p.v = v  # for debug only
        a
    end
end

function α_β_search(env::AbstractEnv, value_function, depth, α, β, maximizing_role)
    if is_terminated(env)
        nothing, reward(env, maximizing_role)
    elseif depth == 0
        nothing, value_function(env)
    elseif current_player(env) == maximizing_role
        legal_actions = legal_action_space(env)
        best_action = legal_actions[1]
        v = -Inf
        for a in legal_actions
            node = child(env, a)
            _, v_node = α_β_search(node, value_function, depth - 1, α, β, maximizing_role)
            if v_node > v
                v = v_node
                best_action = a
            end
            α = max(α, v)
            α >= β && break  # β cut-off
        end
        best_action, v
    else
        legal_actions = legal_action_space(env)
        best_action = legal_actions[1]
        v = Inf
        for a in legal_actions
            node = child(env, a)
            _, v_node = α_β_search(node, value_function, depth - 1, α, β, maximizing_role)
            if v_node < v
                v = v_node
                best_action = a
            end
            β = min(β, v)
            β <= α && break  # α cut-off
        end
        best_action, v
    end
end
