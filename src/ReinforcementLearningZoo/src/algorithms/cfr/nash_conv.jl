export expected_policy_values, nash_conv

function expected_policy_values(π::AbstractPolicy, env::AbstractEnv)
    if get_terminal(env)
        [get_reward(env, p) for p in get_players(env) if p != get_chance_player(env)]
    elseif get_current_player(env) == get_chance_player(env)
        vals = [0.0 for p in get_players(env) if p != get_chance_player(env)]
        for a::ActionProbPair in get_legal_actions(env)
            vals .+= a.prob .* expected_policy_values(π, child(env, a))
        end
        vals
    else
        vals = [0.0 for p in get_players(env) if p != get_chance_player(env)]
        actions = get_actions(env)
        probs = get_prob(π, env)
        @assert length(actions) == length(probs)

        for (a, p) in zip(actions, probs)
            if p > 0 #= ignore illegal action =#
                vals .+= p .* expected_policy_values(π, child(env, a))
            end
        end
        vals
    end
end

function nash_conv(π, env; is_reduce = true, kw...)
    e = copy(env)
    RLBase.reset!(e)

    σ′ = [
        best_response_value(BestResponsePolicy(π, e, i; kw...), e)
        for i in get_players(e) if i != get_chance_player(e)
    ]

    σ = expected_policy_values(π, e)
    if is_reduce
        mapreduce(-, +, σ′, σ)
    else
        map(-, σ′, σ)
    end
end
