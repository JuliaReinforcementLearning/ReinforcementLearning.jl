export expected_policy_values, nash_conv

function expected_policy_values(π::AbstractPolicy, env::AbstractEnv)
    if is_terminated(env)
        [reward(env, p) for p in players(env) if p != chance_player(env)]
    elseif current_player(env) == chance_player(env)
        vals = [0.0 for p in players(env) if p != chance_player(env)]
        for (a, pₐ) in zip(action_space(env), prob(env))
            if pₐ > 0
                vals .+= pₐ .* expected_policy_values(π, child(env, a))
            end
        end
        vals
    else
        vals = [0.0 for p in players(env) if p != chance_player(env)]
        actions = action_space(env)
        probs = prob(π, env)
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
        best_response_value(BestResponsePolicy(π, e, i; kw...), e) for
        i in players(e) if i != chance_player(e)
    ]

    σ = expected_policy_values(π, e)
    if is_reduce
        mapreduce(-, +, σ′, σ)
    else
        map(-, σ′, σ)
    end
end
