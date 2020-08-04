import Base: run

run(agent, env::AbstractEnv, args...) =
    run(DynamicStyle(env), NumAgentStyle(env), agent, env, args...)

function run(
    ::Sequential,
    ::SingleAgent,
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    reset!(env)
    agent(PRE_EPISODE_STAGE, env)
    hook(PRE_EPISODE_STAGE, agent, env)
    action = agent(PRE_ACT_STAGE, env)
    hook(PRE_ACT_STAGE, agent, env, action)

    while true
        env(action)
        agent(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, agent, env)

        if get_terminal(env)
            agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
            hook(POST_EPISODE_STAGE, agent, env)

            stop_condition(agent, env) && break

            reset!(env)
            agent(PRE_EPISODE_STAGE, env)
            hook(PRE_EPISODE_STAGE, agent, env)
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        else
            stop_condition(agent, env) && break
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        end
    end
    hook
end

function run(
    ::Sequential,
    ::SingleAgent,
    agent::AbstractAgent,
    env::MultiThreadEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    while true
        reset!(env)
        action = agent(PRE_ACT_STAGE, env)
        hook(PRE_ACT_STAGE, agent, env, action)

        env(action)
        agent(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, agent, env)

        if stop_condition(agent, env)
            agent(PRE_ACT_STAGE, env)  # let the agent see the last observation
            break
        end
    end
    hook
end

function run(
    ::Sequential,
    ::MultiAgent,
    agents::Tuple{Vararg{<:AbstractAgent}},
    env::AbstractEnv,
    stop_condition,
    hooks = [EmptyHook() for _ in agents],
)
    @assert length(agents) == get_num_players(env)

    reset!(env)
    valid_action = rand(get_actions(env))  # init with a dummy value

    # async here?
    for (agent, hook) in zip(agents, hooks)
        agent(PRE_EPISODE_STAGE, SubjectiveEnv(env, get_role(agent)))
        hook(PRE_EPISODE_STAGE, agent, env)
        action = agent(PRE_ACT_STAGE, SubjectiveEnv(env, get_role(agent)))
        hook(PRE_ACT_STAGE, agent, env, action)
        # for Sequential environments, only one action is valid
        if get_current_player(env) == get_role(agent)
            valid_action = action
        end
    end

    while true
        env(valid_action)

        for (agent, hook) in zip(agents, hooks)
            agent(POST_ACT_STAGE, SubjectiveEnv(env, get_role(agent)))
            hook(POST_ACT_STAGE, agent, env)
        end

        if get_terminal(env)
            for (agent, hook) in zip(agents, hooks)
                agent(POST_EPISODE_STAGE, SubjectiveEnv(env, get_role(agent)))
                hook(POST_EPISODE_STAGE, agent, env)
            end

            stop_condition(agents, env) && break
            reset!(env)
            # async here?
            for (agent, hook) in zip(agents, hooks)
                agent(PRE_EPISODE_STAGE, SubjectiveEnv(env, get_role(agent)))
                hook(PRE_EPISODE_STAGE, agent, env)
                action = agent(PRE_ACT_STAGE, SubjectiveEnv(env, get_role(agent)))
                hook(PRE_ACT_STAGE, agent, env, action)
                # for Sequential environments, only one action is valid
                if get_current_player(env) == get_role(agent)
                    valid_action = action
                end
            end
        else
            stop_condition(agents, env) && break
            for (agent, hook) in zip(agents, hooks)
                action = agent(PRE_ACT_STAGE, SubjectiveEnv(env, get_role(agent)))
                hook(PRE_ACT_STAGE, agent, env, action)
                # for Sequential environments, only one action is valid
                if get_current_player(env) == get_role(agent)
                    valid_action = action
                end
            end
        end
    end
    hooks
end
