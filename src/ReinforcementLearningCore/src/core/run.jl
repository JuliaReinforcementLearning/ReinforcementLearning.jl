export expected_policy_values

import Base: run

run(agent, env::AbstractEnv, args...) = _run(agent, env, args...)

_run(agent, env, args...) = _run(DynamicStyle(env), NumAgentStyle(env), agent, env, args...)

function _run(
    ::Sequential,
    ::SingleAgent,
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    while true # run episodes forever
        reset!(env)
        agent(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, agent, env)

        while !get_terminal(env) # one episode
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)

            env(action)

            agent(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, agent, env)

            stop_condition(agent, env) && return hook # early stop
        end # end of an episode

        agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
        hook(POST_EPISODE_STAGE, agent, env)
    end
    hook
end

function _run(
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

function _run(
    ::Sequential,
    ::MultiAgent,
    agents::Tuple{Vararg{<:AbstractAgent}},
    env::AbstractEnv,
    stop_condition,
    hooks = [EmptyHook() for _ in agents],
)
    @assert length(agents) == get_num_players(env)

    hooks = Dict(get_role(agent) => hook for (agent, hook) in zip(agents, hooks))
    agents = Dict(get_role(agent) => agent for agent in agents)

    while true # run episodes forever
        reset!(env)

        for p in get_players(env)
            agents[p](PRE_EPISODE_STAGE, env)
            hooks[p](PRE_EPISODE_STAGE, agents[p], env)
        end

        while !get_terminal(env) # one episode
            agent = agents[get_current_player(env)]
            hook = hooks[get_current_player(env)]

            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)

            env(action)

            agent(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, agent, env)

            stop_condition(agent, env) && return hooks # early stop
        end # end of an episode

        for p in get_players(env)
            agents[p](POST_EPISODE_STAGE, env)
            hooks[p](POST_EPISODE_STAGE, agents[p], env)
        end
    end
    hooks
end
