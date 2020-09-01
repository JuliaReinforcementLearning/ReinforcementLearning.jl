export expected_policy_values

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

    hooks = Dict(get_role(agent) => hook for (agent, hook) in zip(agents, hooks))
    agents = Dict(get_role(agent) => agent for agent in agents)
    reset!(env)

    agent = agents[get_current_player(env)]
    hook = hooks[get_current_player(env)]

    for p in get_players(env)
        agents[p](PRE_EPISODE_STAGE, env)
        hooks[p](PRE_EPISODE_STAGE, agents[p], env)
    end

    action = agent(PRE_ACT_STAGE, env)
    hook(PRE_ACT_STAGE, agent, env, action)

    while true
        env(action)
        agent(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, agent, env)

        if get_terminal(env)
            for p in get_players(env)
                agents[p](POST_EPISODE_STAGE, env)
                hooks[p](POST_EPISODE_STAGE, agents[p], env)
            end

            stop_condition(agent, env) && break

            reset!(env)

            for p in get_players(env)
                agents[p](PRE_EPISODE_STAGE, env)
                hooks[p](PRE_EPISODE_STAGE, agents[p], env)
            end

            agent = agents[get_current_player(env)]
            hook = hooks[get_current_player(env)]
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        else
            stop_condition(agent, env) && break

            agent = agents[get_current_player(env)]
            hook = hooks[get_current_player(env)]
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        end
    end

    hooks
end

"""
    expected_policy_values(agents, env)

Calculate the expected return of each agent.
"""
function expected_policy_values(agents::Tuple{Vararg{<:AbstractAgent}}, env::AbstractEnv)
    agents = Dict(get_role(agent) => agent for agent in agents)
    values = expected_policy_values(agents, env)
    Dict(zip(get_players(env), values))
end

expected_policy_values(agents::Dict, env::AbstractEnv) = expected_policy_values(agents, env, RewardStyle(env), ChanceStyle(env), DynamicStyle(env))

function expected_policy_values(agents::Dict, env::AbstractEnv, ::TerminalReward, ::Union{ExplicitStochastic,Deterministic}, ::Sequential)
    if get_terminal(env)
        [get_reward(env, get_role(agents[p])) for p in get_players(env)]
    elseif get_current_player(env) == get_chance_player(env)
        vals = zeros(length(agents))
        for a::ActionProbPair in get_legal_actions(env)
            vals .+= a.prob .* expected_policy_values(agents, child(env, a))
        end
        vals
    else
        vals = zeros(length(agents))
        probs = get_prob(agents[get_current_player(env)].policy, env)
        actions = get_actions(env)
        for (a, p) in zip(actions, probs)
            if p > 0 #= ignore illegal action =#
                vals .+= p .* expected_policy_values(agents, child(env, a))
            end
        end
        vals
    end
end
