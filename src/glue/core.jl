import Base: run

using ReinforcementLearningEnvironments: reset!

export run

"""
    run(agent::AbstractAgent, env::AbstractEnv, stop_condition; hook = EmptyHook())

See also [Stop Conditions](@ref), [Hooks](@ref)
"""
function run(agent::AbstractAgent, env::AbstractEnv, stop_condition; hook = EmptyHook())

    reset!(env)
    obs = reset(observe(env); terminal = true)
    hook(PRE_EPISODE_STAGE, agent, env, obs)

    while true
        action = agent(obs)
        hook(PRE_ACT_STAGE, agent, env, obs => action)

        # async here?
        update!(agent, obs => action)
        env(action)
        obs = observe(env)
        hook(POST_ACT_STAGE, agent, env, action => obs)

        if stop_condition(agent, env, obs)
            if get_terminal(obs)
                hook(POST_EPISODE_STAGE, agent, env, obs)
            end
            update!(agent, obs => agent(obs))  # push reward terminal into buffer, state and action are not that important
            break
        end

        if get_terminal(obs)
            hook(POST_EPISODE_STAGE, agent, env, obs)
            r = get_reward(obs)  # !!! deepcopy?
            reset!(env)
            obs = reset(observe(env); reward = r, terminal = true)
            hook(PRE_EPISODE_STAGE, agent, env, obs)
        end
    end
    hook
end

function run(
    agents::Tuple{Vararg{<:AbstractAgent}},
    env::AbstractEnv,
    stop_condition;
    hook = [EmptyHook() for _ in agents],
)
    roles = [agent.role for agent in agents]
    reset!(env)
    observations = [reset(observe(env, agent.role); terminal = true) for agent in agents]
    for (h, agent, obs) in zip(hook, agents, observations)
        h(PRE_EPISODE_STAGE, agent, env, obs)
    end

    while true
        # async here?
        actions = [agent(obs) for (agent, obs) in zip(agents, observations)]
        for (h, agent, obs, action) in zip(hook, agents, observations, actions)
            h(PRE_ACT_STAGE, agent, env, obs => action)
            update!(agent, obs => action)
        end

        env(roles => actions)

        observations = [observe(env, agent.role) for agent in agents]
        for (h, agent, action, obs) in zip(hook, agents, actions, observations)
            h(POST_ACT_STAGE, agents, env, action => obs)
        end

        if stop_condition(agents, env, observations)
            if get_terminal(observations)
                for (h, agent, obs) in zip(hook, agents, observations)
                    h(POST_EPISODE_STAGE, agent, env, obs)
                end
            end
            for (agent, obs) in zip(agents, observations)
                update!(agent, obs => agent(obs))  # push reward terminal into buffer, state and action are not that important
            end
            break
        end

        if get_terminal(observations)
            for (h, agent, obs) in zip(hook, agents, observations)
                h(POST_EPISODE_STAGE, agent, env, obs)
            end

            reset!(env)

            observations = [reset(
                observe(env, agent.role);
                reward = observations[i].reward,
                terminal = observations[i].terminal,
            ) for (i, agent) in enumerate(agents)]

            for (h, agent, obs) in zip(hook, agents, observations)
                h(PRE_EPISODE_STAGE, agent, env, obs)
            end
        end
    end
    hook
end
