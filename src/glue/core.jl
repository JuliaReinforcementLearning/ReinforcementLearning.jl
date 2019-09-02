import Base:run

export run

using ReinforcementLearningEnvironments
using ReinforcementLearningEnvironments:reset!

function run(agent::AbstractAgent, env::AbstractEnv, stop_condition ;hook=EmptyHook())

    reset!(env)
    obs = observe(env)
    hook(PRE_EPISODE_STAGE, agent, env, obs)

    while true
        stop_condition(agent, env, obs) && break

        action = agent(obs)
        hook(PRE_ACT_STAGE, agent, env, obs => action)

        # async here?
        update!(agent, obs => action)
        env(action)

        obs = observe(env)
        hook(POST_ACT_STAGE, agent, env, action => obs)

        if get_terminal(obs)
            hook(POST_EPISODE_STAGE, agent, env, obs)
            r, t = obs.reward, obs.terminal  # !!! deepcopy?
            reset!(env)
            temp_obs = observe(env)
            obs = Observation(r, t, get_state(temp_obs), temp_obs.meta)
            hook(PRE_EPISODE_STAGE, agent, env, obs)
        end
    end
    hook
end

function run(agents::Tuple{Vararg{<:AbstractAgent}}, env::AbstractEnv, stop_condition; hook=EmptyHook())
    roles = [agent.role for agent in agents]
    reset!(env)
    observations = [observe(env, agent.role) for agent in agents]
    for (agent, obs) in zip(agents, observations)
        hook(PRE_EPISODE_STAGE, agent, env, obs)
    end

    while true
        stop_condition(agents, env, observations) && break

        # async here?
        actions = [agent(obs) for (agent, obs) in zip(agents, observations)]
        for (agent, obs, action) in zip(agents, observations, actions)
            hook(PRE_ACT_STAGE, agent, env, obs => action)
            update!(agent, obs => action)
        end

        env(roles => actions)

        observations = [observe(env, agent.role) for agent in agents]
        for (agent, action, obs) in zip(agents, actions, observations)
            hook(POST_ACT_STAGE, agents, env, action => obs)
        end

        if get_terminal(observations)
            for (agent, obs) in zip(agents, observations)
                hook(POST_EPISODE_STAGE, agent, env, obs)
            end

            reset!(env)

            observations = [
                begin
                    temp_obs = observe(env, agent.role)
                    Observation(
                        observations[i].reward,
                        observations[i].terminal,
                        get_state(temp_obs),
                        temp_obs.meta
                    )
                end
                for (i, agent) in enumerate(agents)
            ]

            for (agent, obs) in zip(agents, observations)
                hook(PRE_EPISODE_STAGE, agent, env, obs)
            end
        end
    end
    hook
end