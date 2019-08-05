export train

using ReinforcementLearningEnvironments

function train(
    agent::AbstractAgent,
    env::AbstractEnv
    ;step=0)

    pre_episode(agent, env)
    obs = observe(env)

    is_stop = false
    while !is_stop

        if is_terminal(obs)
            pre_episode(agent, env)
            obs = observe(env)
        end

        action = agent(obs)
        env(action)
        obs = observe(env)

        if is_terminal(obs)
            post_episode(agent, env)
        end

        for sc in stop_conditions
            if sc(agent, env, runtime_info)
                is_stop = true
            end
        end
    end
end