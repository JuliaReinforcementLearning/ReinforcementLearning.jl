export train

using ReinforcementLearningEnvironments

function train(
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition::Function
    ;pre_episode_hook=identity,
    post_episode_hook=identity,
    pre_act_hook=identity,
    post_act_hook=identity
    )

    reset!(env)
    obs = observe(env)
    pre_episode_hook(agent, env, obs)

    while true
        stop_condition(agent, env) && break

        pre_act_hook(agent, env, obs)
        action = agent(obs)
        env(action)
        post_act_hook(agent, env, obs, action)

        if is_terminal(obs)
            post_episode_hook(agent, env, obs)
            reset!(env)
            obs = observe(env)
            pre_episode_hook(agent, env, obs)
        end
    end
end