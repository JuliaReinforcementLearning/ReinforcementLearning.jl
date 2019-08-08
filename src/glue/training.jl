export train

using ReinforcementLearningEnvironments
using ReinforcementLearningEnvironments:reset!

dummy_condition = (xs...) -> nothing

function train(
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition::Function
    ;pre_episode_hook=dummy_condition,
    post_episode_hook=dummy_condition,
    pre_act_hook=dummy_condition,
    post_act_hook=dummy_condition
    )

    reset!(env)
    obs = observe(env)
    pre_episode_hook(agent, env, obs)

    while true
        stop_condition(agent, env) && break

        pre_act_hook(agent, env, obs)
        action = agent(obs)

        # async here?
        update!(agent, obs => action)
        env(action)

        obs = observe(env)
        post_act_hook(agent, env, obs, action)

        if is_terminal(obs)
            post_episode_hook(agent, env, obs)
            r, t = obs.reward, obs.terminal  # !!! deepcopy?
            reset!(env)
            obs = observe(env)
            obs.reward = r
            obs.terminal = t
            pre_episode_hook(agent, env, obs)
        end
    end
end