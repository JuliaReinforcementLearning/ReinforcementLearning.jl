export train

using ReinforcementLearningEnvironments
using ReinforcementLearningEnvironments:reset!

dummy_condition = (xs...) -> nothing

function stage_run()
end

function train(
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition
    ;hook=EmptyHook()
    )

    reset!(env)
    obs = observe(env)
    hook(PRE_EPISODE_STAGE, agent, env, obs)

    while true
        stop_condition(agent, env, obs) && break

        hook(PRE_ACT_STAGE, agent, env, obs)
        action = agent(obs)

        # async here?
        update!(agent, obs => action)
        env(action)

        obs = observe(env)
        hook(POST_ACT_STAGE, agent, env, obs => action)

        if is_terminal(obs)
            hook(POST_EPISODE_STAGE, agent, env, obs)
            r, t = obs.reward, obs.terminal  # !!! deepcopy?
            reset!(env)
            obs = observe(env)
            obs.reward = r
            obs.terminal = t
            hook(PRE_EPISODE_STAGE, agent, env, obs)
        end
    end
end