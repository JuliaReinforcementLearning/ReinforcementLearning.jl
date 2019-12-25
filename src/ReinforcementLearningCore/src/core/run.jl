import Base: run

run(agent::AbstractAgent, env::AbstractEnv, stop_condition, hook::AbstractHook=EmptyHook()) = run(DynamicStyle(agent), agent, env, stop_condition, hook)

function run(::Sequential, agent::AbstractAgent, env::AbstractEnv, stop_condition, hook::AbstractHook)

    reset!(env)
    obs = observe(env)
    hook(PRE_EPISODE_STAGE, agent, env, obs)
    action = agent(PRE_EPISODE_STAGE, obs)
    hook(PRE_ACT_STAGE, agent, env, obs => action)

    while true
        obs = action |> env |> observe
        hook(POST_ACT_STAGE, agent, env, action => obs)

        if stop_condition(agent, env, obs)
            if is_terminal(obs)
                hook(POST_EPISODE_STAGE, agent, env, obs)
            end
            agent(obs)  # let the agent see the last observation
            break
        end

        if is_terminal(obs)
            hook(POST_EPISODE_STAGE, agent, env, obs)
            agent(obs)  # let the agent see the last observation
            reset!(env)
            obs = observe(env)
            hook(PRE_EPISODE_STAGE, agent, env, obs)
        end
    end
end
