include("multi_thread_env.jl")

"""
Many policy gradient based algorithms require that the `env` is a
`MultiThreadEnv` to increase the diversity during training. So the training
pipeline is different from the default one in `RLCore`.
"""
function RLCore._run(
    policy::AbstractPolicy,
    env::MultiThreadEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    while true
        reset!(env)  # this is a soft reset!, only environments reached the end will be reset.
        action = policy(env)
        policy(PRE_ACT_STAGE, env, action)
        hook(PRE_ACT_STAGE, policy, env, action)

        env(action)
        policy(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, policy, env)

        if stop_condition(policy, env)
            break
        end
    end
    action = policy(env)
    policy(PRE_ACT_STAGE, env, action)  # let the policy see the last observation
    hook(PRE_ACT_STAGE, policy, env, action)
    hook(POST_EXPERIMENT_STAGE, policy, env)
    nothing
end
