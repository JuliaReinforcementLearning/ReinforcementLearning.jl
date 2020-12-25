include("multi_thread_env.jl")

"""
Many policy gradient based algorithms require that the `env` is a
`MultiThreadEnv` to increase the diversity during training. So the training
pipeline is different from the default one in `RLCore`.
"""
function RLCore._run(
    ::Sequential,
    ::SingleAgent,
    policy::AbstractPolicy,
    env::MultiThreadEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    while true
        reset!(env)  # this is a soft reset!, only environments reached the end will get reset.
        action = policy(PRE_ACT_STAGE, env)
        hook(PRE_ACT_STAGE, policy, env, action)

        env(action)
        policy(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, policy, env)

        if stop_condition(policy, env)
            policy(PRE_ACT_STAGE, env)  # let the policy see the last observation
            break
        end
    end
    hook
end
