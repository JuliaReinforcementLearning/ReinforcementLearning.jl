include("nfsp.jl")
include("nfsp_manager.jl")


function Base.run(
    policy::NFSPAgentManager,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DefaultStateStyle(env) isa InformationSet

    is_stop = false

    while !is_stop
        RLBase.reset!(env)
        policy(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, policy, env)

        while !is_terminated(env) # one episode
            update!(policy, env) # update policy and env simultaneously.
            hook(POST_ACT_STAGE, policy, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
    hook
end