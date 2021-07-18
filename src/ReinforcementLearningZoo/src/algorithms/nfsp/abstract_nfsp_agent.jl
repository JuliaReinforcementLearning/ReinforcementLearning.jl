include("average_learner.jl")
include("nfsp.jl")
include("nfsp_manager.jl")


function Base.run(
    nfsp::NFSPAgentManager,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DynamicStyle(env) === SEQUENTIAL
    @assert RewardStyle(env) === TERMINAL_REWARD
    @assert ChanceStyle(env) === EXPLICIT_STOCHASTIC
    @assert DefaultStateStyle(env) isa InformationSet

    is_stop = false

    while !is_stop
        RLBase.reset!(env)
        hook(PRE_EPISODE_STAGE, nfsp, env)

        while !is_terminated(env) # one episode
            RLBase.update!(nfsp, env)
            hook(POST_ACT_STAGE, nfsp, env)

            if stop_condition(nfsp, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            hook(POST_EPISODE_STAGE, nfsp, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, nfsp, env)
    hook
end