abstract type NFSPAgents <: MultiAgentManager end

include("average_learner.jl")
include("nfsp.jl")

function Base.run(
    p::NFSPAgents,
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
        hook(PRE_EPISODE_STAGE, policy, env)

        while !is_terminated(env) # one episode
            RLBase.update!(p, env)
            hook(POST_ACT_STAGE, p, env)

            if stop_condition(p, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)  # let the policy see the last observation
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
    hook
end