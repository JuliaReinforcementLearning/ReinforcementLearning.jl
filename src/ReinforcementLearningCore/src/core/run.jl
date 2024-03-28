export Experiment

"""
    Experiment(policy::AbstractPolicy, env::AbstractEnv, stop_condition::AbstractStopCondition, hook::AbstractHook)

A struct to hold the information of an experiment. It is used to run an experiment with the given policy, environment, stop condition and hook.
"""
struct Experiment
    policy::AbstractPolicy
    env::AbstractEnv
    stop_condition::AbstractStopCondition
    hook::AbstractHook
end

Base.show(io::IO, m::MIME"text/plain", t::Experiment) = show(io, m, convert(AnnotatedStructTree, t))

function Base.run(ex::Experiment)
    run(ex.policy, ex.env, ex.stop_condition, ex.hook)
    return ex
end

function Base.run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition::AbstractStopCondition=StopAfterNEpisodes(1),
    hook::AbstractHook=EmptyHook(),
    reset_condition::AbstractResetCondition=ResetIfEnvTerminated()
)
    policy, env = check(policy, env)
    _run(policy, env, stop_condition, hook, reset_condition)
end

"Inject some customized checkings here by overwriting this function"
check(policy, env) = policy, env

function _run(policy::AbstractPolicy,
        env::AbstractEnv,
        stop_condition::AbstractStopCondition,
        hook::AbstractHook,
        reset_condition::AbstractResetCondition)
    push!(hook, PreExperimentStage(), policy, env)
    push!(policy, PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        # NOTE: @timeit_debug statements are used for debug logging
        @timeit_debug timer "reset!"                            reset!(env)
        @timeit_debug timer "push!(policy) PreEpisodeStage"     push!(policy, PreEpisodeStage(), env)
        @timeit_debug timer "optimise! PreEpisodeStage"         optimise!(policy, PreEpisodeStage())
        @timeit_debug timer "push!(hook) PreEpisodeStage"       push!(hook, PreEpisodeStage(), policy, env)


        while !check!(reset_condition, policy, env) # one episode
            @timeit_debug timer "push!(policy) PreActStage"     push!(policy, PreActStage(), env)
            @timeit_debug timer "optimise! PreActStage"         optimise!(policy, PreActStage())
            @timeit_debug timer "push!(hook) PreActStage"       push!(hook, PreActStage(), policy, env)

            action = @timeit_debug timer "plan!"                RLBase.plan!(policy, env)
            @timeit_debug timer "act!"                          act!(env, action)

            @timeit_debug timer "push!(policy) PostActStage"    push!(policy, PostActStage(), env, action)
            @timeit_debug timer "optimise! PostActStage"        optimise!(policy, PostActStage())
            @timeit_debug timer "push!(hook) PostActStage"      push!(hook, PostActStage(), policy, env)

            if check!(stop_condition, policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        @timeit_debug timer "push!(policy) PostEpisodeStage"      push!(policy, PostEpisodeStage(), env)
        @timeit_debug timer "optimise! PostEpisodeStage"          optimise!(policy, PostEpisodeStage())
        @timeit_debug timer "push!(hook) PostEpisodeStage"        push!(hook, PostEpisodeStage(), policy, env)

    end
    push!(policy, PostExperimentStage(), env)
    push!(hook, PostExperimentStage(), policy, env)
    hook
end
