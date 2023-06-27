using TimerOutputs

function _run_with_time(to::TimerOutput,
        policy::AbstractPolicy,
        env::AbstractEnv,
        stop_condition::AbstractStopCondition,
        hook::AbstractHook,
        reset_condition::AbstractResetCondition)
    push!(hook, PreExperimentStage(), policy, env)
    push!(policy, PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        @timeit to "reset!" reset!(env)
        @timeit to "push!(policy) PreEpisodeStage" push!(policy, PreEpisodeStage(), env)
        @timeit to "optimise! PreEpisodeStage" optimise!(policy, PreEpisodeStage())
        @timeit to "push!(hook) PreEpisodeStage" push!(hook, PreEpisodeStage(), policy, env)


        while !reset_condition(policy, env) # one episode
            @timeit to "push!(policy) PreActStage" push!(policy, PreActStage(), env)
            @timeit to "optimise! PreActStage" optimise!(policy, PreActStage())
            @timeit to "push!(hook) PreActStage" push!(hook, PreActStage(), policy, env)

            action = @timeit to "plan!" RLBase.plan!(policy, env)
            @timeit to "act!" act!(env, action)

            @timeit to "push!(policy) PostActStage" push!(policy, PostActStage(), env)
            @timeit to "optimise! PostActStage" optimise!(policy, PostActStage())
            @timeit to "push!(hook) PostActStage" push!(hook, PostActStage(), policy, env)

            if check_stop(stop_condition, policy, env)
                is_stop = true
                @timeit to "push!(policy) PreActStage" push!(policy, PreActStage(), env)
                @timeit to "optimise! PreActStage" optimise!(policy, PreActStage())
                @timeit to "push!(hook) PreActStage" push!(hook, PreActStage(), policy, env)
                @timeit to "plan!" RLBase.plan!(policy, env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        @timeit to "push!(policy) PostEpisodeStage" push!(policy, PostEpisodeStage(), env)  # let the policy see the last observation
        @timeit to "optimise! PostEpisodeStage" optimise!(policy, PostEpisodeStage())
        @timeit to "push!(hook) PostEpisodeStage" push!(hook, PostEpisodeStage(), policy, env)

    end
    push!(policy, PostExperimentStage(), env)
    push!(hook, PostExperimentStage(), policy, env)
    show(to)
    hook
end
