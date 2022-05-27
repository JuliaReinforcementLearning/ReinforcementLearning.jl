import Base: run

function run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)
    check(policy, env)
    _run(policy, env, stop_condition, hook)
end

"Inject some customized checkings here by overwriting this function"
function check(policy, env) end

function _run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook)

    hook(PreExperimentStage(), policy, env)
    policy(PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        policy(PreEpisodeStage(), env)
        hook(PreEpisodeStage(), policy, env)

        while !is_terminated(env) # one episode
            policy(PreActStage(), env)
            hook(PreActStage(), policy, env)

            env |> policy |> env
            optimise!(policy)

            policy(PostActStage(), env)
            hook(PostActStage(), policy, env)

            if stop_condition(policy, env)
                is_stop = true
                policy(PreActStage(), env)
                hook(PreActStage(), policy, env)
                policy(env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(PostEpisodeStage(), env)  # let the policy see the last observation
            hook(PostEpisodeStage(), policy, env)
        end
    end
    policy(PostExperimentStage(), env)
    hook(PostExperimentStage(), policy, env)
    hook
end
