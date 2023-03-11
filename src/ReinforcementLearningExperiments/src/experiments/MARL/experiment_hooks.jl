using ReinforcementLearning

export MeanRewardHook, MeanSTDRewardHook, printMeanRewardhook

#  =========================================================================================================================================================


mutable struct MeanRewardHook <: AbstractHook
    episode::Int
    eval_rate::Int
    eval_episode::Int
    episodes::Vector
    rewards::Vector
end

mutable struct MeanSTDRewardHook <: AbstractHook
    episode::Int
    eval_rate::Int
    eval_episode::Int
    episodes::Vector
    rewards::Vector
    std::Vector
end

function (hook::MeanRewardHook)(::PostEpisodeStage, policy, env)
    if hook.episode % hook.eval_rate == 0
        # evaluate policy's performance
        rew = 0
        for _ in 1:hook.eval_episode
            reset!(env)
            while !is_terminated(env)
                env |> policy |> env
                rew += reward(env)
            end
        end

        push!(hook.episodes, hook.episode)
        push!(hook.mean_rewards, rew / hook.eval_episode)
    end
    hook.episode += 1
end

function (hook::MeanSTDRewardHook)(::PostEpisodeStage, policy, env)
    if hook.episode % hook.eval_rate == 0
        # evaluate policy's performance
        rew = Float32[]
        for _ in 1:hook.eval_episode
            reset!(env)
            loc_rew = 0
            while !is_terminated(env)
                env |> policy |> env
                loc_rew += reward(env)
            end
            push!(rew, loc_rew)
        end

        push!(hook.episodes, hook.episode)
        push!(hook.rewards, Statistics.mean(rew))
        push!(hook.std, Statistics.std(rew))
    end
    hook.episode += 1
end

printMeanRewardhook = DoEveryNEpisode(;n=1000) do t, policy, env
    # In real world cases, the policy is usually wrapped in an Agent,
    # we need to extract the inner policy to run it in the *actor* mode.
    # Here for illustration only, we simply use the origina policy.

    # Note that we create a new instance of CartPoleEnv here to avoid
    # polluting the original env.

    hook = TotalRewardPerEpisode(;is_display_on_exit=false)
    run(policy, ReinforcementLearning.PettingzooEnv("mpe.simple_spread_v2"; seed=123, continuous_actions=true), StopAfterEpisode(10), hook)

    # now you can report the result of the hook.
    println("\navg reward at episode $t is: $(mean(hook.rewards))")
end