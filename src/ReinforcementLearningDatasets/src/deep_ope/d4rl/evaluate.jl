export deep_ope_d4rl_evaluate

using ReinforcementLearningBase
using ReinforcementLearningEnvironments
using PyCall
using UnicodePlots
using Random
using ProgressMeter

"""
    deep_ope_d4rl_evaluate(env_name, agent, epoch; <keyword arguments>)

Return the `UnicodePlot` for the `env_name`, `agent`, `epoch` that is given. Provide `gym_env_name` for specifying the environment explicitly.
`γ` is the discount factor which defaults to 1. Seed of the env can be provided in `env_seed`. 
"""
function deep_ope_d4rl_evaluate(
    env_name::String,
    agent::String,
    epoch::Int;
    gym_env_name::Union{String, Nothing}=nothing,
    rng::AbstractRNG=MersenneTwister(123),
    num_evaluations::Int=10,
    γ::Float64=1.0,
    noisy::Bool=false,
    env_seed::Union{Int, Nothing}=nothing
)   
    policy_folder = "$(env_name)_$(agent)_$(epoch)"

    if gym_env_name === nothing
        for policy in D4RL_POLICIES
            policy_file = split(policy["policy_path"], "/")[end]
            if chop(policy_file, head=0, tail=4) == policy_folder
                gym_env_name = policy["task.task_names"][1]
                break
            end
        end

        if gym_env_name === nothing error("invalid parameters") end
    end

    env = GymEnv(gym_env_name; seed=env_seed)

    model = d4rl_policy(env_name, agent, epoch)
    scores = Vector{Float64}(undef, num_evaluations)

    @showprogress for eval in 1:num_evaluations
        score = 0
        reset!(env)
        while !is_terminated(env)
            s = state(env)
            a = model(s;rng=rng, noisy=noisy)[1]
            s, a , env(a)
            r = reward(env)
            t = is_terminated(env)
            score += r*γ*(1-t)
        end
        scores[eval] = score
    end
    plt = lineplot(1:length(scores), scores, title = "$(gym_env_name) scores", name = "scores", xlabel = "episode", canvas = DotCanvas, ylabel = "score", border=:ascii)
    plt
end