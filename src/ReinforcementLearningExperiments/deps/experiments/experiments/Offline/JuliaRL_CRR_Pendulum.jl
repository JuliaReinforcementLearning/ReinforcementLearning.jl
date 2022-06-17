# ---
# title: JuliaRL\_CRR\_Pendulum
# cover: assets/JuliaRL_CRR_Pendulum_medium.png
# description: CRR applied to Pendulum
# date: 2021-09-17
# author: "[Guoyu Yang](https://github.com/pilgrimygy)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:CRR},
    ::Val{:Pendulum},
    type::AbstractString;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))
    na = 1

    trajectory_num = 10000
    dataset_size = 10000
    batch_size = 128

    env = ActionTransformedEnv(
        inner_env;
        action_mapping = x -> low + (x[1] + 1) * 0.5 * (high - low),
    )
    init = glorot_uniform(rng)

    create_policy_net() = GaussianNetwork(
        pre = Chain(
            Dense(ns, 64, relu), 
            Dense(64, 64, relu),
        ),
        μ = Chain(Dense(64, na, init = init)),
        logσ = Chain(Dense(64, na, init = init)),
    )

    create_q_net() = Chain(
        Dense(ns + na, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
    )

    agent = Agent(
        policy = OfflinePolicy(
            learner = CRRLearner(
                approximator = ActorCritic(
                    actor = create_policy_net() |> cpu,
                    critic = create_q_net() |> cpu,
                    optimizer = ADAM(3e-3),
                ),
                target_approximator = ActorCritic(
                    actor = create_policy_net() |> cpu,
                    critic = create_q_net() |> cpu,
                    optimizer = ADAM(3e-3),
                ),
                γ = 0.99f0,
                batch_size = batch_size,
                policy_improvement_mode = :exp,
                ratio_upper_bound = 20.0f0,
                β = 1.0f0,
                advantage_estimator = :mean,
                m = 4,
                update_freq = 1,
                continuous = true,
            ),
            dataset = gen_JuliaRL_dataset(:SAC, :Pendulum, type; dataset_size = dataset_size),
            continuous = true,
            batch_size = batch_size,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na,),
        ),
    )

    stop_condition = StopAfterStep(trajectory_num, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "CRR <-> Pendulum ($type dataset)")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_CRR_Pendulum(medium)`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_CRR_Pendulum_medium.png") #hide

# ![](assets/JuliaRL_CRR_Pendulum.png)
