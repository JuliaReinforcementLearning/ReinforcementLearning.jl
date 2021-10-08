# ---
# title: JuliaRL\_FQE\_Pendulum
# cover: assets/logo.svg 
# description: FQE applied to CRR policy on PendulumEnv
# date: 2021-9-29
# author: "[Mobius1D](https://github.com/Mobius1D)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:FQE},
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
    
    dataset=gen_JuliaRL_dataset(:SAC, :Pendulum, type; dataset_size = dataset_size)
    
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

    crr_agent = Agent(
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
            dataset = dataset,
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
    hook = EmptyHook()
    run(crr_agent, env, stop_condition, hook)
    
    crr_policy = crr_agent.policy.learner.approximator.actor

    create_fqe_q_net() = NeuralNetworkApproximator(
        model = Chain(
            Dense(ns + na, 64, relu; init = init),
            Dense(64, 64, relu; init = init),
            Dense(64, 1; init = init),
        ),
        optimizer = ADAM(0.003),
    )
    
    fqe = Agent(
        policy = OfflinePolicy(
            learner = FQE(
                policy=crr_policy |> cpu,
                q_network = create_fqe_q_net() |> cpu,
                target_q_network = create_fqe_q_net() |> cpu,
                n_evals = 50,
                γ = 0.99f0,
                batch_size = batch_size,
                update_freq=1,
                update_step=1,
                tar_update_freq=50,
                rng=rng,
            ),
            dataset = dataset,
            continuous = true,
            batch_size = batch_size,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10000,
            state = Vector{Float32} => (ns,),
            action = Vector{Float32} => (na,),
        ),
    )
    stop_condition = StopAfterStep(trajectory_num, is_show_progress=!haskey(ENV, "CI"))
    Experiment(fqe, env, stop_condition, hook, "FQE <-> CRR <-> Pendulum ($type dataset)")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_FQE_Pendulum(medium)`
run(ex)
mean, rewards = ex.policy.policy.learner(ex.env, Val(:Eval))
@info mean, rewards
plot(rewards)
savefig("assets/JuliaRL_FQE_Pendulum.png") #hide

# ![](assets/JuliaRL_FQE_Pendulum.png)