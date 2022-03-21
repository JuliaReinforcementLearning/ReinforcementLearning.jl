# ---
# title: JuliaRL\_SAC\_Pendulum
# cover: assets/JuliaRL_SAC_Pendulum.png
# description: SAC applied to Pendulum
# date: 2021-05-22
# author: "[Roman Bange](https://github.com/rbange)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using IntervalSets
using CUDA

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:SAC},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir=nothing,
    seed=123
)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T=Float32, rng=rng)
    action_dims = inner_env.n_actions
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))
    na = 1

    env = ActionTransformedEnv(
        inner_env;
        action_mapping=x -> low + (x[1] + 1) * 0.5 * (high - low)
    )
    init = glorot_uniform(rng)

    create_policy_net() = NeuralNetworkApproximator(
        model=GaussianNetwork(
            pre=Chain(
                Dense(ns, 30, relu, init=init),
                Dense(30, 30, relu, init=init),
            ),
            μ=Chain(Dense(30, na, init=init)),
            logσ=Chain(Dense(30, na, x -> clamp(x, typeof(x)(-10), typeof(x)(2)), init=init)),
        ),
        optimizer=ADAM(0.003),
    ) |> gpu

    create_q_net() = NeuralNetworkApproximator(
        model=Chain(
            Dense(ns + na, 30, relu; init=init),
            Dense(30, 30, relu; init=init),
            Dense(30, 1; init=init),
        ),
        optimizer=ADAM(0.003),
    ) |> gpu

    agent = Agent(
        policy=SACPolicy(
            policy=create_policy_net(),
            qnetwork1=create_q_net(),
            qnetwork2=create_q_net(),
            target_qnetwork1=create_q_net(),
            target_qnetwork2=create_q_net(),
            γ=0.99f0,
            τ=0.005f0,
            α=0.2f0,
            batch_size=64,
            start_steps=1000,
            start_policy=RandomPolicy(Space([-1.0 .. 1.0 for _ in 1:na]); rng=rng),
            update_after=1000,
            update_freq=1,
            automatic_entropy_tuning=true,
            lr_alpha=0.003f0,
            action_dims=action_dims,
            rng=rng,
            device_rng=CUDA.functional() ? CUDA.CURAND.RNG() : rng
        ),
        trajectory=CircularArraySARTTrajectory(
            capacity=10000,
            state=Vector{Float32} => (ns,),
            action=Vector{Float32} => (na,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "# Play Pendulum with SAC")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_SAC_Pendulum`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_SAC_Pendulum.png") #hide

# ![](assets/JuliaRL_SAC_Pendulum.png)
