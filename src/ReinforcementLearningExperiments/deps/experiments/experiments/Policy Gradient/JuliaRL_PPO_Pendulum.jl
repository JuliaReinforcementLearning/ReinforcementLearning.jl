# ---
# title: JuliaRL\_PPO\_Pendulum
# cover: assets/JuliaRL_PPO_Pendulum.png
# description: PPO applied to Pendulum
# date: 2021-05-22
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Distributions

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right
    ns = length(state(inner_env))

    N_ENV = 8
    UPDATE_FREQ = 2048
    env = MultiThreadEnv([
        PendulumEnv(T = Float32, rng = StableRNG(hash(seed + i))) |>
        env -> ActionTransformedEnv(env, action_mapping = x -> clamp(x * 2, low, high)) for i in 1:N_ENV
    ])

    init = glorot_uniform(rng)

    agent = Agent(
        policy = PPOPolicy(
            approximator = ActorCritic(
                actor = GaussianNetwork(
                    pre = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                    ),
                    μ = Chain(Dense(64, 1, tanh; init = glorot_uniform(rng)), vec),
                    logσ = Chain(Dense(64, 1; init = glorot_uniform(rng)), vec),
                ),
                critic = Chain(
                    Dense(ns, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 1; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(3e-4),
            ) |> gpu,
            γ = 0.99f0,
            λ = 0.95f0,
            clip_range = 0.2f0,
            max_grad_norm = 0.5f0,
            n_epochs = 10,
            n_microbatches = 32,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.00f0,
            dist = Normal,
            rng = rng,
            update_freq = UPDATE_FREQ,
        ),
        trajectory = PPOTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Float32} => (N_ENV,),
            action_log_prob = Vector{Float32} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalBatchRewardPerEpisode(N_ENV)
    Experiment(agent, env, stop_condition, hook, "# Play Pendulum with PPO")
end

#+ tangle=false
using Plots
using Statistics
pyplot() #hide
ex = E`JuliaRL_PPO_Pendulum`
run(ex)
n = minimum(map(length, ex.hook.rewards))
m = mean([@view(x[1:n]) for x in ex.hook.rewards])
s = std([@view(x[1:n]) for x in ex.hook.rewards])
plot(m,ribbon=s)
savefig("assets/JuliaRL_PPO_Pendulum.png") #hide

# ![](assets/JuliaRL_PPO_Pendulum.png)
