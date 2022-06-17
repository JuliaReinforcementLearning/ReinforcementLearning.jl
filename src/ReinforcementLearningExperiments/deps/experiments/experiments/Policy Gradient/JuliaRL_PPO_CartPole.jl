# ---
# title: JuliaRL\_PPO\_CartPole
# cover: assets/JuliaRL_PPO_CartPole.png
# description: PPO applied to CartPole
# date: 2021-05-22
# author: "[Jun Tian](https://github.com/findmyway)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    N_ENV = 8
    UPDATE_FREQ = 32
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = StableRNG(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(state(env[1])), length(action_space(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = PPOPolicy(
            approximator = ActorCritic(
                actor = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na; init = glorot_uniform(rng)),
                ),
                critic = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 1; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(1e-3),
            ) |> gpu,
            γ = 0.99f0,
            λ = 0.95f0,
            clip_range = 0.1f0,
            max_grad_norm = 0.5f0,
            n_epochs = 4,
            n_microbatches = 4,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.001f0,
            update_freq = UPDATE_FREQ,
        ),
        trajectory = PPOTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Int} => (N_ENV,),
            action_log_prob = Vector{Float32} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalBatchRewardPerEpisode(N_ENV)
    Experiment(agent, env, stop_condition, hook, "# PPO with CartPole")
end

#+ tangle=false
using Plots
using Statistics
pyplot() #hide
ex = E`JuliaRL_PPO_CartPole`
run(ex)
n = minimum(map(length, ex.hook.rewards))
m = mean([@view(x[1:n]) for x in ex.hook.rewards])
s = std([@view(x[1:n]) for x in ex.hook.rewards])
plot(m,ribbon=s)
savefig("assets/JuliaRL_PPO_CartPole.png") #hide

# ![](assets/JuliaRL_PPO_CartPole.png)
