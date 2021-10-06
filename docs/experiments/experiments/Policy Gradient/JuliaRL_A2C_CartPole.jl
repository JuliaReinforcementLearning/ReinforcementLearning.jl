# ---
# title: JuliaRL\_A2C\_CartPole
# cover: assets/JuliaRL_A2C_CartPole.png
# description: A2C applied to CartPole
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
    ::Val{:A2C},
    ::Val{:CartPole},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = StableRNG(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(state(env[1])), length(action_space(env[1]))
    RLBase.reset!(env, is_force = true)
    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
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
                ) |> cpu,
                γ = 0.99f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress = true)
    hook = TotalBatchRewardPerEpisode(N_ENV)
    Experiment(agent, env, stop_condition, hook, "# A2C with CartPole")
end

#+ tangle=false
using Plots
using Statistics
pyplot() #hide
ex = E`JuliaRL_A2C_CartPole`
run(ex)
n = minimum(map(length, ex.hook.rewards))
m = mean([@view(x[1:n]) for x in ex.hook.rewards])
s = std([@view(x[1:n]) for x in ex.hook.rewards])
plot(m, ribbon = s)
savefig("assets/JuliaRL_A2C_CartPole.png") #hide

# ![](assets/JuliaRL_A2C_CartPole.png)
