# ---
# title: JuliaRL\_MAC\_CartPole
# cover: assets/JuliaRL_MAC_CartPole.png
# description: MAC applied to CartPole
# date: 2021-05-22
# author: "[Raj Ghugare](https://github.com/RajGhugare19)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:MAC},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    N_ENV = 16
    UPDATE_FREQ = 20
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = StableRNG(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(state(env[1])), length(action_space(env[1]))
    RLBase.reset!(env, is_force = true)

    agent = Agent(
        policy = QBasedPolicy(
            learner = MACLearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 30, relu; init = glorot_uniform(rng)),
                            Dense(30, 30, relu; init = glorot_uniform(rng)),
                            Dense(30, na; init = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(1e-2),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 30, relu; init = glorot_uniform(rng)),
                            Dense(30, 30, relu; init = glorot_uniform(rng)),
                            Dense(30, na; init = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(3e-3),
                    ),
                ) |> cpu,
                γ = 0.99f0,
                bootstrap = true,
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),#= seed = nothing =#
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress = !haskey(ENV, "CI"))
    hook = TotalBatchRewardPerEpisode(N_ENV)
    Experiment(agent, env, stop_condition, hook, "# MAC with CartPole")
end

#+ tangle=false
using Plots
using Statistics
pyplot() #hide
ex = E`JuliaRL_MAC_CartPole`
run(ex)
n = minimum(map(length, ex.hook.rewards))
m = mean([@view(x[1:n]) for x in ex.hook.rewards])
s = std([@view(x[1:n]) for x in ex.hook.rewards])
plot(m, ribbon = s)
savefig("assets/JuliaRL_MAC_CartPole.png") #hide

# ![](assets/JuliaRL_MAC_CartPole.png)
