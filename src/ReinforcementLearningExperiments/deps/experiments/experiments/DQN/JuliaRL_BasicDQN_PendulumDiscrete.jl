# ---
# title: JuliaRL\_BasicDQN\_PendulumDiscrete
# cover: assets/JuliaRL_BasicDQN_PendulumDiscrete.png
# description: BasicDQN can also be applied to discrete Pendulum
# date: 2021-10-20
# author: "[Harley Wiltzer](https://github.com/harwiltz)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:PendulumDiscrete},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    env = PendulumEnv(continuous = false, max_steps = 5000, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, 64, relu; init = glorot_uniform(rng)),
                        Dense(64, na; init = glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 5_000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_BasicDQN_PendulumDiscrete`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_BasicDQN_PendulumDiscrete.png") #hide

#=
## Watch a demo episode with the trained agent

```julia
demo = Experiment(ex.policy,
                  PendulumEnv(continuous=false, max_steps = 1000),
                  StopWhenDone(),
                  RolloutHook(plot, closeall),
                  "DQN <-> Demo")
run(demo)
```
=#

# ![](assets/JuliaRL_BasicDQN_PendulumDiscrete.png)
