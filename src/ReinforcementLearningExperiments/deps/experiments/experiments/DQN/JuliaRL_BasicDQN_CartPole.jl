# ---
# title: JuliaRL\_BasicDQN\_CartPole
# cover: assets/JuliaRL_BasicDQN_CartPole.png
# description: The simplest example to demonstrate how to use BasicDQN
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
    ::Val{:BasicDQN},
    ::Val{:CartPole},
    ::Nothing;
    seed=123
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    policy = Agent(
        policy=QBasedPolicy(
            learner=BasicDQNLearner(
                approximator=NeuralNetworkApproximator(
                    model=Chain(
                        Dense(ns, 128, relu; init=glorot_uniform(rng)),
                        Dense(128, 128, relu; init=glorot_uniform(rng)),
                        Dense(128, na; init=glorot_uniform(rng)),
                    ) |> gpu,
                    optimizer=ADAM(),
                ),
                loss_func=huber_loss,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTTrajectory(
                capacity=1000,
                state=Vector{Float32} => (ns,),
            ),
            sampler=BatchSampler{(:state, :action, :reward, :terminal, :next_state)}(
                batch_size=32
            ),
            controller=AsyncInsertSampleRatioController(
                threshold=100
            )
        )
    )
    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(policy, env, stop_condition, hook, "# BasicDQN <-> CartPole")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_BasicDQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_BasicDQN_CartPole.png") #hide

#=
## Watch a demo episode with the trained agent

```julia
demo = Experiment(ex.policy,
                  CartPoleEnv(),
                  StopWhenDone(),
                  RolloutHook(plot, closeall),
                  "DQN <-> Demo")
run(demo)
```
=#

# ![](assets/JuliaRL_BasicDQN_CartPole.png)
