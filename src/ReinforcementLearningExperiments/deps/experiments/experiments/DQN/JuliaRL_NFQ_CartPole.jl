# ---
# title: JuliaRL\_NFQ\_CartPole
# cover: assets/JuliaRL_BasicDQN_CartPole.png
# description: NFQ applied to the cartpole environment
# date: 2023-06
# author: "[Lucas Bex](https://github.com/CasBex)"
# ---

#+ tangle=true
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using Flux
using Flux: glorot_uniform

using StableRNGs: StableRNG
using Flux.Losses: mse

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:NFQ},
    ::Val{:CartPole},
    seed = 123,
)

    rng = StableRNG(seed)
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=NFQ(
                approximator=Approximator(
                    model=Chain(
                        Dense(ns, 32, σ; init=glorot_uniform(rng)),
                        Dense(32, 32, relu; init=glorot_uniform(rng)),
                        Dense(32, na; init=glorot_uniform(rng)),
                    ),
                    optimiser=RMSProp(),
                ),
                loss_function=mse,
                epochs=10,
                num_iterations=10,
                γ = 0.95f0
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.001,
                warmup_steps=1000,
                decay_steps=3000,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTSTraces(
                capacity=10_000,
                state=Float32 => (ns,),
            ),
            sampler=BatchSampler{SS′ART}(
                batch_size=2048,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=1000,
                ratio=1/10,
                n_sampled=-1
            )
        )
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    Experiment(agent, env, stop_condition, hook)

    end

#+ tangle=false
using Plots
# pyplot() # hide
ex = E`JuliaRL_NFQ_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_NFQ_CartPole.png") #hide

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

# ![](assets/JuliaRL_NFQ_CartPole.png)
