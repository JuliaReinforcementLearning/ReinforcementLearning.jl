
# ---
# title: JuliaRL\_IQN\_CartPole
# cover: assets/JuliaRL_IQN_CartPole.png
# description: IQN applied to CartPole
# date: 2022-06-27
# author: "[Jun Tian](https://github.com/findmyway)"
# ---


using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using ReinforcementLearningEnvironments
using StableRNGs
using Flux
using Flux.Losses

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:IQN},
    ::Val{:CartPole},
    ; seed=123
)
    rng = StableRNG(seed)
    device_rng = rng
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))
    init = glorot_uniform(rng)
    Nₑₘ = 16
    n_hidden = 64
    κ = 1.0f0

    nn_creator() =
        ImplicitQuantileNet(
            ψ=Dense(ns, n_hidden, relu; init=init),
            ϕ=Dense(Nₑₘ, n_hidden, relu; init=init),
            header=Dense(n_hidden, na; init=init),
        )

    agent = Agent(
        policy=QBasedPolicy(
            learner=IQNLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        ImplicitQuantileNet(
                            ψ=Dense(ns, n_hidden, relu; init=init),
                            ϕ=Dense(Nₑₘ, n_hidden, relu; init=init),
                            header=Dense(n_hidden, na; init=init),
                        ),
                        sync_freq=100
                    ),
                    optimiser=Adam(0.001),
                ),
                κ=κ,
                N=8,
                N′=8,
                Nₑₘ=Nₑₘ,
                K=32,
                γ=0.99f0,
                rng=rng,
                device_rng=device_rng,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARSTTraces(
                capacity=1000,
                state=Float32 => (ns,),
            ),
            sampler=BatchSampler{SS′ART}(
                batch_size=32,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=100,
                n_inserted=-1
            )
        )
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end


#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_IQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_IQN_CartPole.png") #hide

# ![](assets/JuliaRL_IQN_CartPole.png)
