# ---
# title: JuliaRL\_DQN\_MPESimple
# cover:
# description: DQN applied to MPE simple
# date: 2023-02-01
# author: "[Panajiotis Keßler](mailto:panajiotis@christoforidis.net)"
# ---

using PyCall
using ReinforcementLearningCore, ReinforcementLearningBase, ReinforcementLearningZoo
using Flux
using Flux: glorot_uniform

using StableRNGs: StableRNG
using Flux.Losses: huber_loss

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQN},
    ::Val{:MPESimple};
    seed=123,
    n=1,
    γ=0.99f0,
    is_enable_double_DQN=true
)
    rng = StableRNG(seed)
    env = discrete2standard_discrete(PettingzooEnv("mpe.simple_v2"; seed=seed))
    ns, na = length(state(env)), length(action_space(env))

    agent = Agent(
        policy=QBasedPolicy(
            learner=DQNLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        Chain(
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, na; init=glorot_uniform(rng)),
                        );
                        sync_freq=100
                    ),
                    optimiser=Adam(),
                ),
                n=n,
                γ=γ,
                is_enable_double_DQN=is_enable_double_DQN,
                loss_func=huber_loss,
                rng=rng,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTTraces(
                capacity=1000,
                state=Float32 => (ns,),
            ),
            sampler=NStepBatchSampler{SS′ART}(
                n=n,
                γ=γ,
                batch_size=32,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=100,
                n_inserted=-1
            )
        )
    )

    stop_condition = StopAfterEpisode(150, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end

using Plots
ex = E`JuliaRL_DQN_MPESimple`
run(ex)
plot(ex.hook.rewards)
savefig("JuliaRL_DQN_MPESimple.png")

