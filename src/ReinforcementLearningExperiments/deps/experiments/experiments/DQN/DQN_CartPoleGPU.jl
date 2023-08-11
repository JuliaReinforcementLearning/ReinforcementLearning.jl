# ---
# title: JuliaRL\_DQNCartPole\_GPU
# cover:
# description: DQN applied to CartPole on GPU
# date: 2023-07-24
# author: "[Panajiotis Keßler](mailto:panajiotis@christoforidis.net)"
# ---

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:DQNCartPole},
    ::Val{:GPU},
    seed=123,
    cap = 100,
    n=12,
    γ=0.99f0
)
    rng = StableRNG(seed)
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    policy = Agent(
        QBasedPolicy(
            learner=DQNLearner(
                approximator=TargetNetwork(
                    Approximator(
                        Chain(
                            Dense(ns, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, 128, relu; init=glorot_uniform(rng)),
                            Dense(128, na; init=glorot_uniform(rng)),
                            ),
                        optimiser=Adam()
                        ),
                    sync_freq=100
                ),
                n=n,
                γ=γ,
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
        Trajectory(
                                   container=CircularArraySARTSTraces(
                                     capacity=cap,
                                     state=Float32 => (ns),
                                   ),
                                   sampler=NStepBatchSampler{SS′ART}(
                                       n=n,
                                       γ=0.99f0,
                                       batch_size=32,
                                       rng=rng
                                   ),
                                   controller=InsertSampleRatioController(
                                       threshold=ceil(1.1*n),
                                       n_inserted=0
                                   )),
    )
    stop_condition = StopAfterEpisode(5, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(policy, env, stop_condition, hook)
end