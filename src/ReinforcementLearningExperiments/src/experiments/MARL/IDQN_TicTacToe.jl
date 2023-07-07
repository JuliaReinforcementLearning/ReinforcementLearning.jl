# ---
# title: JuliaRL\_IDQN\_TicTacToe
# cover:
# description: IDQN applied to TicTacToe competitive
# date: 2023-07-03
# author: "[Panajiotis Keßler](mailto:panajiotis.kessler@gmail.com)"
# ---

using StableRNGs
using ReinforcementLearning
using ReinforcementLearningBase
using ReinforcementLearningZoo
using ReinforcementLearningCore
using Plots
using Flux
using Flux.Losses: huber_loss
using Flux: glorot_uniform

using ProgressMeter


rng = StableRNG(1234)

cap = 100

RLCore.forward(L::DQNLearner, state::A) where {A <: Real} = RLCore.forward(L, [state])

create_policy() = QBasedPolicy(
        learner=DQNLearner(
            approximator=Approximator(
                model=TwinNetwork(
                    Chain(
                        Dense(1, 512, relu; init=glorot_uniform(rng)),
                        Dense(512, 256, relu; init=glorot_uniform(rng)),
                        Dense(256, 9; init=glorot_uniform(rng)),
                    );
                    sync_freq=100
                ),
                optimiser=ADAM(),
            ),
            n=32,
            γ=0.99f0,
            is_enable_double_DQN=true,
            loss_func=huber_loss,
            rng=rng,
        ),
        explorer=EpsilonGreedyExplorer(
            kind=:exp,
            ϵ_stable=0.01,
            decay_steps=500,
            rng=rng,
        ),
    )

e  = TicTacToeEnv();
m = MultiAgentPolicy(NamedTuple((player =>
                           Agent(player != :Cross ? create_policy() : RandomPolicy(;rng=rng),
                               Trajectory(
                                   container=CircularArraySARTTraces(
                                     capacity=cap,
                                     state=Integer => (1,),
                                   ),
                                   sampler=NStepBatchSampler{SS′ART}(
                                       n=1,
                                       γ=0.99f0,
                                       batch_size=1,
                                       rng=rng
                                   ),
                                   controller=InsertSampleRatioController(
                                       threshold=1,
                                       n_inserted=0
                                   ))
                           )
                           for player in players(e)))
                       );
hooks = MultiAgentHook(NamedTuple((p => TotalRewardPerEpisode() for p ∈ players(e))))

episodes_per_step = 25
win_rates = (Cross=Float64[], Nought=Float64[])
@showprogress for i ∈ 1:2
    run(m, e, StopAfterEpisode(episodes_per_step; is_show_progress=false), hooks)
    wr_cross = sum(hooks[:Cross].rewards)/(i*episodes_per_step)
    wr_nought = sum(hooks[:Nought].rewards)/(i*episodes_per_step)
    push!(win_rates[:Cross], wr_cross)
    push!(win_rates[:Nought], wr_nought)
end
p1 = plot([win_rates[:Cross] win_rates[:Nought]], labels=["Cross" "Nought"])
xlabel!("Iteration steps of $episodes_per_step episodes")
ylabel!("Win rate of the player")

p2 = plot([hooks[:Cross].rewards hooks[:Nought].rewards], labels=["Cross" "Nought"])
xlabel!("Overall episodes")
ylabel!("Rewards of the players")

p = plot(p1, p2, layout=(2,1), size=[1000,1000])
savefig("TTT_CROSS_DQN_NOUGHT_RANDOM.png")
