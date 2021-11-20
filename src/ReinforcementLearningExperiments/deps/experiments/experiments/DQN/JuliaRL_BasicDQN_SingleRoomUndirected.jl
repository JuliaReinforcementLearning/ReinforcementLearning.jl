# ---
# title: JuliaRL\_BasicDQN\_SingleRoomUndirected
# cover: assets/JuliaRL_BasicDQN_SingleRoomUndirected.png
# description: A simple example to demonstrate how to use environments in GridWorlds.jl
# date: 2021-07-27
# author: "[Siddharth Bhatia](https://github.com/Sid-Bhatia-0)"
# ---

#+ tangle=true
using ReinforcementLearning
using GridWorlds
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:BasicDQN},
    ::Val{:SingleRoomUndirected},
    ::Nothing;
    seed=123,
)
    rng = StableRNG(seed)

    env = GridWorlds.SingleRoomUndirectedModule.SingleRoomUndirected(rng=rng)
    env = GridWorlds.RLBaseEnv(env)
    env = RLEnvs.StateTransformedEnv(env; state_mapping=x -> vec(Float32.(x)))
    env = RewardTransformedEnv(env; reward_mapping = x -> x - convert(typeof(x), 0.01))
    env = MaxTimeoutEnv(env, 240)

    ns, na = length(state(env)), length(action_space(env))
    agent = Agent(
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
                batch_size=32,
                min_replay_history=100,
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
        trajectory=CircularArraySARTTrajectory(
            capacity=1000,
            state=Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_BasicDQN_SingleRoomUndirected`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_BasicDQN_SingleRoomUndirected.png") #hide

# ![](assets/JuliaRL_BasicDQN_SingleRoomUndirected.png)
