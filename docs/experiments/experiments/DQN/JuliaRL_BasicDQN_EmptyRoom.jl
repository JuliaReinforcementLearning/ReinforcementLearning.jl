# ---
# title: JuliaRL\_BasicDQN\_EmptyRoom
# cover: assets/JuliaRL_BasicDQN_EmptyRoom.png
# description: A simple example to demonstrate how to use environments in GridWorlds.jl
# date: 2021-05-22
# author: Siddharth Bhatia
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
    ::Val{:EmptyRoom},
    ::Nothing;
    seed=123,
)
    rng = StableRNG(seed)

    inner_env = GridWorlds.EmptyRoomDirected(rng=rng)
    action_space_mapping = x -> Base.OneTo(length(RLBase.action_space(inner_env)))
    action_mapping = i -> RLBase.action_space(inner_env)[i]
    env = RLEnvs.ActionTransformedEnv(
        inner_env,
        action_space_mapping=action_space_mapping,
        action_mapping=action_mapping,
    )
    env = RLEnvs.StateTransformedEnv(env;state_mapping=x -> vec(Float32.(x)))
    env = RewardOverriddenEnv(env, x -> x - convert(typeof(x), 0.01))
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
                    ) |> cpu,
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

    stop_condition = StopAfterStep(10_000)
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end

#+ tangle=false
using Plots
abc = E`JuliaRL_BasicDQN_EmptyRoom`
run(abc)
plot(abc.hook.rewards)
savefig("assets/JuliaRL_BasicDQN_EmptyRoom.png") #hide

# ![](assets/JuliaRL_BasicDQN_EmptyRoom.png)
