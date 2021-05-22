# ---
# title: Play EmptyRoom with BasicDQNLearner
# cover: assets/JuliaRL_BasicDQN_CartPole.png
# description: The simplest example to demonstrate how to use BasicDQN
# date: 2021-05-22
# author: Jun Tian
# ---

#+ tangle=true
using GridWorlds

function Experiment(
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

    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                    0
            end
        end,
    )

    description = """
    This experiment uses three dense layers to approximate the Q value.
    The testing environment is EmptyRoom.

    You can view the runtime logs with `tensorboard --logdir $log_dir`.
    Some useful statistics are stored in the `hook` field of this experiment.
    """

    Experiment(agent, env, stop_condition, hook, description)
end
