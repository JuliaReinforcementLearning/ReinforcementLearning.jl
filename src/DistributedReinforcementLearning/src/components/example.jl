#=

Proof of Concept

Here we solve the multi-arm bandit problem with the ϵ-greedy algorithm
in a distributed manner.

Each worker starts with a random policy. They collect actions and corresponding
rewards and then forward them to a trajectory proxy.

A trajectory stores transitions from works in a local buffer and periodically
send a batch to a optimizer.

A optimizer updates its policy and broadcast the latest policy to workers periodically.
=#

using Flux

"""
    UploadTrajectoryEveryNStep(;mailbox, n, sealer=deepcopy)
"""
Base.@kwdef mutable struct UploadTrajectoryEveryNStep{M,S} <: AbstractHook
    mailbox::M
    n::Int
    t::Int = 0
    sealer::S = deepcopy
end

function (hook::UploadTrajectoryEveryNStep)(::PostActStage, agent, env)
    hook.t += 1
    if hook.t % hook.n == 0
        put!(hook.mailbox, InsertTrajectoryMsg(hook.sealer(agent.trajectory)))
    end
end

struct LoadParamsHook <: AbstractHook
    buffer::Channel{Any}
end

function (hook::LoadParamsHook)(::PostActStage, agent, env)
    ps = nothing
    while isready(hook.buffer)
        ps = take!(hook.buffer)
    end
    isnothing(ps) || Flux.loadparams!(agent.policy, ps)
end

env = CartPoleEnv(; T = Float32)
ns, na = length(get_state(env)), length(get_actions(env))

optimizer = actor(
    Optimizer(;
        policy=BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = glorot_uniform),
                    Dense(128, 128, relu; initW = glorot_uniform),
                    Dense(128, na; initW = glorot_uniform),
                ) |> cpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
        )
    )
)

trajectory_proxy = actor(
    TrajectoryProxy(
        trajectory = VectSARTSATrajectory(;state_type=Any),
        sampler = UniformBatchSampler(32),
        inserter = NStepInserter(),
    )
)

worker = actor(
    Worker() do
        Experiment(
            Agent(
                policy = StaticPolicy(
                        QBasedPolicy(
                        learner = BasicDQNLearner(
                            approximator = NeuralNetworkApproximator(
                                model = Chain(
                                    Dense(ns, 128, relu; initW = glorot_uniform),
                                    Dense(128, 128, relu; initW = glorot_uniform),
                                    Dense(128, na; initW = glorot_uniform),
                                ) |> cpu,
                                optimizer = ADAM(),
                            ),
                            batch_size = 32,
                            min_replay_history = 100,
                            loss_func = huber_loss,
                        ),
                        explorer = EpsilonGreedyExplorer(
                            kind = :exp,
                            ϵ_stable = 0.01,
                            decay_steps = 500,
                        ),
                    ),
                ),
                trajectory = CircularCompactSARTSATrajectory(
                    capacity = 10,
                    state_type = Float32,
                    state_size = (ns,),
                ),
            ),
            CartPoleEnv(; T = Float32),
            StopSignal(),
            UploadTrajectoryEveryNStep(mailbox=trajectory_proxy, n=11),
            "experimenting..."
            )
    end
)
