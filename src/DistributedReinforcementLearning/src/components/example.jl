using Distributed, ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments, ReinforcementLearningZoo, DistributedReinforcementLearning, Flux

struct LoadParamsHook <: AbstractHook
    buffer::Channel
end

function (hook::LoadParamsHook)(::PostActStage, agent, env)
    ps = nothing
    while isready(hook.buffer)
        ps = take!(hook.buffer).data
    end
    Flux.loadparams!(agent.policy, ps)
end

(msg::LoadParamMsg)(x::LoadParamsHook) = put!(x.buffer, msg)

Base.@kwdef mutable struct BlockingLoadParamsHook <: AbstractHook
    buffer::RemoteChannel
    target::RemoteChannel
    n::Int = 0
    freq::Int = 1
end

(msg::LoadParamMsg)(x::BlockingLoadParamsHook) = put!(x.buffer, msg)

function (hook::LoadParamsHook)(::PostActStage, agent, env)
    hook.n += 1
    if hook.n % hook.freq == 0
        if isready(hook.buffer)
            # some other workers have sent the request, so we just reuse it
            while isready(hook.buffer)
                ps = take!(hook.buffer).data
            end
            Flux.loadparams!(agent.policy, ps)
        else
            # blocking
            put!(target, FetchParamMsg(hook.buffer))
            ps = take!(hook.buffer).data
            Flux.loadparams!(agent.policy, ps)
        end
    end
end

#####
# Example
#####

env = CartPoleEnv(; T = Float32)
ns, na = length(get_state(env)), length(get_actions(env))

_trainer = Trainer(;
    policy=BasicDQNLearner(
        approximator = NeuralNetworkApproximator(
            model = Chain(
                Dense(ns, 128, relu; initW = glorot_uniform),
                Dense(128, 128, relu; initW = glorot_uniform),
                Dense(128, na; initW = glorot_uniform),
            ) |> cpu,
            optimizer = ADAM(),
        ),
        loss_func = huber_loss,
    )
)

trainer = actor(_trainer)

_trajectory_proxy = TrajectoryManager(
    trajectory = CircularSARTSATrajectory(;capacity=1_000, state_type=Any, ),
    sampler = UniformBatchSampler(32),
    inserter = NStepInserter(),
)

trajectory_proxy = actor(_trajectory_proxy)

_orchestrator = Orchestrator(
    trainer = trainer,
    trajectory_proxy = trajectory_proxy,
    limiter = InsertSampleRateLimiter(
        ;min_size_to_sample=100,
        sample_insert_ratio=1,
    )
)

orchestrator = actor(_orchestrator)

_worker = Worker() do
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
                    capacity = 1,
                    state_type = Float32,
                    state_size = (ns,),
                ),
            ),
            CartPoleEnv(; T = Float32),
            ComposedStopCondition(
                StopAfterStep(50_000),
                StopSignal(),
            ),
            ComposedHook(
                UploadTrajectoryEveryNStep(mailbox=orchestrator, n=1, sealer=x -> InsertTrajectoryMsg(deepcopy(x))),
                LoadParamsHook(;to=trainer),
                TotalRewardPerEpisode(),
            ),
            "experimenting..."
            )
    end

worker = actor(_worker)
