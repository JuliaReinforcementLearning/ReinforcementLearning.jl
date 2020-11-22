@testset "core.jl" begin

@testset "Trainer" begin
    _trainer = Trainer(;
        policy=BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(4, 128, relu; initW = glorot_uniform),
                    Dense(128, 128, relu; initW = glorot_uniform),
                    Dense(128, 2; initW = glorot_uniform),
                ) |> cpu,
                optimizer = ADAM(),
            ),
            loss_func = huber_loss,
        )
    )

    trainer = actor(_trainer)

    put!(trainer, FetchParamMsg())
    ps = take!(self())
    original_sum = sum(sum, ps.data)

    for x in ps.data
        fill!(x, 0.)
    end

    put!(trainer, FetchParamMsg())
    ps = take!(self())
    new_sum = sum(sum, ps.data)

    # make sure no state sharing between messages
    @test original_sum == new_sum

    batch_data = (
        state = rand(4, 32),
        action = rand(1:2, 32),
        reward = rand(32),
        terminal = rand(Bool, 32),
        next_state = rand(4,32),
        next_action = rand(1:2, 32)
    )

    put!(trainer, BatchDataMsg(batch_data))

    put!(trainer, FetchParamMsg())
    ps = take!(self())
    updated_sum = sum(sum, ps.data)
    @test original_sum != updated_sum
end

@testset "TrajectoryManager" begin
    _trajectory_proxy = TrajectoryManager(
        trajectory = CircularSARTSATrajectory(;capacity=5, state_type=Any, ),
        sampler = UniformBatchSampler(3),
        inserter = NStepInserter(),
    )

    trajectory_proxy = actor(_trajectory_proxy)

    # 1. init traj for testing
    traj = CircularCompactSARTSATrajectory(
        capacity = 2,
        state_type = Float32,
        state_size = (4,),
    )
    push!(traj;state=rand(Float32, 4), action=rand(1:2))
    push!(traj;reward=rand(), terminal=rand(Bool),state=rand(Float32, 4), action=rand(1:2))
    push!(traj;reward=rand(), terminal=rand(Bool),state=rand(Float32, 4), action=rand(1:2))

    # 2. insert
    put!(trajectory_proxy, InsertTrajectoryMsg(deepcopy(traj)))  #!!! we used deepcopy here

    # 3. make sure the above message is already been handled
    put!(trajectory_proxy, PingMsg())
    take!(self())

    # 4. test that updating traj will not affect data in trajectory_proxy
    s_tp = _trajectory_proxy.trajectory[:state]
    s_traj = traj[:state]

    @test s_tp[1] == s_traj[:, 1]

    push!(traj;reward=rand(), terminal=rand(Bool),state=rand(Float32, 4), action=rand(1:2))

    @test s_tp[1] != s_traj[:, 1]

    s = sample(_trajectory_proxy.trajectory, _trajectory_proxy.sampler)
    fill!(s[:state], 0.)
    @test any(x -> sum(x) == 0, s_tp) == false  # make sure sample create an independent copy
end

@testset "Worker" begin
    _worker = Worker() do worker_proxy
        Experiment(
            Agent(
                policy = StaticPolicy(
                        QBasedPolicy(
                        learner = BasicDQNLearner(
                            approximator = NeuralNetworkApproximator(
                                model = Chain(
                                    Dense(4, 128, relu; initW = glorot_uniform),
                                    Dense(128, 128, relu; initW = glorot_uniform),
                                    Dense(128, 2; initW = glorot_uniform),
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
                    capacity = 10,
                    state_type = Float32,
                    state_size = (4,),
                ),
            ),
            CartPoleEnv(; T = Float32),
            ComposedStopCondition(
                StopAfterStep(1_000),
                StopSignal(),
            ),
            ComposedHook(
                UploadTrajectoryEveryNStep(mailbox=worker_proxy, n=10, sealer=x -> InsertTrajectoryMsg(deepcopy(x))),
                LoadParamsHook(),
                TotalRewardPerEpisode(),
            ),
            "experimenting..."
        )
    end

    worker = actor(_worker)
    tmp_mailbox = Channel(100)
    put!(worker, StartMsg(tmp_mailbox))
end

@testset "WorkerProxy" begin
    target = RemoteChannel(() -> Channel(10))
    workers = [RemoteChannel(()->Channel(10)) for _ in 1:10]
    _wp = WorkerProxy(workers)
    wp = actor(_wp)

    put!(wp, StartMsg(target))
    for w in workers
        # @test take!(w).args[1] === wp
        @test Distributed.channel_from_id(remoteref_id(take!(w).args[1])) === Distributed.channel_from_id(remoteref_id(wp))
    end

    msg = InsertTrajectoryMsg(1)
    put!(wp, msg)
    @test take!(target) === msg

    for w in workers
        put!(wp, FetchParamMsg(w))
    end
    # @test take!(target).from === wp
    @test Distributed.channel_from_id(remoteref_id(take!(target).from)) === Distributed.channel_from_id(remoteref_id(wp))

    # make sure target only received one FetchParamMsg
    msg = PingMsg()
    put!(target, msg)
    @test take!(target) === msg

    msg = LoadParamMsg([])
    put!(wp, msg)
    for w in workers
        @test take!(w) === msg
    end
end

@testset "Orchestrator" begin
    # TODO
    # Add an integration test
end

end