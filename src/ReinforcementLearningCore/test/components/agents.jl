@testset "Agent" begin
    action_space = DiscreteSpace(3)
    agent = Agent(;
        policy = RandomPolicy(action_space),
        trajectory = VectorialCompactSARTSATrajectory(),
    )

    @testset "loading/saving Agent" begin
        mktempdir() do dir
            RLCore.save(dir, agent)
            @test length(readdir(dir)) != 0
            agent = RLCore.load(dir, Agent)
        end
    end
end
