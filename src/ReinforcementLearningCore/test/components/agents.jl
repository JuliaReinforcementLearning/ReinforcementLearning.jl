@testset "Agent" begin
    env = CartPoleEnv(;T=Float32)
    agent = Agent(;
        policy = RandomPolicy(env),
        trajectory = CircularArraySARTTrajectory(;
            capacity = 10_000,
            state = Vector{Float32} => (4,),
        ),
    )

    # TODO: test de/serialization
end
