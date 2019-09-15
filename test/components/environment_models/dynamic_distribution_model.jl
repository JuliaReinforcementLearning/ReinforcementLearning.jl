@testset "dynamic_distribution_model" begin
    f(s, a) = (nextstate = 1, reward = 1.0, prob = 1.0)
    m = DynamicDistributionModel(f, 2, 2)

    @test length(observation_space(m)) == 2
    @test length(action_space(m)) == 2

    for s = 1:2
        for a = 1:2
            @test m(s, a) == f(s, a)
        end
    end
end