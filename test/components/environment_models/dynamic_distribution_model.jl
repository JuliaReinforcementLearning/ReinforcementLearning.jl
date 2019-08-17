@testset "dynamic_distribution_model" begin
    f(s, a) = (nextstate=1, reward=1.0, prob=1.0)
    m = DynamicDistributionModel(f, 2, 2)

    @test get_states(m) == 1:2
    @test get_actions(m) == 1:2

    for s in 1:2
        for a in 1:2
            @test m(s, a) == f(s, a)
        end
    end
end