@testset "deterministic_distribution_model" begin
    table = Array{Vector{NamedTuple{(:nextstate, :reward, :prob), Tuple{Int, Float64, Float64}}}, 2}(undef, 2, 2)

    table[1] = [(nextstate=1, reward=1.0, prob=0.5), (nextstate=2, reward=0.0, prob=0.5)]
    table[2] = [(nextstate=2, reward=-1.0, prob=0.25), (nextstate=1, reward=0.0, prob=0.75)]
    table[3] = [(nextstate=2, reward=-1.0, prob=0.25), (nextstate=1, reward=0.5, prob=0.75)]
    table[4] = [(nextstate=1, reward=1.0, prob=0.75), (nextstate=2, reward=-0.5, prob=0.25)]

    m = DeterministicDistributionModel(table)

    @test get_states(m) == 1:2
    @test get_actions(m) == 1:2

    @test m(1, 1) == table[1, 1]
    @test m(1, 2) == table[1, 2]
    @test m(2, 1) == table[2, 1]
    @test m(2, 2) == table[2, 2]
end