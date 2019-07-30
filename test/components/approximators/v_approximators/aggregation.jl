@testset "AggregationV" begin
    table = [-1., 1.]
    V = AggregationV(table, s -> s < 0 ? 1 : 2)

    @test V(-5) == -1.
    @test V(5) == 1.

    update!(V, -1 => 0.5)
    update!(V, 1 => -0.5)

    @test V(-5) == -.5
    @test V(5) == 0.5
end