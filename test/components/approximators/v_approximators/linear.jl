@testset "LinearV" begin
    features = [1 0 1; 0 1 1]
    weights = [1., 2., 3.]
    V = LinearV(features, weights)

    @test V(1) ≈ 1. + 3.
    @test V(2) ≈ 2. + 3.

    update!(V, 1 => 0.5)
    @test V(1) ≈ 1.5 + 3.5
    @test V(2) ≈ 2. + 3.5
end