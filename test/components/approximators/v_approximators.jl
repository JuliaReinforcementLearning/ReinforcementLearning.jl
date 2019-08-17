@testset "v_approximators" begin
    @testset "LinearVApproximator" begin
        features = [1 0 1; 0 1 1]
        weights = [1., 2., 3.]
        V = LinearVApproximator(features, weights)

        @test V(1) ≈ 1. + 3.
        @test V(2) ≈ 2. + 3.

        update!(V, 1 => 0.5)
        @test V(1) ≈ 1.5 + 3.5
        @test V(2) ≈ 2. + 3.5
    end

    @testset "TabularVApproximator" begin
        V = TabularVApproximator([1., 2., 3., 2., 1.])
        @test V(1) == 1.
        @test V(3) == 3.

        update!(V, 3 => -1.)
        @test V(3) == 2.
    end
end