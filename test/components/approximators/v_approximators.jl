@testset "v_approximators" begin
    @testset "LinearVApproximator" begin
        weights = [1., 2., 3.]
        V = LinearVApproximator(weights)

        @test V([1, 1, 1]) ≈ 6

        update!(V, [1, 1, 1] => [0.1, 0.2, 0.3])
        @test V([1, 1, 1]) ≈ 6.6
    end

    @testset "TabularVApproximator" begin
        V = TabularVApproximator([1., 2., 3., 2., 1.])
        @test V(1) == 1.
        @test V(3) == 3.

        update!(V, 3 => -1.)
        @test V(3) == 2.
    end
end