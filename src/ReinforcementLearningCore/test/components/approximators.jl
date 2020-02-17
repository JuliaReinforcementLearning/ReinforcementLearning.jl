@testset "Approximators" begin

    @testset "TabularApproximator" begin
        A = TabularApproximator(ones(3))

        @test A(1) == 1.0
        @test A(2) == 1.0

        update!(A, 2 => 3.0)
        @test A(2) == 4.0
    end

end
