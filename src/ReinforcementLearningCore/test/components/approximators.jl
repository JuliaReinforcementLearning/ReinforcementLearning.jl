@testset "Approximators" begin

@testset "TabularApproximator" begin
    A = TabularApproximator(;n_state=3)

    @test A(1) == 0.
    @test A(2) == 0.

    update!(A, 2 => 3.)
    @test A(2) == 3.
end

end