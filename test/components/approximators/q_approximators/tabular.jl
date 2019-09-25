@testset "TabularQApproximator" begin
    A = TabularQApproximator([1 2; 3 4])

    @test A(1) == [1.0, 2.0]
    @test A(2) == [3.0, 4.0]
    @test A(2, 2) == 4

    update!(A, (2, 2) => -1.0)
    @test A(2) == [3.0, 3.0]
    @test A(2, 2) == 3

    update!(A, 1 => [-1.0, -1.0])
    @test A(1) == [0.0, 1.0]
    @test A(1, 1) == 0
    @test A(1, 2) == 1
end