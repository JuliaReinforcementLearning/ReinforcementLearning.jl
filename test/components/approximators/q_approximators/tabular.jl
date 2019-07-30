@testset "TabularQ" begin
    A = TabularQ([1 2; 3 4])

    @test A(1) == [1., 2.]
    @test A(2) == [3., 4.]
    @test A(2, 2) == 4

    update!(A, (2, 2) => -1.)
    @test A(2) == [3., 3.]
    @test A(2, 2) == 3

    update!(A, 1 => [-1., -1.])
    @test A(1) == [0., 1.]
    @test A(1, 1) == 0
    @test A(1, 2) == 1
end