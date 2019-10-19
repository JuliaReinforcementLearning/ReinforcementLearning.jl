@testset "alternate selector" begin
    N = 3
    s = AlternateSelector(n=N)
    values = zeros(N)

    @test s(values; step = 1) == 1
    @test s(values; step = 2) == 2
    @test s(values; step = 3) == 3

    @test s(values; step = 4) == 1
    @test s(values; step = 5) == 2
    @test s(values; step = 6) == 3
end