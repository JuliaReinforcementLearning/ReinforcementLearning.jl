@testset "UpperConfidneceBound" begin
    s = UCBSelector(2)
    values = [1, 2]
    @test s(values; step=1) == 2
    @test s(values; step=2) == 1
    @test s(values; step=3) == 2
    @test s(values; step=4) == 2
    @test s(values; step=5) == 1
end