@testset "UpperConfidneceBound" begin
    s = UCBSelector(2)
    values = [1, 2]
    @test s(values) == 2
    @test s(values) == 1
    @test s(values) == 2
    @test s(values) == 2
    @test s(values) == 1
end