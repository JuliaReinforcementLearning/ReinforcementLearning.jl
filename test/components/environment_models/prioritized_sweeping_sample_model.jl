@testset "prioritized_sweeping_sample_model" begin
    m = PrioritizedSweepingSampleModel(0.5)

    update!(m, 1, 1, 1.0, true, 2, 0.0)
    @test sample(m) === nothing

    update!(m, 1, 1, 1.0, true, 2, 1.0)
    @test sample(m) == (1, 1, 1.0, true, 2)

    update!(m, 1, 1, 1.0, true, 2, 1.0)
    update!(m, 2, 2, 2.0, true, 3, 1.1)
    @test sample(m) == (2, 2, 2.0, true, 3)
    @test sample(m) == (1, 1, 1.0, true, 2)
end