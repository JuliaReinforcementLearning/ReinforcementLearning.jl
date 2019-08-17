@testset "experience_sample_model" begin
    m = ExperienceBasedSampleModel()

    update!(m, 1, 1, 1.0, false, 2)

    @test haskey(m.experiences, 1)
    @test haskey(m.experiences[1], 1)

    update!(m, 2, 2, -1.0, true, 3)

    @test haskey(m.experiences, 1)
    @test haskey(m.experiences, 2)
    @test haskey(m.experiences[1], 1)
    @test haskey(m.experiences[2], 2)

    @test m.experiences[2][2] == (reward=-1.0, terminal=true, nextstate=3)
end