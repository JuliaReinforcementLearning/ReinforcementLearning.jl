@testset "time_based_sample_model" begin
    m = TimeBasedSampleModel(2, 1.0)

    update!(m, 1, 1, 1.0, false, 2)
    update!(m, 2, 1, 1.0, false, 3)

    @test m.experiences[1][1] == (reward=1.0, terminal=false, nextstate=2)
    @test m.experiences[2][1] == (reward=1.0, terminal=false, nextstate=3)
    @test m.last_visit[(1,1)] == 1
    @test m.last_visit[(2,1)] == 2
end