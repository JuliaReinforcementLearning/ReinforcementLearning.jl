@testset "preprocessors" begin
    obs1 = (state = [1, 2, 3],)
    p = CloneStatePreprocessor()
    obs2 = p(obs1)

    @test get_state(obs1) !== get_state(obs2)
    @test get_state(obs1) == get_state(obs2)
end
