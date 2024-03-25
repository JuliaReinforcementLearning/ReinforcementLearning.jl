@testset "Player" begin
    @test Player(1) == Player(Symbol(1))
    @test Player("test").name == :test
end
