@testset "Basic PlayerNamedTuple tests" begin
    nt = PlayerNamedTuple(Player(Symbol(1)) => 3, Player(Symbol(2)) => 4)
    @test nt.data == (; Symbol(1) => 3, Symbol(2) => 4)
    @test typeof(nt).parameters == typeof(nt.data).parameters
    @test nt[Player(Symbol(1))] == 3
end
