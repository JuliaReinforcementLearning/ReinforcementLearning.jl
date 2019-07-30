@testset "TilingsV" begin
    init_tiling = Tiling((0:5:50, 0:10:100))
    tilings = [init_tiling, init_tiling - [1, 2]]
    V = TilingsV(tilings)

    update!(V, [0, 0] => 1.)
    @test V([0,0]) == 2.
    @test V([4,8]) == 1.
    @test V([5,10]) == 0.
end
