@testset "TilingsQ" begin
    init_tiling = Tiling((0:5:50, 0:10:100))
    tilings = [init_tiling, init_tiling - [1, 2]]
    Q = TilingsQ(tilings, 2)
    update!(Q, ([0, 0], 1) => 1.)

    @test Q([0, 0]) ≈ [2., 0.]
    @test Q([0, 0], 1) ≈ 2.
end