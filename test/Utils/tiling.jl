@testset "Tiling" begin
    t = Tiling((1:2:5, 10:5:20))
    @test encode(t, (2, 12)) == 1
    @test encode(t, (2, 18)) == 3

    t2 = t - (1, 3)
    @test t2.ranges == (0:2:4, 7:5:17)
end