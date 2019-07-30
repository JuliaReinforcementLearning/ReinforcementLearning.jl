@testset "TabularV" begin
    V = TabularV([1., 2., 3., 2., 1.])
    @test V(1) == 1.
    @test V(3) == 3.

    update!(V, 3 => -1.)
    @test V(3) == 2.
end