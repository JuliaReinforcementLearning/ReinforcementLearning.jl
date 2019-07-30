@testset "FourierV" begin
    V = FourierV([1, 1, 1])
    @test V(1) ≈ 1.0
    @test V(2) ≈ 3.0
    
    update!(V, 1 => 0.5)
    @test V(1) ≈ 2.5
    @test V(2) ≈ 3.5
end