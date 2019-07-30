@testset "PolynomialV" begin
    V = PolynomialV([1., 2., 3.])

    @test V(2) ≈ 1 + 2. * 2  + 3. * 2^2

    update!(V, 2, 1.)
    @test V(2) ≈ 2 + 4. * 2 + 7. * 2^2
end