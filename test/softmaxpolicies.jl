let x = [-.1, .5, .8, .8]
    @test getactionprobabilities(SoftmaxPolicy(x -> x), x) ≈ exp.(x)./sum(exp.(x))
    @test getactionprobabilities(SoftmaxPolicy(x -> x, β = 2.), x) ≈ exp.(2x)./sum(exp.(2x))
    @test getactionprobabilities(SoftmaxPolicy(x -> x, β = Inf64), x) == [0., 0., .5, .5]
    @test getactionprobabilities(SoftmaxPolicy(x -> x), [1, Inf64, Inf64]) == [0., .5, .5]
    @test isapprox(empiricalactionprop(SoftmaxPolicy(x -> x), x), exp.(x)./sum(exp.(x)), atol = .05)
    @test isapprox(empiricalactionprop(SoftmaxPolicy(x -> x, β = Inf64), x), [0, 0, .5, .5], atol = .05)
end
