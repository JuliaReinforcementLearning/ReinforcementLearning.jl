import ReinforcementLearning: getactionprobabilities
using ReinforcementLearningBase: DiscreteSpace

@testset "policies" begin

function empiricalactionprop(p, v; n = 10^6)
    res = [p(v) for _ in 1:n]
    map(x -> length(findall(i -> i == x, res)), 1:length(v))./n
end

@testset "ϵ-greedy policies" begin
    for (v, rO, rVO, r, rP) in (([-9., 12., Inf64], [0, .5, .5], [0, 0., 1.], 
                                                    [0., 0., 1.], [0., 1., 0.]),
                                ([-100, Inf64, Inf64], [1/3, 1/3, 1/3], 
                                                    [0., 0.5, 0.5], 
                                                    [0., 0.5, 0.5], 
                                                    [1., 0., 0.]))
        @test getactionprobabilities(EpsilonGreedyPolicy(0., DiscreteSpace(3, 1), x -> x, kind = :optimistic), v) == rO
        @test getactionprobabilities(EpsilonGreedyPolicy(0., DiscreteSpace(3, 1), x -> x), v) == rVO
        @test getactionprobabilities(EpsilonGreedyPolicy(0., DiscreteSpace(3, 1), x -> x, kind = :pessimistic), v) == rP
        @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(0., DiscreteSpace(3, 1), x -> x, kind = :optimistic), v), rO, atol = .05)
        @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(0., DiscreteSpace(3, 1), x -> x), v), rVO, atol = .05)
        @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(0., DiscreteSpace(3, 1), x -> x, kind = :pessimistic), v), rP, atol = .05)
        @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(.2, DiscreteSpace(3, 1), x -> x, kind = :optimistic), v),
                    getactionprobabilities(EpsilonGreedyPolicy(.2, DiscreteSpace(3, 1), x -> x, kind = :optimistic), v),
                    atol = .05)
    end
end

@testset "SoftmaxPolicy" begin
    let x = [-.1, .5, .8, .8]
        @test getactionprobabilities(SoftmaxPolicy(x -> x), x) ≈ exp.(x)./sum(exp.(x))
        @test getactionprobabilities(SoftmaxPolicy(x -> x, β = 2.), x) ≈ exp.(2x)./sum(exp.(2x))
        @test getactionprobabilities(SoftmaxPolicy(x -> x, β = Inf64), x) == [0., 0., .5, .5]
        @test getactionprobabilities(SoftmaxPolicy(x -> x), [1, Inf64, Inf64]) == [0., .5, .5]
        @test isapprox(empiricalactionprop(SoftmaxPolicy(x -> x), x), exp.(x)./sum(exp.(x)), atol = .05)
        @test isapprox(empiricalactionprop(SoftmaxPolicy(x -> x, β = Inf64), x), [0, 0, .5, .5], atol = .05)
    end
end

end