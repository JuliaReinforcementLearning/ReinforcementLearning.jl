import ReinforcementLearning: selectaction

function empiricalactionprop(p, v; n = 10^6)
    res = [selectaction(p, v) for _ in 1:n]
    map(x -> length(findall(i -> i == x, res)), 1:length(v))./n
end

for (v, rO, rVO, r, rP) in (([-9., 12., Inf64], [0, .5, .5], [0, 0., 1.], 
                                                [0., 0., 1.], [0., 1., 0.]),
                            ([-100, Inf64, Inf64], [1/3, 1/3, 1/3], 
                                                   [0., 0.5, 0.5], 
                                                   [0., 0.5, 0.5], 
                                                   [1., 0., 0.]))
    @test getactionprobabilities(EpsilonGreedyPolicy(0., kind = :optimistic), v) == rO
    @test getactionprobabilities(EpsilonGreedyPolicy(0.), v) == rVO
    @test getactionprobabilities(EpsilonGreedyPolicy(0., kind = :pessimistic), v) == rP
    @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(0., kind = :optimistic), v),
                   rO, atol = .05)
    @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(0.), v),
                   rVO, atol = .05)
    @test isapprox(empiricalactionprop(EpsilonGreedyPolicy(0., kind = :pessimistic), v),
                   rP, atol = .05)
    @test isapprox(empiricalactionprop(EpsilonGreedyPolicy{:optimistic}(.2), v),
                   getactionprobabilities(EpsilonGreedyPolicy{:optimistic}(.2), v),
                   atol = .05)
end


