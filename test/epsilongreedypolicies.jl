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
    @test getactionprobabilities(OptimisticEpsilonGreedyPolicy(0.), v) == rO
    @test getactionprobabilities(VeryOptimisticEpsilonGreedyPolicy(0.), v) == rVO
    @test getactionprobabilities(PesimisticEpsilonGreedyPolicy(0.), v) == rP
    @test isapprox(empiricalactionprop(OptimisticEpsilonGreedyPolicy(0.), v),
                   rO, atol = .05)
    @test isapprox(empiricalactionprop(VeryOptimisticEpsilonGreedyPolicy(0.), v),
                   rVO, atol = .05)
    @test isapprox(empiricalactionprop(PesimisticEpsilonGreedyPolicy(0.), v),
                   rP, atol = .05)
    @test isapprox(empiricalactionprop(OptimisticEpsilonGreedyPolicy(.2), v),
                   getactionprobabilities(OptimisticEpsilonGreedyPolicy(.2), v),
                   atol = .05)
end


