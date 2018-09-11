import ReinforcementLearning:update!

function tdlearnintest()
    episode = [(1, 2, 0., false, 3, 1),
               (3, 1, 1., false, 1, 2),
               (1, 2, 0., false, 2, 2),
               (2, 2, 1., false, 3, 2)]
    γ = .9
    λ = .8
    α = .1
    γλ = γ*λ
    results = Dict()
    results[QLearning, NoTraces] = [Inf  Inf     1.; 
                                      0. 1. + γ  Inf]
    results[Sarsa, NoTraces] = [Inf64 Inf64 1.; 
                                0.    1.    Inf]
    δ2 = 1.
    δ3 = - α * γλ
    δ4 = 1. + γ * (δ2 + α * δ3 * γλ)
    δ4Sarsa = 1.
    tmp = [Inf64 Inf64 0.; 0. δ4 Inf]
    tmp[2, 1] += α * (δ2 * γλ + δ3 * (1 + γλ^2) + δ4 * (γλ + γλ^3))
    tmp[1, 3] += δ2 + α * (δ3 * γλ + δ4 * γλ^2)
    results[QLearning, AccumulatingTraces] = deepcopy(tmp)
    tmp[2, 1] -=  α * (δ3 * γλ^2 + δ4 * γλ^3)
    results[QLearning, ReplacingTraces] = deepcopy(tmp)
    tmp[2, 2] = δ4Sarsa
    tmp[2, 1] += α * (δ4Sarsa - δ4) * γλ
    tmp[1, 3] += α * (δ4Sarsa - δ4) * γλ^2
    results[Sarsa, ReplacingTraces] = deepcopy(tmp)
    tmp[2, 1] += α * (δ3 * γλ^2 + δ4Sarsa * γλ^3)
    results[Sarsa, AccumulatingTraces] = deepcopy(tmp)
    for tdkind in [QLearning, Sarsa] #, ExpectedSarsa]
        for tracekind in [NoTraces, AccumulatingTraces, ReplacingTraces]
            buffer = CircularTurnBuffer{Int, Int, Float64, Bool}(1)
            learner = tdkind(ns = 3, na = 2, γ = γ, λ = λ, α = α, initvalue = Inf64,
                             tracekind = tracekind)
            println(learner.params, results[tdkind, tracekind])
            for t in episode
                push!(buffer, Turn(t...))
                update!(learner, buffer)
                println(learner.params, results[tdkind, tracekind])
            end
            @test isapprox(learner.params, results[tdkind, tracekind], atol = 1e-15) 
        end
    end
end
tdlearnintest()