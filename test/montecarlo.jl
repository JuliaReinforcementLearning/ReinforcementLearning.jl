import ReinforcementLearning: EpisodeBuffer, pushstateaction!, pushreturn!
function mctest()
    γ = .9
    learner = MonteCarlo(ns = 4, na = 1, γ = γ, initvalue = Inf64)
    buffer = EpisodeBuffer()
    pushstateaction!(buffer, 1, 1)
    for i in 2:4
        pushreturn!(buffer, iseven(i), i == 4)
        pushstateaction!(buffer, i, 1)
        update!(learner, buffer)
    end
    @test learner.Q == [1 + γ^2 γ 1 Inf64]
end
mctest()
