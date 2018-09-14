import ReinforcementLearning:update!

@testset "montecarl" begin

γ = .9
learner = MonteCarlo(ns = 4, na = 1, γ = γ, initvalue = Inf64)
buffer = EpisodeTurnBuffer{Turn{Int, Int, Bool, Bool}}()
s, a = 1, 1
push!(buffer, s, a)
for i in 2:4
    r, d = iseven(i), i == 4
    next_s, next_a = i, 1
    push!(buffer, r, d, next_s, next_a)
    update!(learner, buffer)
    s, a = next_s, next_a
end
@test learner.Q == [1 + γ^2 γ 1 Inf64]

end