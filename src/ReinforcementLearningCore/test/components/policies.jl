@testset "policies" begin

@testset "RandomPolicy" begin
    p = RandomPolicy(;action_space=DiscreteSpace(3))
    obs = (reward=0., terminal=false, state=1)

    Random.seed!(p, 321)
    actions = [p(obs) for _ in 1:100]
    Random.seed!(p, 321)
    new_actions = [p(obs) for _ in 1:100]
    @test actions == new_actions
end

end