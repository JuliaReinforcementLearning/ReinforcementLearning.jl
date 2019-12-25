@testset "explorers" begin

@testset "AlternateExplorer" begin
    explorer = AlternateExplorer(;n=3)
    @test get_distribution(explorer, nothing) == [1, 0, 0]

    # make sure that `get_distribution` has no side-effect
    @test get_distribution(explorer, nothing) == [1, 0, 0]
    @test get_distribution(explorer, nothing) == [1, 0, 0]

    @test [explorer(nothing) for _ in 1:9] == repeat([1, 2, 3], 3)

    @test explorer(nothing) == 1
    @test explorer(nothing) == 2
    @test get_distribution(explorer, nothing) == [0, 1, 0]

    reset!(explorer)
    @test explorer(nothing) == 1
    @test get_distribution(explorer, nothing) == [1, 0, 0]
end

@testset "EpsilonGreedyExplorer" begin
    @testset "API" begin
        explorer = EpsilonGreedyExplorer(0.1)
        Random.seed!(explorer, 123)

        values = [0, 1, 2, -1]
        target_distribution = [0.025, 0.025, 0.925, 0.025]

        @test get_distribution(explorer, values) == target_distribution

        actions = [explorer(values) for _ in 1:10000]
        action_counts = countmap(actions)

        @test all(
            isapprox.(
                [action_counts[i] for i in 1:length(values)] ./ 10000,
                target_distribution;
                atol=0.005
            )
        )

        explorer_copy = copy(explorer)
        reset!(explorer_copy)
        Random.seed!(explorer_copy, 123)

        new_actions = [explorer_copy(values) for _ in 1:10000]

        @test actions == new_actions
    end

    @testset "linear" begin
        explorer = EpsilonGreedyExplorer(; ϵ_stable=0.1, ϵ_init=0.9, warmup_steps=3, decay_steps=8, kind=:linear)
        Ε = [RLCore.get_ϵ(explorer, i) for i in 1:15]
        @test Ε ≈ [0.9, 0.9, range(0.9; stop=0.1, step=-0.1)..., 0.1, 0.1, 0.1, 0.1]
    end

    @testset "exp" begin
        explorer = EpsilonGreedyExplorer(; ϵ_stable=0.1, ϵ_init=0.9, warmup_steps=3, decay_steps=8, kind=:exp)
        @test isapprox(RLCore.get_ϵ(explorer, 100), 0.1; atol=1e-5)
    end

end

end