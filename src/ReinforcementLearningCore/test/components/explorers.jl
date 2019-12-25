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
    @test get_distribution(explorer, nothing) == [0, 0, 1]
    @test explorer(nothing) == 3

    reset!(explorer)
    @test get_distribution(explorer, nothing) == [1, 0, 0]
    @test explorer(nothing) == 1
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
        xs = [0, 1, 2, -1, 2]
        mask = [true, true, false, true, false]
        E = [0.9, 0.9, range(0.9; stop=0.1, step=-0.1)..., 0.1, 0.1, 0.1, 0.1]

        for ϵ in E
            @test RLCore.get_ϵ(explorer) ≈ ϵ
            @test isapprox(get_distribution(explorer, xs), [ϵ/5, ϵ/5, ϵ/5+(1-ϵ)/2, ϵ/5, ϵ/5+(1-ϵ)/2])
            explorer(xs)
        end

        reset!(explorer)

        for ϵ in E
            @test RLCore.get_ϵ(explorer) ≈ ϵ
            @test isapprox(get_distribution(explorer, xs, mask), [ϵ/3, (1-ϵ) + ϵ/3, 0., ϵ/3, 0.])
            explorer(xs)
        end

        reset!(explorer)
        for i in 1:100
            @test mask[explorer(xs, mask)]
        end
    end

    @testset "exp" begin
        explorer = EpsilonGreedyExplorer(; ϵ_stable=0.1, ϵ_init=0.9, warmup_steps=3, decay_steps=8, kind=:exp)
        xs = [0, 1, 2, -1, 2]
        mask = [true, true, false, true, false]
        for i in 1:10
            explorer(xs)
        end
        ϵ = 0.1 + (0.9 - 0.1) * exp(-1)
        @test isapprox(get_distribution(explorer, xs), [ϵ/5, ϵ/5, ϵ/5+(1-ϵ)/2, ϵ/5, ϵ/5+(1-ϵ)/2])

        for i in 1:100
            explorer(xs)
        end
        ϵ = 0.1
        @test isapprox(get_distribution(explorer, xs), [ϵ/5, ϵ/5, ϵ/5+(1-ϵ)/2, ϵ/5, ϵ/5+(1-ϵ)/2]; atol=1e-5)

        reset!(explorer)
        for i in 1:100
            @test mask[explorer(xs, mask)]
        end
    end
end

end