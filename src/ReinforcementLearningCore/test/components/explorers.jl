@testset "explorers" begin

    @testset "EpsilonGreedyExplorer" begin
        @testset "API" begin
            explorer = EpsilonGreedyExplorer(0.1; is_break_tie = true)
            Random.seed!(explorer, 123)

            values = [0, 1, 2, -1]
            target_prob =
                DiscreteNonParametric(1:length(values), [0.025, 0.025, 0.925, 0.025])

            # https://github.com/JuliaLang/julia/issues/10391#issuecomment-488642687
            # @test isapprox(get_prob(explorer, values), target_prob)
            @test isapprox(probs(get_prob(explorer, values)), probs(target_prob))

            actions = [explorer(values) for _ in 1:10000]
            action_counts = countmap(actions)

            @test all(isapprox.(
                [action_counts[i] for i in 1:length(values)] ./ 10000,
                probs(target_prob);
                atol = 0.005,
            ))

            explorer_copy = copy(explorer)
            reset!(explorer_copy)
            Random.seed!(explorer_copy, 123)

            new_actions = [explorer_copy(values) for _ in 1:10000]

            @test actions == new_actions
        end

        @testset "linear" begin
            explorer = EpsilonGreedyExplorer(;
                ϵ_stable = 0.1,
                ϵ_init = 0.9,
                warmup_steps = 3,
                decay_steps = 8,
                kind = :linear,
                is_break_tie = true,
            )
            xs = [0, 1, 2, -1, 2]
            mask = [true, true, false, true, false]
            E = [0.9, 0.9, range(0.9; stop = 0.1, step = -0.1)..., 0.1, 0.1, 0.1, 0.1]

            for ϵ in E
                @test RLCore.get_ϵ(explorer) ≈ ϵ
                @test isapprox(
                    probs(get_prob(explorer, xs)),
                    [ϵ / 5, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2],
                )
                explorer(xs)
            end

            reset!(explorer)

            for ϵ in E
                @test RLCore.get_ϵ(explorer) ≈ ϵ
                @test isapprox(
                    probs(get_prob(explorer, xs, mask)),
                    [ϵ / 3, (1 - ϵ) + ϵ / 3, 0.0, ϵ / 3, 0.0],
                )
                explorer(xs)
            end

            reset!(explorer)
            for i in 1:100
                @test mask[explorer(xs, mask)]
            end
        end

        @testset "exp" begin
            explorer = EpsilonGreedyExplorer(;
                ϵ_stable = 0.1,
                ϵ_init = 0.9,
                warmup_steps = 3,
                decay_steps = 8,
                kind = :exp,
                is_break_tie = true,
            )
            xs = [0, 1, 2, -1, 2]
            mask = [true, true, false, true, false]
            for i in 1:10
                explorer(xs)
            end
            ϵ = 0.1 + (0.9 - 0.1) * exp(-1)
            @test isapprox(
                probs(get_prob(explorer, xs)),
                [ϵ / 5, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2],
            )

            for i in 1:100
                explorer(xs)
            end
            ϵ = 0.1
            @test isapprox(
                probs(get_prob(explorer, xs)),
                [ϵ / 5, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2];
                atol = 1e-5,
            )

            reset!(explorer)
            for i in 1:100
                @test mask[explorer(xs, mask)]
            end
        end
    end

end
