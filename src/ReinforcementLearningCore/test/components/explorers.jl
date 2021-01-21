@testset "explorers" begin

    @testset "EpsilonGreedyExplorer" begin
        @testset "API" begin
            explorer = EpsilonGreedyExplorer(0.1; is_break_tie = true)
            Random.seed!(explorer, 123)

            values = [0, 1, 2, -1]
            tarprob = [0.025, 0.025, 0.925, 0.025]

            # https://github.com/JuliaLang/julia/issues/10391#issuecomment-488642687
            # @test isapprox(prob(explorer, values), tarprob)
            @test isapprox(probs(prob(explorer, values)), tarprob)

            actions = [explorer(values) for _ in 1:10000]
            action_counts = countmap(actions)

            @test all(
                isapprox.(
                    [action_counts[i] for i in 1:length(values)] ./ 10000,
                    tarprob;
                    atol = 0.005,
                ),
            )
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
                    probs(prob(explorer, xs)),
                    [ϵ / 5, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2],
                )
                explorer(xs)
            end

            explorer = EpsilonGreedyExplorer(;
                ϵ_stable = 0.1,
                ϵ_init = 0.9,
                warmup_steps = 3,
                decay_steps = 8,
                kind = :linear,
                is_break_tie = true,
            )
            for ϵ in E
                @test RLCore.get_ϵ(explorer) ≈ ϵ
                @test isapprox(
                    probs(prob(explorer, xs, mask)),
                    [ϵ / 3, (1 - ϵ) + ϵ / 3, 0.0, ϵ / 3, 0.0],
                )
                explorer(xs)
            end

            explorer = EpsilonGreedyExplorer(;
                ϵ_stable = 0.1,
                ϵ_init = 0.9,
                warmup_steps = 3,
                decay_steps = 8,
                kind = :linear,
                is_break_tie = true,
            )
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
                probs(prob(explorer, xs)),
                [ϵ / 5, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2],
            )

            for i in 1:100
                explorer(xs)
            end
            ϵ = 0.1
            @test isapprox(
                probs(prob(explorer, xs)),
                [ϵ / 5, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2, ϵ / 5, ϵ / 5 + (1 - ϵ) / 2];
                atol = 1e-5,
            )
        end
    end

end
