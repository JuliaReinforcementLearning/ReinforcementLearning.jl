@testset "EpsilonGreedySelector" begin
    @testset "basic" begin
        Random.seed!(123)
        ϵ = 0.1
        s = EpsilonGreedySelector(ϵ)
        values = [1, 2, 0, -1]
        N = 10000
        action_counts = countmap([s(values) for i = 1:N])
        @test isapprox(action_counts[1] / N, ϵ / length(values), atol = 0.01)
        @test isapprox(action_counts[2] / N, 1 - ϵ + ϵ / length(values), atol = 0.01)
        @test isapprox(action_counts[3] / N, ϵ / length(values), atol = 0.01)
        @test isapprox(action_counts[4] / N, ϵ / length(values), atol = 0.01)
    end

    @testset "linear method" begin
        ϵ, ϵ_init, warmup_steps, decay_steps = 0.1, 0.9, 10, 4
        s = EpsilonGreedySelector{:linear}(
            ;
            ϵ_stable = ϵ,
            ϵ_init = ϵ_init,
            warmup_steps = warmup_steps,
            decay_steps = decay_steps,
        )

        for i = 1:warmup_steps
            @test ReinforcementLearning.get_ϵ(s, i) == ϵ_init
        end

        for (i, expected_ϵ) in zip(
            warmup_steps+1:warmup_steps+decay_steps,
            range(0.7, step = -0.2, 0.1),
        )
            @test ReinforcementLearning.get_ϵ(s, i) ≈ expected_ϵ
        end

        for i = warmup_steps+decay_steps:warmup_steps+decay_steps*2
            @test ReinforcementLearning.get_ϵ(s, i) == ϵ
        end
    end

    @testset "get_prob" begin
        ϵ = 0.4
        s = EpsilonGreedySelector(ϵ)
        values = [1, 2, 2, -1]
        @test get_prob(s, values) == [0.1, 0.4, 0.4, 0.1]
    end
end