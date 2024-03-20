using Test
using Distributions: Categorical
using ReinforcementLearningCore: EpsilonGreedyExplorer, GreedyExplorer, get_ϵ
using Random

@testset "EpsilonGreedyExplorer" begin
    @testset "get_ϵ for linear kind" begin
        @test get_ϵ(EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 50) ≈ 0.9
        @test get_ϵ(EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 100) ≈ 0.9
        @test get_ϵ(EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 150) ≈ 0.5
        @test get_ϵ(EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 200) ≈ 0.1
    end

    @testset "get_ϵ for exp kind" begin
        @test get_ϵ(EpsilonGreedyExplorer(kind=:exp, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 50) ≈ 0.9
        @test get_ϵ(EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 100) ≈ 0.9
        @test get_ϵ(EpsilonGreedyExplorer(kind=:exp, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 150) ≈ 0.5852245277701068
        @test get_ϵ(EpsilonGreedyExplorer(kind=:exp, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100), 2000) ≈ 0.1 atol=1e-2
    end

    @testset "EpsilonGreedyExplorer Tests" begin
        # Test plan! for is_break_tie=true
        rng = Random.default_rng(123)
        s = EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100, is_break_tie=true, rng=rng)
        values = [0.1, 0.5, 0.5, 0.3]
        actions = []
        for _ in 1:300
            push!(actions, RLBase.plan!(s, values))
        end
        @test length(unique(actions)) == 4
    end

    @testset "EpsilonGreedyExplorer Tests" begin
        # Test plan! for is_break_tie=false
        rng = Random.default_rng(123)
        s = EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100, is_break_tie=false, rng=rng)
        values = [0.1, 0.5, 0.5, 0.3]
        actions = []
        for _ in 1:300
            push!(actions, RLBase.plan!(s, values))
        end
        @test length(unique(actions)) == 4
    end

    @testset "prob for is_break_tie=true" begin
        s = EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100, is_break_tie=true)
        values = [0.1, 0.5, 0.5, 0.3]
        @test RLBase.prob(s, values) ≈ Categorical([0.225, 0.275, 0.275, 0.225])
        @test RLBase.prob(s, values, 2) ≈ 0.275
    end

    @testset "prob for is_break_tie=false" begin
        s = EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100, is_break_tie=false)
        values = [0.1, 0.5, 0.5, 0.3]
        @test RLBase.prob(s, values) ≈ Categorical([0.225, 0.32499999999999996, 0.225, 0.225])
        @test RLBase.prob(s, values, 2) ≈ 0.32500000000000007
    end
end

@testset "GreedyExplorer" begin
    @testset "plan!" begin
        s = GreedyExplorer()
        values = [0.1, 0.5, 0.5, 0.3]
        @test RLBase.plan!(s, values) == 2
    end

    @testset "prob" begin
        s = GreedyExplorer()
        values = [0.1, 0.5, 0.5, 0.3]
        @test RLBase.prob(s, values) ≈ Categorical([0.0, 1.0, 0.0, 0.0])
        @test RLBase.prob(s, values, 2) == 1.0
    end
end
