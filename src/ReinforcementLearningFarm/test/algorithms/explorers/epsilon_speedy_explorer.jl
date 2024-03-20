using ReinforcementLearningFarm: EpsilonSpeedyExplorer, get_ϵ
using Random

@testset "EpsilonSpeedyExplorer" begin
    using Test

    @testset "EpsilonSpeedyExplorer" begin
        @testset "constructor" begin
            explorer = EpsilonSpeedyExplorer(0.1)
            @test explorer.β == 0.1
            @test explorer.β_neg == -0.1
            @test explorer.step[] == 1
            @test explorer.rng === Random.GLOBAL_RNG
        end
    
        @testset "get_ϵ" begin
            explorer = EpsilonSpeedyExplorer(0.1)
            @test get_ϵ(explorer) ≈ exp(-0.1)
            explorer.step[] = 10
            @test get_ϵ(explorer) ≈ exp(-1.0)
        end
    
        @testset "plan" begin
            explorer = EpsilonSpeedyExplorer(0.1)
            values = [1, 2, 3, 4, 5]
            mask = [true, false, true, false, true]
    
            @testset "without mask" begin
                action = RLBase.plan!(explorer, values)
                @test action ∈ 1:length(values)
            end
    
            @testset "with mask" begin
                action = RLBase.plan!(explorer, values, mask)
                @test action ∈ findall(mask)
            end
    
            @testset "with true mask" begin
                true_mask = [true, true, true, true, true]
                action = RLBase.plan!(explorer, values, true_mask)
                @test action ∈ findall(true_mask)
            end
        end
    
        @testset "prob" begin
            explorer = EpsilonSpeedyExplorer(0.1)
            values = [1, 2, 3, 4, 5]
            mask = [true, false, true, false, true]
    
            @testset "without mask" begin
                prob_dist = RLBase.prob(explorer, values)
                @test length(prob_dist.p) == length(values)
            end
    
            @testset "with mask" begin
                prob_dist = RLBase.prob(explorer, values, mask)
                @test length(prob_dist.p) == length(values)
            end
    
            @testset "with true mask" begin
                true_mask = [true, true, true, true, true]
                prob_dist = RLBase.prob(explorer, values, true_mask)
                @test length(prob_dist.p) == length(values)
            end
        end
    end
    
    @testset "EpsilonSpeedyExplorer correctness" begin
        explorer = RLFarm.EpsilonSpeedyExplorer(1e-5)
        explorer.step[] = Int(1e5)
        @test RLFarm.get_ϵ(explorer) ≈ 0.36787944117144233
    end
end

