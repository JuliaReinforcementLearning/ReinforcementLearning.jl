using Test, LinearAlgebra, Distributions, ReinforcementLearningCore

@testset "utils/distributions" begin
    @testset "On CPU" begin
        @testset "1D Gaussian" begin
            μ1 = 10f0
            σ1 = 5f0
            x = 4f0
            d1 = Normal(μ1,σ1)
            @testset "normlogpdf" begin 
                @test logpdf(d1, x) ≈ normlogpdf(μ1, σ1, x)
            end
            μ2 = 11f0
            σ2 = 6f0
            x = 4f0
            d2 = Normal(μ2,σ2)
            @testset "normkldivergence" begin
                @test kldivergence(d1, d2) ≈ normkldivergence(μ1, σ1, μ2, σ2)
                @test kldivergence(d2, d1) ≈ normkldivergence(μ2, σ2, μ1, σ1)
            end
        end
        @testset "Diagonal Gaussian" begin
            μ1 = [10f0, 10f0]
            σ1 = [5f0, 6f0]
            x = [4f0,4f0]
            d1 = MvNormal(μ1, LinearAlgebra.Diagonal(map(abs2, σ1)))
            μ2 = [11f0, 11f0]
            σ2 = [6f0, 6f0]
            x = [4f0, 4f0]
            d2 = MvNormal(μ2, LinearAlgebra.Diagonal(map(abs2, σ2)))
            @testset "diagnormlogpdf" begin
                @test logpdf(d1, x) ≈ diagnormlogpdf(μ1, σ1, x)
            end
            @testset "diagnormkldivergence" begin 
                @test kldivergence(d1, d2) ≈ diagnormkldivergence(μ1, σ1, μ2, σ2)
                @test kldivergence(d2, d1) ≈ diagnormkldivergence(μ2, σ2, μ1, σ1)
            end
        end
        @testset "Full covariance Gaussian" begin
            μ1 = [10f0, 10f0]
            Σ1 = [2f0 -1f0; -1f0 2f0]
            L1 = cholesky(Σ1).L
            x = [4f0,4f0]
            d1 = MvNormal(μ1,Σ1)
            μ2 = [11f0, 11f0]
            Σ2 = [2f0 -1f0; -1f0 2f0]
            L2 = cholesky(Σ2).L
            x = [4f0, 4f0]
            d2 = MvNormal(μ2,Σ2)
            @testset "mvnormlogpdf" begin
                @test logpdf(d1, x) ≈ only(mvnormlogpdf(μ1, L1, x))
            end
            @testset "mvnormkldivergence" begin
                @test kldivergence(d1, d2) ≈ mvnormkldivergence(μ1, L1, μ2, L2)
                @test kldivergence(d2, d1) ≈ mvnormkldivergence(μ2, L2, μ1, L1)
            end
        end
    end
    @testset "CUDA" begin
        if CUDA.functional()
            CUDA.allowscalar(false)
            @testset "1D Gaussian" begin
                μ1 = 10f0
                σ1 = 5f0
                x = 4f0
                d1 = Normal(μ1,σ1)
                @testset "normlogpdf" begin 
                    @test logpdf(d1, x) ≈ normlogpdf(cu(μ1), cu(σ1), cu(x))
                end
                μ2 = 11f0
                σ2 = 6f0
                x = 4f0
                d2 = Normal(μ2,σ2)
                @testset "normkldivergence" begin
                    @test kldivergence(d1, d2) ≈ normkldivergence(cu(μ1), cu(σ1), cu(μ2), cu(σ2))
                    @test kldivergence(d2, d1) ≈ normkldivergence(cu(μ2), cu(σ2), cu(μ1), cu(σ1))
                end
            end
            @testset "Diagonal Gaussian" begin
                μ1 = [10f0, 10f0]
                σ1 = [5f0, 6f0]
                x = [4f0,4f0]
                d1 = MvNormal(μ1, LinearAlgebra.Diagonal(map(abs2, σ1)))
                μ2 = [11f0, 11f0]
                σ2 = [6f0, 6f0]
                x = [4f0, 4f0]
                d2 = MvNormal(μ2, LinearAlgebra.Diagonal(map(abs2, σ2)))
                @testset "diagnormlogpdf" begin
                    @test logpdf(d1, x) ≈ diagnormlogpdf(cu(μ1), cu(σ1), cu(x))
                end
                @testset "diagnormkldivergence" begin 
                    @test kldivergence(d1, d2) ≈ diagnormkldivergence(cu(μ1), cu(σ1), cu(μ2), cu(σ2))
                    @test kldivergence(d2, d1) ≈ diagnormkldivergence(cu(μ2), cu(σ2), cu(μ1), cu(σ1))
                end
            end
            @testset "Full covariance Gaussian" begin
                μ1 = [10f0, 10f0]
                Σ1 = [2f0 -1f0; -1f0 2f0]
                L1 = cholesky(Σ1).L
                x = [4f0,4f0]
                d1 = MvNormal(μ1,Σ1)
                μ2 = [11f0, 11f0]
                Σ2 = [2f0 -1f0; -1f0 2f0]
                L2 = cholesky(Σ2).L
                x = [4f0, 4f0]
                d2 = MvNormal(μ2,Σ2)
                @testset "mvnormlogpdf" begin
                    @test logpdf(d1, x) ≈ sum(mvnormlogpdf(cu(μ1), cu(L1), cu(x)))
                end
                @testset "mvnormkldivergence" begin
                    @test kldivergence(d1, d2) ≈ mvnormkldivergence(cu(μ1), cu(L1), cu(μ2), cu(L2))
                    @test kldivergence(d2, d1) ≈ mvnormkldivergence(cu(μ2), cu(L2), cu(μ1), cu(L1))
                end
            end
        end
    end
end
