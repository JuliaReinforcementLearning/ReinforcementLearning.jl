using Test, LinearAlgebra, Distributions, ReinforcementLearningCore
import ReinforcementLearningCore: normlogpdf, mvnormlogpdf, diagnormlogpdf, mvnormkldivergence, diagnormkldivergence, normkldivergence

@testset "utils/distributions" begin
    #Scalar
    μ1 = 10f0
    σ1 = 5f0
    x = 4f0
    d1 = Normal(μ1,σ1)
    @test logpdf(d1, x) ≈ normlogpdf(μ1, σ1, x)
    μ2 = 11f0
    σ2 = 6f0
    x = 4f0
    d2 = Normal(μ2,σ2)
    @test kldivergence(d1, d2) ≈ normkldivergence(μ1, σ1, μ2, σ2)
    @test kldivergence(d2, d1) ≈ normkldivergence(μ2, σ2, μ1, σ1)
    
    #diagonal
    μ1 = [10f0, 10f0]
    σ1 = [5f0, 6f0]
    x = [4f0,4f0]
    d1 = MvNormal(μ1, LinearAlgebra.Diagonal(map(abs2, σ1)))
    @test logpdf(d1, x) ≈ diagnormlogpdf(μ1, σ1, x)
    μ2 = [11f0, 11f0]
    σ2 = [6f0, 6f0]
    x = [4f0, 4f0]
    d2 = MvNormal(μ2, LinearAlgebra.Diagonal(map(abs2, σ2)))
    @test kldivergence(d1, d2) ≈ diagnormkldivergence(μ1, σ1, μ2, σ2)
    @test kldivergence(d2, d1) ≈ diagnormkldivergence(μ2, σ2, μ1, σ1)

    #full
    μ1 = [10f0, 10f0]
    Σ1 = [2f0 -1f0; -1f0 2f0]
    L1 = cholesky(Σ1).L
    x = [4f0,4f0]
    d1 = MvNormal(μ1,Σ1)
    @test logpdf(d1, x) ≈ only(mvnormlogpdf(μ1, L1, x))
    μ2 = [11f0, 11f0]
    Σ2 = [2f0 -1f0; -1f0 2f0]
    L2 = cholesky(Σ2).L
    x = [4f0, 4f0]
    d2 = MvNormal(μ2,Σ2)
    @test kldivergence(d1, d2) ≈ mvnormkldivergence(μ1, L1, μ2, L2)
    @test kldivergence(d2, d1) ≈ mvnormkldivergence(μ2, L2, μ1, L1)
end
