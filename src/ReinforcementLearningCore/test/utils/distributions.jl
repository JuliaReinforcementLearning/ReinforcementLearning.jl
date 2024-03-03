using Test, LinearAlgebra, Distributions, Flux

@testset "utils/distributions" begin
    @testset "logdetLorU" begin
        M = [2f0 -1f0; -1f0 2f0]
        C = cholesky(M)
        L, U = C.L, C.U
        logdetM = logdet(M)
        @test logdetM == ReinforcementLearningCore.logdetLorU(L) 
        @test logdetM == ReinforcementLearningCore.logdetLorU(U)
        if (@isdefined CUDA) && CUDA.functional()
            L_d = cu(L)
            U_d = cu(U)
            @test logdetM == ReinforcementLearningCore.logdetLorU(L_d) 
            @test logdetM == ReinforcementLearningCore.logdetLorU(U_d)
        end
    end
    @testset "On CPU" begin
        @testset "1D Gaussian" begin
            μ1 = 10f0
            σ1 = 5f0
            x = 4f0
            d1 = Normal(μ1,σ1)
            @testset "normlogpdf" begin 
                @test logpdf(d1, x) ≈ normlogpdf(μ1, σ1, x)
                g = gradient(x) do x
                    sum(normlogpdf(μ1, σ1, x))
                end
            end
            μ2 = 11f0
            σ2 = 6f0
            x = 4f0
            d2 = Normal(μ2,σ2)
            @testset "normkldivergence" begin
                @test kldivergence(d1, d2) ≈ normkldivergence(μ1, σ1, μ2, σ2)
                @test kldivergence(d2, d1) ≈ normkldivergence(μ2, σ2, μ1, σ1)
            end
            g = gradient(μ1) do μ1
                sum(normkldivergence(μ1, σ1, μ2, σ2))
            end
        end
        @testset "Diagonal Gaussian" begin
            @testset "2D" begin
                μ1 = [10f0 10f0; 1f0 1f0]
                σ1 = [5f0 5f0; 6f0 6f0]
                x = [4f0 4f0; 3f0 3f0]
                d1 = MvNormal(μ1[:, 1], LinearAlgebra.Diagonal(map(abs2, σ1[:, 1])))
                μ2 = [11f0 11f0; 1f0 1f0]
                σ2 = [6f0 6f0; 5f0 5f0]
                d2 = MvNormal(μ2[:, 1], LinearAlgebra.Diagonal(map(abs2, σ2[:, 1])))
                @testset "diagnormlogpdf" begin
                    logpdfs = diagnormlogpdf(μ1, σ1, x)
                    for i in 1:2
                        @test size(logpdfs) == (1, 2)
                        @test logpdf(d1, x[:, i]) ≈ logpdfs[i]
                    end
                    g = gradient(x) do x
                        sum(diagnormlogpdf(μ1, σ1, x))
                    end
                end
                @testset "diagnormkldivergence" begin 
                    @test all(kldivergence(d1, d2) .≈ diagnormkldivergence(μ1, σ1, μ2, σ2))
                    @test all(kldivergence(d2, d1) .≈ diagnormkldivergence(μ2, σ2, μ1, σ1))
                    g = gradient(μ1) do μ1
                        sum(diagnormkldivergence(μ1, σ1, μ2, σ2))
                    end
                end
            end
            @testset "3D" begin
                μ1 = [10f0; 1f0;;; 10f0; 1f0]
                σ1 = [5f0; 6f0;;; 5f0 ;6f0]
                x = [4f0; 3f0;;; 4f0; 3f0]
                d1 = MvNormal(μ1[:, 1, 1], LinearAlgebra.Diagonal(map(abs2, σ1[:, 1, 1])))
                μ2 = [11f0; 1f0;;; 11f0; 1f0]
                σ2 = [6f0; 5f0;;; 6f0; 5f0]
                d2 = MvNormal(μ2[:, 1, 1], LinearAlgebra.Diagonal(map(abs2, σ2[:, 1, 1])))
                @testset "diagnormlogpdf" begin
                    logpdfs = diagnormlogpdf(μ1, σ1, x)
                    for i in 1:2
                        @test size(logpdfs) == (1, 1, 2)
                        @test logpdf(d1, x[:, 1, i]) ≈ logpdfs[i]
                    end
                    g = gradient(x) do x
                        sum(diagnormlogpdf(μ1, σ1, x))
                    end
                end
                @testset "diagnormkldivergence" begin 
                    @test all(kldivergence(d1, d2) .≈ diagnormkldivergence(μ1, σ1, μ2, σ2))
                    @test all(kldivergence(d2, d1) .≈ diagnormkldivergence(μ2, σ2, μ1, σ1))
                    g = gradient(μ1) do μ1
                        sum(diagnormkldivergence(μ1, σ1, μ2, σ2))
                    end
                end
            end
        end
        @testset "Full covariance Gaussian" begin
            #Only exists in 3D
            μ1 = [10f0; 1f0 ;;; 10f0; 1f0]
            Σ1 = [2f0 -1f0; -1f0 2f0]
            _L1 = cholesky(Σ1).L
            L1 = [_L1 ;;; _L1]
            x = [4f0; 3f0;;;4f0; 3f0]
            d1 = MvNormal(μ1[:, 1, 1],Σ1[:,:,1])
            μ2 = [11f0 ;2f0;;; 11f0; 2f0]
            Σ2 = [2f0 ;-1f0;; -1f0; 2f0]
            _L2 = cholesky(Σ2).L
            L2 = [_L2;;; _L2]
            d2 = MvNormal(μ2[:, 1, 1],Σ2[:,:,1])
            @testset "mvnormlogpdf" begin
                logpdfs = mvnormlogpdf(μ1, L1, x)
                @test size(logpdfs) == (1, 1, 2)
                for i in 1:2
                    @test logpdf(d1, x[:, 1, i]) ≈ logpdfs[i]
                end
                g = gradient(x) do x
                    sum(mvnormlogpdf(μ1, L1, x))
                end
            end
            @testset "mvnormkldivergence" begin 
                @test all(kldivergence(d1, d2) .≈ mvnormkldivergence(μ1, L1, μ2, L2))
                @test all(kldivergence(d2, d1) .≈ mvnormkldivergence(μ2, L2, μ1, L1))
                g = gradient(μ1) do μ1
                    sum(mvnormkldivergence(μ1, L1, μ2, L2))
                end
            end
        end
    end
    @testset "CUDA" begin
        if (@isdefined CUDA) && CUDA.functional()
            CUDA.allowscalar(false)
            #These only test GPU compatibility, exactness of results is tested above on the CPU
            @testset "Diagonal Gaussian" begin
                @testset "2D" begin
                    μ1 = [10f0; 1f0;; 10f0; 1f0]
                    σ1 = [5f0; 6f0;; 5f0; 6f0]
                    x = [4f0; 3f0;; 4f0; 3f0]
                    d1 = MvNormal(μ1[:, 1], LinearAlgebra.Diagonal(map(abs2, σ1[:, 1])))
                    μ2 = [11f0; 1f0;; 11f0; 1f0]
                    σ2 = [6f0; 5f0;; 6f0; 5f0]
                    d2 = MvNormal(μ2[:, 1], LinearAlgebra.Diagonal(map(abs2, σ2[:, 1])))
                    @testset "diagnormlogpdf" begin
                        logpdfs = diagnormlogpdf(cu(μ1), cu(σ1), cu(x)) |> collect
                        @test size(logpdfs) == (1, 2)
                        for i in 1:2
                            @test logpdf(d1, x[:, 1]) ≈ logpdfs[i]
                        end
                        g = gradient(x) do x
                            sum(diagnormlogpdf(cu(μ1), cu(σ1), cu(x)))
                        end
                    end
                    @testset "diagnormkldivergence" begin 
                        @test all(kldivergence(d1, d2) .≈ collect(diagnormkldivergence(cu(μ1), cu(σ1), cu(μ2), cu(σ2))))
                        @test all(kldivergence(d2, d1) .≈ collect(diagnormkldivergence(cu(μ2), cu(σ2), cu(μ1), cu(σ1))))
                        g = gradient(μ1) do μ1
                            sum(diagnormkldivergence(cu(μ1), cu(σ1), cu(μ2), cu(σ2)))
                        end
                    end
                end
                @testset "3D" begin
                    μ1 = [10f0; 1f0;;; 10f0; 1f0]
                    σ1 = [5f0; 6f0;;; 5f0 ;6f0]
                    x = [4f0; 3f0;;; 4f0; 3f0]
                    d1 = MvNormal(μ1[:, 1, 1], LinearAlgebra.Diagonal(map(abs2, σ1[:, 1, 1])))
                    μ2 = [11f0; 1f0;;; 11f0; 1f0]
                    σ2 = [6f0; 5f0;;; 6f0; 5f0]
                    d2 = MvNormal(μ2[:, 1, 1], LinearAlgebra.Diagonal(map(abs2, σ2[:, 1, 1])))
                    @testset "diagnormlogpdf" begin
                        logpdfs = diagnormlogpdf(cu(μ1), cu(σ1), cu(x)) |> collect
                        @test size(logpdfs) == (1,1,2)
                        for i in 1:2
                            @test logpdf(d1, x[:,1, i]) ≈ logpdfs[i]
                        end
                        g = gradient(x) do x
                            sum(diagnormlogpdf(cu(μ1), cu(σ1), cu(x)))
                        end
                    end
                    @testset "diagnormkldivergence" begin 
                        @test all(kldivergence(d1, d2) .≈ diagnormkldivergence(cu(μ1), cu(σ1), cu(μ2), cu(σ2)))
                        @test all(kldivergence(d2, d1) .≈ diagnormkldivergence(cu(μ2), cu(σ2), cu(μ1), cu(σ1)))
                        g = gradient(μ1) do μ1
                            sum(diagnormkldivergence(cu(μ1), cu(σ1), cu(μ2), cu(σ2)))
                        end
                    end

                end
            end
            @testset "Full covariance Gaussian" begin
                #Only exists in 3D
                μ1 = [10f0; 1f0 ;;; 10f0; 1f0]
                Σ1 = [2f0 -1f0; -1f0 2f0]
                _L1 = cholesky(Σ1).L
                L1 = [_L1 ;;; _L1]
                x = [4f0; 3f0;;;4f0; 3f0]
                d1 = MvNormal(μ1[:, 1, 1],Σ1[:,:,1])
                μ2 = [11f0 ;2f0;;; 11f0; 2f0]
                Σ2 = [2f0 ;-1f0;; -1f0; 2f0]
                _L2 = cholesky(Σ2).L
                L2 = [_L2;;; _L2]
                d2 = MvNormal(μ2[:, 1, 1],Σ2[:,:,1])
                @testset "mvnormlogpdf" begin
                    logpdfs = mvnormlogpdf(cu(μ1), cu(L1), cu(x))
                    @test size(logpdfs) == (1, 1, 2)
                    for i in 1:2
                        @test logpdf(d1, x[:, 1, i]) ≈ collect(logpdfs)[i]
                    end
                    g = gradient(x) do x
                        sum(mvnormlogpdf(cu(μ1), cu(L1), cu(x)))
                    end
                end
                @testset "mvnormkldivergence" begin 
                    @test all(kldivergence(d1, d2) .≈ mvnormkldivergence(cu(μ1), cu(L1), cu(μ2), cu(L2)))
                    @test all(kldivergence(d2, d1) .≈ mvnormkldivergence(cu(μ2), cu(L2), cu(μ1), cu(L1)))
                    g = gradient(μ1) do μ1
                        sum(mvnormkldivergence(cu(μ1), cu(L1), cu(μ2), cu(L2)))
                    end
                end
            end
        end
    end
end
