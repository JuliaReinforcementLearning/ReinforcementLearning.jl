@testset "Zygote" begin
    grads = IdDict()
    grads[:x] = [-3.0 0.0 0.0; 4.0 0.0 0.0]
    ps = Zygote.Params([:x])
    gs = Zygote.Grads(grads, ps)
    clip_by_global_norm!(gs, ps, 4.0f0)
    @test isapprox(gs[:x], [-2.4 0.0 0.0; 3.2 0.0 0.0])

    gs.grads[:x] = [1.0 0.0 0.0; 1.0 0.0 0.0]
    clip_by_global_norm!(gs, ps, 4.0f0)
    @test isapprox(gs[:x], [1.0 0.0 0.0; 1.0 0.0 0.0])

    gs.grads[:x] = [0.0 0.0 0.0; 0.0 0.0 0.0]
    clip_by_global_norm!(gs, ps, 4.0f0)
    @test isapprox(gs[:x], [0.0 0.0 0.0; 0.0 0.0 0.0])
end


@testset "Distributions" begin
    @testset "normlogpdf" begin
        @test isapprox(logpdf(Normal(), 2), normlogpdf(0, 1, 2))
        @test isapprox(
            logpdf.([Normal(), Normal()], [2, 10]),
            normlogpdf([0, 0], [1, 1], [2, 10]),
        )

        # Test numeric stability for 0 sigma
        @test isnan(normlogpdf(0, 0, 2, ϵ = 0))
        @test !isnan(normlogpdf(0, 0, 2))

        if CUDA.functional()
            cpu_grad = Zygote.gradient([0.2, 0.5]) do x
                sum(logpdf.([Normal(1, 0.1), Normal(2, 0.2)], x))
            end
            gpu_grad = Zygote.gradient(cu([0.2, 0.5])) do x
                sum(normlogpdf(cu([1, 2]), cu([0.1, 0.2]), x))
            end
            @test isapprox(cpu_grad[1], gpu_grad[1] |> Array)
        end
    end
    @testset "mvnormlogpdf" begin
        softplus(x) = log(1 + exp(x))
        #2D,CPU
        μ = rand(5,1)
        L = tril(softplus.(rand(5,5)))
        Σ = L*L'
        x = zeros(5,3)
        d = MvNormal(vec(μ), Σ)
        logp_true = logpdf(d, x)
        logp = mvnormlogpdf(μ,L,x)
        @test logp_true ≈ logp
        g = Flux.gradient(Flux.Params([L])) do 
            mean(mvnormlogpdf(μ,L,x))
        end
        
        #3D,CPU
        
        μ = rand(20,1,128)
        L = mapslices(tril, softplus.(rand(20,20,128)), dims = (1,2))
        Σ = mapslices(l -> l*l', L, dims = (1,2))
        x = zeros(20,40,128)
        d = map(z -> MvNormal(vec(z[1]), Matrix(z[2])), zip(eachslice(μ, dims = 3), eachslice(Σ, dims =3)))

        logp_true = map(logpdf, d, eachslice(x, dims = 3))
        logp = mvnormlogpdf(μ,L,x)
        @test collect(dropdims(logp, dims = 1)') ≈ Flux.stack(logp_true,1)
        g = Flux.gradient(Flux.Params([L])) do 
            mean(mvnormlogpdf(μ,L,x))
        end
        #3D, GPU
        if CUDA.functional()
            μ_d = cu(μ)
            L_d = cu(L)
            x_d = cu(x)
            Σ_d = cu(Σ)
            logp_d = mvnormlogpdf(μ_d, L_d, x_d)
            @test logp ≈ Array(logp_d) atol = 0.001 #there is a fairly high numerical imprecision when working with CUDA. This is not due to the implementation of logdet as can be seen in the related test below.

            g_d = Flux.gradient(Flux.Params([L_d])) do 
                mean(mvnormlogpdf(μ_d,L_d,x_d))
            end
            CUDA.@allowscalar @test (mapslices(tril!, g_d[L_d], dims=(1,2)) |> Array) ≈ mapslices(tril!, g[L], dims=(1,2))
        end
    end
end

@testset "logdetLorU" begin
    A = rand(5,10)
    Σ = A*A'
    L = cholesky(Σ).L
    @test logdet(Σ) ≈ RLCore.logdetLorU(L)
    if CUDA.functional()
        @test logdet(Σ) ≈ RLCore.logdetLorU(cu(L)) atol = 1f-4
    end
end
