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
