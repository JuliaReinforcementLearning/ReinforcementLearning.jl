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

gs.grads[:x] = [.0 0.0 0.0; .0 0.0 0.0]
clip_by_global_norm!(gs, ps, 4.0f0)
@test isapprox(gs[:x], [.0 0.0 0.0; .0 0.0 0.0])
end