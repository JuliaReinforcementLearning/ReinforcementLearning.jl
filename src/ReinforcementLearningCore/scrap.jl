using ReinforcementLearningCore
using Test, Flux, ChainRulesCore, LinearAlgebra, Distributions
using Flux: params, gradient
# CUDA, 
pre = Dense(20,15)
μ = Dense(15,10)
Σ = Dense(15,10*11÷2)
gn = CovGaussianNetwork(pre, μ, Σ)
@test Flux.params(gn) == Flux.Params([pre.weight, pre.bias, μ.weight, μ.bias, Σ.weight, Σ.bias])
state = rand(20,3) #batch of 3 states
#Check that it works in 2D
m, L = gn(state)
@test size(m) == (10,3)
@test size(L) == (10, 10,3)
a, logp = gn(state, is_sampling = true, is_return_log_prob = true)
@test size(a) == (10,3)
@test size(logp) == (1,3)
logp2d = gn(state,a)
@test size(logp2d) == (1,3)
#rest is 3D
m, L = gn(Flux.unsqueeze(state,dims = 2))
@test size(m) == (10,1,3)
@test size(L) == (10, 10,3)
a, logp = gn(Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
@test size(a) == (10,1,3)
@test size(logp) == (1,1,3)

@test logp ≈ mvnormlogpdf(m, L, a)
@test logp ≈ gn(Flux.unsqueeze(state,dims = 2), a)
as, logps = gn(Flux.unsqueeze(state,dims = 2), 5) #sample 5 actions
@test size(as) == (10,5,3)
@test size(logps) == (1,5,3)
logps2 = gn(Flux.unsqueeze(state,dims = 2), as)
@test logps2 ≈ logps
s = Flux.stack(map(l -> l*l', eachslice(L, dims=3)); dims=3)
mvnormals = map(z -> MvNormal(Array(vec(z[1])), Array(z[2])), zip(eachslice(m, dims = 3), eachslice(s, dims = 3)))
logp_truth = [logpdf(mvn, a) for (mvn, a) in zip(mvnormals, eachslice(as, dims = 3))]
@test Flux.stack(logp_truth; dims=2) ≈ dropdims(logps,dims = 1) #test against ground truth
action_saver = []
g = Flux.gradient(Flux.params(gn)) do 
    a, logp = gn(Flux.unsqueeze(state,dims = 2), is_sampling = true, is_return_log_prob = true)
    ChainRulesCore.ignore_derivatives() do 
        push!(action_saver, a)
    end
    mean(logp)
end
g2 = Flux.gradient(Flux.params(gn)) do
    logp = gn(Flux.unsqueeze(state,dims = 2), only(action_saver))
    mean(logp)
end
for (grad1, grad2) in zip(g,g2)
    @test grad1 ≈ grad2
end
empty!(action_saver)
g3 = Flux.gradient(Flux.params(gn)) do 
    a, logp = gn(Flux.unsqueeze(state,dims = 2), 3)
    ChainRulesCore.ignore_derivatives() do 
        push!(action_saver, a)
    end
    mean(logp)
end
g4 = Flux.gradient(Flux.params(gn)) do
    logp = gn(Flux.unsqueeze(state,dims = 2), only(action_saver))
    mean(logp)
end
for (grad1, grad2) in zip(g4,g3)
    @test grad1 ≈ grad2
end
