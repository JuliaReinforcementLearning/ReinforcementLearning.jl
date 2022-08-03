using Distributions: DiscreteDistribution, ContinuousDistribution
using Flux: softmax, loadmodel!, params
using ReverseDiff: GradientTape, GradientConfig, gradient!, compile, DiffResults
import ReverseDiff
import Zygote
import ForwardDiff
using JLD2
using NNlib: logsoftmax

export action_distribution, policy_gradient_estimate, IsPolicyGradient
export conjugate_gradient!, kldiv

struct IsPolicyGradient end
IsPolicyGradient(::T) where T = IsPolicyGradient(T)
IsPolicyGradient(T::Type) = error("Type $T does not implement IsPolicyGradient")

"""
    action_distribution(dist, model_output)

Compute the action distribution using the distribution type and output from a model.
"""
action_distribution(dist, model_output) =
    throw(ArgumentError("dist ($dist) is not a ContinuousDistribution or DiscreteDistribution, not implemented"))

"""
    action_distribution(dist::Type{T}, model_output) where {T<:DiscreteDistribution}

When `dist` is a subtype of `DiscreteDistribution`, assume `model_output` is a batch of unnormalized log probabilities.
# Examples
```jldoctest
julia> model_output = reshape(1:10, 5, 2)
5×2 reshape(::UnitRange{Int64}, 5, 2) with eltype Int64:
 1   6
 2   7
 3   8
 4   9
 5  10
julia> action_distribution(Categorical, model_output)
2-element Vector{Categorical{Float64, Vector{Float64}}}:
 Categorical{Float64, Vector{Float64}}(
support: Base.OneTo(5)
p: [0.011656230956039605, 0.03168492079612427, 0.0861285444362687, 0.23412165725273662, 0.6364086465588308]
)

 Categorical{Float64, Vector{Float64}}(
support: Base.OneTo(5)
p: [0.011656230956039605, 0.03168492079612427, 0.0861285444362687, 0.23412165725273662, 0.6364086465588308]
)
```
"""
action_distribution(dist::Type{T}, model_output) where {T<:DiscreteDistribution} = 
    map(col -> dist(col; check_args=false), eachcol(softmax(model_output)))

"""
    action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution}

When `dist` is a subtype of `ContinuousDistribution`, assume `model_output` are a batch of parameters to be supplied to `dist`.
# Examples
```jldoctest
julia> model_output = reshape(1:10, 2, 5)
2×5 reshape(::UnitRange{Int64}, 2, 5) with eltype Int64:
 1  3  5  7   9
 2  4  6  8  10
julia> action_distribution(Normal, model_output)
5-element Vector{Normal{Float64}}:
 Normal{Float64}(μ=1.0, σ=2.0)
 Normal{Float64}(μ=3.0, σ=4.0)
 Normal{Float64}(μ=5.0, σ=6.0)
 Normal{Float64}(μ=7.0, σ=8.0)
 Normal{Float64}(μ=9.0, σ=10.0)
```
"""
action_distribution(dist::Type{T}, model_output) where {T<:ContinuousDistribution} = 
    map(col -> dist(col...), eachcol(model_output))

"""
    policy_gradient_estimate(policy::AbstractPolicy, states, actions, advantage)
Estimate the policy gradient from a batch of aligned states, actions, and advantages.
"""
policy_gradient_estimate(policy::AbstractPolicy, states, actions, advantage) =
    policy_gradient_estimate(IsPolicyGradient(policy), policy, states, actions, advantage)

function policy_gradient_estimate(::IsPolicyGradient, policy, states, actions, advantage)
    local action_distribut
    gs = gradient(params(policy.approximator)) do
        action_distribut = action_distribution(policy.dist, policy.approximator(states))
        total_loss = logpdf.(action_distribut, actions) .* advantage
        loss = -mean(total_loss)
        loss
    end
    gs, action_distribut
end

function gvp(f, θ, x)
	(Zygote.gradient(f, θ)[1])' * x
end

function hvp(f, θ, x)
	g(y) = gvp(f, y, x)
	inputs = (θ,)
	cfg = GradientConfig(inputs)
	ReverseDiff.gradient(g, inputs, cfg)[1]
end

function hvp_f(f, θ, x)
    g(y) = gvp(f, y, x)
    ForwardDiff.gradient(g, θ)
end

function gvp_rev(f, θ, x)
    inputs = (θ,)
    cfg = GradientConfig(inputs)
    ReverseDiff.gradient(f, inputs, cfg)[1]' * x
end

function gvp_direct(model, states, action_distribution, x)
    gs = Flux.gradient(Flux.params(model)) do 
        kld_direct(model, states, action_distribution)
    end
    mapreduce(vec, vcat, gs)' * x
end

function hvp_direct(model, states, action_distribution, x)
    gs = Flux.gradient(Flux.params(model)) do
        gvp_direct(model, states, action_distribution, x)
    end
    out = mapreduce(vec, vcat, gs)
    if any(isnan, out)
        jldsave("error.jld2"; model, states, action_distribution, x)
        error("hvp isnan, saved objects")
    end
    out .+ (1f-2 .* x)
end


function kld_direct(model, states, action_distribution)
    new_action_distribution = model(states)
    #println("action dist is $action_distribution,\nnew action dist is $new_action_distribution")
    map(eachcol(new_action_distribution), eachcol(action_distribution)) do a, b
        softmax(a) .* (logsoftmax(a) .- logsoftmax(b)) |> sum
    end |> mean
end

function backtrack_line_search(θₖ, α, δ, K, condition)
    #println("\nbacktracking_line_search begin:")
    #println("θₖ = $θₖ, α = $α, δ = $δ, K = $K")
    Δ = copy(δ)
    local θ
    for i in 1:K
        #println("try number $i")
        θ = θₖ + Δ
        condition(θ) && break
        Δ = Δ * α
        i == K && println("linesearch failed")
    end
    θ
end

function surrogate_advantage(model, states, actions, advantage, action_logits)
    π_θₖ = map(eachcol(softmax(action_logits)), actions) do a, b
        a[b]
    end
    π_θ = map(eachcol(softmax(model(states))), actions) do a, b
        a[b]
    end
    @show π_θₖ
    @show π_θ
    @show advantage
    (π_θ ./ π_θₖ) .* advantage |> mean
end

function conjugate_gradient(A, b; max_iter = 10, ϵ=1e-10)
    x = zeros(Float32, length(b))
    r = b - A(x)
    p = copy(r)
    rsold = r' * r

	iters = min(max_iter, length(b))

    for _ in 1:iters
        Ap = A(p)
        α = rsold / (p' * Ap)
        x .+= α .* p
        r .-= α .* Ap
        rsnew = r' * r
        if sqrt(rsnew) < ϵ
            break
        end
        p .= r .+ (rsnew ./ rsold) .* p
        rsold = rsnew
    end
    any(isnan, x) && error("cg isnan, b = $b")
    x
end