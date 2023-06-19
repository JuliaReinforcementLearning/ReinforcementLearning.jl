export IQNLearner, ImplicitQuantileNet

using Functors: @functor
using Flux: params, unsqueeze, gradient
import Random
using StatsBase: mean
using ChainRulesCore: ignore_derivatives

"""
    ImplicitQuantileNet(;ψ, ϕ, header)

```
        quantiles (n_action, n_quantiles, batch_size)
           ↑
         header
           ↑
feature ↱  ⨀   ↰ transformed embedding
       ψ       ϕ
       ↑       ↑
       s        τ
```
"""
Base.@kwdef struct ImplicitQuantileNet{A,B,C}
    ψ::A
    ϕ::B
    header::C
end

@functor ImplicitQuantileNet

function (net::ImplicitQuantileNet)(s, emb)
    features = net.ψ(s)  # (n_feature, batch_size)
    emb_aligned = net.ϕ(emb)  # (n_feature, N * batch_size)
    merged = unsqueeze(features, dims=2) .* reshape(emb_aligned, size(features, 1), :, size(features, 2))  # (n_feature, N, batch_size)
    quantiles = net.header(reshape(merged, size(merged)[1:end-2]..., :)) # flattern last two dimension first
    reshape(quantiles, :, size(merged, 2), size(merged, 3))  # (n_action, N, batch_size)
end

Base.@kwdef mutable struct IQNLearner{A<:Approximator{<:TwinNetwork}, R1, R2} <: AbstractLearner
    approximator::A
    γ::Float32 = 0.99f0
    κ::Float32 = 1.0f0
    N::Int = 32
    N′::Int = 32
    Nₑₘ::Int = 64
    K::Int = 32
    rng::R1 = default_rng()
    device_rng::R2 = rng
    # for logging
    loss::Float32 = 0.0f0
end

@functor IQNLearner (approximator, device_rng)

embed(x, Nₑₘ) = cos.(Float32(π) .* (1:Nₑₘ) .* reshape(x, 1, :))

# the last dimension is batch_size
function RLCore.forward(learner::IQNLearner, s::A) where {A<:AbstractArray}
    batch_size = size(s)[end]
    τ = rand(learner.device_rng, Float32, learner.K, batch_size)
    τₑₘ = embed(τ, learner.Nₑₘ)
    quantiles = RLCore.forward(learner.approximator, s, τₑₘ)
    dropdims(mean(quantiles; dims=2); dims=2)
end

function RLCore.forward(L::IQNLearner, env::E) where {E<:AbstractEnv}
    s = env |> state |> send_to_device(L.approximator)
    q = s |> unsqueeze(dims=ndims(s) + 1) |> x -> RLCore.forward(L, x) |> vec
    q
end

function RLCore.optimise!(learner::IQNLearner, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        optimise!(learner, batch)
    end
end

function RLBase.optimise!(learner::IQNLearner, batch::NamedTuple)
    A = learner.approximator
    Z = A.model.source
    Zₜ = A.model.target
    N = learner.N
    N′ = learner.N′
    Nₑₘ = learner.Nₑₘ
    κ = learner.κ
    
    s, s′, a, r, t = map(x -> batch[x], SS′ART)
    batch_size = length(t)
    τ′ = rand(learner.device_rng, Float32, N′, batch_size)  # TODO: support β distribution
    τₑₘ′ = embed(τ′, Nₑₘ)
    zₜ = Zₜ(s′, τₑₘ′)
    avg_zₜ = mean(zₜ, dims=2)

    if haskey(batch, :next_legal_actions_mask)
        masked_value = similar(batch.next_legal_actions_mask, Float32)
        masked_value = fill!(masked_value, typemin(Float32))
        masked_value[batch.next_legal_actions_mask] .= 0
        avg_zₜ .+= masked_value
    end

    aₜ = argmax(avg_zₜ, dims=1)
    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0:0, 0:N′-1, 0:0)))
    qₜ = reshape(zₜ[aₜ], :, batch_size)
    target = reshape(r, 1, batch_size) .+ learner.γ * reshape(1 .- t, 1, batch_size) .* qₜ  # reshape to allow broadcast

    τ = rand(learner.device_rng, Float32, N, batch_size)
    τₑₘ = embed(τ, Nₑₘ)
    a = CartesianIndex.(repeat(a, inner=N), 1:(N*batch_size))

    gs = gradient(params(A)) do
        z_raw = Z(s, τₑₘ)
        z = reshape(z_raw, size(z_raw)[1:end-2]..., :)
        q = z[a]

        TD_error = reshape(target, N′, 1, batch_size) .- reshape(q, 1, N, batch_size)
        # can't apply huber_loss in RLCore directly here
        abs_error = abs.(TD_error)
        quadratic = min.(abs_error, κ)
        linear = abs_error .- quadratic
        huber_loss = 0.5f0 .* quadratic .* quadratic .+ κ .* linear

        # dropgrad
        raw_loss =
            abs.(reshape(τ, 1, N, batch_size) .- ignore_derivatives(TD_error .< 0)) .*
            huber_loss ./ κ
        loss_per_quantile = reshape(sum(raw_loss; dims=1), N, batch_size)
        loss_per_element = mean(loss_per_quantile; dims=1)  # use as priorities
        loss = mean(loss_per_element)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)
end
