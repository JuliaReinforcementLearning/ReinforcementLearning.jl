export QRDQNLearner, quantile_huber_loss

using ChainRulesCore: ignore_derivatives
import Random
using StatsBase: mean
using Functors: @functor
using Flux
using Flux: gradient, params

function quantile_huber_loss(ŷ, y; κ=1.0f0)
    N, B = size(y)
    Δ = reshape(y, N, 1, B) .- reshape(ŷ, 1, N, B)
    abs_error = abs.(Δ)
    quadratic = min.(abs_error, κ)
    linear = abs_error .- quadratic
    huber_loss = 0.5f0 .* quadratic .* quadratic .+ κ .* linear

    cum_prob = ignore_derivatives() do
        send_to_device(device(y), range(0.5f0 / N; length=N, step=1.0f0 / N))
    end
    loss = ignore_derivatives(abs.(cum_prob .- (Δ .< 0))) .* huber_loss
    mean(sum(loss; dims=1))
end

Base.@kwdef mutable struct QRDQNLearner{A<:Approximator{<:TwinNetwork}, F, R} <: AbstractLearner
    approximator::A
    n_quantile::Int
    loss_func::F = quantile_huber_loss
    γ::Float32 = 0.99f0
    rng::R = Random.default_rng()
    # for recording
    loss::Float32 = 0.0f0
end

@functor QRDQNLearner (approximator,)

RLCore.forward(L::QRDQNLearner, s::A) where {A<:AbstractArray} = vec(mean(reshape(RLCore.forward(L.approximator, s), L.n_quantile, :), dims=1))

function RLBase.optimise!(learner::QRDQNLearner, batch::NamedTuple)
    A = learner.approximator
    Q = A.model.source
    Qₜ = A.model.target
    γ = learner.γ
    N = learner.n_quantile
    loss_func = learner.loss_func

    s, s′, a, r, t = map(x -> batch[x], SS′ART)
    batch_size = length(r)
    a = CartesianIndex.(a, 1:batch_size)

    target_quantiles = reshape(Qₜ(s′), N, :, batch_size)
    qₜ = dropdims(mean(target_quantiles; dims=1); dims=1)
    aₜ = dropdims(argmax(qₜ, dims=1); dims=1)
    @views target_quantile_aₜ = target_quantiles[:, aₜ]
    y = reshape(r, 1, batch_size) .+ γ .* reshape(1 .- t, 1, batch_size) .* target_quantile_aₜ

    gs = gradient(params(A)) do
        q = reshape(Q(s), N, :, batch_size)
        @views ŷ = q[:, a]

        loss = loss_func(ŷ, y)

        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)
end
