export REMDQNLearner

using Random: AbstractRNG, GLOBAL_RNG
using Flux
using Flux.Losses: huber_loss
using Flux: gradient, params
using Functors: @functor

Base.@kwdef mutable struct REMDQNLearner{A<:Union{Approximator,TargetNetwork}} <: AbstractLearner
    approximator::A
    ensemble_num::Int
    ensemble_method::Symbol
    n::Int = 1
    γ::Float32 = 0.99f0
    loss_func::Any = huber_loss
    rng::AbstractRNG = GLOBAL_RNG
    # for logging
    loss::Float32 = 0.0f0
end

@functor REMDQNLearner (approximator,)

function RLCore.forward(L::REMDQNLearner, s::A) where {A<:AbstractArray}
    q = reshape(RLCore.forward(L.approximator, s), :, L.ensemble_num)
    vec(mean(q, dims=2))
end

function RLCore.optimise!(learner::REMDQNLearner, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        optimise!(learner, batch)
    end
end

function RLBase.optimise!(learner::REMDQNLearner, batch::NamedTuple)
    A = learner.approximator
    Q = model(A)
    Qₜ = RLCore.target(A)
    γ = learner.γ
    n = learner.n
    loss_func = learner.loss_func
    ensemble_num = learner.ensemble_num

    # Build a convex polygon to make a combination of multiple Q-value estimates as a Q-value estimate.
    if learner.ensemble_method == :rand
        convex_polygon = rand(learner.rng, Float32, (1, ensemble_num))
    else
        convex_polygon = ones(Float32, (1, ensemble_num))
    end

    convex_polygon ./= sum(convex_polygon)

    s, s′, a, r, t = map(x -> batch[x], SS′ART)
    a = CartesianIndex.(a, 1:length(a))
    batchsize = length(a)

    qₜ = Qₜ(s′)
    qₜ = convex_polygon .* reshape(qₜ, :, ensemble_num, batchsize)
    qₜ = dropdims(sum(qₜ, dims=2), dims=2)

    if haskey(batch, :next_legal_actions_mask)
        qₜ .+= ifelse.(batch[:next_legal_actions_mask], 0.0f0, typemin(Float32))
    end

    q′ = dropdims(maximum(qₜ; dims=1), dims=1)
    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(A)) do
        q = Q(s)
        q = convex_polygon .* reshape(q, :, ensemble_num, batchsize)
        q = dropdims(sum(q, dims=2), dims=2)[a]

        loss = loss_func(G, q)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)
end

