export DQNLearner

using Random: AbstractRNG
using Functors: @functor

Base.@kwdef mutable struct DQNLearner{A<:Approximator{<:TwinNetwork}, F, R} <: AbstractLearner
    approximator::A
    loss_func::F
    n::Int = 1
    γ::Float32 = 0.99f0
    is_enable_double_DQN::Bool = true
    rng::R = Random.default_rng()
    # for logging
    loss::Float32 = 0.0f0
end

RLCore.forward(L::DQNLearner, s::A) where {A<:AbstractArray}  = RLCore.forward(L.approximator, s)

@functor DQNLearner (approximator,)

function RLBase.optimise!(learner::DQNLearner, batch::NamedTuple)
    A = learner.approximator
    Q = A.model.source
    Qₜ = A.model.target
    γ = learner.γ
    loss_func = learner.loss_func
    n = learner.n

    s, s′, a, r, t = map(x -> batch[x], SS′ART)
    a = CartesianIndex.(a, 1:length(a))

    q′ = learner.is_enable_double_DQN ? Q(s′) : Qₜ(s′)

    if haskey(batch, :next_legal_actions_mask)
        q′ .+= ifelse.(batch[:next_legal_actions_mask], 0.0f0, typemin(Float32))
    end

    q′ₐ = learner.is_enable_double_DQN ? Qₜ(s′)[dropdims(argmax(q′, dims=1), dims=1)] : dropdims(maximum(q′; dims=1), dims=1)

    G = r .+ γ^n .* (1 .- t) .* q′ₐ

    gs = gradient(params(A)) do
        qₐ = Q(s)[a]
        loss = loss_func(G, qₐ)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)
end
