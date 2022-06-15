export DQNLearner

using Setfield: @set
using Random: AbstractRNG, GLOBAL_RNG
import Functors

Base.@kwdef mutable struct DQNLearner{A<:Approximator{<:TwinNetwork}} <: AbstractLearner
    approximator::A
    loss_func::Any
    n::Int = 1
    γ::Float32 = 0.99f0
    is_enable_double_DQN::Bool = true
    rng::AbstractRNG = GLOBAL_RNG
    # for logging
    loss::Float32 = 0.0f0
end

(L::DQNLearner)(s::AbstractArray) = L.approximator(s)

Functors.functor(x::DQNLearner) = (; approximator=x.approximator), y -> @set x.approximator = y.approximator

function RLBase.optimise!(learner::DQNLearner, batch::Union{NamedTuple{SS′ART},NamedTuple{SS′L′ART}})
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
        ignore() do
            learner.loss = loss
        end
        loss
    end

    optimise!(A, gs)
end
