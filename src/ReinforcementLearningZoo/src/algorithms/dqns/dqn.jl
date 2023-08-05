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

function RLCore.optimise!(learner::DQNLearner, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        optimise!(learner, batch)
    end
end

function RLBase.optimise!(learner::DQNLearner, batch::NamedTuple)
    A = learner.approximator
    Q = A.model.source
    Qₜ = A.model.target

    
    γ = learner.γ
    loss_func = learner.loss_func
    n = learner.n

    s, s_next, a, r, t = map(x -> batch[x], SS′ART)
    a = CartesianIndex.(a, 1:length(a))
    s, s_next, a, r, t = gpu((s, s_next, a, r, t))

    q_next = learner.is_enable_double_DQN ? Q(s_next) : Qₜ(s_next)

    if haskey(batch, :next_legal_actions_mask)
        q_next .+= ifelse.(batch[:next_legal_actions_mask], 0.0f0, typemin(Float32))
    end

    q_next_action = learner.is_enable_double_DQN ? Qₜ(s_next)[dropdims(argmax(q_next, dims=1), dims=1)] : dropdims(maximum(q_next; dims=1), dims=1)

    R = r .+ γ^n .* (1 .- t) .* q_next_action

    gs = gradient(params(A)) do
        qₐ = Q(s)[a]
        loss = loss_func(R, qₐ)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)
end
