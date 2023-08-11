export DQNLearner

using Random: AbstractRNG
using Functors: @functor

Base.@kwdef mutable struct DQNLearner{A<:Union{Approximator,TargetNetwork}, F, R} <: AbstractLearner
    q_network::A
    loss_func::F
    n::Int = 1
    γ::Float32 = 0.99f0
    rng::R = Random.default_rng()
    # for logging
    loss::Float32 = 0.0f0
end

RLCore.forward(L::DQNLearner, s::A) where {A<:AbstractArray}  = RLCore.forward(L.q_network, s)

@functor DQNLearner (q_network,)

function RLCore.optimise!(learner::DQNLearner, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        optimise!(learner, batch)
    end
end

function RLBase.optimise!(learner::DQNLearner, batch::NamedTuple)
    Q = learner.q_network
    Qt = target(learner.q_network)
    
    γ = learner.γ
    loss_func = learner.loss_func
    n = learner.n

    s, s_next, a, r, t = map(x -> batch[x], SS′ART)
    a = CartesianIndex.(a, 1:length(a))
    s, s_next, a, r, t = send_to_device(device(Q), (s, s_next, a, r, t))

    q_next = forward(Qt, s_next)

    if haskey(batch, :next_legal_actions_mask)
        q_next .+= ifelse.(batch[:next_legal_actions_mask], 0.0f0, typemin(Float32))
    end

    q_next_action =  dropdims(maximum(q_next; dims=1), dims=1)

    R = r .+ γ^n .* (1 .- t) .* q_next_action

    gs = gradient(params(A)) do
        qₐ = forward(Q, s)[a]
        loss = loss_func(R, qₐ)
        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)
end
