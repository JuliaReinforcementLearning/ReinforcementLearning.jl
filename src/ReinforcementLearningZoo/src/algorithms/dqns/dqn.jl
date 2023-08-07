export DQNLearner

using Random: AbstractRNG
using Functors: @functor

Base.@kwdef struct DQNLearner{A<:Approximator{<:TwinNetwork}, F, R} <: AbstractLearner
    approximator::A
    loss_func::F
    n::Int = 1
    γ::Float32 = 0.99f0
    is_enable_double_DQN::Bool = true
    rng::R = Random.default_rng()
    # for logging
    loss::Vector{Float32} = Float32[0.0f0]
end


@functor DQNLearner (approximator,)

RLCore.forward(L::DQNLearner, s::A) where {A<:AbstractArray}  = RLCore.forward(L.approximator, s)

function RLCore.optimise!(learner::DQNLearner, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        optimise!(learner, batch)
    end
end

function generate_q_function(n::Int64)
    @eval function q_function(r::Float32, γ::Float32, t::Bool, q_next_action::Float32)
        return r + γ ^ $n * (1f0 - t) * q_next_action
    end
    return q_function
end

function RLBase.optimise!(learner::DQNLearner, batch::NamedTuple)
    optimiser_state = learner.approximator.optimiser_state
    A = learner.approximator
    Q = A.model.source
    Qₜ = A.model.target

    γ = learner.γ
    loss_func = learner.loss_func
    n = learner.n

    s, s_next, a, r, t = map(x -> batch[x], SS′ART) |> Flux.gpu
    a = CartesianIndex.(a, 1:length(a))

    q_next = learner.is_enable_double_DQN ? Q(s_next) : Qₜ(s_next)

    if haskey(batch, :next_legal_actions_mask)
        q_next .+= ifelse.(batch[:next_legal_actions_mask], 0.0f0, typemin(Float32))
    end

    q_next_action = learner.is_enable_double_DQN ? Qₜ(s_next)[dropdims(argmax(q_next, dims=1), dims=1)] : dropdims(maximum(q_next; dims=1), dims=1)

    q_function_ = generate_q_function(n)
    R = q_function_.(r, (γ,), t, q_next_action)

    grads = gradient(Q) do Q
        qₐ = Q(s)[a]
        loss = loss_func(R, qₐ)
        ignore_derivatives() do
            learner.loss[1] = loss
        end
        loss
    end |> Flux.cpu

    # Optimization step
    Flux.update!(optimiser_state, Flux.cpu(Q), grads[1])
end
