export TDLearner

using LinearAlgebra: dot
using Distributions: pdf

Base.@kwdef struct TDLearner{A} <: AbstractLearner
    approximator::A
    γ::Float64 = 1.0
    method::Symbol
    n::Int = 0
end

(L::TDLearner)(env::AbstractEnv) = L.approximator(state(env))
(L::TDLearner)(s) = L.approximator(s)
(L::TDLearner)(s, a) = L.approximator(s, a)

## update policies

function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    if p.learner.method === :ExpectedSARSA && s === PRE_ACT_STAGE
        # A special case
        update!(p.learner, (t, pdf(prob(p, e))), e, s)
    else
        update!(p.learner, t, e, s)
    end
end


function RLBase.update!(L::TDLearner, t::AbstractTrajectory, ::AbstractEnv, s::PreActStage)
    _update!(L, L.approximator, Val(L.method), t, s)
end

function RLBase.update!(
    L::TDLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    s::PostEpisodeStage,
)
    _update!(L, L.approximator, Val(L.method), t, s)
end

# for ExpectedSARSA
function RLBase.update!(
    L::TDLearner,
    t::Tuple,
    ::AbstractEnv,
    s::Union{PreActStage,PostEpisodeStage},
)
    _update!(L, L.approximator, Val(L.method), t, s)
end

## update trajectories

function RLBase.update!(
    t::AbstractTrajectory,
    ::Union{
        QBasedPolicy{<:TDLearner},
        NamedPolicy{<:QBasedPolicy{<:TDLearner}},
        VBasedPolicy{<:TDLearner},
    },
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

## implementations

function _update!(
    L::TDLearner,
    ::Union{TabularQApproximator,LinearQApproximator},
    ::Union{Val{:SARSA},Val{:ExpectedSARSA},Val{:SARS}},
    t::Trajectory,
    ::PostEpisodeStage,
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0
    for i in 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s, a = S[end-i], A[end-i]
        update!(Q, (s, a) => Q(s, a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::Union{TabularQApproximator,LinearQApproximator},
    ::Val{:SARSA},
    t::Trajectory,
    ::PreActStage,
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′, a′ = S[end-n-1], A[end-n-1], S[end], A[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * Q(s′, a′)
        update!(Q, (s, a) => Q(s, a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::TabularQApproximator,
    ::Val{:ExpectedSARSA},
    experience,
    ::PreActStage,
)
    t, p = experience

    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * dot(Q(s′), p)
        update!(Q, (s, a) => Q(s, a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::Union{TabularQApproximator,LinearQApproximator},
    ::Val{:SARS},
    t::AbstractTrajectory,
    ::PreActStage,
)
    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * maximum(Q(s′))
        update!(Q, (s, a) => Q(s, a) - G)
    end
end

function _update!(
    L::TDLearner,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::Val{:SRS},
    t::Trajectory,
    ::PostEpisodeStage,
)
    S, R = t[:state], t[:reward]
    n, γ, V = L.n, L.γ, L.approximator
    G = 0.0
    w = 1.0
    for i in 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s = S[end-i]
        if haskey(t, :weight)
            w *= t[:weight][end-i]
        end
        update!(V, s => w * (V(s) - G))
    end
end

function _update!(
    L::TDLearner,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::Val{:SRS},
    t::AbstractTrajectory,
    ::PreActStage,
)
    S = t[:state]
    R = t[:reward]

    n, γ, V = L.n, L.γ, L.approximator
    if length(R) >= n + 1
        s, s′ = S[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * V(s′)
        if haskey(t, :weight)
            W = t[:weight]
            @views w = reduce(*, W[end-n-1:end-1])
        else
            w = 1.0
        end
        update!(V, s => w * (V(s) - G))
    end
end

#####
# DynaAgent
#####

function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    m::Union{ExperienceBasedSamplingModel,TimeBasedSamplingModel},
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::Union{PreActStage,PostEpisodeStage},
)
    if p.learner.method == :SARS
        transition = sample(m)
        if !isnothing(transition)
            s, a, r, t, s′ = transition
            traj = VectorSARTTrajectory()
            push!(traj; state = s, action = a, reward = r, terminal = t)
            push!(traj; state = s′, action = a)  # here a is a dummy one
            update!(p.learner, traj, env, t ? POST_EPISODE_STAGE : PRE_ACT_STAGE)
        end
    else
        @error "unsupported method $(p.learner.method)"
    end
end

function RLBase.update!(
    p::QBasedPolicy{<:TDLearner},
    m::PrioritizedSweepingSamplingModel,
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::Union{PreActStage,PostEpisodeStage},
)
    if p.learner.method == :SARS
        transition = sample(m)
        if !isnothing(transition)
            s, a, r, t, s′ = transition
            traj = VectorSARTTrajectory()
            push!(traj; state = s, action = a, reward = r, terminal = t)
            push!(traj; state = s′, action = a)  # here a is a dummy one
            update!(p.learner, traj, env, t ? POST_EPISODE_STAGE : PRE_ACT_STAGE)

            # update priority
            for (s̄, ā, r̄, d̄) in m.predecessors[s]
                P = RLBase.priority(p.learner, (s̄, ā, r̄, d̄, s))
                if P ≥ m.θ
                    m.PQueue[(s̄, ā)] = P
                end
            end
        end
    else
        @error "unsupported method $(p.learner.method)"
    end
end

function RLBase.priority(L::TDLearner, transition::Tuple)
    if L.method == :SARS
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        Δ = d ? (r - Q(s, a)) : (r + γ^(L.n + 1) * maximum(Q(s′)) - Q(s, a))
        Δ = [Δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
        abs(Δ[])
    else
        @error "unsupported method"
    end
end

#####
# TDλReturnLearner
#####

export TDλReturnLearner

Base.@kwdef struct TDλReturnLearner{Tapp<:AbstractApproximator} <: AbstractLearner
    approximator::Tapp
    γ::Float64 = 1.0
    λ::Float64
end

(L::TDλReturnLearner)(env::AbstractEnv) = L(state(env))
(L::TDλReturnLearner)(s) = L.approximator(s)
(L::TDλReturnLearner)(s, a) = L.approximator(s, a)

function RLBase.update!(
    L::TDλReturnLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PreActStage,
) end

function RLBase.update!(
    L::TDλReturnLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    λ, γ, V = L.λ, L.γ, L.approximator
    R = t[:reward]
    S = @view t[:state][1:end-1]
    S′ = @view t[:state][2:end]
    T = length(R)
    for t in 1:T
        G = 0.0
        for n in 1:(T-t)
            G +=
                λ^(n - 1) *
                (discount_rewards_reduced(@view(R[t:t+n-1]), γ) + γ^n * V(S′[t+n-1]))
        end
        G *= 1 - λ
        G +=
            λ^(T - t) *
            (discount_rewards_reduced(@view(R[t:T]), γ) + γ^(T - t + 1) * V(S′[T]))
        sₜ = S[t]
        update!(V, sₜ => V(sₜ) - G)
    end
end

function RLBase.update!(
    t::AbstractTrajectory,
    ::VBasedPolicy{<:TDλReturnLearner},
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end
