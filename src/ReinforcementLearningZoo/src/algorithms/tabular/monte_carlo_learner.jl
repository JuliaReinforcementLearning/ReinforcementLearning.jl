export MonteCarloLearner,
    FIRST_VISIT,
    EVERY_VISIT,
    NO_SAMPLING,
    ORDINARY_IMPORTANCE_SAMPLING,
    WEIGHTED_IMPORTANCE_SAMPLING

using StatsBase: countmap

abstract type AbstractVisitType end
struct FirstVisit <: AbstractVisitType end
struct EveryVisit <: AbstractVisitType end
const FIRST_VISIT = FirstVisit()
const EVERY_VISIT = EveryVisit()

abstract type AbstractSamplingStyle end
struct NoSampling <: AbstractSamplingStyle end
struct OrdinaryImportanceSampling <: AbstractSamplingStyle end
struct WeightedImportanceSampling <: AbstractSamplingStyle end
const NO_SAMPLING = NoSampling()
const ORDINARY_IMPORTANCE_SAMPLING = OrdinaryImportanceSampling()
const WEIGHTED_IMPORTANCE_SAMPLING = WeightedImportanceSampling()

"""
    MonteCarloLearner(;kwargs...)

Use monte carlo method to estimate state value or state-action value.

# Fields
- `approximator`::[`TabularApproximator`](@ref), can be either
  `TabularVApproximator` or `TabularQApproximator`.
- `γ=1.0`, discount rate.
- `kind=FIRST_VISIT`. Optional values are `FIRST_VISIT` or `EVERY_VISIT`.
- `sampling=NO_SAMPLING`. Optional values are `NO_SAMPLING`,
  `WEIGHTED_IMPORTANCE_SAMPLING` or `ORDINARY_IMPORTANCE_SAMPLING`.
"""
Base.@kwdef struct MonteCarloLearner{A,K,S} <: AbstractLearner
    approximator::A
    γ::Float64 = 1.0
    kind::K = FIRST_VISIT
    sampling::S = NO_SAMPLING
end

(learner::MonteCarloLearner)(env::AbstractEnv) = learner(state(env))
(learner::MonteCarloLearner)(s) = learner.approximator(s)
(learner::MonteCarloLearner)(s, a) = learner.approximator(s, a)

function RLBase.update!(::VBasedPolicy{<:MonteCarloLearner}, ::AbstractTrajectory) end

"Only update at the end of an episode"
function RLBase.update!(
    p::VBasedPolicy{<:MonteCarloLearner},
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    update!(p.learner, t)
end

function RLBase.update!(
    L::MonteCarloLearner,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::PostEpisodeStage,
)
    update!(L, t)
end

"Empty the trajectory at the end of an episode"
function RLBase.update!(
    t::AbstractTrajectory,
    ::Union{
        VBasedPolicy{<:MonteCarloLearner},
        QBasedPolicy{<:MonteCarloLearner},
        NamedPolicy{<:VBasedPolicy{<:MonteCarloLearner}},
    },
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

function RLBase.update!(L::MonteCarloLearner, t::AbstractTrajectory)
    _update!(L.kind, L.approximator, L.sampling, L, t)
end

function _update!(
    ::FirstVisit,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::NoSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R = t[:state], t[:reward]
    V, G, γ = L.approximator, 0.0, L.γ
    state_counts = countmap(@view(S[1:end-1]))
    for i in length(R):-1:1
        s, r = S[i], R[i]
        G = γ * G + r
        if state_counts[S[i]] == 1
            update!(V, s => V(s) - G)  # first visit
        else
            state_counts[s] -= 1
        end
    end
end

function _update!(
    ::EveryVisit,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::NoSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R = t[:state], t[:reward]
    V, G, γ = L.approximator, 0.0, L.γ
    for i in length(R):-1:1
        s, r = S[i], R[i]
        G = γ * G + r
        update!(V, s => V(s) - G)
    end
end

function _update!(
    ::EveryVisit,
    ::TabularQApproximator,
    ::NoSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, A, R = t[:state], t[:action], t[:reward]
    γ, Q, G = L.γ, L.approximator, 0.0
    for i in length(R):-1:1
        s, a, r = S[i], A[i], R[i]
        G = γ * G + R[i]
        update!(Q, (s, a) => (Q(s, a) - G))
    end
end

function _update!(
    ::FirstVisit,
    ::TabularQApproximator,
    ::NoSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, A, R = t[:state], t[:action], t[:reward]
    γ, Q, G = L.γ, L.approximator, 0.0
    seen_pairs = countmap(zip(@view(S[1:end-1]), @view(A[1:end-1])))

    for i in length(R):-1:1
        s, a, r = S[i], A[i], R[i]
        pair = (s, a)
        G = γ * G + r
        if seen_pairs[pair] == 1  # first visit
            update!(Q, (s, a) => (Q(s, a) - G))
        else
            seen_pairs[pair] -= 1
        end
    end
end

function _update!(
    ::FirstVisit,
    ::Tuple{
        <:Union{TabularVApproximator,LinearVApproximator},
        <:Union{TabularVApproximator,LinearVApproximator},
    },
    ::OrdinaryImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W = t[:state], t[:reward], t[:weight]
    (V, G), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0
    seen_states = countmap(@view(S[1:end-1]))

    # @info "debug" S R W seen_states G V t[:action]
    for i in length(R):-1:1
        s, r = S[i], R[i]
        g = γ * g + r
        ρ *= W[i]
        if seen_states[s] == 1  # first visit
            update!(G, s => G(s) - ρ * g)
            update!(V, s => V(s) - G(s))
        else
            seen_states[s] -= 1
        end
    end
end

function _update!(
    ::FirstVisit,
    ::Tuple,
    ::WeightedImportanceSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R, W = t[:state], t[:reward], t[:weight]
    (V, G, Ρ), g, γ, ρ = L.approximator, 0.0, L.γ, 1.0
    seen_states = countmap(@view(S[1:end-1]))

    for i in length(R):-1:1
        s, r = S[i], R[i]
        g = γ * g + r
        ρ *= W[i]
        if seen_states[s] == 1  # first visit
            update!(G, s => G(s) - ρ * g)
            update!(Ρ, s => Ρ(s) - ρ)
            val = Ρ(s) == 0 ? 0 : G(s) / Ρ(s)
            update!(V, s => V(s) - val)
        else
            seen_states[s] -= 1
        end
    end
end
