"""
    mutable struct MonteCarlo <: AbstractReinforcementLearner
        ns::Int64 = 10
        na::Int64 = 4
        γ::Float64 = .9
        initvalue = 0.
        Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
        Q::Array{Float64, 2} = zeros(na, ns) + initvalue


Estimate Q values by averaging over returns.
"""
@with_kw struct MonteCarlo
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = .9
    initvalue = 0.
    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
    Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue
end
function defaultbuffer(learner::MonteCarlo, env, preprocessor)
    EpisodeBuffer(statetype = typeof(preprocessstate(preprocessor,
                                                     getstate(env)[1])))
end

export MonteCarlo

function update!(learner::MonteCarlo, buffer)
    rewards = buffer.rewards
    states = buffer.states
    actions = buffer.actions
    if learner.Q[actions[end-1], states[end-1]] == Inf64
        learner.Q[actions[end-1], states[end-1]] = 0.
    end
    if buffer.done[end]
        G = 0.
        for t in length(rewards):-1:1
            G = learner.γ * G + rewards[t]
            n = learner.Nsa[actions[t], states[t]] += 1
            learner.Q[actions[t], states[t]] *= (1 - 1/n)
            learner.Q[actions[t], states[t]] += 1/n * G
        end
    end
end
