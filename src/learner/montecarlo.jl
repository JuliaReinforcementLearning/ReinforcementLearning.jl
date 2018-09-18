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
    EpisodeTurnBuffer{typeof(getstate(env).observation), typeof(actionspace(env)), Float64, Bool}()
end

export MonteCarlo

function update!(learner::MonteCarlo, buffer)
    if learner.Q[buffer[:actions, end], buffer[:states, end]] == Inf64
        learner.Q[buffer[:actions, end], buffer[:states, end]] = 0.
    end
    if buffer[:isdone, end]
        G = 0.
        for t in length(buffer):-1:1
            turn = buffer[t]
            G = learner.γ * G + buffer[:rewards, t]
            n = learner.Nsa[buffer[:actions, t], buffer[:states,t]] += 1
            learner.Q[buffer[:actions, t], buffer[:states, t]] *= (1 - 1/n)
            learner.Q[buffer[:actions, t], buffer[:states, t]] += 1/n * G
        end
    end
end
