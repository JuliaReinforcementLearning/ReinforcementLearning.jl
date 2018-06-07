@with_kw mutable struct TDLearner{T,Tp}
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = .9
    λ::Float64 = .8
    α::Float64 = .1
    nsteps::Int64 = 1
    initvalue::Float64 = 0.
    unseenvalue::Float64 = initvalue == Inf64 ? 0. : initvalue
    params::Array{Float64, 2} = zeros(na, ns) + initvalue
    tracekind = DataType = ReplacingTraces
    traces::T = λ == 0 || tracekind == NoTraces ? NoTraces() : tracekind(ns, na, λ, γ)
    endvaluepolicy::Tp = :Sarsa
end
Sarsa(; kargs...) = TDLearner(; kargs...)
QLearning(; kargs...) = TDLearner(; endvaluepolicy = :QLearning, kargs...)
ExpectedSarsa(; kargs...) = TDLearner(; endvaluepolicy = VeryOptimisticEpsilonGreedyPolicy(.1), kargs...)
export Sarsa, QLearning, ExpectedSarsa

@inline function selectaction(learner::Union{TDLearner, AbstractPolicyGradient}, 
                              policy,
                              state)
    selectaction(policy, getvalue(learner.params, state))
end
params(learner::TDLearner) = learner.params
reconstructwithparams(learner::TDLearner, w) = reconstruct(learner, params = w)

# td error

@inline getvaluecheckinf(learner, a, s) = checkinf(learner, getvalue(learner.params, a, s))
@inline getvaluecheckinf(learner, a, s::AbstractArray) = getvalue(learner.params, a, s)
@inline checkinf(learner, value) = (value == Inf64 ? learner.unseenvalue : value)

@inline function futurevalue(learner, buffer)
    a = buffer.actions[end]
    s = buffer.states[end]
    if learner.endvaluepolicy == :QLearning
        checkinf(learner, maximumbelowInf(getvalue(learner.params, s)))
    elseif learner.endvaluepolicy == :Sarsa
        getvaluecheckinf(learner, a, s)
    else
        actionprobabilites = getactionprobabilities(learner.endvaluepolicy,
                                                    getvalue(learner.params, s))
        m = 0.
        for (a, w) in enumerate(actionprobabilites)
            if w != 0.
                m += w * getvaluecheckinf(learner, a, s)
            end
        end
        m
    end
end

@inline function discountedrewards(rewards, done, γ)
    gammaeff = 1.
    discr = 0.
    for (r, done) in zip(rewards, done)
        discr += gammaeff * r
        done && return [discr; 0.]
        gammaeff *= γ
    end
    [discr; gammaeff]
end
@inline function tderror(rewards, done, γ, startvalue, endvalue)
    discr, gammaeff = discountedrewards(rewards, done, γ)
    discr + gammaeff * endvalue - startvalue
end

function tderror(learner, buffer)
    tderror(buffer.rewards, buffer.done, learner.γ,
            getvaluecheckinf(learner, buffer.actions[1], buffer.states[1]),
            futurevalue(learner, buffer))
end

# update params

@inline function updateparam!(learner, s, a, δ)
    if learner.params[a, s] == Inf64
        learner.params[a, s] = learner.unseenvalue + δ
    else
        learner.params[a, s] += learner.α * δ
    end
end
@inline function updateparam!(learner, s::Vector, a, δ)
    na, ns = size(learner.params)
    BLAS.axpy!(learner.α * δ, s, 1:ns, learner.params, a:na:na * (ns - 1) + a)
end
@inline function updateparam!(learner, s::SparseVector, a, δ)
    @simd for i in 1:length(s.nzind)
        learner.params[a, s.nzind[i]] += learner.α * δ * s.nzval[i]
    end
end

@inline updatetraceandparams!(learner::TDLearner{NoTraces, <:Any}, s, a, δ, done) =
    updateparam!(learner, s, a, δ)
@inline function updatetraceandparams!(learner, s, a, δ, done)
    increasetrace!(learner.traces, s, a)
    updatetraceandparams!(learner.traces, learner.params, learner.α * δ)
    if learner.initvalue == Inf && learner.params[a, s] == Inf
        learner.params[a, s] = learner.unseenvalue + δ
    end
    if done; resettraces!(learner.traces); end
end

# update

function update!(learner::TDLearner, buffer)
    !isfull(buffer) && return
    updatetraceandparams!(learner, 
                          buffer.states[1], 
                          buffer.actions[1],
                          tderror(learner, buffer),
                          buffer.done[1])
end
 
function getvalues(learner::TDLearner)
    [maximum(learner.params[:, i]) for i in 1:size(learner.params, 2)]
end
export getvalues

