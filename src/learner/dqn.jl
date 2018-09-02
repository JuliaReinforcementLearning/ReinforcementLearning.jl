"""
    mutable struct DQN{Tnet,TnetT,ToptT,Topt}
        γ::Float64 = .99
        na::Int64
        net::TnetT
        targetnet::Tnet = Flux.mapleaves(Flux.Tracker.data, deepcopy(net))
        policynet::Tnet = Flux.mapleaves(Flux.Tracker.data, net)
        updatetargetevery::Int64 = 500
        t::Int64 = 0
        updateevery::Int64 = 1
        opttype::ToptT = Flux.ADAM
        opt::Topt = opttype(Flux.params(net))
        startlearningat::Int64 = 10^3
        minibatchsize::Int64 = 32
        doubledqn::Bool = true
        nmarkov::Int64 = 1
        nsteps::Int64 = 1
        replaysize::Int64 = 10^4
        loss::Function = Flux.mse
        sampling::Ts
"""
@with_kw mutable struct DQN{Tnet,TnetT,ToptT,Topt,Ts}
    γ::Float64 = .99
    na::Int64
    net::TnetT
    policynet::Tnet = Flux.mapleaves(Flux.Tracker.data, net)
    targetnet::Tnet = deepcopy(policynet)
    updatetargetevery::Int64 = 500
    t::Int64 = 0
    updateevery::Int64 = 1
    opttype::ToptT = Flux.ADAM
    opt::Topt = opttype(Flux.params(net))
    startlearningat::Int64 = 10^3
    minibatchsize::Int64 = 32
    doubledqn::Bool = true
    nmarkov::Int64 = 1
    nsteps::Int64 = 1
    replaysize::Int64 = 10^4
    loss::Function = Flux.mse
    sampling::Ts = UniformSampling()
end
export DQN
function DQN(net; kargs...)
    na = 0
    try
        if haskey(Dict(kargs), :na)
            na = Dict(kargs)[:na]
        elseif typeof(net) == Flux.Chain
            na = size(net.layers[end].W, 1)
        else
            na = size(net.W, 1)
        end
    catch
        error("Could not infer the number of actions na. Please provide them as
               a keyword argument of the form `na = ...`.")
    end
    DQN(; net = Flux.gpu(net), na = na, kargs...)
end
function defaultbuffer(learner::Union{DQN, DeepActorCritic}, env, preprocessor)
    state = preprocessstate(preprocessor, getstate(env)[1])
    ArrayStateBuffer(capacity = typeof(learner) <: DQN ? learner.replaysize :
                                                         learner.nsteps + learner.nmarkov, 
                     arraytype = typeof(state).name.wrapper,
                     datatype = typeof(state[1]),
                     elemshape = size(state))
end
function defaultpolicy(learner::Union{DQN, DeepActorCritic}, buffer)
    if learner.nmarkov == 1
        typeof(learner) <: DQN ? EpsilonGreedyPolicy(.1) : SoftmaxPolicy()
    else
        a = buffer.states.data
        data = getindex(a, map(x -> 1:x, size(a)[1:end-1])..., 1:learner.nmarkov)
        NMarkovPolicy(typeof(learner) <: DQN ? EpsilonGreedyPolicy(.1) : 
                                               SoftmaxPolicy(),
                      ArrayCircularBuffer(data, learner.nmarkov, 0, 0, false))
    end
end

@with_kw struct NMarkovPolicy{Tpol, Tbuf}
    policy::Tpol = EpsilonGreedyPolicy(.1)
    buffer::Tbuf
end
@inline setepsilon(policy::NMarkovPolicy, val) = policy.policy.ϵ = val
@inline incrementepsilon(policy::NMarkovPolicy, val) = policy.policy.ϵ += val

huberloss(yhat, y::Flux.TrackedArray) = -2*dot(clamp.(yhat - y.data, -1, 1), y)/length(y)
huberloss(yhat, y, w) = huberloss(yhat, y)
huberloss(yhat, y::Flux.TrackedArray, w::AbstractArray) = -2*dot(w .* clamp.(yhat - y.data, -1, 1), y)/length(y)
export huberloss
import Flux: mse
mse(yhat, y, w) = mse(yhat, y)
mse(yhat, y, w::AbstractArray) = sum(w .* (yhat .- y).^2)/length(y)

@inline function selectaction(learner::Union{DQN, DeepActorCritic}, policy, state)
    if learner.nmarkov == 1
        selectaction(policy, learner.na, learner.policynet, state)
    else
        push!(policy.buffer, state)
        selectaction(policy.policy, learner.na,
                     b -> learner.policynet(nmarkovgetindex(b, 
                                                            lastindex(b),
                                                            learner.nmarkov)),
                     policy.buffer)
    end
end
function selecta(q, a)
    na, t = size(q)
    q[na * collect(0:t-1) .+ a]
end
import StatsBase
function update!(learner::DQN, b)
    learner.t += 1
    if learner.t % learner.updatetargetevery == 0
        learner.targetnet = deepcopy(learner.policynet)
    end
    (learner.t < learner.startlearningat || 
     learner.t % learner.updateevery != 0) && return
    indices, weights = sample(learner.sampling, 
                              1:length(b.rewards) - learner.nsteps + 1, 
                              learner.minibatchsize, b)
    qa = learner.net(nmarkovgetindex(b.states, indices, learner.nmarkov))
    qat = learner.targetnet(nmarkovgetindex(b.states, 
                                            indices .+ learner.nsteps, 
                                            learner.nmarkov))
    q = selecta(qa, b.actions[indices])
    rs = Float64[]
    for (k, i) in enumerate(indices)
        r, γeff = discountedrewards(b.rewards[i:i + learner.nsteps - 1], 
                                    b.done[i:i + learner.nsteps - 1], 
                                    learner.γ)
        if γeff > 0
            if learner.doubledqn
                r += γeff * qat[argmax(qa.data[:,k]), k]
            else
                r += γeff * maximum(qat[:, k])
            end
        end
        push!(rs, r)
    end
    updatesampling!(learner.sampling, rs .- q.data, indices, b)
    Flux.back!(learner.loss(Flux.gpu(rs), q))
    learner.opt()
end

struct UniformSampling end
function sample(::UniformSampling, range, N, buffer)
    StatsBase.sample(range, N, replace = false), nothing
end
export UniformSampling
updatesampling!(::UniformSampling, values, indices, buffer) = nothing
mutable struct PrioritizedSampling{T}
    α::Float64
    β::Float64
    ϵ::Float64
    w::T
    maxp::Float64
    lastidx::Int64
end
function PrioritizedSampling(; α = .6, β = .5, ϵ = 1e-8, N = 10^6)
    PrioritizedSampling(α, β, ϵ, SumTree(N), 0., 0)
end
export PrioritizedSampling
function sample(p::PrioritizedSampling, range, N, buffer)
    # initialize
    b = buffer.states
    if p.lastidx == 0
        for i in 1:b.counter - 1
            r = abs(buffer.rewards[i])
            p.w[i] = r
            if r > p.maxp
                p.maxp = r
            end
        end
        p.lastidx = b.counter - 1
    end
    idxs = [wsample(p.w) for _ in 1:N]
    w = ([p.w[i] for i in idxs] ./ sum(p.w)) .^ (-p.β)
    w ./= maximum(w)
    (idxs .- b.start .+ b.capacity .- 2) .% (b.capacity - 1) .+ 1, w
end
function updatesampling!(p::PrioritizedSampling, values, indices, buffer)
    b = buffer.states
    @. indices = (indices + b.start - 1) % (b.capacity - 1) + 1
    # update
    for (i, v) in zip(indices, values)
        pi = (abs(v) + p.ϵ)^p.α
        p.w[i] = pi
        if pi > p.maxp
            p.maxp = pi
        end
    end
    if b.counter < p.lastidx
        for i in p.lastidx:b.capacity - 1
            p.w[i] = p.maxp
        end
        p.lastidx = 0
    end
    for i in p.lastidx + 1:b.counter - 1
        p.w[i] = p.maxp
    end
    p.lastidx = b.counter - 1
end


struct SumTree{T, N}
    data::Array{T, 1}
end
SumTree(T, N) = SumTree{T, N}(zeros(T, 2*N-1))
SumTree(N) = SumTree(Float64, N)

import Base: setindex!, getindex, sum
function setindex!(tree::SumTree{T, N}, value, idx::Int) where {T, N}
    idx += N - 1
    change = value - tree.data[idx]
    tree.data[idx] = value
    while idx > 1
        idx >>>= 1
        tree.data[idx] += change
    end
    value
end

getindex(tree::SumTree{T, N}, idx) where {T, N} = tree.data[idx .+ N .- 1]
sum(tree::SumTree) = tree.data[1]

import StatsBase: wsample
function wsample(tree::SumTree{T, N}) where {T, N}
    s = tree.data[1] * rand()
    idx = 1
    while (idx << 1) < 2*N - 1
        idx <<= 1
        if tree.data[idx] < s
            s -= tree.data[idx]
            idx += 1
        end
    end
    idx - N + 1
end
