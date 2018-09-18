export huberloss

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
"""
@with_kw mutable struct DQN{Tnet,TnetT,ToptT,Topt}
    γ::Float64 = .99
    na::Int64
    net::TnetT
    targetnet::Tnet = deepcopy(Flux.mapleaves(Flux.Tracker.data, net))
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
    state = preprocessstate(preprocessor, getstate(env).observation)
    action = sample(actionspace(env))
    CircularTurnBuffer{Turn{typeof(state), typeof(action), Float64, Bool}}(
        typeof(learner) <: DQN ? learner.replaysize : learner.nsteps + learner.nmarkov - 1,
        size(state))
end
function defaultpolicy(learner::DQN, actionspace, buffer)
    π = EpsilonGreedyPolicy(.1, actionspace, 
                            Flux.mapleaves(Flux.Tracker.data, learner.net))
    defaultnmarkovpolicy(learner, buffer, π)
end

huberloss(yhat, y::Flux.TrackedArray) = -2*dot(clamp.(yhat - y.data, -1, 1), y)/length(y)

function selecta(q, a)
    na, t = size(q)
    q[na * collect(0:t-1) .+ a]
end

function update!(learner::DQN, b)
    learner.t += 1
    if learner.t % learner.updatetargetevery == 0
        learner.targetnet = deepcopy(Flux.mapleaves(Flux.Tracker.data, learner.net))
    end
    (learner.t < learner.startlearningat || 
     learner.t % learner.updateevery != 0) && return
    indices = StatsBase.sample(1:length(b) - learner.nsteps + 1, 
                               learner.minibatchsize, 
                               replace = false)

    sd = viewconsecutive(b, :states, indices, learner.nmarkov)
    qa = learner.net(convert(Array, reshape(sd, size(sd)[1:ndims(sd)-2]..., size(sd)[end-1] * size(sd)[end])))

    st = viewconsecutive(b, :nextstates, indices .+ (learner.nsteps - 1), learner.nmarkov)
    qat = learner.targetnet(convert(Array, reshape(st, size(st)[1:ndims(st)-2]..., size(st)[end-1] * size(st)[end])))

    q = selecta(qa, b[:actions, indices])
    rs = Float64[]
    for (k, i) in enumerate(indices)
        r, γeff = discountedrewards(b[:rewards, i:i + learner.nsteps - 1], 
                                    b[:isdone, i:i + learner.nsteps - 1], 
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
    Flux.back!(learner.loss(Flux.gpu(rs), q))
    learner.opt()
end
