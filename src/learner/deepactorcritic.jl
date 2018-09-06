"""
    mutable struct DeepActorCritic{Tnet, Tpl, Tplm, Tvl, ToptT, Topt}
        nh::Int64 = 4
        na::Int64 = 2
        γ::Float64 = .9
        nsteps::Int64 = 5
        net::Tnet
        policylayer::Tpl = Linear(nh, na)
        policynet::Tplm = Flux.Chain(Flux.mapleaves(Flux.Tracker.data, net),
                                 Flux.mapleaves(Flux.Tracker.data, policylayer))
        valuelayer::Tvl = Linear(nh, 1)
        params::Array{Any, 1} = vcat(map(Flux.params, [net, policylayer, valuelayer])...)
        t::Int64 = 0
        updateevery::Int64 = 1
        opttype::ToptT = Flux.ADAM
        opt::Topt = opttype(params)
        αcritic::Float64 = .1
        nmarkov::Int64 = 1
"""
@with_kw mutable struct DeepActorCritic{Tnet, Tpl, Tplm, Tvl, ToptT, Topt}
    nh::Int64 = 4
    na::Int64 = 2
    γ::Float64 = .9
    nsteps::Int64 = 5
    net::Tnet
    policylayer::Tpl = Linear(nh, na)
    policynet::Tplm = Flux.Chain(Flux.mapleaves(Flux.Tracker.data, net),
                             Flux.mapleaves(Flux.Tracker.data, policylayer))
    valuelayer::Tvl = Linear(nh, 1)
    params::Array{Any, 1} = vcat(map(Flux.params, [net, policylayer, valuelayer])...)
    t::Int64 = 0
    updateevery::Int64 = 1
    opttype::ToptT = Flux.ADAM
    opt::Topt = opttype(params)
    αcritic::Float64 = .1
    nmarkov::Int64 = 1
end
export DeepActorCritic
DeepActorCritic(net; kargs...) = DeepActorCritic(; net = net, kargs...)

function defaultpolicy(learner::DeepActorCritic, actionspace, buffer)
    π = SoftmaxPolicy(Flux.Chain(Flux.mapleaves(Flux.Tracker.data, learner.net),
                                 Flux.mapleaves(Flux.Tracker.data, learner.policylayer)))
    defaultnmarkovpolicy(learner, buffer, π)
end

function update!(learner::DeepActorCritic, b)
    learner.t += 1
    (!isfull(b) || learner.t % learner.updateevery != 0) && return
    h1 = learner.net(nmarkovgetindex(b.states, learner.nmarkov, learner.nmarkov))
    p1 = learner.policylayer(h1)
    v1 = learner.valuelayer(h1)[:]
    r, γeff = discountedrewards(view(b.rewards, learner.nmarkov:lastindex(b.rewards)), 
                                view(b.done, learner.nmarkov:lastindex(b.done)), 
                                learner.γ)
    advantage = r - v1.data[1]
    if γeff > 0
        h2 = learner.net(nmarkovgetindex(b.states, lastindex(b.states), learner.nmarkov))
        v2 = learner.valuelayer(h2)
        advantage += γeff * v2.data[1] 
    end
    Flux.back!(advantage * (-Flux.logsoftmax(p1)[b.actions[learner.nmarkov]] + 
                            learner.αcritic * v1[1]))
    learner.opt()
end
