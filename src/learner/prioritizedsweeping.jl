"""
    mutable struct SmallBackups <: AbstractReinforcementLearner
        ns::Int64 = 10
        na::Int64 = 4
        γ::Float64 = .9
        initvalue::Float64 = Inf64
        maxcount::UInt64 = 3
        minpriority::Float64 = 1e-8
        M::Int64 = 1
        counter::Int64 = 0
        Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue
        V::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
        U::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
        Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
        Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1} = [Dict{Tuple{Int64, Int64}, Int64}() for _ in 1:ns]
        queue::PriorityQueue = PriorityQueue(Base.Order.Reverse, zip(Int64[], Float64[]))

See [Harm Van Seijen, Rich Sutton ; Proceedings of the 30th International Conference on Machine Learning, PMLR 28(3):361-369, 2013.](http://proceedings.mlr.press/v28/vanseijen13.html)

`maxcount` defines the maximal number of backups per action, `minpriority` is
the smallest priority still added to the queue.
"""
@with_kw mutable struct SmallBackups
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = .9
    initvalue::Float64 = Inf64
    maxcount::UInt64 = 3
    minpriority::Float64 = 1e-8
    M::Int64 = 1
    counter::Int64 = 0
    Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue
    V::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
    U::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
    Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1} = [Dict{Tuple{Int64, Int64}, Int64}() for _ in 1:ns]
    queue::PriorityQueue = PriorityQueue(Base.Order.Reverse, zip(Int64[], Float64[]))
end
export SmallBackups
function defaultpolicy(learner::Union{SmallBackups, MonteCarlo}, actionspace,
                       buffer)
    EpsilonGreedyPolicy(.1, actionspace, s -> getvalue(learner.Q, s))
end

function addtoqueue!(q, s, p)
    if haskey(q, s) 
        if q[s] > p; q[s] = p; end
    else
        enqueue!(q, s, p)
    end
end

function processqueue!(learner)
    while length(learner.queue) > 0 && learner.counter < learner.maxcount
        learner.counter += 1
        s1 = dequeue!(learner.queue)
        ΔV = learner.V[s1] - learner.U[s1]
        learner.U[s1] = learner.V[s1]
        if length(learner.Ns1a0s0[s1]) > 0
            for ((a0, s0), n) in learner.Ns1a0s0[s1]
                if learner.Nsa[a0, s0] >= learner.M
                    learner.Q[a0, s0] += learner.γ * ΔV * n/learner.Nsa[a0, s0]
                    learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
                    p = abs(learner.V[s0] - learner.U[s0])
                    if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
                end
            end
        end
    end
    learner.counter = 0
end


function update!(learner::SmallBackups, buffer)
    a0 = buffer[:actions, 1]
    a1 = buffer[:nextactions, 1]
    s0 = buffer[:states, 1]
    s1 = buffer[:nextstates, 1]
    r = buffer[:rewards, 1]
    if buffer[:isdone, 1]
        learner.Nsa[a0, s0] += 1
        if learner.Q[a0, s0] == Inf; learner.Q[a0, s0] = 0; end
        if learner.Nsa[a0, s0] >= learner.M
            learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Nsa[a0, s0] - 1) + r) / 
                               learner.Nsa[a0, s0]
        end
    else
        learner.Nsa[a0, s0] += 1
        if haskey(learner.Ns1a0s0[s1], (a0, s0))
            learner.Ns1a0s0[s1][(a0, s0)] += 1
        else
            learner.Ns1a0s0[s1][(a0, s0)] = 1
        end
        if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
        nextv = learner.γ * learner.U[s1]
        if learner.Nsa[a0, s0] >= learner.M
            learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Nsa[a0, s0] - 1) + 
                                r + nextv) / learner.Nsa[a0, s0]
        end
    end
    learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
    p = abs(learner.V[s0] - learner.U[s0])
    if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
    processqueue!(learner)
end
