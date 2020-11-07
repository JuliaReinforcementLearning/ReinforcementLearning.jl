#=

Proof of Concept

Here we solve the multi-arm bandit problem with the ϵ-greedy algorithm
in a distributed manner.

Each worker starts with a random policy. They collect actions and corresponding
rewards and then forward them to a trajectory proxy.

A trajectory stores transitions from works in a local buffer and periodically
send a batch to a optimizer.

A optimizer updates its policy and broadcast the latest policy to workers periodically.
=#

using StatsBase

K = 10
TRUE_VALUES = rand(K)
SZ = 100
N_WORKER = 10
ϵ = 0.1

#####
# Optimizer
#####

Base.@kwdef mutable struct Optimizer
    total_rewards::Vector{Float64} = fill(0., K)
    total_counts::Vector{Int} = ones(Int, K)
    workers::Vector{Channel{Any}} = []
    freq::Int = 10
    n::Int = 0
end

struct SetWorkersMsg
    workers::Vector{Channel{Any}}
end

(opt::Optimizer)(msg::SetWorkersMsg) = opt.workers = msg.workers

struct BatchTrainingDataMsg
    data::Vector{Pair{Int,Float64}}
end

struct SyncParamMsg
    policy::Vector{Float64}
end

function (opt::Optimizer)(msg::BatchTrainingDataMsg)
    for (a,r) in msg.data
        opt.total_counts[a] += 1
        opt.total_rewards[a] += r
    end
    opt.n += 1
    if opt.n % opt.freq == 0
        _, ind = findmax([r/c for (r, c) in zip(opt.total_rewards, opt.total_counts)])
        policy = fill(ϵ/K, K)
        policy[ind] += 1 - ϵ
        for w in opt.workers
            put!(w, SyncParamMsg(policy))
        end
    end
end

opt = Optimizer()

optimizer = Channel(SZ) do ch
    while true
        msg = take!(ch)
        opt(msg)
        yield()
    end
end

#####
# Trajectory
#####

Base.@kwdef struct Trajectory
    container::Vector{Pair{Int, Float64}} = []
    batch_size::Int = 32
    optimizer::Channel{Any} = optimizer
end

struct TransitionMsg
    data::Pair{Int, Float64}
end

function (traj::Trajectory)(msg::TransitionMsg)
    push!(traj.container, msg.data)
    if length(traj.container) >= traj.batch_size
        put!(traj.optimizer, BatchTrainingDataMsg(traj.container[:]))
        empty!(traj.container)
    end
end

traj = Trajectory()

trajectory = Channel(SZ) do ch
    while true
        msg = take!(ch)
        traj(msg)
        yield()
    end
end

#####
# Worker
#####

mutable struct Worker
    policy_buffer::Channel{Any}
    is_terminate::Ref{Bool}

    function Worker(traj; init_policy=fill(1/K, K), true_values=TRUE_VALUES)
        policy_buffer = Channel(SZ)
        is_terminate = Ref(false)
        task = Threads.@spawn begin
            policy = init_policy
            while true
                action = sample(Weights(policy, 1.0))
                reward = true_values[action] + randn() * 0.1
                put!(traj, TransitionMsg(action => reward))
                is_terminate[] && break
                while isready(policy_buffer)
                    policy = take!(policy_buffer)
                end
                yield()
            end
        end
        new(policy_buffer, is_terminate)
    end
end

function (w::Worker)(msg::SyncParamMsg)
    put!(w.policy_buffer, msg.policy)
end

workers = [
    Channel(SZ) do ch
        w = Worker(trajectory)

        while true
            msg = take!(ch)
            w(msg)
            yield()
        end
    end
    for _ in 1:N_WORKER
]

put!(optimizer, SetWorkersMsg(workers))