export BatchSampleMsg,
    BatchDataMsg,
    FetchParamMsg,
    LoadParamMsg,
    InsertTrajectoryMsg,
    Trainer,
    TrajectoryManager,
    Worker,
    WorkerProxy,
    Orchestrator,
    InsertSampleLoadRateLimiter

using Flux

#####
# Messages
#####

struct BatchSampleMsg <: AbstractMessage
    from::RemoteChannel
end

struct BatchDataMsg{D} <: AbstractMessage
    data::D
end

Base.@kwdef struct FetchParamMsg <: AbstractMessage
    from::RemoteChannel = self()
end

struct LoadParamMsg{P} <: AbstractMessage
    data::P
end

struct InsertTrajectoryMsg{B} <: AbstractMessage
    data::B
end

#####
# Trainer
#####

"""
    Trainer(;policy, sealer=deepcopy)

A wrapper around `AbstractPolicy`, the `sealer` is used to create an immutable
object from the inner parameters of `policy` when received a `FetchParamMsg`.
"""
Base.@kwdef struct Trainer{P,S}
    policy::P
    sealer::S = deepcopy
end

Trainer(p) = Trainer(; policy = p)

function (trainer::Trainer)(msg::BatchDataMsg)
    update!(trainer.policy, msg.data)
end

function (trainer::Trainer)(msg::FetchParamMsg)
    ps = trainer.sealer(params(trainer.policy))
    put!(msg.from, LoadParamMsg(ps))
end

#####
# TrajectoryManager
#####

Base.@kwdef mutable struct TrajectoryManager{T,S,I}
    trajectory::T
    sampler::S
    inserter::I
end

(t::TrajectoryManager)(msg::InsertTrajectoryMsg) = push!(t.trajectory, msg.data, t.inserter)

function (t::TrajectoryManager)(msg::BatchSampleMsg)
    s = sample(t.trajectory, t.sampler)  # !!! sample must ensure no sharing data
    put!(msg.from, BatchDataMsg(s))
end

#####
# Worker
#####

"""
    Worker(()->ex::Experiment)
"""
mutable struct Worker
    init::Any
    experiment::Experiment
    task::Task
    Worker(f) = new(f)
end

function (w::Worker)(msg::StartMsg)
    w.experiment = w.init(msg.args...; msg.kwargs...)
    w.task = Threads.@spawn run(w.experiment)
end

function (w::Worker)(msg::StopMsg)
    msg(w.experiment.stop_condition)
    wait(w.task)
end

(w::Worker)(msg::LoadParamMsg) = msg(w.experiment.hook)

#####
# WorkerProxy
#####

mutable struct WorkerProxy
    is_fetch_msg_sent::Ref{Bool}
    workers::Vector{<:RemoteChannel}
    target::RemoteChannel
    WorkerProxy(workers) = new(Ref(false), workers)
end

function (wp::WorkerProxy)(msg::StartMsg)
    wp.target = msg.args[1]
    for w in wp.workers
        put!(w, StartMsg(self()))
    end
end

(wp::WorkerProxy)(msg::InsertTrajectoryMsg) = put!(wp.target, msg)

function (wp::WorkerProxy)(::FetchParamMsg)
    if !wp.is_fetch_msg_sent[]
        put!(wp.target, FetchParamMsg(self()))
        wp.is_fetch_msg_sent[] = true
    end
end

function (wp::WorkerProxy)(msg::LoadParamMsg)
    for w in wp.workers
        put!(w, msg)
    end
    wp.is_fetch_msg_sent[] = false
end

#####
# Orchestrator
#####

Base.@kwdef mutable struct InsertSampleLoadRateLimiter
    min_insert_before_sampling::Int = 100
    n_sample::Int = 0
    n_insert::Int = 0
    n_load::Int = 0
    sample_insert_ratio::Int = 1
    sample_load_ratio::Int = 1
end

Base.@kwdef struct Orchestrator
    trainer::RemoteChannel
    trajectory_proxy::RemoteChannel
    worker::RemoteChannel{Channel{Any}}
    limiter::InsertSampleLoadRateLimiter = InsertSampleLoadRateLimiter()
end

function (orc::Orchestrator)(msg::StartMsg)
    put!(orc.worker, StartMsg(self()))
end

function (orc::Orchestrator)(msg::InsertTrajectoryMsg)
    L = orc.limiter
    put!(orc.trajectory_proxy, msg)
    L.n_insert += 1
    if L.n_insert > L.min_size_to_sample
        for i in 1:L.sample_insert_ratio
            put!(orc.trajectory_proxy, BatchSampleMsg(orc.trainer))
            L.n_sample += 1
            if L.n_sample == (L.n_load + 1) * L.sample_load_ratio
                put!(
                    orc.trajectory_proxy,
                    ProxyMsg(to = orc.trainer, msg = FetchParamMsg(orc.worker)),
                )
                L.n_load += 1
            end
        end
    end
end
