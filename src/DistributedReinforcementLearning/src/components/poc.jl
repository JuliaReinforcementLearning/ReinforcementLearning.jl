export actor,
    Trainer,
    TrajectoryManager,
    Worker,
    Orchestrator,
    InsertSampleRateLimiter,
    StartMsg,
    StopMsg,
    InsertTrajectoryMsg,
    LoadParamMsg,
    BatchDataMsg,
    FetchParamMsg,
    BatchSampleMsg

using Distributed
using ReinforcementLearningBase
using ReinforcementLearningCore
using Flux

#####
# Messages
#####

abstract type AbstractMessage end

struct StartMsg <: AbstractMessage
    args
    kwargs
    StartMsg(args...;kwargs...) = new(args, kwargs)
end


struct StopMsg <: AbstractMessage end

struct BatchSampleMsg <: AbstractMessage
    from
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

Base.@kwdef struct ProxyMsg{M} <: AbstractMessage
    to::RemoteChannel
    msg::M
end

#####
# Message Extensions
#####

(msg::StopMsg)(x) = nothing

(msg::StopMsg)(x::StopSignal) = x[] = true

function (msg::StopMsg)(x::ComposedStopCondition)
    for s in x.stop_conditions
        msg(s)
    end
end

(msg::LoadParamMsg)(x) = nothing

function (msg::LoadParamMsg)(x::ComposedHook)
    for h in x.hooks
        msg(h)
    end
end

#####
# Actor Model
# Each actor is a RemoteChannel by default
#  TODO: switch to https://github.com/JuliaActors/Actors.jl
#####

const DEFAULT_MAILBOX_SIZE = 32

"""
    actor(f;sz=DEFAULT_MAILBOX_SIZE)

Create a task to handle messages one-by-one by calling `f(msg)`.
A mailbox (`RemoteChannel`) is returned.
"""
function actor(f;sz=DEFAULT_MAILBOX_SIZE)
    RemoteChannel() do
        Channel(sz;spawn=true) do ch
            task_local_storage("MAILBOX", RemoteChannel(() -> ch))
            while true
                msg = take!(ch)
                f(msg)
                msg isa StopMsg && break
            end
        end
    end
end

"""
    self()

Get the mailbox in current task.
"""
self() = task_local_storage("MAILBOX")

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

Trainer(p) = Trainer(;policy=p)

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
(t::TrajectoryManager)(msg::ProxyMsg) = put!(msg.to, msg)

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
    w.experiment = w.init(msg.args...;msg.kwargs...)
    w.task = Threads.@spawn run(w.experiment)
end

(w::Worker)(msg::StopMsg) = msg(w.experiment.stop_condition)
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

function (wp::WorkerProxy)(msg::FetchParamMsg)
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

Base.@kwdef mutable struct InsertSampleRateLimiter
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
    limiter::InsertSampleRateLimiter = InsertSampleRateLimiter()
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
                put!(orc.trajectory_proxy, ProxyMsg(to=orc.trainer, msg=FetchParamMsg(orc.worker)))
                L.n_load += 1
            end
        end
    end
end
