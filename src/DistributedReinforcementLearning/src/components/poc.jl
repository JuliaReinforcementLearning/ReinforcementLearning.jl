export actor, Optimizer, TrajectoryProxy, Worker, InsertTrajectoryMsg, FetchParamMsg, BatchSampleMsg

using Distributed
using ReinforcementLearning
using Flux

#=

Basic Idea

1. Create an Optimizer
2. Create a TrajectoryProxy
3. Create an Orchestrator to collect transitions.
4. Use the Orchestrator above to initialize workers.
5. Send a StartMsg to workers
6. Create a new worker to evaluate the latest params in Optimizer.

The Orchestrator controls the speed of:

- Insert transitions into trajectory
- Sample batches from trajectory
- Fetch parameters from Optimizer to workers

=#

#####
# Messages
#####

abstract type AbstractMessage end

struct StartMsg <: AbstractMessage
    arg
    kwargs
end

struct StopMsg <: AbstractMessage end

const MailBox = Channel{AbstractMessage}

#####
# Actor Model
#  TODO: switch to https://github.com/pbayer/YAActL.jl
#####

function actor(x;sz=32)
    RemoteChannel() do
        Channel(sz;spawn=true) do ch
            task_local_storage("MAILBOX", RemoteChannel(() -> ch))
            while true
                msg = take!(ch)
                x(msg)
                msg isa StopMsg && break
            end
        end
    end
end

self() = task_local_storage("MAILBOX")

#####
# Optimizer
#####

Base.@kwdef struct Optimizer{P,S}
    policy::P
    sealer::S = deepcopy
end

struct BatchDataMsg{D} <: AbstractMessage
    data::D
end

(opt::Optimizer)(msg::BatchDataMsg) = update!(opt.policy, msg.data)

Base.@kwdef struct FetchParamMsg <: AbstractMessage
    from::MailBox = self()
end

struct LoadParamMsg{P} <: AbstractMessage
    data::P
end

function (opt::Optimizer)(msg::FetchParamMsg)
    ps = opt.sealer(params(opt.policy))
    put!(msg.from, LoadParamMsg(ps))  # blocking?
end

#####
# TrajectoryProxy
#####

Base.@kwdef struct TrajectoryProxy{T,S,I}
    trajectory::T
    sampler::S
    inserter::I
end

struct InsertTrajectoryMsg{B} <: AbstractMessage
    bulk::B
end

(t::TrajectoryProxy)(msg::InsertTrajectoryMsg) = push!(t.trajectory, msg.bulk, t.inserter)

struct BatchSampleMsg <: AbstractMessage
    from
end

function (t::TrajectoryProxy)(msg::BatchSampleMsg)
    s = sample(t.trajectory, t.sampler)
    put!(msg.from, s)  # blocking?
end

#####
# Worker
#####

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

(w::Worker)(msg::StopMsg) = w.experiment.stop_condition(msg)
(w::Worker)(msg::LoadParamMsg) = w.experiment.hook(msg)

#####
# Orchestrator
#####

# optimizer = actor(Optimizer(RandomPolicy()))
# trajectory = actor(TrajectoryProxy(CircularSARTSATrajectory()))
# worker = actor() do
#     Experiment(
#         agent = Agent(
#             policy=RandomPolicy(),
#             trajectory = CircularSARTSATrajectory()
#         ),
#         env = CartPoleEnv()
#         stop_condition = 
#         hook = 
#     )
# end