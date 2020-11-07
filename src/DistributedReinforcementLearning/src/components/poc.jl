using Distributed

#####
# Messages
#####

const MailBox = Union{Channel{AbstractMessage}, RemoteChannel{Channel{AbstractMessage}}}

abstract type AbstractMessage end

struct StopMsg <: AbstractMessage end

struct RequestMsg <: AbstractMessage
    msg
    from
end

function request!(mailbox::MailBox, msg::AbstractMessage)
    tmp_mailbox = Channel(1)
    if mailbox isa RemoteChannel
        tmp_mailbox = RemoteChannel(() -> tmp_mailbox)
    end
    put!(mailbox, RequestMsg(msg, tmp_mailbox))
    take!(mailbox, tmp_mailbox)
end

_handle(x, msg) = x(msg)

function _handle(x, msg::RequestMsg)
    res = x(msg.data)
    put!(ResponseMsg(res, self()))
end

#####
# Actor Model
#  TODO: switch to https://github.com/pbayer/YAActL.jl
#####

function actor(x;sz=32, is_local=true)
    mailbox = Channel(sz)
    task_local_storage("MAILBOX", mailbox)
    c = Channel(sz;spawn=true) do ch
        while true
            msg = take!(ch)
            _handle(x, msg)
            msg isa StopMsg && break
            yield()
        end
    end
    is_local ? c : RemoteChannel(() -> c)
end

function self(;is_local=true)
    if is_local
        task_local_storage("MAILBOX")
    else
        RemoteChannel(() -> task_local_storage("MAILBOX"))
    end
end

#####
# Optimizer
#####

struct Optimizer{P}
    policy::P
end

Base.@kwdef struct AsyncOptimizer{P,S}
    policy::P
    sealer::S = deepcopy
end

struct BatchDataMsg{D} <: AbstractMessage
    data::D
end

(opt::Union{Optimizer, AsyncOptimizer})(msg::BatchDataMsg) = update!(opt.policy, msg.data)

Base.@kwdef struct FetchParamMsg <: AbstractMessage
    from::MailBox = self()
end

struct LoadParamMsg{P} <: AbstractMessage
    data::P
end

function (opt::Optimizer)(msg::FetchParamMsg)
    ps = params(opt.policy)
    request!(msg.from, LoadParamMsg(ps))
end

function (opt::AsyncOptimizer)(msg::FetchParamMsg)
    ps = opt.sealer(params(opt.policy))
    put!(msg.from, LoadParamMsg(ps))
end

#####
# TrajectoryProxy
#####

Base.@kwdef mutable struct TrajectoryProxy{T}
    trajectory::T
end

struct InsertBulkMsg{B} <: AbstractMessage
    bulk::B
end

(t::TrajectoryProxy)(msg::InsertBulkMsg) = push!(t.trajectory, msg.bulk)

struct BatchSampleMsg <: AbstractMessage
    batch_size::Int
    from
end

(t::TrajectoryProxy)(msg::BatchSampleMsg) = put!(msg.from, sample(t.trajectory, msg.batch_size))

#####
# Worker
#####

struct FetchParamHook <: AbstractHook
    params_buffer::Channel{Any}
end

function (h::FetchParamHook)(::Training{PostActStage}, agent, env)
    ps = nothing
    while isready(params_buffer)
        ps = take!(params_buffer)
    end
    isnothing(ps) || load_params!(agent, ps)
end

mutable struct Worker
    params_buffer::Channel{Any}
    is_exit::Ref{Bool}
    task::Task
    function Worker(ex)
        agent, env, stop_condition, hook = ex()
        is_exit = Ref(false)
        params_buffer = Channel{Any}()
        sc = ComposedStopCondition(
            stop_condition,
            (args...) -> is_exit[]
        )
        h = ComposedHook(
            hook,
            FetchParamHook(params_buffer)
        )
        task = Threads.@spawn run(agent,env,sc,h)
        new(params_buffer, is_exit, task)
    end
end

(w::Worker)(msg::LoadParamMsg) = put!(w.params_buffer, msg.data)

function (w::Worker)(::StopMsg)
    w.is_exit[] = true
    wait(w.task)
end
