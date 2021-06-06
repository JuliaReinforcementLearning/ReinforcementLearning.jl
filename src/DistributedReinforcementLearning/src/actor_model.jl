export AbstractMessage,
    StartMsg,
    StopMsg,
    PingMsg,
    PongMsg,
    ProxyMsg,
    actor,
    self


abstract type AbstractMessage end

struct StartMsg{A, K} <: AbstractMessage
    args::A
    kwargs::K
    StartMsg(args...;kwargs...) = new{typeof(args), typeof(kwargs)}(args, kwargs)
end

struct StopMsg <: AbstractMessage end

Base.@kwdef struct ProxyMsg{M} <: AbstractMessage
    to::RemoteChannel
    msg::M
end

Base.@kwdef struct PingMsg <: AbstractMessage
    from::RemoteChannel = self()
end

Base.@kwdef struct PongMsg <: AbstractMessage
    from::RemoteChannel = self()
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
                _handle(f, msg)
                msg isa StopMsg && break
            end
        end
    end
end

_handle(f, msg) = f(msg)
_handle(f, msg::ProxyMsg) = put!(msg.to, msg.msg)
_handle(f, msg::PingMsg) = put!(msg.from, PongMsg())

"""
    self()

Get the mailbox in current task.
"""
function self()
    try
        task_local_storage("MAILBOX")
    catch
        mailbox = RemoteChannel(() -> Channel(DEFAULT_MAILBOX_SIZE))
        task_local_storage("MAILBOX", mailbox)
        mailbox
    end
end
