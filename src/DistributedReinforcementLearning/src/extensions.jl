export LoadParamsHook, FetchParamsHook

#####
# StopMsg
#####

(msg::StopMsg)(x::StopSignal) = x[] = true

(msg::StopMsg)(x) = nothing

function (msg::StopMsg)(x::ComposedStopCondition)
    for s in x.stop_conditions
        msg(s)
    end
end

#####
# LoadParamMsg
#####

(msg::LoadParamMsg)(x) = nothing

function (msg::LoadParamMsg)(x::ComposedHook)
    for h in x.hooks
        msg(h)
    end
end

struct LoadParamsHook <: AbstractHook
    buffer::Channel
end

LoadParamsHook() = LoadParamsHook(Channel(DEFAULT_MAILBOX_SIZE))

function (hook::LoadParamsHook)(::PostActStage, agent, env)
    ps = nothing
    while isready(hook.buffer)
        ps = take!(hook.buffer).data
    end
    isnothing(ps) || Flux.loadparams!(agent.policy, ps)
end

(msg::LoadParamMsg)(x::LoadParamsHook) = put!(x.buffer, msg)

#####
# FetchParamsHook
#####

Base.@kwdef mutable struct FetchParamsHook <: AbstractHook
    target::RemoteChannel
    buffer::RemoteChannel = RemoteChannel(() -> Channel(DEFAULT_MAILBOX_SIZE))
    n::Int = 0
    freq::Int = 1
    is_blocking::Bool = false
end

(msg::LoadParamMsg)(x::FetchParamsHook) = put!(x.buffer, msg)

function (hook::FetchParamsHook)(::PostActStage, agent, env)
    hook.n += 1
    if hook.n % hook.freq == 0
        if isready(hook.buffer)
            # some other workers have sent the request since our last request
            # so we just reuse it instead of creating new requests to reduce 
            # message traffic
            while isready(hook.buffer)
                ps = take!(hook.buffer).data
            end
            Flux.loadparams!(agent.policy, ps)
            put!(target, FetchParamMsg(hook.buffer))
        else
            put!(target, FetchParamMsg(hook.buffer))
            if hook.is_blocking
                ps = take!(hook.buffer).data
                Flux.loadparams!(agent.policy, ps)
            end
        end
    end
end