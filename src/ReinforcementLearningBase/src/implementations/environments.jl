export SubjectiveEnv,
    MultiThreadEnv,
    StateOverriddenEnv,
    RewardOverriddenEnv,
    ActionTransformedEnv,
    StateCachedEnv

using MacroTools: @forward
using Random

import Base.Threads.@spawn

#####
# SubjectiveEnv
#####

struct SubjectiveEnv{E<:AbstractEnv,P} <: AbstractEnv
    env::E
    player::P
end

(env::SubjectiveEnv)(action; kwargs...) = env.env(action, env.player; kwargs...)

# partial constructor to allow chaining
SubjectiveEnv(player) = env -> SubjectiveEnv(env, player)

for f in ENV_API
    @eval $f(x::SubjectiveEnv, args...; kwargs...) = $f(x.env, args...; kwargs...)
end

for f in MULTI_AGENT_ENV_API
    @eval $f(x::SubjectiveEnv, args...; kwargs...) = $f(x.env, x.player, args...; kwargs...)
end

#####
# StateOverriddenEnv
#####

struct StateOverriddenEnv{P,E<:AbstractEnv} <: AbstractEnv
    processors::P
    env::E
end

(env::StateOverriddenEnv)(args...; kwargs...) = env.env(args...; kwargs...)

# partial constructor to allow chaining
StateOverriddenEnv(processors...) = env -> StateOverriddenEnv(processors, env)

for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    if f != :get_state
        @eval $f(x::StateOverriddenEnv, args...; kwargs...) = $f(x.env, args...; kwargs...)
    end
end

get_state(env::StateOverriddenEnv, args...; kwargs...) =
    foldl(|>, env.processors; init = get_state(env.env, args...; kwargs...))

#####
# StateCachedEnv
#####

"""
Cache the state so that `get_state(env)` will always return the same result before the next interaction with `env`.
"""
mutable struct StateCachedEnv{S,E<:AbstractEnv} <: AbstractEnv
    s::S
    env::E
    is_state_cached::Bool
end

StateCachedEnv(env) = StateCachedEnv(get_state(env), env, true)

function (env::StateCachedEnv)(args...; kwargs...)
    env.env(args...; kwargs...)
    env.is_state_cached = false
end

function get_state(env::StateCachedEnv, args...; kwargs...)
    if env.is_state_cached
        env.s
    else
        env.s = get_state(env.env, args...; kwargs...)
        env.is_state_cached = true
        env.s
    end
end

for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    if f != :get_state
        @eval $f(x::StateCachedEnv, args...; kwargs...) = $f(x.env, args...; kwargs...)
    end
end

#####
# RewardOverriddenEnv
#####

struct RewardOverriddenEnv{P,E<:AbstractEnv} <: AbstractEnv
    processors::P
    env::E
end

(env::RewardOverriddenEnv)(args...; kwargs...) = env.env(args...; kwargs...)

# partial constructor to allow chaining
RewardOverriddenEnv(processors...) = env -> RewardOverriddenEnv(processors, env)

for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    if f != :get_reward
        @eval $f(x::RewardOverriddenEnv, args...; kwargs...) = $f(x.env, args...; kwargs...)
    end
end

get_reward(env::RewardOverriddenEnv, args...; kwargs...) =
    foldl(|>, env.processors; init = get_reward(env.env, args...; kwargs...))

#####
# ActionTransformedEnv
#####

struct ActionTransformedEnv{P,E<:AbstractEnv} <: AbstractEnv
    processors::P
    env::E
end

# partial constructor to allow chaining
ActionTransformedEnv(processors...) = env -> ActionTransformedEnv(processors, env)

for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    @eval $f(x::ActionTransformedEnv, args...; kwargs...) = $f(x.env, args...; kwargs...)
end

(env::ActionTransformedEnv)(action, args...; kwargs...) =
    env.env(foldl(|>, env.processors; init = action), args...; kwargs...)

#####
# MultiThreadEnv
#####

"""
    MultiThreadEnv(envs::Vector{<:AbstractEnv})

Wrap multiple environments into one environment.
Each environment will run in parallel by leveraging `Threads.@spawn`.
So remember to set the environment variable `JULIA_NUM_THREADS`!
"""
struct MultiThreadEnv{E} <: AbstractEnv
    envs::Vector{E}
end

MultiThreadEnv(f, n) = MultiThreadEnv([f() for _ in 1:n])

@forward MultiThreadEnv.envs Base.getindex, Base.length, Base.setindex!

function (env::MultiThreadEnv)(actions)
    @sync for i in 1:length(env)
        @spawn begin
            env[i](actions[i])
        end
    end
end

function reset!(env::MultiThreadEnv; is_force = false)
    if is_force
        for i in 1:length(env)
            reset!(env[i])
        end
    else
        @sync for i in 1:length(env)
            if get_terminal(env[i])
                @spawn begin
                    reset!(env[i])
                end
            end
        end
    end
end

const MULTI_THREAD_ENV_CACHE = IdDict{AbstractEnv,Dict{Symbol,Array}}()

# TODO:using https://github.com/oxinabox/AutoPreallocation.jl ?
for f in
    (:get_state, :get_terminal, :get_reward, :get_legal_actions, :get_legal_actions_mask)
    @eval function $f(env::MultiThreadEnv, args...; kwargs...)
        sample = $f(env[1], args...; kwargs...)
        m, n = length(sample), length(env)
        env_cache = get!(MULTI_THREAD_ENV_CACHE, env, Dict{Symbol,Array}())
        cache = get!(
            env_cache,
            Symbol($f, args, kwargs),
            Array{eltype(sample)}(undef, size(sample)..., n),
        )
        selectdim(cache, ndims(cache), 1) .= sample
        for i in 2:n
            selectdim(cache, ndims(cache), i) .= $f(env[i], args...; kwargs...)
        end
        cache
    end
end

get_actions(env::MultiThreadEnv, args...; kwargs...) =
    TupleSpace([get_actions(x, args...; kwargs...) for x in env.envs])
get_current_player(env::MultiThreadEnv) = [get_current_player(x) for x in env.envs]

function Base.show(io::IO, t::MIME"text/markdown", env::MultiThreadEnv)
    show(io, t, Markdown.parse("""
    # MultiThreadEnv

    ## Num of threads

    $(Threads.nthreads())

    ## Num of inner environments

    $(length(env.envs)) replicates of `$(get_name(env.envs[1]))`

    ## Traits of inner environment
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in get_env_traits()], "\n"))

    ## Actions of inner environment
    $(get_actions(env[1]))

    ## Players
    $(join(["- `$p`" for p in get_players(env.envs[1])], "\n"))

    ## Current Player
    $(join(["`$x`" for x in get_current_player(env)], ","))

    ## Is Environment Terminated?
    $(get_terminal(env))
    """))
end

# !!! some might not be meaningful, use with caution.
#=
for f in vcat(ENV_API, MULTI_AGENT_ENV_API)
    if f ∉ (:get_state, :get_terminal, :get_reward, :get_legal_actions, :get_legal_actions_mask)
        @eval $f(x::MultiThreadEnv, args...;kwargs...) = $f(x.envs[1], args...;kwargs...)
    end
end
=#

Base.summary(
    io::IO,
    env::T,
) where {T<:Union{
    SubjectiveEnv,
    MultiThreadEnv,
    StateOverriddenEnv,
    RewardOverriddenEnv,
    ActionTransformedEnv,
}} = print(io, T.name)
