export MultiThreadEnv

using Base.Threads: @spawn

"""
    MultiThreadEnv(envs::Vector{<:AbstractEnv})

Wrap multiple instances of the same environment type into one environment.
Each environment will run in parallel by leveraging `Threads.@spawn`.
So remember to set the environment variable `JULIA_NUM_THREADS`!
"""
struct MultiThreadEnv{E,S,R,AS,SS,L} <: AbstractEnv
    envs::Vector{E}
    states::S
    rewards::R
    terminals::BitArray{1}
    action_space::AS
    state_space::SS
    legal_action_space_mask::L
end

function Base.show(io::IO, t::MIME"text/markdown", env::MultiThreadEnv)
    print(io, "MultiThreadEnv($(length(env)) x $(nameof(env[1])))")
end

"""
    MultiThreadEnv(f, n::Int)

`f` is a lambda function which creates an `AbstractEnv` by calling `f()`.
"""
MultiThreadEnv(f, n::Int) = MultiThreadEnv([f() for _ in 1:n])

function MultiThreadEnv(envs::Vector{<:AbstractEnv})
    n = length(envs)
    S = state_space(envs[1])
    s = state(envs[1])
    if S isa Space
        S_batch = similar(S, size(S)..., n)
        s_batch = similar(s, size(s)..., n)
        for j in 1:n
            Sₙ = state_space(envs[j])
            sₙ = state(envs[j])
            for i in CartesianIndices(size(S))
                S_batch[i, j] = Sₙ[i]
                s_batch[i, j] = sₙ[i]
            end
        end
    else
        S_batch = Space(state_space.(envs))
        s_batch = state.(envs)
    end

    A = action_space(envs[1])
    if A isa Space
        A_batch = similar(A, size(A)..., n)
        for j in 1:n
            Aⱼ = action_space(envs[j])
            for i in CartesianIndices(size(A))
                A_batch[i, j] = Aⱼ[i]
            end
        end
    else
        A_batch = Space(action_space.(envs))
    end

    r_batch = reward.(envs)
    t_batch = is_terminated.(envs)
    if ActionStyle(envs[1]) === FULL_ACTION_SET
        m_batch = BitArray(undef, size(A)..., n)
        for j in 1:n
            L = legal_action_space_mask(envs[j])
            for i in CartesianIndices(size(A))
                m_batch[i, j] = L[i]
            end
        end
    else
        m_batch = nothing
    end
    MultiThreadEnv(envs, s_batch, r_batch, t_batch, A_batch, S_batch, m_batch)
end

MacroTools.@forward MultiThreadEnv.envs Base.getindex, Base.length, Base.iterate

function (env::MultiThreadEnv)(actions)
    N = ndims(actions)
    @sync for i in 1:length(env)
        @spawn begin
            if N == 1
                env[i](actions[i])
            else
                env[i](selectdim(actions, N, i))
            end
        end
    end
end

function RLBase.reset!(env::MultiThreadEnv; is_force = false)
    if is_force
        for i in 1:length(env)
            reset!(env[i])
        end
    else
        @sync for i in 1:length(env)
            if is_terminated(env[i])
                @spawn begin
                    reset!(env[i])
                end
            end
        end
    end
end

const MULTI_THREAD_ENV_CACHE = IdDict{AbstractEnv,Dict{Symbol,Array}}()

function RLBase.state(env::MultiThreadEnv)
    N = ndims(env.states)
    @sync for i in 1:length(env)
        @spawn selectdim(env.states, N, i) .= state(env[i])
    end
    env.states
end

function RLBase.reward(env::MultiThreadEnv)
    env.rewards .= reward.(env.envs)
    env.rewards
end

function RLBase.is_terminated(env::MultiThreadEnv)
    env.terminals .= is_terminated.(env.envs)
    env.terminals
end

function RLBase.legal_action_space_mask(env::MultiThreadEnv)
    N = ndims(env.states)
    @sync for i in 1:length(env)
        @spawn selectdim(env.legal_action_space_mask, N, i) .=
            legal_action_space_mask(env[i])
    end
    env.legal_action_space_mask
end

RLBase.action_space(env::MultiThreadEnv) = env.action_space
RLBase.state_space(env::MultiThreadEnv) = env.state_space
RLBase.legal_action_space(env::MultiThreadEnv) = Space(legal_action_space.(env.envs))
# RLBase.current_player(env::MultiThreadEnv) = current_player.(env.envs)

for f in RLBase.ENV_API
    if endswith(String(f), "Style")
        @eval RLBase.$f(x::MultiThreadEnv) = $f(x[1])
    end
end

#####
# Patches
#####

(env::MultiThreadEnv)(action::EnrichedAction) = env(action.action)

function (π::QBasedPolicy)(env::MultiThreadEnv, ::MinimalActionSet, A)
    [A[i][a] for (i, a) in enumerate(π.explorer(π.learner(env)))]
end

function (π::QBasedPolicy)(env::MultiThreadEnv, ::FullActionSet, A)
    [
        A[i][a] for
        (i, a) in enumerate(π.explorer(π.learner(env), legal_action_space_mask(env)))
    ]
end

function (π::QBasedPolicy)(
    env::MultiThreadEnv,
    ::MinimalActionSet,
    ::Space{<:Vector{<:Base.OneTo{<:Integer}}},
)
    π.explorer(π.learner(env))
end

function (π::QBasedPolicy)(
    env::MultiThreadEnv,
    ::FullActionSet,
    ::Space{<:Vector{<:Base.OneTo{<:Integer}}},
)
    π.explorer(π.learner(env), legal_action_space_mask(env))
end
