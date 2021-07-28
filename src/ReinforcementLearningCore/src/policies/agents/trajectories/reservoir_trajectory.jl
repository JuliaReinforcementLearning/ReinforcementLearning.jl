export ReservoirTrajectory

using MacroTools: @forward
using Random

mutable struct ReservoirTrajectory{B,R<:AbstractRNG} <: AbstractTrajectory
    buffer::B
    n::Int
    capacity::Int
    rng::R
end

@forward ReservoirTrajectory.buffer Base.keys, Base.haskey, Base.getindex

Base.length(x::ReservoirTrajectory) = length(x.buffer[1])

function ReservoirTrajectory(capacity; n = 0, rng = Random.GLOBAL_RNG, kw...)
    buffer = VectorTrajectory(; kw...)
    ReservoirTrajectory(buffer, n, capacity, rng)
end

function Base.push!(b::ReservoirTrajectory; kw...)
    b.n += 1
    if b.n <= b.capacity
        push!(b.buffer; kw...)
    else
        i = rand(b.rng, 1:b.n)
        if i <= b.capacity
            for (k, v) in kw
                b.buffer[k][i] = v
            end
        end
    end
end

function RLBase.update!(
    trajectory::ReservoirTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    if haskey(trajectory.buffer, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory; :state => s, :action => action, :legal_actions_mask => lasm)
    else
        push!(trajectory; :state => s, :action => action)
    end
end
