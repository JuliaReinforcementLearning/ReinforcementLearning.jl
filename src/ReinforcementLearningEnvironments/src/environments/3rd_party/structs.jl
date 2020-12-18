struct GymEnv{T,Ta,To,P} <: AbstractEnv
    pyenv::P
    observation_space::To
    action_space::Ta
    state::P
end
export GymEnv

mutable struct AtariEnv{IsGrayScale,TerminalOnLifeLoss,N,S<:AbstractRNG} <: AbstractEnv
    ale::Ptr{Nothing}
    name::String
    screens::Tuple{Array{UInt8,N},Array{UInt8,N}}  # for max-pooling
    actions::Vector{Int}
    action_space::Base.OneTo{Int}
    observation_space::Space{Array{ClosedInterval{UInt8},N}}
    noopmax::Int
    frame_skip::Int
    reward::Float32
    lives::Int
    rng::S
end
export AtariEnv

mutable struct OpenSpielEnv{S,G} <: AbstractEnv
    state::S
    game::G
end
export OpenSpielEnv

# ??? can we safely ignore the `game` field here
Base.hash(e::OpenSpielEnv, h::UInt) = hash(e.state, h)
Base.:(==)(e::OpenSpielEnv, ee::OpenSpielEnv) = e.state == ee.state

mutable struct SnakeGameEnv{A,N,G} <: AbstractEnv
    game::G
    latest_snakes_length::Vector{Int}
    latest_actions::Vector{CartesianIndex{2}}
    is_terminated::Bool
end
export SnakeGameEnv
