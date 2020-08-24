struct GymEnv{T,Ta<:AbstractSpace,To<:AbstractSpace,P} <: AbstractEnv
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
    action_space::DiscreteSpace{UnitRange{Int}}
    observation_space::MultiDiscreteSpace{Array{UInt8,N}}
    noopmax::Int
    frame_skip::Int
    reward::Float32
    lives::Int
    rng::S
end
export AtariEnv

mutable struct POMDPEnv{M,S,A,O,R} <: AbstractEnv
    model::M
    state::S
    action::A
    observation::O
    rng::R
end
export POMDPEnv

mutable struct MDPEnv{M,S,A,R} <: AbstractEnv
    model::M
    state::S
    action::A
    rng::R
end
export MDPEnv

mutable struct OpenSpielEnv{S,T,ST,G,R} <: AbstractEnv
    state::ST
    game::G
    rng::R
end
export OpenSpielEnv

mutable struct SnakeGameEnv{A,N,G} <: AbstractEnv
    game::G
    latest_snakes_length::Vector{Int}
    latest_actions::Vector{CartesianIndex{2}}
    is_terminated::Bool
end
export SnakeGameEnv