struct GymEnv{T,Ta<:AbstractSpace,To<:AbstractSpace,P} <: AbstractEnv
    pyenv::P
    observation_space::To
    action_space::Ta
    state::P
end
export GymEnv

mutable struct AtariEnv{IsGrayScale,TerminalOnLifeLoss,N,S<:AbstractRNG} <: AbstractEnv
    ale::Ptr{Nothing}
    screens::Tuple{Array{UInt8,N},Array{UInt8,N}}  # for max-pooling
    actions::Vector{Int}
    action_space::DiscreteSpace{UnitRange{Int}}
    observation_space::MultiDiscreteSpace{Array{UInt8,N}}
    noopmax::Int
    frame_skip::Int
    reward::Float32
    lives::Int
    seed::S
end
export AtariEnv

mutable struct POMDPEnv{M,S,O,I,R,RNG<:AbstractRNG} <: AbstractEnv
    model::M
    state::S
    observation::O
    info::I
    reward::R
    rng::RNG
end
export POMDPEnv

mutable struct MDPEnv{M,S,I,R,RNG<:AbstractRNG} <: AbstractEnv
    model::M
    state::S
    info::I
    reward::R
    rng::RNG
end
export MDPEnv

mutable struct OpenSpielEnv{O,D,S,G,R} <: AbstractEnv
    state::S
    game::G
    rng::R
end
export OpenSpielEnv

struct OpenSpielObs{O,D,S}
    state::S
    player::Int32
end
