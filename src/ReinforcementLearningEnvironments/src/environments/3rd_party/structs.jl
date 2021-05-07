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

struct AcrobotEnvParams{T}
    link_length_a::T # [m]
    link_length_b::T # [m]
    link_mass_a::T # : [kg] mass of link 1
    link_mass_b::T # : [kg] mass of link 2
    # : [m] position of the center of mass of link 1
    link_com_pos_a::T
    # : [m] position of the center of mass of link 2
    link_com_pos_b::T
    # : Rotation related parameters
    link_moi::T
    max_torque_noise::T
    # : [m/s] maximum velocity of link 1
    max_vel_a::T
    # : [m/s] maximum velocity of link 2
    max_vel_b::T
    # : [m/s2] acceleration due to gravity
    g::T
    # : [s] timestep
    dt::T
    # : maximum steps in episode
    max_steps::Int
end

export AcrobotEnvParams

mutable struct AcrobotEnv{T,R<:AbstractRNG} <: AbstractEnv
    params::AcrobotEnvParams{T}
    state::Vector{T}
    action::Int
    done::Bool
    t::Int
    rng::R
    reward::T
    # difference in second link angular acceleration equation
    # as per python gym
    book_or_nips::String
    # array of available torques based on actions
    avail_torque::Vector{T}
end

export AcrobotEnv
