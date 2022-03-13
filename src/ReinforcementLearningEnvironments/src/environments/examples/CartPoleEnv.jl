export CartPoleEnv

struct CartPoleEnvParams{T}
    gravity::T
    masscart::T
    masspole::T
    totalmass::T
    halflength::T
    polemasslength::T
    forcemag::T
    dt::T
    thetathreshold::T
    xthreshold::T
    max_steps::Int
end

Base.show(io::IO, params::CartPoleEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(CartPoleEnvParams)], ","),
)

function CartPoleEnvParams(;
    T = Float64,
    gravity = 9.8,
    masscart = 1.0,
    masspole = 0.1,
    halflength = 0.5,
    forcemag = 10.0,
    max_steps = 200,
    dt = 0.02,
    thetathreshold = 12.0,
    xthreshold = 2.4
)
    CartPoleEnvParams{T}(
        gravity,
        masscart,
        masspole,
        masscart + masspole,
        halflength,
        masspole * halflength,
        forcemag,
        dt,
        thetathreshold * π / 180,
        xthreshold,
        max_steps,
    )
end

mutable struct CartPoleEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::CartPoleEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    done::Bool
    t::Int
    rng::R
end

"""
    CartPoleEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `continuous = false`
- `rng = Random.GLOBAL_RNG`
- `gravity = T(9.8)`
- `masscart = T(1.0)`
- `masspole = T(0.1)`
- `halflength = T(0.5)`
- `forcemag = T(10.0)`
- `max_steps = 200`
- `dt = 0.02`
- `thetathreshold = 12.0 # degrees`
- `xthreshold` = 2.4`
"""
function CartPoleEnv(;
    T = Float64,
    continuous = false,
    rng = Random.GLOBAL_RNG,
    kwargs...
)
    params = CartPoleEnvParams(; T = T, kwargs...)
    action_space = continuous ? ClosedInterval{T}(-1.0, 1.0) : Base.OneTo(2)
    state_space = Space(
        ClosedInterval{T}[
            (-2*params.xthreshold)..(2*params.xthreshold),
            typemin(T)..typemax(T),
            (-2*params.thetathreshold)..(2*params.thetathreshold),
            typemin(T)..typemax(T),
        ],
    )
    env = CartPoleEnv(
        params,
        action_space,
        state_space,
        zeros(T, 4),
        rand(action_space),
        false,
        0,
        rng,
    )
    reset!(env)
    env
end

CartPoleEnv{T}(; kwargs...) where {T} = CartPoleEnv(T = T, kwargs...)

Random.seed!(env::CartPoleEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::CartPoleEnv) = env.action_space
RLBase.state_space(env::CartPoleEnv) = env.observation_space
RLBase.reward(env::CartPoleEnv{A,T}) where {A,T} = env.done ? zero(T) : one(T)
RLBase.is_terminated(env::CartPoleEnv) = env.done
RLBase.state(env::CartPoleEnv) = env.state

function RLBase.reset!(env::CartPoleEnv{A,T}) where {A,T}
    env.state[:] = T(0.1) * rand(env.rng, T, 4) .- T(0.05)
    env.t = 0
    env.action = rand(env.rng, env.action_space)
    env.done = false
    nothing
end

function (env::CartPoleEnv{<:ClosedInterval})(a::AbstractFloat)
    @assert a in env.action_space
    env.action = a
    _step!(env, a)
end

function (env::CartPoleEnv{<:Base.OneTo{Int}})(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a == 2 ? 1 : -1)
end

function _step!(env::CartPoleEnv, a)
    env.t += 1
    force = a * env.params.forcemag
    x, xdot, theta, thetadot = env.state
    costheta = cos(theta)
    sintheta = sin(theta)
    tmp = (force + env.params.polemasslength * thetadot^2 * sintheta) / env.params.totalmass
    thetaacc =
        (env.params.gravity * sintheta - costheta * tmp) / (
            env.params.halflength *
            (4 / 3 - env.params.masspole * costheta^2 / env.params.totalmass)
        )
    xacc = tmp - env.params.polemasslength * thetaacc * costheta / env.params.totalmass
    env.state[1] += env.params.dt * xdot
    env.state[2] += env.params.dt * xacc
    env.state[3] += env.params.dt * thetadot
    env.state[4] += env.params.dt * thetaacc
    env.done =
        abs(env.state[1]) > env.params.xthreshold ||
        abs(env.state[3]) > env.params.thetathreshold ||
        env.t > env.params.max_steps
    nothing
end
