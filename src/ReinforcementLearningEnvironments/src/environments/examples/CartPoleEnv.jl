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

function CartPoleEnvParams{T}(;
    gravity=9.8,
    masscart=1.0,
    masspole=0.1,
    halflength=0.5,
    forcemag=10.0,
    max_steps=200,
    dt=0.02,
    thetathreshold=12.0,
    xthreshold=2.4
) where {T}
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

mutable struct CartPoleEnv{T,ACT} <: AbstractEnv
    params::CartPoleEnvParams{T}
    state::Vector{T}
    action::ACT
    done::Bool
    t::Int
    rng::AbstractRNG
end

"""
    CartPoleEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `continuous = false`
- `rng = Random.default_rng()`
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
function CartPoleEnv(; T=Float64, continuous=false, rng=Random.default_rng(), kwargs...)
    params = CartPoleEnvParams{T}(; kwargs...)
    env = CartPoleEnv(params, zeros(T, 4), continuous ? zero(T) : zero(Int), false, 0, rng)
    reset!(env)
    env
end

CartPoleEnv{T}(; kwargs...) where {T} = CartPoleEnv(T=T, kwargs...)

Random.seed!(env::CartPoleEnv, seed) = Random.seed!(env.rng, seed)
RLBase.reward(env::CartPoleEnv{T}) where {T} = env.done ? zero(T) : one(T)
RLBase.is_terminated(env::CartPoleEnv) = env.done
RLBase.state(env::CartPoleEnv, ::Observation, ::DefaultPlayer) = env.state

function RLBase.state_space(env::CartPoleEnv{T}) where {T}
    ((-2 * env.params.xthreshold) .. (2 * env.params.xthreshold)) ×
    (typemin(T) .. typemax(T)) ×
    ((-2 * env.params.thetathreshold) .. (2 * env.params.thetathreshold)) ×
    (typemin(T) .. typemax(T))
end

RLBase.action_space(env::CartPoleEnv{<:AbstractFloat,Int}, ::DefaultPlayer) = Base.OneTo(2)
RLBase.action_space(env::CartPoleEnv{<:AbstractFloat,<:AbstractFloat}, ::DefaultPlayer) = -1.0 .. 1.0

function RLBase.reset!(env::CartPoleEnv{T}) where {T}
    env.state[:] = T(0.1) * rand(env.rng, T, 4) .- T(0.05)
    env.t = 0
    env.action = rand(env.rng, action_space(env))
    env.done = false
    nothing
end

function RLBase.act!(env::CartPoleEnv, a::AbstractFloat)
    @assert a in action_space(env)
    env.action = a
    _step!(env, a)
end

function RLBase.act!(env::CartPoleEnv, a::Int)
    @assert a in action_space(env)
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
