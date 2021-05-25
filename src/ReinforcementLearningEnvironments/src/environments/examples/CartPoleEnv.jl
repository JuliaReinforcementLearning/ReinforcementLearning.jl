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

mutable struct CartPoleEnv{T,R<:AbstractRNG} <: AbstractEnv
    params::CartPoleEnvParams{T}
    state::Array{T,1}
    action::Int
    done::Bool
    t::Int
    rng::R
end

"""
    CartPoleEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `gravity = T(9.8)`
- `masscart = T(1.0)`
- `masspole = T(0.1)`
- `halflength = T(0.5)`
- `forcemag = T(10.0)`
- `max_steps = 200`
- 'dt = 0.02'
- `rng = Random.GLOBAL_RNG`
"""
function CartPoleEnv(;
    T = Float64,
    gravity = 9.8,
    masscart = 1.0,
    masspole = 0.1,
    halflength = 0.5,
    forcemag = 10.0,
    max_steps = 200,
    dt = 0.02,
    rng = Random.GLOBAL_RNG,
)
    params = CartPoleEnvParams{T}(
        gravity,
        masscart,
        masspole,
        masscart + masspole,
        halflength,
        masspole * halflength,
        forcemag,
        dt,
        2 * 12 * π / 360,
        2.4,
        max_steps,
    )
    high = cp = CartPoleEnv(params, zeros(T, 4), 2, false, 0, rng)
    reset!(cp)
    cp
end

CartPoleEnv{T}(; kwargs...) where {T} = CartPoleEnv(; T = T, kwargs...)

function RLBase.reset!(env::CartPoleEnv{T}) where {T<:Number}
    env.state[:] = T(0.1) * rand(env.rng, T, 4) .- T(0.05)
    env.t = 0
    env.action = 2
    env.done = false
    nothing
end

RLBase.action_space(env::CartPoleEnv) = Base.OneTo(2)

RLBase.state_space(env::CartPoleEnv{T}) where {T} = Space(
    ClosedInterval{T}[
        (-2*env.params.xthreshold)..(2*env.params.xthreshold),
        -1e38..1e38,
        (-2*env.params.thetathreshold)..(2*env.params.thetathreshold),
        -1e38..1e38,
    ],
)

RLBase.reward(env::CartPoleEnv{T}) where {T} = env.done ? zero(T) : one(T)
RLBase.is_terminated(env::CartPoleEnv) = env.done
RLBase.state(env::CartPoleEnv) = env.state

function (env::CartPoleEnv)(a)
    @assert a in (1, 2)
    env.action = a
    env.t += 1
    force = a == 2 ? env.params.forcemag : -env.params.forcemag
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

Random.seed!(env::CartPoleEnv, seed) = Random.seed!(env.rng, seed)
