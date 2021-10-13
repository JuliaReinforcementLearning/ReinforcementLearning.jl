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

mutable struct CartPoleEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    params::CartPoleEnvParams{T}
    action_space::A
    state_space::Space{Vector{ClosedInterval{T}}}
    state::Array{T,1}
    last_action::T
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
    continuous::Bool = false,
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
    high = T.([2 * params.xthreshold, 1e38, 2 * params.thetathreshold, 1e38])
    action_space = continuous ? -forcemag..forcemag : Base.OneTo(2)
    env = CartPoleEnv(
        params, 
        action_space,
        Space(ClosedInterval{T}.(-high, high)),
        zeros(T, 4), 
        zero(T),
        false, 
        0, 
        rng,
    )
    reset!(env)
    env
end

CartPoleEnv{T}(; kwargs...) where {T} = CartPoleEnv(; T = T, kwargs...)

function RLBase.reset!(env::CartPoleEnv{A,T}) where {A,T<:Number}
    env.state .= T(0.1) .* rand(env.rng, T, 4) .- T(0.05)
    env.t = 0
    env.done = false
    nothing
end

RLBase.action_space(env::CartPoleEnv) = env.action_space

RLBase.state_space(env::CartPoleEnv) = env.state_space

RLBase.reward(env::CartPoleEnv{A,T}) where {A,T<:Number} = env.done ? zero(T) : one(T)
RLBase.is_terminated(env::CartPoleEnv) = env.done
RLBase.state(env::CartPoleEnv) = env.state

function (env::CartPoleEnv{<:ClosedInterval})(a)
    @assert a in env.action_space
    _step!(env, a)
end

function (env::CartPoleEnv{<:Base.OneTo})(a)
    @assert a in env.action_space
    force = a == 2 ? env.params.forcemag : -env.params.forcemag
    _step!(env, force)
end

function _step!(env::CartPoleEnv, force)
    env.last_action = force
    env.t += 1
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
