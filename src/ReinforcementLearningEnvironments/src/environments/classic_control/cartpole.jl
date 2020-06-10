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
    action_space::DiscreteSpace{UnitRange{Int64}}
    observation_space::MultiContinuousSpace{Vector{T}}
    state::Array{T,1}
    action::Int
    done::Bool
    t::Int
    rng::R
end

Base.show(io::IO, env::CartPoleEnv{T}) where {T} =
    print(io, "CartPoleEnv{$T}($(env.params))")

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
- `seed = nothing`
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
    seed = nothing,
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
    high = [2 * params.xthreshold, T(1e38), 2 * params.thetathreshold, T(1e38)]
    cp = CartPoleEnv(
        params,
        DiscreteSpace(2),
        MultiContinuousSpace(-high, high),
        zeros(T, 4),
        2,
        false,
        0,
        MersenneTwister(seed),
    )
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

RLBase.observe(env::CartPoleEnv{T}) where {T} =
    (reward = env.done ? zero(T) : one(T), terminal = env.done, state = env.state)

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

function plotendofepisode(x, y, d)
    if d
        setmarkercolorind(7)
        setmarkertype(-1)
        setmarkersize(6)
        polymarker([x], [y])
    end
    return nothing
end
function RLBase.render(env::CartPoleEnv)
    s, a, d = env.state, env.action, env.done
    x, xdot, theta, thetadot = s
    l = 2 * env.params.halflength
    clearws()
    setviewport(0, 1, 0, 1)
    xthreshold = env.params.xthreshold
    setwindow(-xthreshold, xthreshold, -.1, l + 0.1)
    fillarea([x - 0.5, x - 0.5, x + 0.5, x + 0.5], [-.05, 0, 0, -.05])
    setlinecolorind(4)
    setlinewidth(3)
    polyline([x, x + l * sin(theta)], [0, l * cos(theta)])
    setlinecolorind(2)
    drawarrow(x + (a == 1) - 0.5, -.025, x + 1.4 * (a == 1) - 0.7, -.025)
    plotendofepisode(xthreshold - 0.2, l, d)
    updatews()
end
