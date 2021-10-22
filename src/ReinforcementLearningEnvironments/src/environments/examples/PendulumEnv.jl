export PendulumEnv

struct PendulumEnvParams{T}
    max_speed::T
    max_torque::T
    g::T
    m::T
    l::T
    dt::T
    max_steps::Int
end

mutable struct PendulumEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    params::PendulumEnvParams{T}
    action_space::A
    action::T
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    done::Bool
    t::Int
    rng::R
    reward::T
    n_actions::Int
end

"""
    PendulumEnv(;kwargs...)

# Keyword arguments

- `T = Float64`
- `max_speed = T(8)`
- `max_torque = T(2)`
- `g = T(10)`
- `m = T(1)`
- `l = T(1)`
- `dt = T(0.05)`
- `max_steps = 200`
- `continuous::Bool = true`
- `n_actions::Int = 3`
- `rng = Random.GLOBAL_RNG`
"""
function PendulumEnv(;
    T = Float64,
    max_speed = T(8),
    max_torque = T(2),
    g = T(10),
    m = T(1),
    l = T(1),
    dt = T(0.05),
    max_steps = 200,
    continuous::Bool = true,
    n_actions::Int = 3,
    rng = Random.GLOBAL_RNG,
)
    high = T.([1, 1, max_speed])
    action_space = continuous ? -2.0..2.0 : Base.OneTo(n_actions)
    env = PendulumEnv(
        PendulumEnvParams(max_speed, max_torque, g, m, l, dt, max_steps),
        action_space,
        zero(T),
        Space(ClosedInterval{T}.(-high, high)),
        zeros(T, 2),
        false,
        0,
        rng,
        zero(T),
        n_actions,
    )
    reset!(env)
    env
end

Random.seed!(env::PendulumEnv, seed) = Random.seed!(env.rng, seed)

pendulum_observation(s) = [cos(s[1]), sin(s[1]), s[2]]
angle_normalize(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π

RLBase.action_space(env::PendulumEnv) = env.action_space
RLBase.state_space(env::PendulumEnv) = env.observation_space
RLBase.reward(env::PendulumEnv) = env.reward
RLBase.is_terminated(env::PendulumEnv) = env.done
RLBase.state(env::PendulumEnv) = pendulum_observation(env.state)

function RLBase.reset!(env::PendulumEnv{A,T}) where {A,T}
    env.state[1] = 2 * π * (rand(env.rng, T) .- 1)
    env.state[2] = 2 * (rand(env.rng, T) .- 1)
    env.action = zero(T)
    env.t = 0
    env.done = false
    env.reward = zero(T)
    nothing
end

function (env::PendulumEnv)(a::Union{Int, AbstractFloat})
    @assert a in env.action_space
    env.action = torque(env, a)
    _step!(env, env.action)
end

function _step!(env::PendulumEnv, a)
    env.t += 1
    th, thdot = env.state
    a = clamp(a, -env.params.max_torque, env.params.max_torque)
    costs = angle_normalize(th)^2 + 0.1 * thdot^2 + 0.001 * a^2
    newthdot =
        thdot +
        (
            -3 * env.params.g / (2 * env.params.l) * sin(th + pi) +
            3 * a / (env.params.m * env.params.l^2)
        ) * env.params.dt
    th += newthdot * env.params.dt
    newthdot = clamp(newthdot, -env.params.max_speed, env.params.max_speed)
    env.state[1] = th
    env.state[2] = newthdot
    env.done = env.t >= env.params.max_steps
    env.reward = -costs
    nothing
end

function torque(env::PendulumEnv{<:Base.OneTo}, a::Int)
    return (4 / (env.n_actions - 1)) * (a - (env.n_actions - 1) / 2 - 1)
end

torque(env::PendulumEnv{<:ClosedInterval}, a::AbstractFloat) = a
