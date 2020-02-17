using Random

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

mutable struct PendulumEnv{T,R<:AbstractRNG} <: AbstractEnv
    params::PendulumEnvParams{T}
    action_space::ContinuousSpace
    observation_space::MultiContinuousSpace{Vector{T}}
    state::Vector{T}
    done::Bool
    t::Int
    rng::R
    reward::T
end

function PendulumEnv(;
    T = Float64,
    max_speed = T(8),
    max_torque = T(2),
    g = T(10),
    m = T(1),
    l = T(1),
    dt = T(0.05),
    max_steps = 200,
    seed = nothing,
)
    high = T.([1, 1, max_speed])
    env = PendulumEnv(
        PendulumEnvParams(max_speed, max_torque, g, m, l, dt, max_steps),
        ContinuousSpace(-2.0, 2.0),
        MultiContinuousSpace(-high, high),
        zeros(T, 2),
        false,
        0,
        MersenneTwister(seed),
        zero(T),
    )
    reset!(env)
    env
end

Random.seed!(env::PendulumEnv, seed) = Random.seed!(env.rng, seed)

pendulum_observation(s) = [cos(s[1]), sin(s[1]), s[2]]
angle_normalize(x) = ((x + pi) % (2 * pi)) - pi

RLBase.observe(env::PendulumEnv) =
    (reward = env.reward, state = pendulum_observation(env.state), terminal = env.done)

function RLBase.reset!(env::PendulumEnv{T}) where {T}
    env.state[:] = 2 * rand(env.rng, T, 2) .- 1
    env.t = 0
    env.done = false
    env.reward = zero(T)
    nothing
end

function (env::PendulumEnv)(a)
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
