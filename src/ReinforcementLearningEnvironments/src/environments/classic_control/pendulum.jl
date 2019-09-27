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
    observation_space::MultiContinuousSpace{(3,),1}
    state::Array{T,1}
    done::Bool
    t::Int
    rng::R
    reward::T
end

function PendulumEnv(
    ;
    T = Float64,
    max_speed = T(8),
    max_torque = T(2),
    g = T(10),
    m = T(1),
    l = T(1),
    dt = T(.05),
    max_steps = 200,
)
    high = T.([1, 1, max_speed])
    env = PendulumEnv(
        PendulumEnvParams(max_speed, max_torque, g, m, l, dt, max_steps),
        ContinuousSpace(-2., 2.),
        MultiContinuousSpace(-high, high),
        zeros(T, 2),
        false,
        0,
        Random.GLOBAL_RNG,
        zero(T),
    )
    reset!(env)
    env
end

pendulum_observation(s) = [cos(s[1]), sin(s[1]), s[2]]
angle_normalize(x) = ((x + pi) % (2 * pi)) - pi

function observe(env::PendulumEnv)
    Observation(
        reward = env.reward,
        state = pendulum_observation(env.state),
        terminal = env.done,
    )
end

function reset!(env::PendulumEnv{T}) where {T}
    env.state[:] = 2 * rand(env.rng, T, 2) .- 1
    env.t = 0
    env.done = false
    env.reward = zero(T)
    nothing
end

function interact!(env::PendulumEnv, a)
    env.t += 1
    th, thdot = env.state
    a = clamp(a, -env.params.max_torque, env.params.max_torque)
    costs = angle_normalize(th)^2 + .1 * thdot^2 + .001 * a^2
    newthdot = thdot +
               (-3 * env.params.g / (2 * env.params.l) * sin(th + pi) +
                3 * a / (env.params.m * env.params.l^2)) * env.params.dt
    th += newthdot * env.params.dt
    newthdot = clamp(newthdot, -env.params.max_speed, env.params.max_speed)
    env.state[1] = th
    env.state[2] = newthdot
    env.done = env.t >= env.params.max_steps
    env.reward = -costs
    nothing
end