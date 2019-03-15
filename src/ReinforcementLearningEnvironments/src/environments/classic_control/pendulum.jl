using Random

export PendulumEnv

struct PendulumEnvParams{T}
    max_speed::T
    max_torque::T
    g::T
    m::T
    l::T
    dt::T
    max_steps::Int64
end

mutable struct PendulumEnv{T, R<:AbstractRNG} <: AbstractEnv
    params::PendulumEnvParams{T}
    action_space::ContinuousSpace
    observation_space::MultiContinuousSpace{(3,), 1}
    state::Array{T, 1}
    done::Bool
    t::Int64
    rng::R
end

function PendulumEnv(; T = Float64, max_speed = T(8), max_torque = T(2), 
                    g = T(10), m = T(1), l = T(1), dt = T(.05), max_steps = 200)
    high = T.([1, 1, max_speed])
    env = PendulumEnv(PendulumEnvParams(max_speed, max_torque, g, m, l, dt, max_steps), 
                   ContinuousSpace(-2., 2.),
                   MultiContinuousSpace(-high, high),
                   zeros(T, 2), false, 0, Random.GLOBAL_RNG)
    reset!(env)
    env
end

action_space(env::PendulumEnv) = env.action_space
observation_space(env::PendulumEnv) = env.observation_space

pendulum_observation(s) = [cos(s[1]), sin(s[1]), s[2]]
angle_normalize(x) = ((x + pi) % (2*pi)) - pi

observe(env::PendulumEnv) = (observation=pendulum_observation(env.state), isdone=env.done)

function reset!(env::PendulumEnv{T}) where T
    env.state[:] = 2 * rand(env.rng, T, 2) .- 1
    env.t = 0
    env.done = false
    nothing
end

function interact!(env::PendulumEnv, a)
    env.t += 1
    th, thdot = env.state
    a = clamp(a, -env.params.max_torque, env.params.max_torque)
    costs = angle_normalize(th)^2 + .1 * thdot^2 + .001 * a^2
    newthdot = thdot + (-3 * env.params.g/(2*env.params.l) * sin(th + pi) + 
                        3 * a/(env.params.m * env.params.l^2)) * env.params.dt
    th += newthdot * env.params.dt
    newthdot = clamp(newthdot, -env.params.max_speed, env.params.max_speed)
    env.state[1] = th
    env.state[2] = newthdot
    env.done = env.t >= env.params.max_steps
    (observation=pendulum_observation(env.state), reward=-costs, isdone=env.done)
end