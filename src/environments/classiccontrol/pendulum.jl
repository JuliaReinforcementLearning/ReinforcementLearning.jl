using ReinforcementLearning
import ReinforcementLearning.interact!,
ReinforcementLearning.getstate,
ReinforcementLearning.reset!

struct PendulumParams{T}
    maxspeed::T
    maxtorque::T
    g::T
    m::T
    l::T
    dt::T
    maxsteps::Int64
end
mutable struct Pendulum{T}
    params::PendulumParams{T}
    observation_space::ReinforcementLearning.Box{T}
    state::Array{T, 1}
    done::Bool
    t::Int64
end
function Pendulum(; T = Float64, maxspeed = T(8), maxtorque = T(2), 
                    g = T(10), m = T(1), l = T(1), dt = T(.05), maxsteps = 200)
    high = T.([1, 1, maxspeed])
    env = Pendulum(PendulumParams(maxspeed, maxtorque, g, m, l, dt, maxsteps), 
                   ReinforcementLearning.Box(-high, high),
                   zeros(T, 2), false, 0)
    reset!(env)
    env
end

pendulumobservation(s) = [cos(s[1]), sin(s[1]), s[2]]
anglenormalize(x) = ((x + pi) % (2*pi)) - pi

getstate(env::Pendulum) = (pendulumobservation(env.state), env.done)
function reset!(env::Pendulum{T}) where T
    box = env.observation_space
    env.state[:] = 2 * rand(T, 2) - 1
    env.t = 0
    env.done = false
    pendulumobservation(env.state)
end

function interact!(a, env::Pendulum)
    if env.done
        reset!(env)
        return pendulumobservation(env.state), 0., env.done
    end
    env.t += 1
    th, thdot = env.state
    a = clamp(a, -env.params.maxtorque, env.params.maxtorque)
    costs = anglenormalize(th)^2 + .1 * thdot^2 + .001 * a^2
    newthdot = thdot + (-3 * env.params.g/(2*env.params.l) * sin(th + pi) + 
                        3 * a/(env.params.m * env.params.l^2)) * env.params.dt
    th += newthdot * env.params.dt
    newthdot = clamp(newthdot, -env.params.maxspeed, env.params.maxspeed)
    env.state[1] = th
    env.state[2] = newthdot
    env.done = env.t >= env.params.maxsteps
    pendulumobservation(env.state), -costs, env.done
end
