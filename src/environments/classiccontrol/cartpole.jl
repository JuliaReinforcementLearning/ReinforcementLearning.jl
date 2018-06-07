using ReinforcementLearning
import ReinforcementLearning.interact!,
ReinforcementLearning.getstate,
ReinforcementLearning.reset!

struct CartPoleParams{T}
    gravity::T
    masscart::T
    masspole::T
    totalmass::T
    halflength::T
    polemasslength::T
    forcemag::T
    tau::T
    thetathreshold::T
    xthreshold::T
    maxsteps::Int64
end
mutable struct CartPole{T}
    params::CartPoleParams{T}
    observation_space::ReinforcementLearning.Box{T}
    state::Array{T, 1}
    done::Bool
    t::Int64
end
function CartPole(; T = Float64, gravity = T(9.8), masscart = T(1.), 
                  masspole = T(.1), halflength = T(.5), forcemag = T(10.),
                  maxsteps = 200)
    params = CartPoleParams(gravity, masscart, masspole, masscart + masspole,
                            halflength, masspole * halflength, forcemag,
                            T(.02), T(2 * 12 * π /360), T(2.4), maxsteps)
    high = [2 * params.xthreshold, T(1e38),
            2 * params.thetathreshold, T(1e38)]
    cp = CartPole(params, ReinforcementLearning.Box(-high, high), 
                  zeros(T, 4), false, 0)
    reset!(cp)
    cp
end

function reset!(env::CartPole{T}) where T <: Number
    env.state[:] = T(.1) * rand(T, 4) - T(.05)
    env.t = 0
    env.done = false
    env.state
end

function getstate(env::CartPole)
    env.state, env.done
end

function interact!(a, env::CartPole{T}) where T <: Number
    if env.done
        reset!(env)
        return env.state, 1., env.done
    end
    env.t += 1
    force = a == 2 ? env.params.forcemag : -env.params.forcemag
    x, xdot, theta, thetadot = env.state
    costheta = cos(theta)
    sintheta = sin(theta)
    tmp = (force + env.params.polemasslength * thetadot^2 * sintheta) /
        env.params.totalmass
    thetaacc = (env.params.gravity * sintheta - costheta * tmp) / 
        (env.params.halflength * 
            (4/3 - env.params.masspole * costheta^2/env.params.totalmass))
    xacc = tmp - env.params.polemasslength * thetaacc * costheta / 
        env.params.totalmass
    env.state[1] += env.params.tau * xdot
    env.state[2] += env.params.tau * xacc
    env.state[3] += env.params.tau * thetadot
    env.state[4] += env.params.tau * thetaacc
    env.done = abs(env.state[1]) > env.params.xthreshold ||
               abs(env.state[3]) > env.params.thetathreshold ||
               env.t >= env.params.maxsteps
    env.state, 1., env.done
end
