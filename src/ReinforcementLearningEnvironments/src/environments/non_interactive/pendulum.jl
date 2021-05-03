export PendulumNonInteractiveEnv

struct PendulumNonInteractiveEnvParams{Fl<:AbstractFloat}
    gravity::Fl
    length::Fl
    mass::Fl
    step_size::Fl
    maximum_time::Fl
end


"""
A non-interactive pendulum environment.

Accepts only `nothing` actions, which result in the system being simulated for one time step.
Sets `env.done` to `true` once `maximum_time` is reached. Resets to a random position and momentum.
Always returns zero rewards.

Useful for debugging and development purposes, particularly in model-based reinforcement learning.
"""
mutable struct PendulumNonInteractiveEnv{
    Fl<:AbstractFloat,
    VFl<:AbstractVector{Fl},
    R<:AbstractRNG,
} <: NonInteractiveEnv
    parameters::PendulumNonInteractiveEnvParams{Fl}
    state::VFl
    done::Bool
    t::Int
    rng::R
end

"""
    PendulumNonInteractiveEnv(;kwargs...)

# Keyword arguments

- `float_type = Float64`
- `gravity = 9.8`
- `length = 2.0`
- `mass = 1.0`
- `step_size = 0.01`
- `maximum_time = 10.0`
- `rng = Random.GLOBAL_RNG`
"""
function PendulumNonInteractiveEnv(;
    float_type = Float64,
    gravity = 9.8,
    length = 2.0,
    mass = 1.0,
    step_size = 0.01,
    maximum_time = 10.0,
    rng = Random.GLOBAL_RNG,
)
    parameters = PendulumNonInteractiveEnvParams{float_type}(
        gravity,
        length,
        mass,
        step_size,
        maximum_time,
    )
    env = PendulumNonInteractiveEnv(parameters, zeros(float_type, 2), false, 0, rng)
    reset!(env)
    env
end

Random.seed!(env::PendulumNonInteractiveEnv, seed) = Random.seed!(env.rng, seed)

RLBase.reward(env::PendulumNonInteractiveEnv) = 0
RLBase.is_terminated(env::PendulumNonInteractiveEnv) = env.done
RLBase.state(env::PendulumNonInteractiveEnv) = env.state
RLBase.state_space(env::PendulumNonInteractiveEnv{T}) where {T} =
    Space([typemin(T)..typemax(T), typemin(T)..typemax(T)])

function RLBase.reset!(env::PendulumNonInteractiveEnv{Fl}) where {Fl}
    env.state .= (Fl(2 * pi) * rand(env.rng, Fl), randn(env.rng, Fl))
    env.t = 0
    env.done = false

    nothing
end

function (env::PendulumNonInteractiveEnv{Fl})(a::Nothing) where {Fl}
    (g, l, m) = (env.parameters.gravity, env.parameters.length, env.parameters.mass)
    (dt, T) = (env.parameters.step_size, env.parameters.maximum_time)
    (theta, p_theta) = env.state

    # H = p_theta^2 / (2*m*l^2) + m*g*l*(1 - cos(theta))
    p_theta -= (dt / 2) * m * g * l * sin(theta)
    theta += dt * p_theta / (m * l^2)
    p_theta -= (dt / 2) * m * g * l * sin(theta)

    theta = mod(theta, 2 * Fl(pi))

    env.t += 1
    env.state .= (theta, p_theta)
    env.done = env.t * dt > T ? true : false

    nothing
end
