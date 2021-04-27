export MountainCarEnv, ContinuousMountainCarEnv

struct MountainCarEnvParams{T}
    min_pos::T
    max_pos::T
    max_speed::T
    goal_pos::T
    goal_velocity::T
    power::T
    gravity::T
    max_steps::Int
end

function MountainCarEnvParams(;
    T = Float64,
    min_pos = -1.2,
    max_pos = 0.6,
    max_speed = 0.07,
    goal_pos = 0.5,
    max_steps = 200,
    goal_velocity = 0.0,
    power = 0.001,
    gravity = 0.0025,
)
    MountainCarEnvParams{T}(
        min_pos,
        max_pos,
        max_speed,
        goal_pos,
        goal_velocity,
        power,
        gravity,
        max_steps,
    )
end

mutable struct MountainCarEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::MountainCarEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    action::ACT
    done::Bool
    t::Int
    rng::R
end

"""
    MountainCarEnv(;kwargs...)

# Keyword arguments

- `T = Float64`
- `continuous = false`
- `rng = Random.GLOBAL_RNG`
- `min_pos = -1.2`
- `max_pos = 0.6`
- `max_speed = 0.07`
- `goal_pos = 0.5`
- `max_steps = 200`
- `goal_velocity = 0.0`
- `power = 0.001`
- `gravity = 0.0025`
"""
function MountainCarEnv(;
    T = Float64,
    continuous = false,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)
    if continuous
        params = MountainCarEnvParams(; goal_pos = 0.45, power = 0.0015, T = T, kwargs...)
    else
        params = MountainCarEnvParams(; T = T, kwargs...)
    end
    action_space = continuous ? ClosedInterval{T}(-1.0, 1.0) : Base.OneTo(3)
    env = MountainCarEnv(
        params,
        action_space,
        Space([params.min_pos..params.max_pos, -params.max_speed..params.max_speed]),
        zeros(T, 2),
        rand(action_space),
        false,
        0,
        rng,
    )
    reset!(env)
    env
end

ContinuousMountainCarEnv(; kwargs...) = MountainCarEnv(; continuous = true, kwargs...)

Random.seed!(env::MountainCarEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::MountainCarEnv) = env.action_space
RLBase.state_space(env::MountainCarEnv) = env.observation_space
RLBase.reward(env::MountainCarEnv{A,T}) where {A,T} = env.done ? zero(T) : -one(T)
RLBase.is_terminated(env::MountainCarEnv) = env.done
RLBase.state(env::MountainCarEnv) = env.state

function RLBase.reset!(env::MountainCarEnv{A,T}) where {A,T}
    env.state[1] = 0.2 * rand(env.rng, T) - 0.6
    env.state[2] = 0.0
    env.done = false
    env.t = 0
    nothing
end

function (env::MountainCarEnv{<:ClosedInterval})(a::AbstractFloat)
    @assert a in env.action_space
    env.action = a
    _step!(env, a)
end

function (env::MountainCarEnv{<:Base.OneTo{Int}})(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a - 2)
end

function _step!(env::MountainCarEnv, force)
    env.t += 1
    x, v = env.state
    v += force * env.params.power + cos(3 * x) * (-env.params.gravity)
    v = clamp(v, -env.params.max_speed, env.params.max_speed)
    x += v
    x = clamp(x, env.params.min_pos, env.params.max_pos)
    if x == env.params.min_pos && v < 0
        v = 0
    end
    env.done =
        x >= env.params.goal_pos && v >= env.params.goal_velocity ||
        env.t >= env.params.max_steps
    env.state[1] = x
    env.state[2] = v
    nothing
end

# adapted from https://github.com/JuliaML/Reinforce.jl/blob/master/src/envs/mountain_car.jl
height(xs) = sin(3 * xs) * 0.45 + 0.55
rotate(xs, ys, θ) = xs * cos(θ) - ys * sin(θ), ys * cos(θ) + xs * sin(θ)
translate(xs, ys, t) = xs .+ t[1], ys .+ t[2]

function GR.plot(env::MountainCarEnv)
    s = env.state
    d = env.done
    clearws()
    setviewport(0, 1, 0, 1)
    setwindow(
        env.params.min_pos - 0.1,
        env.params.max_pos + 0.2,
        -.1,
        height(env.params.max_pos) + 0.2,
    )
    xs = LinRange(env.params.min_pos, env.params.max_pos, 100)
    ys = height.(xs)
    polyline(xs, ys)
    x = s[1]
    θ = cos(3 * x)
    carwidth = 0.05
    carheight = carwidth / 2
    clearance = 0.2 * carheight
    xs = [-carwidth / 2, -carwidth / 2, carwidth / 2, carwidth / 2]
    ys = [0, carheight, carheight, 0]
    ys .+= clearance
    xs, ys = rotate(xs, ys, θ)
    xs, ys = translate(xs, ys, [x, height(x)])
    fillarea(xs, ys)
    plotendofepisode(env.params.max_pos + 0.1, 0, d)
    updatews()
end
