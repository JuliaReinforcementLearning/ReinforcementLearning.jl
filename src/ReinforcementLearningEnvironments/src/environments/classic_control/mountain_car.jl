@reexport module MountainCar
using Random
using ..ReinforcementLearningEnvironments
const RLEnv = ReinforcementLearningEnvironments
export MountainCarEnv

struct MountainCarEnvParams{T}
    min_pos::T
    max_pos::T
    max_speed::T
    goal_pos::T
    max_steps::Int64
end

mutable struct MountainCarEnv{T, R<:AbstractRNG} <: AbstractEnv
    params::MountainCarEnvParams{T}
    action_space::DiscreteSpace
    observation_space::MultiContinuousSpace{(2,), 1}
    state::Array{T, 1}
    action::Int64
    done::Bool
    t::Int64
    rng::R
end

function MountainCarEnv(; T = Float64, min_pos = T(-1.2), max_pos = T(.6),
                       max_speed = T(.07), goal_pos = T(.5), max_steps = 200)
    env = MountainCarEnv(MountainCarEnvParams(min_pos, max_pos, max_speed, goal_pos, max_steps),
                      DiscreteSpace(3),
                      MultiContinuousSpace([min_pos, -max_speed], [max_pos, max_speed]),
                      zeros(T, 2),
                      1,
                      false,
                      0,
                      Random.GLOBAL_RNG)
    reset!(env)
    env
end

RLEnv.action_space(env::MountainCarEnv) = env.action_space
RLEnv.observation_space(env::MountainCarEnv) = env.observation_space
RLEnv.observe(env::MountainCarEnv) = (observation=env.state, isdone=env.done)

function RLEnv.reset!(env::MountainCarEnv{T}) where T
    env.state[1] = .2 * rand(env.rng, T) - .6
    env.state[2] = 0.
    env.done = false
    env.t = 0
    nothing
end

function RLEnv.interact!(env::MountainCarEnv, a)
    env.t += 1
    x, v = env.state
    v += (a - 2)*0.001 + cos(3*x)*(-0.0025)
    v = clamp(v, -env.params.max_speed, env.params.max_speed)
    x += v
    x = clamp(x, env.params.min_pos, env.params.max_pos)
    if x == env.params.min_pos && v < 0 v = 0 end
    env.done = x >= env.params.goal_pos || env.t >= env.params.max_steps
    env.state[1] = x
    env.state[2] = v
    (observation=env.state, reward=-1., isdone=env.done)
end

# adapted from https://github.com/JuliaML/Reinforce.jl/blob/master/src/envs/mountain_car.jl
height(xs) = sin(3 * xs)*0.45 + 0.55
rotate(xs, ys, θ) = xs*cos(θ) - ys*sin(θ), ys*cos(θ) + xs*sin(θ)
translate(xs, ys, t) = xs .+ t[1], ys .+ t[2]
function RLEnv.render(env::MountainCarEnv)
    s = env.state
    d = env.done
    clearws()
    setviewport(0, 1, 0, 1)
    setwindow(env.params.min_pos - .1, env.params.max_pos + .2, -.1,
              height(env.params.max_pos) + .2)
    xs = LinRange(env.params.min_pos, env.params.max_pos, 100)
    ys = height.(xs)
    polyline(xs, ys)
    x = s[1]
    θ = cos(3*x)
    carwidth = 0.05
    carheight = carwidth/2
    clearance = .2*carheight
    xs = [-carwidth/2, -carwidth/2, carwidth/2, carwidth/2]
    ys = [0, carheight, carheight, 0]
    ys .+= clearance
    xs, ys = rotate(xs, ys, θ)
    xs, ys = translate(xs, ys, [x, height(x)])
    fillarea(xs, ys)
    plotendofepisode(env.params.max_pos + .1, 0, d) 
    updatews()
end
end