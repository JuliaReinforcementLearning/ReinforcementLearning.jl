module DiscreteMaze
using Random, StatsBase, SparseArrays, GR, ..ReinforcementLearningEnvironments
export DiscreteMazeEnv

function emptymaze(dimx, dimy)
    maze = ones(Int, dimx, dimy)
    maze[1,:] .= maze[end,:] .= maze[:, 1] .= maze[:, end] .= 0 # borders
    return maze
end
iswall(maze, pos) = maze[pos] == 0
isinsideframe(maze, i::Int) = isinsideframe(maze, CartesianIndices(maze)[i])
isinsideframe(maze, i) = i[1] > 1 && i[2] > 1 && i[1] < size(maze, 1) && i[2] < size(maze, 2)

const UP = CartesianIndex(0, -1)
const DOWN = CartesianIndex(0, 1)
const LEFT = CartesianIndex(-1, 0)
const RIGHT = CartesianIndex(1, 0)
function orthogonal_directions(dir)
    dir[1] == 0 && return (LEFT, RIGHT)
    return (UP, DOWN)
end

function is_wall_neighbour(maze, pos)
    for dir in (UP, DOWN, LEFT, RIGHT, UP + RIGHT, UP + LEFT, DOWN + RIGHT, DOWN + LEFT)
        iswall(maze, pos + dir) && return true
    end
    return false
end
function is_wall_tangential(maze, pos, dir)
    for ortho_dir in orthogonal_directions(dir)
        iswall(maze, pos + ortho_dir) && return true
    end
    return false
end
is_wall_ahead(maze, pos, dir) = iswall(maze, pos + dir)

function addrandomwall!(maze; rng = Random.GLOBAL_RNG)
    potential_startpos = filter(x -> !is_wall_neighbour(maze, x),
                                findall(x -> x != 0, maze))
    if potential_startpos == []
        @warn("Cannot add a random wall.")
        return maze
    end
    pos = rand(rng, potential_startpos)
    direction = rand(rng, (UP, DOWN, LEFT, RIGHT))
    while true
        maze[pos] = 0
        pos += direction
        is_wall_tangential(maze, pos, direction) && break
        if is_wall_ahead(maze, pos, direction)
            maze[pos] = 0
            break
        end
    end
    return maze
end

function n_effective(n, f, list)
    N = n === nothing ? div(length(list), Int(1/f)) : n
    min(N, length(list))
end
function breaksomewalls!(m; f = 1/50, n = nothing, rng = Random.GLOBAL_RNG)
    wallpos = Int[]
    for i in 1:length(m)
        iswall(m, i) && isinsideframe(m, i) && push!(wallpos, i)
    end
    pos = sample(rng, wallpos, n_effective(n, f, wallpos), replace = false)
    m[pos] .= 1
    m
end
function addobstacles!(m; f = 1/100, n = nothing, rng = Random.GLOBAL_RNG)
    nz = findall(x -> x == 1, reshape(m, :))
    pos = sample(rng, nz, n_effective(n, f, nz), replace = false)
    m[pos] .= 0
    m
end
function setTandR!(d)
    for s in LinearIndices(d.maze)[findall(x -> x != 0, d.maze)]
        setTandR!(d, s)
    end
end
function setTandR!(d, s)
    T = d.mdp.trans_probs
    R = d.mdp.reward
    goals = d.goals
    ns = length(d.mdp.observation_space)
    maze = d.maze
    if s in goals
        idx_goals = findfirst(x -> x == s, goals)
        R.value[s] = d.goalrewards[idx_goals]
    end
    pos = CartesianIndices(maze)[s]
    for (aind, a) in enumerate((UP, DOWN, LEFT, RIGHT))
        nextpos = maze[pos + a] == 0 ? pos : pos + a
        if d.neighbourstateweight > 0
            positions = [nextpos]
            weights = [1.]
            for dir in (UP, DOWN, LEFT, RIGHT)
                if maze[nextpos + dir] != 0
                    push!(positions, nextpos + dir)
                    push!(weights, d.neighbourstateweight)
                end
            end
            states = LinearIndices(maze)[positions]
            weights /= sum(weights)
            T[aind, s] = sparsevec(states, weights, ns)
        else
            nexts = LinearIndices(maze)[nextpos]
            T[aind, s] = sparsevec([nexts], [1.], ns)
        end
    end
end

"""
    struct DiscreteMazeEnv
        mdp::MDP
        maze::Array{Int, 2}
        goals::Array{Int, 1}
        statefrommaze::Array{Int, 1}
        mazefromstate::Array{Int, 1}
"""
struct DiscreteMazeEnv{T}
    mdp::T
    maze::Array{Int, 2}
    goals::Array{Int, 1}
    goalrewards::Array{Float64, 1}
    neighbourstateweight::Float64
end
"""
    DiscreteMazeEnv(; nx = 40, ny = 40, nwalls = div(nx*ny, 20), ngoals = 1,
                      goalrewards = 1, stepcost = 0, stochastic = false,
                      neighbourstateweight = .05, rng = Random.GLOBAL_RNG)

Returns a `DiscreteMazeEnv` of width `nx` and height `ny` with `nwalls` walls and
`ngoals` goal locations with reward `goalreward` (a list of different rewards
for the different goal states or constant reward for all goals), cost of moving
`stepcost` (reward = -`stepcost`); if `stochastic = true` the actions lead with
a certain probability to a neighbouring state, where `neighbourstateweight`
controls this probability.
"""
function DiscreteMazeEnv(; nx = 40, ny = 40, nwalls = div(nx*ny, 20),
                        rng = Random.GLOBAL_RNG, kwargs...)
    m = emptymaze(nx, ny)
    for _ in 1:nwalls
        addrandomwall!(m, rng = rng)
    end
    breaksomewalls!(m, rng = rng)
    DiscreteMazeEnv(m; rng = rng, kwargs...)
end
function DiscreteMazeEnv(maze; ngoals = 1,
                            goalrewards = 1.,
                            stepcost = 0,
                            stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.,
                            rng = Random.GLOBAL_RNG)
    na = 4
    ns = length(maze)
    legalstates = LinearIndices(maze)[findall(x -> x != 0, maze)]
    T = Array{SparseVector{Float64,Int}}(undef, na, ns)
    goals = sort(sample(rng, legalstates, ngoals, replace = false))
    R = DeterministicNextStateReward(fill(-stepcost, ns))
    isterminal = zeros(Int, ns); isterminal[goals] .= 1
    isinitial = setdiff(legalstates, goals)
    res = DiscreteMazeEnv(SimpleMDPEnv(DiscreteSpace(ns, 1),
                                    DiscreteSpace(na, 1),
                                    rand(rng, legalstates),
                                    T, R,
                                    isinitial,
                                    isterminal,
                                    rng),
                       maze,
                       goals,
                       typeof(goalrewards) <: Number ? fill(goalrewards, ngoals) :
                                                       goalrewards,
                       neighbourstateweight)
    setTandR!(res)
    res
end

ReinforcementLearningEnvironments.interact!(env::DiscreteMazeEnv, a) = interact!(env.mdp, a)
ReinforcementLearningEnvironments.reset!(env::DiscreteMazeEnv) = reset!(env.mdp)
ReinforcementLearningEnvironments.observe(env::DiscreteMazeEnv) = observe(env.mdp)
ReinforcementLearningEnvironments.action_space(env::DiscreteMazeEnv) = action_space(env.mdp)
ReinforcementLearningEnvironments.observation_space(env::DiscreteMazeEnv) = observation_space(env.mdp)
function ReinforcementLearningEnvironments.render(env::DiscreteMazeEnv)
    goals = env.goals
    m = copy(env.maze)
    m[goals] .= 3
    m[env.mdp.state] = 2
    imshow(m, colormap = 21, size = (400, 400))
end
end
