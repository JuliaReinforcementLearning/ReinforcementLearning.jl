# Maze
module Maze
using ReinforcementLearning, StatsBase
function getemptymaze(dimx, dimy)
    maze = ones(Int64, dimx, dimy)
    maze[1,:] .= maze[end,:] .= 0
    maze[:, 1] .= maze[:, end] .= 0
    maze
end

function setwall!(maze, startpos, endpos)
    dimx, dimy = startpos - endpos
    if dimx == 0
        maze[startpos[1], startpos[2]:endpos[2]] .= 0
    else
        maze[startpos[1]:endpos[1], startpos[2]] .= 0
    end
end

function indto2d(maze, pos)
    dimx = size(maze, 1)
    [rem(pos, dimx), div(pos, dimx) + 1]
end
function posto1d(maze, pos)
    dimx = size(maze, 1)
    (pos[2] - 1) * dimx + pos[1]
end

function checkpos(maze, pos)
    count = 0
    for dx in -1:1
        for dy in -1:1
            count += maze[(pos + [dx, dy])...] == 0
        end
    end
    count
end

function addrandomwall!(maze)
    startpos = rand(find(maze))
    startpos = indto2d(maze, startpos)
    starttouch = checkpos(maze, startpos) 
    if starttouch > 0
        return 0
    end
    endx, endy = startpos
    if rand(0:1) == 0 # horizontal
        while checkpos(maze, [endx, startpos[2]]) == 0
            endx += 1
        end
        if maze[endx + 1, startpos[2]] == 1 &&
            maze[endx + 1, startpos[2] + 1] == 
            maze[endx + 1, startpos[2] - 1] == 0
            endx -= 1
        end
    else
        while checkpos(maze, [startpos[1], endy]) == 0
            endy += 1
        end
        if maze[startpos[1], endy + 1] == 1 &&
            maze[startpos[1] + 1, endy + 1] == 
            maze[startpos[1] - 1, endy + 1] == 0
            endx -= 1
        end
    end
    setwall!(maze, startpos, [endx, endy])
    return 1
end

function mazetomdp(maze, ngoalstates = 1, goalrewards = 0, stochastic = false,
                   neighbourstateweight = .05)
    stochastic && goalrewards != 0 && 
    error("Non-zero goalrewards are not implemented for stochastic mazes.")
    na = 4
    nzpos = find(maze)
    mapping = cumsum(maze[:])
    ns = length(nzpos)
    T = Array{SparseVector}(na, ns)
    goals = sort(sample(1:ns, ngoalstates, replace = false))
    R = -ones(na, ns)
    isterminal = zeros(Int64, ns); isterminal[goals] = 1
    isinitial = collect(1:ns); deleteat!(isinitial, goals)
    for s in 1:ns
        for (aind, a) in enumerate(([0, 1], [1, 0], [0, -1], [-1, 0]))
            pos = indto2d(maze, nzpos[s])
            nextpos = maze[(pos + a)...] == 0 ? pos : pos + a
            if stochastic
                positions = []
                push!(positions, nextpos)
                weights = [1.]
                for dir in ([0, 1], [1, 0], [0, -1], [-1, 0])
                    if maze[(nextpos + dir)...] != 0
                        push!(positions, nextpos + dir)
                        push!(weights, neighbourstateweight)
                    end
                end
                states = map(p -> mapping[posto1d(maze, p)], positions)
                weights /= sum(weights)
                T[aind, s] = SparseVector(ns, states, weights)
            else
                nexts = mapping[posto1d(maze, nextpos)]
                T[aind, s] = ReinforcementLearning.getprobvecdeterministic(ns,
                                                                       nexts,
                                                                       nexts)
                if nexts in goals
                    R[a, s] = goalrewards <: Number ? goalrewards : 
                                goalrewards[findfirst(x -> x == nexts, goals)]
                end
            end
        end
    end
    MDP(ns, na, rand(1:ns), T, R, isinitial, isterminal), goals, nzpos
end

function breaksomewalls(m; f = 1/50, 
                        n = div(length(find(1 - m[2:end-1, 2:end-1])), 1/f))
    nx, ny = size(m)
    zeros = find(1 - m)
    i = 1
    while i < n
        candidate = rand(zeros)
        if candidate > nx && candidate < nx * (ny - 1) &&
            candidate % nx != 0 && candidate % nx != 1
            m[candidate] = 1
            i += 1
        end
    end
end
    
function MazeMDP(; nx = 40, ny = 40, returnall = false, 
                   nwalls = div(nx*ny, 10), 
                   offset = 0., stochastic = false, ngoals = 1,
                   neighbourstateweight = .05, goalrewards = 0)
    m = getemptymaze(nx, ny)
    [addrandomwall!(m) for _ in 1:nwalls]
    breaksomewalls(m)
    mdp, goals, mapping = mazetomdp(m, ngoals, goalrewards,
                                    stochastic, neighbourstateweight)
    mdp.reward .+= offset
    if returnall
        m, mdp, goals, mapping
    else
        mdp
    end
end
export MazeMDP
end
using Maze

# this function requires
# using PyPlot, PyCall
# @pyimport matplotlib.colors as matcolors
function plotmazemdp(maze, goal, state, mapping; 
                     showvalues = false,
                     values = zeros(length(mapping)))
    maze[mapping[goal]] = 3
    maze[mapping[state]] = 2
    figure(figsize = (4, 4))
    cmap = matcolors.ListedColormap(["gray", "white", "blue", "red"], "A")
    if showvalues
        m = zeros(size(maze)...)
        m[mapping] .= values
        imshow(m, cmap = "Spectral_r")
    end
    imshow(maze, interpolation = "none", cmap = cmap, alpha = (1 - .5showvalues))
    plt[:tick_params](top="off", bottom="off",
                      labelbottom="off", labeltop="off",
                      labelleft="off", left="off")
    maze[mapping[goal]] = 1
    maze[mapping[state]] = 1
end

# this function requires
# using PlotlyJS
function plotmazemdp(maze, goals, state, mapping)
    m = deepcopy(maze)
    m[mapping[goals]] = 3
    m[mapping[state]] = 2
    data = heatmap(z = m, colorscale = [[0, "gray"], [1/3, "white"], 
                                        [2/3, "blue"], [1., "red"]], 
                  showscale = false)
    w, h = size(m)
    layout = Layout(autosize = false, width = 600, height = 600 * h/w)
    plot(data, layout)
end
function updatemazeplot(p, state, mapping)
    oldstate = findfirst(x -> x == 2, p.plot.data[1][:z])
    p.plot.data[1][:z][oldstate] = 1
    p.plot.data[1][:z][mapping[state]] = 2
    p
end


