export GraphShortestPathEnv

using Random
using SparseArrays
using LinearAlgebra


mutable struct GraphShortestPathEnv{G, R} <: AbstractEnv
    graph::G
    pos::Int
    goal::Int
    max_steps::Int
    rng::R
    reward::Int
    step::Int
end

"""
    GraphShortestPathEnv([rng]; n=10, sparsity=0.1, max_steps=10)

Quoted **A.3** in the the paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).

> We give details of the illustrative example discussed in the introduction.
> The task is to find theshortest path on a fixed directed graph, which can be
> formulated as an MDP where reward is0whenthe agent is at the goal node
> and−1otherwise.  The observation is the integer index of the graphnode the
> agent is in. The action is the integer index of the graph node to move to
> next. The transitiondynamics transport the agent to the action’s node index if
> there is an edge in the graph, while theagent remains at the past node
> otherwise. The returns-to-go in this problem correspond to negativepath
> lengths and maximizing them corresponds to generating shortest paths.

"""
function GraphShortestPathEnv(rng=Random.GLOBAL_RNG; n=20, sparsity=0.1, max_steps=10)
    graph = sprand(rng, Bool, n, n, sparsity) .| I(n)

    goal = rand(rng, 1:n)
    pos = rand(rng, 1:n)
    while pos == goal
        pos = rand(rng, 1:n)
    end
    GraphShortestPathEnv(graph, pos, goal, max_steps, rng, 0, 0)
end

function (env::GraphShortestPathEnv)(action)
    env.step += 1
    if env.graph[action, env.pos]
        env.pos = action
    end
    env.reward = env.pos == env.goal ? 0 : -1
end

RLBase.state(env::GraphShortestPathEnv) = env.pos
RLBase.state_space(env::GraphShortestPathEnv) = axes(env.graph, 2)
RLBase.action_space(env::GraphShortestPathEnv) = axes(env.graph, 2)
RLBase.legal_action_space(env::GraphShortestPathEnv) = (env.graph[:, env.pos]).nzind
RLBase.reward(env::GraphShortestPathEnv) = env.reward
RLBase.is_terminated(env::GraphShortestPathEnv) = env.pos == env.goal || env.step >= env.max_steps

function RLBase.reset!(env::GraphShortestPathEnv)
    env.step = 0
    env.reward = 0
    env.goal = rand(env.rng, state_space(env))
    env.pos = rand(env.rng, state_space(env))
    while env.pos == env.goal
        env.pos = rand(env.rng, state_space(env))
    end
end

Random.seed!(env::GraphShortestPathEnv, seed) = Random.seed!(env.rng, seed)

function floyd_shortest_path(env::GraphShortestPathEnv)
    n = size(env.graph, 1)
    M = fill(Inf, n, n)
    for idx in CartesianIndices(M)
        if idx[1] == idx[2]
            M[idx] = 0
        elseif env.graph[idx]
            M[idx] = 1
        end
    end
    for k in 1:n
        for i in 1:n
            for j in 1:n
                if M[i, j] > M[i, k] + M[k, j]
                    M[i, j] = M[i, k] + M[k, j]
                end
            end
        end
    end
    M
end

#=

using ReinforcementLearning
using Random

rng = MersenneTwister(123)

env = GraphShortestPathEnv(rng)
policy = RandomPolicy(rng=rng)
M = RLEnvs.floyd_shortest_path(env)


Base.@kwdef struct ShortestPathCount <: AbstractHook
    shortest_paths::Vector{Float64} = []
end

(h::ShortestPathCount)(::PreEpisodeStage, policy, env) = push!(h.shortest_paths, M[env.goal, env.pos])

h = run(policy, env, StopAfterEpisode(1_000), ComposedHook(StepsPerEpisode(), ShortestPathCount()))

using UnicodePlots

barplot(x, [sum(h[2].shortest_paths .== i) for i in x])  # shortest path

#       ┌                                        ┐ 
#   1.0 ┤■■■■■■■■■■■ 113                           
#   2.0 ┤■■■■■■■■■■■■■ 134                         
#   3.0 ┤■■■■■■■■■■■■■■ 147                        
#   4.0 ┤■■■■■■■■■■■ 111                           
#   5.0 ┤■■■■■■■ 76                                
#   6.0 ┤■■■■ 40                                   
#   7.0 ┤■■ 19                                     
#   8.0 ┤ 3                                        
#   9.0 ┤ 0                                        
#   Inf ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 357   
#       └                                        ┘ 
# 

barplot(1:10, [sum(h[1].steps .== i) for i in 1:10])  # random walk

#      ┌                                        ┐ 
#    1 ┤■■ 39                                     
#    2 ┤■■ 38                                     
#    3 ┤■ 17                                      
#    4 ┤■ 23                                      
#    5 ┤■ 28                                      
#    6 ┤■ 24                                      
#    7 ┤■ 25                                      
#    8 ┤■ 21                                      
#    9 ┤■ 16                                      
#   10 ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 769   
#      └                                        ┘ 
# 
=#