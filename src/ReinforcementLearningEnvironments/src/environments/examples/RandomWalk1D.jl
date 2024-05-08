export RandomWalk1D

import CommonRLInterface

"""
    RandomWalk1D(;rewards=-1. => 1.0, N=7, start_pos=(N+1) ÷ 2, actions=[-1,1])

An agent is placed at the `start_pos` and can move left or right (stride is
defined in actions). The game terminates when the agent reaches either end and
receives a reward correspondingly.

Compared to the [`MultiArmBanditsEnv`](@ref):

1. The state space is more complicated (well, not that complicated though).
1. It's a sequential game of multiple action steps.
1. It's a deterministic game instead of stochastic game.
"""
Base.@kwdef mutable struct RandomWalk1D <: AbstractEnv
    rewards::Pair{Float64,Float64} = -1.0 => 1.0
    N::Int = 7
    actions::Vector{Int} = [-1, 1]
    start_pos::Int = (N + 1) ÷ 2
    pos::Int = start_pos

    action_space::Base.OneTo = Base.OneTo(length(actions))
    state_space::Base.OneTo = Base.OneTo(N)
end

RLBase.action_space(env::RandomWalk1D) = env.action_space

function RLBase.act!(env::RandomWalk1D, action::Int)
    env.pos += env.actions[action]
    if env.pos > env.N
        env.pos = env.N
    elseif env.pos < 1
        env.pos = 1
    end
    return
end

RLBase.state(env::RandomWalk1D, ::Observation, ::DefaultPlayer) = env.pos
RLBase.state_space(env::RandomWalk1D) = env.state_space
RLBase.is_terminated(env::RandomWalk1D) = env.pos == 1 || env.pos == env.N
RLBase.reset!(env::RandomWalk1D) = env.pos = env.start_pos

RLBase.reward(env::RandomWalk1D) = random_walk_reward(env.pos, env.rewards, env.N)

function random_walk_reward(pos::Int, rewards::Pair{Float64,Float64}, N::Int)
    if pos == 1
        return random_walk_reward_first(rewards)
    elseif pos == N
        return random_walk_reward_last(rewards)
    else
        return 0.0
    end
end

random_walk_reward_first(rewards::Pair{Float64,Float64}) = first(rewards)
random_walk_reward_last(rewards::Pair{Float64,Float64}) = last(rewards)

RLBase.NumAgentStyle(::RandomWalk1D) = SINGLE_AGENT
RLBase.DynamicStyle(::RandomWalk1D) = SEQUENTIAL
RLBase.ActionStyle(::RandomWalk1D) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::RandomWalk1D) = PERFECT_INFORMATION
RLBase.StateStyle(::RandomWalk1D) = Observation{Int}()
RLBase.RewardStyle(::RandomWalk1D) = TERMINAL_REWARD
RLBase.UtilityStyle(::RandomWalk1D) = GENERAL_SUM
RLBase.ChanceStyle(::RandomWalk1D) = DETERMINISTIC
