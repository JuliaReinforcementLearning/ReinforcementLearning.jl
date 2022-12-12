export RockPaperScissorsEnv

"""
    RockPaperScissorsEnv()

[Rock Paper Scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors) is a
simultaneous, zero sum game.
"""
Base.@kwdef mutable struct RockPaperScissorsEnv <: AbstractEnv
    reward::Tuple{Int,Int} = (0, 0)
    is_done::Bool = false
end

RLBase.players(::RockPaperScissorsEnv) = (1, 2)

"""
Note that although this is a two player game. The current player is always a
dummy simultaneous player.
"""
RLBase.current_player(::RockPaperScissorsEnv) = SIMULTANEOUS_PLAYER

# Defining the `action_space` of each independent player can help to transform
# this SIMULTANEOUS environment into a SEQUENTIAL environment with
# [`simultaneous2sequential`](@ref).
RLBase.action_space(::RockPaperScissorsEnv, ::Int) = ('💎', '📃', '✂')

RLBase.action_space(::RockPaperScissorsEnv, ::SimultaneousPlayer) =
    Tuple((i, j) for i in ('💎', '📃', '✂') for j in ('💎', '📃', '✂'))

RLBase.action_space(env::RockPaperScissorsEnv) = action_space(env, SIMULTANEOUS_PLAYER)

RLBase.legal_action_space(env::RockPaperScissorsEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

"Since it's a one-shot game, the state space doesn't have much meaning."
RLBase.state_space(::RockPaperScissorsEnv, ::Observation, p) = Base.OneTo(1)

"""
For multi-agent environments, we usually implement the most detailed one.
"""
RLBase.state(::RockPaperScissorsEnv, ::Observation, p) = 1

RLBase.reward(env::RockPaperScissorsEnv) = env.is_done ? env.reward : (0, 0)
RLBase.reward(env::RockPaperScissorsEnv, p::Int) = reward(env)[p]

RLBase.is_terminated(env::RockPaperScissorsEnv) = env.is_done
RLBase.reset!(env::RockPaperScissorsEnv) = env.is_done = false

function (env::RockPaperScissorsEnv)((x, y))
    if x == y
        env.reward = (0, 0)
    elseif x == '💎' && y == '✂' || x == '✂' && y == '📃' || x == '📃' && y == '💎'
        env.reward = (1, -1)
    else
        env.reward = (-1, 1)
    end
    env.is_done = true
end

RLBase.NumAgentStyle(::RockPaperScissorsEnv) = MultiAgent(2)
RLBase.DynamicStyle(::RockPaperScissorsEnv) = SIMULTANEOUS
RLBase.ActionStyle(::RockPaperScissorsEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::RockPaperScissorsEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::RockPaperScissorsEnv) = Observation{Int}()
RLBase.RewardStyle(::RockPaperScissorsEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::RockPaperScissorsEnv) = ZERO_SUM
RLBase.ChanceStyle(::RockPaperScissorsEnv) = DETERMINISTIC
