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

players(::RockPaperScissorsEnv) = (1, 2)

"""
Note that although this is a two player game. The current player is always a
dummy simultaneous player.
"""
current_player(::RockPaperScissorsEnv) = SIMULTANEOUS_PLAYER

action_space(::RockPaperScissorsEnv, ::Int) = ('💎', '📃', '✂')
action_space(::RockPaperScissorsEnv, ::SimultaneousPlayer) =
    Tuple((i, j) for i in ('💎', '📃', '✂') for j in ('💎', '📃', '✂'))
action_space(env::RockPaperScissorsEnv) = action_space(env, SIMULTANEOUS_PLAYER)

legal_action_space(env::RockPaperScissorsEnv, p) =
    is_terminated(env) ? () : action_space(env, p)

"Since it's a one-shot game, the state space doesn't have much meaning."
state_space(::RockPaperScissorsEnv, ::Observation, p) = Base.OneTo(1)

"""
For multi-agent environments, we usually implement the most detailed one.
"""
state(::RockPaperScissorsEnv, ::Observation, p) = 1

reward(env::RockPaperScissorsEnv) = env.is_done ? env.reward : (0, 0)
reward(env::RockPaperScissorsEnv, p::Int) = reward(env)[p]

is_terminated(env::RockPaperScissorsEnv) = env.is_done
reset!(env::RockPaperScissorsEnv) = env.is_done = false

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

NumAgentStyle(::RockPaperScissorsEnv) = MultiAgent(2)
DynamicStyle(::RockPaperScissorsEnv) = SIMULTANEOUS
ActionStyle(::RockPaperScissorsEnv) = MINIMAL_ACTION_SET
InformationStyle(::RockPaperScissorsEnv) = IMPERFECT_INFORMATION
StateStyle(::RockPaperScissorsEnv) = Observation{Int}()
RewardStyle(::RockPaperScissorsEnv) = TERMINAL_REWARD
UtilityStyle(::RockPaperScissorsEnv) = ZERO_SUM
ChanceStyle(::RockPaperScissorsEnv) = DETERMINISTIC
