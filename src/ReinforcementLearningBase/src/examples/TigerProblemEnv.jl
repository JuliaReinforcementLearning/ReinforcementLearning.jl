export TigerProblemEnv

const REWARD_OF_LISTEN = -1
const REWARD_OF_TIGER = -100
const REWARD_OF_TREASURE = 10

"""
    TigerProblemEnv(;rng=Random>GLOBAL_RNG)

Here we use the [The Tiger
Proglem](https://cw.fel.cvut.cz/old/_media/courses/a4m36pah/pah-pomdp-tiger.pdf)
to demonstrate how to write a [POMDP](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) problem.
"""
Base.@kwdef mutable struct TigerProblemEnv <: AbstractEnv
    obs_prob::Float64 = 0.85
    action::Union{Nothing,Symbol} = nothing
    rng::AbstractRNG = Random.GLOBAL_RNG
    tiger_pos::Int = rand(rng, 1:2)
end

Random.seed!(env::TigerProblemEnv, s) = seed!(env.rng, s)

action_space(::TigerProblemEnv) = (:listen, :open_left, :open_right)

(env::TigerProblemEnv)(action) = env.action = action

function reward(env::TigerProblemEnv)
    if env.action == :listen
        REWARD_OF_LISTEN
    elseif (env.action == :open_left && env.tiger_pos == 1) ||
           (env.action == :open_right && env.tiger_pos == 2)
        REWARD_OF_TIGER
    else
        REWARD_OF_TREASURE
    end
end

is_terminated(env::TigerProblemEnv) = !isnothing(env.action) && env.action != :listen

function reset!(env::TigerProblemEnv)
    env.tiger_pos = rand(env.rng, 1:2)
    env.action = nothing
end

"""
The main difference compared to other environments is that, now we have two
kinds of *states*. The **observation** and the **internal state**. By default we
return the **observation**.
"""
state(env::TigerProblemEnv) = state(env, Observation{Int}())

function state(env::TigerProblemEnv, ::Observation)
    if isnothing(env.action)
        # game not started yet
        1
    elseif env.action == :listen
        if rand(env.rng) < env.obs_prob
            env.tiger_pos + 1 # shift by 1
        else
            2 - env.tiger_pos + 1 # shift by 1
        end
    else
        4 # terminal state, a dummy state
    end
end

state(env::TigerProblemEnv, ::InternalState) = env.tiger_pos

state_space(env::TigerProblemEnv) = state_space(env, Observation{Int}())
state_space(env::TigerProblemEnv, ::Observation) = 1:4
state_space(env::TigerProblemEnv, ::InternalState) = 1:2

NumAgentStyle(::TigerProblemEnv) = SINGLE_AGENT
DynamicStyle(::TigerProblemEnv) = SEQUENTIAL
ActionStyle(::TigerProblemEnv) = MINIMAL_ACTION_SET
InformationStyle(::TigerProblemEnv) = IMPERFECT_INFORMATION
StateStyle(::TigerProblemEnv) = (Observation{Int}(), InternalState{Int}())
RewardStyle(::TigerProblemEnv) = STEP_REWARD
UtilityStyle(::TigerProblemEnv) = GENERAL_SUM
ChanceStyle(::TigerProblemEnv) = STOCHASTIC
