@doc """
[ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl) (**RLBase**)
provides some common constants, traits, abstractions and interfaces in developing reinforcement learning algorithms in 
Julia. From the concept level, they can be organized in the following parts:

- [Policy](@ref)
- [EnvironmentModel](@ref)
- [Environment](@ref)
  - [Traits for Environment](@ref)
  - [Observation of Environment](@ref)
  - [Observation Space and Action Space](@ref)
- [ EnvironmentModel](@ref)

""" RLBase

using Random

#####
# Policy
#####

"""
    (π::AbstractPolicy)(obs) -> action

Policy is the most basic concept in reinforcement learning. A policy is a functional object which takes in an observation and generate an action.
"""
@interface abstract type AbstractPolicy end
@interface (π::AbstractPolicy)(obs)


"""
    update!(π::AbstractPolicy, experience)

Update the policy `π` with online/offline experience.
"""
@interface update!(π::AbstractPolicy, experience) = nothing

"""
    get_prob(π::AbstractPolicy, obs)

Get the probability distribution of actions based on policy `π` given an observation `obs`. 
"""
@interface get_prob(π::AbstractPolicy, obs)

"""
    get_prob(π::AbstractPolicy, obs, action)

Only valid for environments with discrete action space.
"""
@interface get_prob(π::AbstractPolicy, obs, action)

"""
    get_priority(π::AbstractPolicy, experience)

Usually used in offline policies.
"""
@interface get_priority(π::AbstractPolicy, experience)

#####
# Environment
#####

"""
    (env::AbstractEnv)(action) = env(get_current_player(env), action) -> nothing
    (env::AbstractEnv)(player, action) -> nothing

Super type of all reinforcement learning environments.
"""
@interface abstract type AbstractEnv end

@interface (env::AbstractEnv)(action) = env(get_current_player(env), action)
@interface (env::AbstractEnv)(player, action)

#####
## Traits for Environment
## mostly borrowed from https://github.com/deepmind/open_spiel/blob/master/open_spiel/spiel.h
#####

#####
### DynamicStyle
#####

abstract type AbstractDynamicStyle end

@interface struct Sequential <: AbstractDynamicStyle end
@interface struct Simultaneous <: AbstractDynamicStyle end

"Environment with the [`DynamicStyle`](@ref) of `SEQUENTIAL` must takes actions from different players one-by-one."
@interface const SEQUENTIAL = Sequential()

"Environment with the [`DynamicStyle`](@ref) of `SIMULTANEOUS` must take in actions from some (or all) players at one time"
@interface const SIMULTANEOUS = Simultaneous()

"""
    DynamicStyle(env::AbstractEnv) = SEQUENTIAL

Determine whether the players can play simultaneous or not. Default value is [`SEQUENTIAL`](@ref)
"""
@interface DynamicStyle(env::AbstractEnv) = SEQUENTIAL

#####
### ChanceStyle
#####

abstract type AbstractChanceStyle end

@interface struct Deterministic <: AbstractChanceStyle end
@interface struct Stochastic <: AbstractChanceStyle end

"Observations are solely determined by action"
@interface const DETERMINISTIC = Deterministic()

"Observations are stochastic given by the same action"
@interface const STOCHASTIC = Stochastic()

"Either [`DETERMINISTIC`](@ref) or [`STOCHASTIC`](@ref)."
@interface ChanceStyle(env::AbstractEnv)

#####
### InformationStyle
#####

abstract type AbstractInformationStyle end

@interface struct PerfectInformation <: AbstractInformationStyle end
@interface struct ImperfectInformation <: AbstractInformationStyle end

"All players observe the same state"
@interface const PERFECT_INFORMATION = PerfectInformation()

"The inner state of some players' observations may be different"
@interface const IMPERFECT_INFORMATION = ImperfectInformation()

@interface InformationStyle(env::AbstractEnv)

#####
### RewardStyle
#####

abstract type AbstractRewardStyle end

@interface struct StepReward <: AbstractRewardStyle end
@interface struct TerminalReward <: AbstractRewardStyle end

"We can get reward after each step"
@interface const STEP_REWARD = StepReward()

"Only get reward at the end of environment"
@interface const TERMINAL_REWARD = TerminalReward()

#####
### UtilityStyle
#####

abstract type AbstractUtilityStyle end

@interface struct ZeroSum <: AbstractUtilityStyle end
@interface struct ConstantSum <: AbstractUtilityStyle end
@interface struct GeneralSum <: AbstractUtilityStyle end
@interface struct IdenticalUtility <: AbstractUtilityStyle end

"Rewards of all players sum to 0"
@interface const ZERO_SUM = ZeroSum()

"Rewards of all players sum to a constant"
@interface const CONSTANT_SUM = ConstantSum()

"Total rewards of all players may be different in each step"
@interface const GENERAL_SUM = GeneralSum()

"Every player gets the same reward"
@interface const IDENTICAL_REWARD = IdenticalUtility()

@interface UtilityStyle(env::AbstractEnv)

#####
### ActionStyle
#####

abstract type AbstractActionStyle end
@interface struct FullActionSet <: AbstractActionStyle end

"The action space of the environment may contains illegal actions"
@interface const FULL_ACTION_SET = FullActionSet()

@interface struct MinimalActionSet <: AbstractActionStyle end

"All actions in the action space of the environment are legal"
@interface const MINIMAL_ACTION_SET = MinimalActionSet()

"""
    ActionStyle(env::AbstractEnv)
    ActionStyle(obs)

Specify whether the observation contains a full action set or a minimal action set.
By default the [`MINIMAL_ACTION_SET`](@ref) is returned.
"""
@interface ActionStyle(obs) = MINIMAL_ACTION_SET

ActionStyle(obs::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) = FULL_ACTION_SET
ActionStyle(obs::NamedTuple{(:reward, :terminal, :state, :legal_actions_mask)}) =
    FULL_ACTION_SET
ActionStyle(
    obs::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)},
) = FULL_ACTION_SET

#####
## general
#####

"""
    get_action_space(env::AbstractEnv) -> AbstractSpace
"""
@interface get_action_space(env::AbstractEnv) = env.action_space

"""
    get_observation_space(env::AbstractEnv) -> AbstractSpace
"""
@interface get_observation_space(env::AbstractEnv) = env.observation_space

@interface get_current_player(env::AbstractEnv)

"""
    get_player_id(player, env::AbstractEnv) -> Int

Get the index of current player. Result should be an Int and starts from 1.
Usually used in multi-agent environments.
"""
@interface get_player_id(player, env::AbstractEnv)

"""
    get_num_players(env::AbstractEnv) -> Int
"""
@interface get_num_players(env::AbstractEnv) = 1

"Show the environment in a user-friendly manner"
@interface render(env::AbstractEnv)

"Reset the internal state of an environment"
@interface reset!(env::AbstractEnv)

@interface Random.seed!(env::AbstractEnv, seed)

@interface Base.copy(env::AbstractEnv)

"Get all actions in each ply"
@interface get_history(env::AbstractEnv)

#####
## Observation
#####

"""
    observe(env::AbstractEnv) = observe(env, get_current_player(env))
    observe(::AbstractEnv, player)

Get an observation of the `env` from the perspective of an `player`.

!!! note
    This is a very deliberate decision to adopt the duck-typing here
    to describe an observation from an environment.
    By default, we assume an observation is a NamedTuple,
    which is the most common case.
    But of course it can be of any type, as long as it implemented 
    the necessay methods described in this section.
"""
@interface observe(env::AbstractEnv) = observe(env, get_current_player(env))
@interface observe(::AbstractEnv, player)

"""
    get_legal_actions_mask(obs) -> Bool[]

Only valid for observations of [`FULL_ACTION_SET`](@ref).
"""
@interface get_legal_actions_mask(obs) = obs.legal_actions_mask

"""
    get_legal_actions(obs)

Only valid for observations of [`FULL_ACTION_SET`](@ref).
"""
@interface get_legal_actions(obs) = findall(get_legal_actions_mask(obs))

get_legal_actions(
    obs::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)},
) = obs.legal_actions
get_legal_actions(obs::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) =
    obs.legal_actions

"""
    get_terminal(obs) -> bool
"""
@interface get_terminal(obs) = obs.terminal

"""
    get_reward(obs) -> Number
"""
@interface get_reward(obs) = obs.reward

"""
    get_state(obs) -> Array
"""
@interface get_state(obs) = obs.state

#####
## Space
#####

"""
Describe the span of observations and actions.
Usually the following methods are implemented:

- `Base.length`
- `Base.in`
- `Random.rand`
- `Base.eltype`
"""
@interface abstract type AbstractSpace end

@interface Base.length(::AbstractSpace)
@interface Base.in(x, s::AbstractSpace)
@interface Random.rand(rng::AbstractRNG, s::AbstractSpace)
@interface Base.eltype(s::AbstractSpace)

#####
# EnvironmentModel
#####

"""
Describe how to model a reinforcement learning environment.
TODO: need more investigation
Ref: https://bair.berkeley.edu/blog/2019/12/12/mbpo/
- Analytic gradient computation
- Sampling-based planning
- Model-based data generation
- Value-equivalence prediction
"""
@interface abstract type AbstractEnvironmentModel end
