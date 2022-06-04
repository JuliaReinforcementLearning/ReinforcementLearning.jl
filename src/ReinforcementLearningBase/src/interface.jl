@doc """
[ReinforcementLearningBase.jl](https://juliareinforcementlearning.org/docs/rlbase/)
(**RLBase**) provides some common constants, traits, abstractions and interfaces
in developing reinforcement learning algorithms in Julia. 

Basically, we defined the following two main concepts in reinforcement learning:

- [`AbstractPolicy`](@ref)
- [`AbstractEnv`](@ref)
""" RLBase

import Base: copy, copyto!, nameof
import Random: seed!, rand, AbstractRNG
import AbstractTrees: children, has_children
import Markdown

#####
# Policy
#####

"""
    (π::AbstractPolicy)(env) -> action

Policy is the most basic concept in reinforcement learning. Unlike the
definition in some other packages, here a policy is defined as a functional
object which takes in an environment and returns an action.

!!! note
    See discussions
    [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues/86)
    if you are wondering why we define the input as `AbstractEnv` instead of
    state.

!!! warning
    The policy `π` may change its internal state but it shouldn't change `env`.
    When it's really necessary, remember to make a copy of `env` to keep the
    original `env` untouched.
"""
@api abstract type AbstractPolicy end
@api (π::AbstractPolicy)(env)

"""
    update!(π::AbstractPolicy, experience)

Update the policy `π` with online/offline experience or parameters.
"""
@api update!(π::AbstractPolicy, experience)

"""
    prob(π::AbstractPolicy, env) -> Distribution

Get the probability distribution of actions based on policy `π` given an `env`.
"""
@api prob(π::AbstractPolicy, env)

"""
    prob(π::AbstractPolicy, env, action)

Only valid for environments with discrete actions.
"""
@api prob(π::AbstractPolicy, env, action)

"""
    priority(π::AbstractPolicy, experience)

Usually used in offline policies.
"""
@api priority(π::AbstractPolicy, experience)

#####
# Environment
#####

"""
    (env::AbstractEnv)(action, player=current_player(env))

Super type of all reinforcement learning environments.
"""
@api abstract type AbstractEnv end

abstract type AbstractEnvStyle end

#####
## Traits for Environment
## mostly borrowed from https://github.com/deepmind/open_spiel/blob/master/open_spiel/spiel.h
#####

#####
### NumAgentStyle
#####

abstract type AbstractNumAgentStyle <: AbstractEnvStyle end

@api struct SingleAgent <: AbstractNumAgentStyle end
@api const SINGLE_AGENT = SingleAgent()

@api struct MultiAgent{N} <: AbstractNumAgentStyle end

"""
    MultiAgent(n::Integer) -> MultiAgent{n}()

`n` must be ≥ 2.
"""
function MultiAgent(n::Integer)
    if n < 0
        throw(ArgumentError("number of agents must be > 1, get $n"))
    elseif n == 1
        throw(ArgumentError("do you mean `SINGLE_AGENT`?"))
    else
        MultiAgent{convert(Int, n)}()
    end
end

@api const TWO_AGENT = MultiAgent(2)

"""
    NumAgentStyle(env)

Number of agents involved in the `env`. Possible returns are:

- [`SINGLE_AGENT`](@ref). This is the default return.
- [`MultiAgent`][@ref].
"""
@env_api NumAgentStyle(env::T) where {T<:AbstractEnv} = NumAgentStyle(T)
NumAgentStyle(env::Type{<:AbstractEnv}) = SINGLE_AGENT

#####
### DynamicStyle
#####

abstract type AbstractDynamicStyle <: AbstractEnvStyle end

@api struct Sequential <: AbstractDynamicStyle end
@api struct Simultaneous <: AbstractDynamicStyle end

"Environment with the [`DynamicStyle`](@ref) of `SEQUENTIAL` must takes actions from different players one-by-one."
@api const SEQUENTIAL = Sequential()

"Environment with the [`DynamicStyle`](@ref) of `SIMULTANEOUS` must take in actions from some (or all) players at one time"
@api const SIMULTANEOUS = Simultaneous()

"""
    DynamicStyle(env::AbstractEnv) = SEQUENTIAL

Only valid in environments with a [`NumAgentStyle`](@ref) of
[`MultiAgent`](@ref). Determine whether the players can play simultaneously or
not. Possible returns are:

- [`SEQUENTIAL`](@ref). This is the default return.
- [`SIMULTANEOUS`](@ref).
"""
@env_api DynamicStyle(env::T) where {T<:AbstractEnv} = DynamicStyle(T)
DynamicStyle(::Type{<:AbstractEnv}) = SEQUENTIAL

#####
### InformationStyle
#####

abstract type AbstractInformationStyle <: AbstractEnvStyle end

@api struct PerfectInformation <: AbstractInformationStyle end
@api struct ImperfectInformation <: AbstractInformationStyle end

"All players observe the same state"
@api const PERFECT_INFORMATION = PerfectInformation()

"The inner state of some players' observations may be different"
@api const IMPERFECT_INFORMATION = ImperfectInformation()

"""
    InformationStyle(env) = IMPERFECT_INFORMATION

Distinguish environments between [`PERFECT_INFORMATION`](@ref) and
[`IMPERFECT_INFORMATION`](@ref). [`IMPERFECT_INFORMATION`](@ref) is returned by default.
"""
@env_api InformationStyle(env::T) where {T<:AbstractEnv} = InformationStyle(T)
InformationStyle(::Type{<:AbstractEnv}) = IMPERFECT_INFORMATION

#####
### ChanceStyle
#####

abstract type AbstractChanceStyle <: AbstractEnvStyle end
abstract type AbstractStochasticChanceStyle <: AbstractChanceStyle end

@api struct Deterministic <: AbstractChanceStyle end
@api struct Stochastic <: AbstractStochasticChanceStyle end
@api struct ExplicitStochastic <: AbstractStochasticChanceStyle end
@api struct SampledStochastic <: AbstractStochasticChanceStyle end

"No chance player in the environment. And the game is fully deterministic."
@api const DETERMINISTIC = Deterministic()

"""
No chance player in the environment. And the game is stochastic. To help
increase reproducibility, these environments should generally accept a
`AbstractRNG` as a keyword argument. For some third-party environments, at least
a `seed` is exposed in the constructor.
"""
@api const STOCHASTIC = Stochastic()

"""
Usually used to describe [extensive-form game](https://en.wikipedia.org/wiki/Extensive-form_game).
The environment contains a chance player and the corresponding probability is known.
Therefore, [`prob`](@ref)`(env, player=chance_player(env))` must be defined.
"""
@api const EXPLICIT_STOCHASTIC = ExplicitStochastic()

"""
Environment contains chance player and the probability is unknown. Usually only
a dummy action is allowed in this case.

!!! note
    The chance player ([`chance_player`](@ref)`(env)`) must appears in the result of
    [`players`](@ref)`(env)`.
    The result of `action_space(env, chance_player)` should only contains one
    dummy action.
"""
@api const SAMPLED_STOCHASTIC = SampledStochastic()

"""
    ChanceStyle(env) = STOCHASTIC

Specify which role the chance plays in the `env`. Possible returns are:

- [`STOCHASTIC`](@ref). This is the default return.
- [`DETERMINISTIC`](@ref)
- [`EXPLICIT_STOCHASTIC`](@ref)
- [`SAMPLED_STOCHASTIC`](@ref)
"""
@env_api ChanceStyle(env::T) where {T<:AbstractEnv} = ChanceStyle(T)
ChanceStyle(::Type{<:AbstractEnv}) = STOCHASTIC

#####
### RewardStyle
#####

abstract type AbstractRewardStyle <: AbstractEnvStyle end

@api struct StepReward <: AbstractRewardStyle end
@api struct TerminalReward <: AbstractRewardStyle end

"We can get reward after each step"
@api const STEP_REWARD = StepReward()

"Only get reward at the end of environment"
@api const TERMINAL_REWARD = TerminalReward()

"""
Specify whether we can get reward after each step or only at the end of an game.
Possible values are [`STEP_REWARD`](@ref) (the default one) or
[`TERMINAL_REWARD`](@ref).

!!! note
    Environments of [`TERMINAL_REWARD`](@ref) style can be viewed as a subset of
    environments of [`STEP_REWARD`](@ref) style. For some algorithms, like MCTS,
    we may have some a more efficient implementation for environments of
    [`TERMINAL_REWARD`](@ref) style.
"""
@env_api RewardStyle(env::T) where {T<:AbstractEnv} = RewardStyle(T)
RewardStyle(::Type{<:AbstractEnv}) = STEP_REWARD

#####
### UtilityStyle
#####

abstract type AbstractUtilityStyle <: AbstractEnvStyle end

@api struct ZeroSum <: AbstractUtilityStyle end
@api struct ConstantSum <: AbstractUtilityStyle end
@api struct GeneralSum <: AbstractUtilityStyle end
@api struct IdenticalUtility <: AbstractUtilityStyle end

"Rewards of all players sum to 0. A special case of [`CONSTANT_SUM`]."
@api const ZERO_SUM = ZeroSum()

"Rewards of all players sum to a constant"
@api const CONSTANT_SUM = ConstantSum()

"Total rewards of all players may be different in each step"
@api const GENERAL_SUM = GeneralSum()

"Every player gets the same reward"
@api const IDENTICAL_UTILITY = IdenticalUtility()

"""
    UtilityStyle(env::AbstractEnv)

Specify the utility style in multi-agent environments. Possible values are:

- [GENERAL_SUM](@ref). The default return.
- [ZERO_SUM](@ref)
- [CONSTANT_SUM](@ref)
- [IDENTICAL_UTILITY](@ref)
"""
@env_api UtilityStyle(env::T) where {T<:AbstractEnv} = UtilityStyle(T)
UtilityStyle(::Type{<:AbstractEnv}) = GENERAL_SUM

#####
### ActionStyle
#####

abstract type AbstractActionStyle <: AbstractEnvStyle end
abstract type AbstractDiscreteActionStyle <: AbstractActionStyle end
@api struct FullActionSet <: AbstractDiscreteActionStyle end

"The action space of the environment may contains illegal actions"
@api const FULL_ACTION_SET = FullActionSet()

@api struct MinimalActionSet <: AbstractDiscreteActionStyle end

"All actions in the action space of the environment are legal"
@api const MINIMAL_ACTION_SET = MinimalActionSet()

"""
    ActionStyle(env::AbstractEnv)

For environments of discrete actions, specify whether the current state of `env`
contains a full action set or a minimal action set. By default the
[`MINIMAL_ACTION_SET`](@ref) is returned.
"""
@env_api ActionStyle(env::T) where {T<:AbstractEnv} = ActionStyle(T)
ActionStyle(::Type{<:AbstractEnv}) = MINIMAL_ACTION_SET

#####
# StateStyle
#####

abstract type AbstractStateStyle end

"See the definition of [information set](https://en.wikipedia.org/wiki/Information_set_(game_theory))"
@api struct InformationSet{T} <: AbstractStateStyle end
InformationSet() = InformationSet{Any}()

"Use it to represent the internal state."
@api struct InternalState{T} <: AbstractStateStyle end
InternalState() = InternalState{Any}()

"Use it to represent the [goal state](http://proceedings.mlr.press/v37/schaul15.pdf)"
@api struct GoalState{T} <: AbstractStateStyle end
GoalState() = GoalState{Any}()

"""
Sometimes people from different field talk about the same thing with a different
name. Here we set the `Observation{Any}()` as the default state style in this
package.

See discussions [here](https://ai.stackexchange.com/questions/5970/what-is-the-difference-between-an-observation-and-a-state-in-reinforcement-learn)
"""
@api struct Observation{T} <: AbstractStateStyle end
Observation() = Observation{Any}()

"""
    StateStyle(env::AbstractEnv)

Define the possible styles of `state(env)`. Possible values are:

- [`Observation{T}`](@ref). This is the default return.
- [`InternalState{T}`](@ref)
- [`Information{T}`](@ref)
- You can also define your customized state style when necessary.

Or a tuple contains several of the above ones.

This is useful for environments which provide more than one kind of state.
"""
@env_api StateStyle(env::AbstractEnv) = Observation{Any}()

"""
Specify the default state style when calling `state(env)`.
"""
@env_api DefaultStateStyle(env::AbstractEnv) = DefaultStateStyle(StateStyle(env))
DefaultStateStyle(ss::AbstractStateStyle) = ss
DefaultStateStyle(ss::Tuple{Vararg{<:AbstractStateStyle}}) = first(ss)

# EpisodeStyle
# Episodic
# NeverEnding

#####
# General
#####

@api struct DefaultPlayer end
@api const DEFAULT_PLAYER = DefaultPlayer()

@api struct ChancePlayer end
@api const CHANCE_PLAYER = ChancePlayer()

@api struct SimultaneousPlayer end
@api const SIMULTANEOUS_PLAYER = SimultaneousPlayer()

@api struct Spector end
@api const SPECTOR = Spector()

@api (env::AbstractEnv)(action, player = current_player(env))

"""
Make an independent copy of `env`, 

!!! note
    rng (if `env` has) is also copied!
"""
@api copy(env::AbstractEnv) = deepcopy(env)
@api copyto!(dest::AbstractEnv, src::AbstractEnv)

# checking the state of all players in env is enough?
"""
    Base.:(==)(env1::T, env2::T) where T<:AbstractEnv
!!! warning
    Only check the state of all players in the env.
"""
function Base.:(==)(env1::T, env2::T) where T<:AbstractEnv
    len = length(players(env1))
    len == length(players(env2)) && 
    all(state(env1, player) == state(env2, player) for player in players(env1))
end
Base.hash(env::AbstractEnv, h::UInt) = hash([state(env, player) for player in players(env)], h)

@api nameof(env::AbstractEnv) = nameof(typeof(env))

"""
Get the action distribution of chance player.

!!! note
    Only valid for environments of [`EXPLICIT_STOCHASTIC`](@ref) style. The
    current player of `env` must be the chance player.
"""
@env_api prob(env::AbstractEnv, player = chance_player(env))

"""
    action_space(env, player=current_player(env))

Get all available actions from environment. See also:
[`legal_action_space`](@ref)
"""
@multi_agent_env_api action_space(env::AbstractEnv, player = current_player(env))

"""
    legal_action_space(env, player=current_player(env))

For environments of [`MINIMAL_ACTION_SET`](@ref), the result is the same with
[`action_space`](@ref).
"""
@multi_agent_env_api legal_action_space(env::AbstractEnv, player = current_player(env)) =
    legal_action_space(ActionStyle(env), env, player)

legal_action_space(::MinimalActionSet, env, player) = action_space(env)

"""
    legal_action_space_mask(env, player=current_player(env)) -> AbstractArray{Bool}

Required for environments of [`FULL_ACTION_SET`](@ref). As a default implementation,
     [`legal_action_space_mask`](@ref) creates a mask of [`action_space`](@ref) with
     the subset [`legal_action_space`](@ref).
"""
@multi_agent_env_api legal_action_space_mask(env::AbstractEnv, player = current_player(env)) = 
    map(action_space(env, player)) do action
        action in legal_action_space(env, player)
    end

"""
    state(env, style=[DefaultStateStyle(env)], player=[current_player(env)])

The state can be of any type. However, most neural network based algorithms
assume an `AbstractArray` is returned. For environments with many different states
provided (inner state, information state, etc), users need to provide `style`
to declare which kind of state they want.
            
!!! warning
    The state **may** be reused and be mutated at each step. Always remember to make a copy
    if this is not what you expect.
"""
@multi_agent_env_api state(env::AbstractEnv) = state(env, DefaultStateStyle(env))
state(env::AbstractEnv, ss::AbstractStateStyle) = state(env, ss, current_player(env))
state(env::AbstractEnv, player) = state(env, DefaultStateStyle(env), player)

"""
    state_space(env, style=[DefaultStateStyle(env)], player=[current_player(env)])
    
Describe all possible states.
"""
@multi_agent_env_api state_space(env::AbstractEnv) =
    state_space(env, DefaultStateStyle(env))
state_space(env::AbstractEnv, ss::AbstractStateStyle) =
    state_space(env, ss, current_player(env))
state_space(env::AbstractEnv, player) = state_space(env, DefaultStateStyle(env), player)

"""
    current_player(env)

Return the next player to take action. For [Extensive Form
Games](https://en.wikipedia.org/wiki/Extensive-form_game), a *chance player* may
be returned. (See also [`chance_player`](@ref)) For [SIMULTANEOUS](@ref)
environments, a *simultaneous player* is always returned. (See also
[`simultaneous_player`](@ref)).
"""
@env_api current_player(env::AbstractEnv) = DEFAULT_PLAYER

"""
    chance_player(env)

Only valid for environments with a chance player.
"""
@env_api chance_player(env::AbstractEnv) = CHANCE_PLAYER

"""
    simultaneous_player(env)

Only valid for environments of [`SIMULTANEOUS`](@ref) style.
"""
@env_api simultaneous_player(env) = SIMULTANEOUS_PLAYER

"""
    spectator_player(env)

Used in imperfect multi-agent environments.
"""
@env_api spectator_player(env::AbstractEnv)

@env_api players(env::AbstractEnv) = (DEFAULT_PLAYER,)

"Reset the internal state of an environment"
@env_api reset!(env::AbstractEnv)

"Set the seed of internal rng"
@env_api seed!(env::AbstractEnv, seed)

"""
    is_terminated(env, player=current_player(env))
"""
@env_api is_terminated(env::AbstractEnv)

"""
    reward(env, player=current_player(env))
"""
@multi_agent_env_api reward(env::AbstractEnv, player = current_player(env))

"""
    child(env::AbstractEnv, action)

Treat the `env` as a game tree. Create an independent child after applying
`action`.
"""
@api function child(env::AbstractEnv, action)
    new_env = copy(env)
    new_env(action)
    new_env
end

@api has_children(env::AbstractEnv) = !is_terminated(env)

@api children(env::AbstractEnv) = (child(env, action) for action in legal_action_space(env))

"""
    walk(f, env::AbstractEnv)

Call `f` with `env` and its descendants. Only use it with small games.
"""
@api function walk(f, env::AbstractEnv)
    f(env)
    if has_children(env)
        for x in children(env)
            walk(f, x)
        end
    end
end

#####
# EnvironmentModel
#####

"""
TODO:

Describe how to model a reinforcement learning environment. TODO: need more
investigation Ref: https://bair.berkeley.edu/blog/2019/12/12/mbpo/
- Analytic gradient computation
- Sampling-based planning
- Model-based data generation
- Value-equivalence prediction [Model-based Reinforcement Learning: A
  Survey.](https://arxiv.org/pdf/2006.16712.pdf) [Tutorial on Model-Based
  Methods in Reinforcement
  Learning](https://sites.google.com/view/mbrl-tutorial)
"""
@api abstract type AbstractEnvironmentModel end
