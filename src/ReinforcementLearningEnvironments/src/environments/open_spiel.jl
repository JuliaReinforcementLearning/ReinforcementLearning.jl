module OpenSpielWrapper

export OpenSpielEnv

using ReinforcementLearningBase
using OpenSpiel
using Random
using StatsBase: sample, weights

abstract type AbstractObservationType end

mutable struct OpenSpielEnv{O,D,S,G,R} <: AbstractEnv
    state::S
    game::G
    rng::R
end

"""
    OpenSpielEnv(name; observation_type=nothing, kwargs...)

# Arguments

- `name`::`String`, you can call `resigtered_names()` to see all the supported names. Note that the name can contains parameters, like `"goofspiel(imp_info=True,num_cards=4,points_order=descending)"`. Because the parameters part is parsed by the backend C++ code, the bool variable must be `True` or `False` (instead of `true` or `false`). Another approach is to just specify parameters in `kwargs` in the Julia style.
- `observation_type`::`Union{Symbol,Nothing}`, Supported values are [`:information`](https://github.com/deepmind/open_spiel/blob/1ad92a54f3b800394b2bc7f178ccdff62d8369e1/open_spiel/spiel.h#L342-L367), [`:observation`](https://github.com/deepmind/open_spiel/blob/1ad92a54f3b800394b2bc7f178ccdff62d8369e1/open_spiel/spiel.h#L397-L408) or `nothing`. The default value is `nothing`, which means `:information` if the game ` provides_information_state_tensor`. If not, it means `:observation`.
"""
function OpenSpielEnv(name; seed = nothing, observation_type = nothing, kwargs...)
    game = load_game(name, kwargs...)
    game_type = get_type(game)

    has_info_state = provides_information_state_tensor(game_type)
    has_obs_state = provides_observation_tensor(game_type)
    has_info_state || has_obs_state ||
    @error "the environment neither provides information tensor nor provides observation tensor"
    if isnothing(observation_type)
        observation_type = has_info_state ? :information : :observation
    end
    if observation_type == :observation
        has_obs_state ||
        @error "the environment doesn't support observation_type of $observation_type"
    elseif observation_type == :information
        has_info_state ||
        @error "the environment doesn't support observation_type of $observation_type"
    else
        @error "unknown observation_type $observation_type"
    end

    d = dynamics(game_type)
    dynamic_style = if d === OpenSpiel.SEQUENTIAL
        RLBase.SEQUENTIAL
    elseif d === OpenSpiel.SIMULTANEOUS
        RLBase.SIMULTANEOUS
    else
        @error "unknown dynamic style of $d"
    end

    state = new_initial_state(game)

    rng = MersenneTwister(seed)

    env =
        OpenSpielEnv{observation_type,dynamic_style,typeof(state),typeof(game),typeof(rng)}(
            state,
            game,
            rng,
        )
    reset!(env)
    env
end

RLBase.DynamicStyle(env::OpenSpielEnv{O,D}) where {O,D} = D

function RLBase.reset!(env::OpenSpielEnv)
    state = new_initial_state(env.game)
    _sample_external_events!(env.rng, state)
    env.state = state
end

function _sample_external_events!(rng::AbstractRNG, state)
    while is_chance_node(state)
        outcomes_with_probs = chance_outcomes(state)
        actions, probs = zip(outcomes_with_probs...)
        action = actions[sample(rng, weights(collect(probs)))]
        apply_action(state, action)
    end
end

function (env::OpenSpielEnv)(action)
    apply_action(env.state, action)
    _sample_external_events!(env.rng, env.state)
end

(env::OpenSpielEnv)(player, action) = env(DynamicStyle(env), player, action)

function (env::OpenSpielEnv)(::Sequential, player, action)
    if get_current_player(env) == player
        apply_action(env.state, action)
    else
        apply_action(env.state, OpenSpiel.INVALID_ACTION[])
    end
    _sample_external_events!(env.rng, env.state)
end

(env::OpenSpielEnv)(::Simultaneous, player, action) =
    @error "Simultaneous environments can not take in the actions from players seperately"

struct OpenSpielObs{O,D,S,P}
    state::S
    player::P
end

RLBase.observe(env::OpenSpielEnv{O,D,S}, player::P) where {O,D,S,P} =
    OpenSpielObs{O,D,S,P}(env.state, player)

RLBase.get_action_space(env::OpenSpielEnv) =
    DiscreteSpace(0:num_distinct_actions(env.game)-1)

function RLBase.get_observation_space(env::OpenSpielEnv{:information})
    s = information_state_tensor_size(env.game)
    MultiContinuousSpace(fill(typemin(Float64), s...), fill(typemax(Float64), s...))
end

function RLBase.get_observation_space(env::OpenSpielEnv{:observation})
    s = observation_tensor_size(env.game)
    MultiContinuousSpace(fill(typemin(Float64), s...), fill(typemax(Float64), s...))
end

RLBase.get_current_player(env::OpenSpielEnv) = current_player(env.state)

RLBase.get_num_players(env::OpenSpielEnv) = num_players(env.game)

Random.seed!(env::OpenSpielEnv, seed) = Random.seed!(env.rng, seed)

RLBase.ActionStyle(::OpenSpielObs) = FULL_ACTION_SET

RLBase.get_legal_actions(obs::OpenSpielObs) = legal_actions(obs.state, obs.player)

RLBase.get_legal_actions_mask(obs::OpenSpielObs) = legal_actions_mask(obs.state, obs.player)

RLBase.get_terminal(obs::OpenSpielObs) = OpenSpiel.is_terminal(obs.state)

RLBase.get_reward(obs::OpenSpielObs) = rewards(obs.state)[obs.player+1]  # player starts with 0

RLBase.get_state(obs::OpenSpielObs{:information}) =
    information_state_tensor(obs.state, obs.player)

RLBase.get_state(obs::OpenSpielObs{:observation}) =
    observation_tensor(obs.state, obs.player)

RLBase.get_invalid_action(obs::OpenSpielObs) = convert(Int, OpenSpiel.INVALID_ACTION[])

end

using .OpenSpielWrapper
export OpenSpielEnv
