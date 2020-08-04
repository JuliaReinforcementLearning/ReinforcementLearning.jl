import .OpenSpiel:
    load_game,
    get_type,
    provides_information_state_tensor,
    provides_observation_tensor,
    dynamics,
    new_initial_state,
    chance_mode,
    is_chance_node,
    information,
    information_state_tensor,
    information_state_tensor_size,
    information_state_string,
    num_distinct_actions,
    num_players,
    apply_action,
    current_player,
    player_reward,
    legal_actions,
    legal_actions_mask,
    rewards,
    reward_model,
    history,
    observation_tensor_size,
    observation_tensor,
    observation_string,
    chance_mode,
    chance_outcomes,
    utility
using StatsBase: sample, weights


"""
    OpenSpielEnv(name; state_type=nothing, kwargs...)

# Arguments

- `name`::`String`, you can call `ReinforcementLearningEnvironments.OpenSpiel.registered_names()` to see all the supported names. Note that the name can contains parameters, like `"goofspiel(imp_info=True,num_cards=4,points_order=descending)"`. Because the parameters part is parsed by the backend C++ code, the bool variable must be `True` or `False` (instead of `true` or `false`). Another approach is to just specify parameters in `kwargs` in the Julia style.
- `state_type`::`Union{Symbol,Nothing}`, Supported values are [`:information`](https://github.com/deepmind/open_spiel/blob/1ad92a54f3b800394b2bc7f178ccdff62d8369e1/open_spiel/spiel.h#L342-L367), [`:observation`](https://github.com/deepmind/open_spiel/blob/1ad92a54f3b800394b2bc7f178ccdff62d8369e1/open_spiel/spiel.h#L397-L408) or `nothing`. The default value is `nothing`, which means `:information` if the game ` provides_information_state_tensor`. If not, it means `:observation`.
- `rng::AbstractRNG`, used to initial the `rng` for chance nodes. And the `rng` will only be used if the environment contains chance node, else it is set to `nothing`. To set the seed of inner environment, you may check the documentation of each specific game. Usually adding a keyword argument named `seed` should work.
- `is_chance_agent_required::Bool=false`, by default, no chance agent is required. An internal `rng` will be used to automatically generate actions for chance node. If set to `true`, you need to feed the action of chance agent to environment explicitly. And the `seed` will be ignored.
"""
function OpenSpielEnv(
    name;
    rng = Random.GLOBAL_RNG,
    state_type = nothing,
    is_chance_agent_required = false,
    kwargs...,
)
    game = load_game(name; kwargs...)
    game_type = get_type(game)

    has_info_state = provides_information_state_tensor(game_type)
    has_obs_state = provides_observation_tensor(game_type)
    has_info_state ||
        has_obs_state ||
        @error "the environment neither provides information tensor nor provides observation tensor"
    if isnothing(state_type)
        state_type = has_info_state ? :information : :observation
    end

    if state_type == :observation
        has_obs_state || @error "the environment doesn't support state_typeof $state_type"
    elseif state_type == :information
        has_info_state || @error "the environment doesn't support state_typeof $state_type"
    else
        @error "unknown state_type $state_type"
    end

    state = new_initial_state(game)

    c = if chance_mode(game_type) == OpenSpiel.DETERMINISTIC
        RLBase.DETERMINISTIC
    elseif is_chance_agent_required
        if chance_mode(game_type) == OpenSpiel.EXPLICIT_STOCHASTIC
            RLBase.EXPLICIT_STOCHASTIC
        else
            RLBase.SAMPLED_STOCHASTIC
        end
    else
        RLBase.STOCHASTIC
    end

    d = dynamics(game_type) == OpenSpiel.SEQUENTIAL ? RLBase.SEQUENTIAL :
        RLBase.SIMULTANEOUS

    i = information(game_type) == OpenSpiel.PERFECT_INFORMATION ?
        RLBase.PERFECT_INFORMATION : RLBase.IMPERFECT_INFORMATION

    n = MultiAgent(num_players(game))

    r = reward_model(game_type) == OpenSpiel.REWARDS ? RLBase.STEP_REWARD :
        RLBase.TERMINAL_REWARD

    u = if utility(game_type) == OpenSpiel.ZERO_SUM
        RLBase.ZERO_SUM
    elseif utility(game_type) == OpenSpiel.CONSTANT_SUM
        RLBase.CONSTANT_SUM
    elseif utility(game_type) == OpenSpiel.GENERAL_SUM
        RLBase.GENERAL_SUM
    elseif utility(game_type) == OpenSpiel.IDENTICAL_SUM
        RLBase.IDENTICAL_SUM
    end

    env =
        OpenSpielEnv{state_type,Tuple{c,d,i,n,r,u},typeof(state),typeof(game),typeof(rng)}(
            state,
            game,
            rng,
        )
    reset!(env)
    env
end

RLBase.ActionStyle(env::OpenSpielEnv) = FULL_ACTION_SET
RLBase.ChanceStyle(env::OpenSpielEnv{S,Tuple{C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = C
RLBase.InformationStyle(env::OpenSpielEnv{S,Tuple{C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = I
RLBase.NumAgentStyle(env::OpenSpielEnv{S,Tuple{C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = N
RLBase.RewardStyle(env::OpenSpielEnv{S,Tuple{C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = R
RLBase.UtilityStyle(env::OpenSpielEnv{S,Tuple{C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = U

Base.copy(env::OpenSpielEnv{S,T,ST,G,R}) where {S,T,ST,G,R} =
    OpenSpielEnv{S,T,ST,G,R}(copy(env.state), env.game, env.rng)

function RLBase.reset!(env::OpenSpielEnv)
    state = new_initial_state(env.game)
    ChanceStyle(env) === STOCHASTIC && _sample_external_events!(env.rng, state)
    env.state = state
end

_sample_external_events!(::Nothing, state) = nothing

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
    ChanceStyle(env) === STOCHASTIC && _sample_external_events!(env.rng, env.state)
end

RLBase.get_actions(env::OpenSpielEnv) = 0:num_distinct_actions(env.game)-1
RLBase.get_current_player(env::OpenSpielEnv) = current_player(env.state)
RLBase.get_chance_player(env::OpenSpielEnv) = convert(Int, OpenSpiel.CHANCE_PLAYER)
RLBase.get_players(env::OpenSpielEnv) = 0:(num_players(env.game)-1)

function Random.seed!(env::OpenSpielEnv, seed)
    if ChanceStyle(env) === STOCHASTIC
        Random.seed!(env.rng, seed)
    else
        @error "only environments of STOCHASTIC are supported, perhaps initialize the environment with a seed argument instead?"
    end
end

RLBase.get_legal_actions(env::OpenSpielEnv, player) = legal_actions(env.state, player)

function RLBase.get_legal_actions_mask(env::OpenSpielEnv, player)
    n = player == convert(Int, OpenSpiel.CHANCE_PLAYER) ? max_chance_outcomes(env.game) :
        num_distinct_actions(env.game)
    mask = BitArray(undef, n)
    for a in legal_actions(env.state, player)
        mask[a+1] = true
    end
    mask
end

RLBase.get_terminal(env::OpenSpielEnv, player) = OpenSpiel.is_terminal(env.state)

function RLBase.get_reward(env::OpenSpielEnv, player)
    if DynamicStyle(env) === SIMULTANEOUS &&
       player == convert(Int, OpenSpiel.SIMULTANEOUS_PLAYER)
        rewards(env.state)
    else
        player_reward(env.state, player)
    end
end

RLBase.get_state(env::OpenSpielEnv) = env.state
RLBase.get_state(env::OpenSpielEnv, player::Integer) = env.state

RLBase.get_history(env::OpenSpielEnv) = history(env.state)
