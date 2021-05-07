import .OpenSpiel:
    load_game,
    get_type,
    provides_information_state_tensor,
    provides_observation_tensor,
    provides_information_state_string,
    provides_observation_string,
    dynamics,
    new_initial_state,
    chance_mode,
    is_chance_node,
    information,
    information_state_tensor,
    information_state_tensor_size,
    information_state_tensor_shape,
    information_state_string,
    num_distinct_actions,
    num_players,
    apply_action,
    player_reward,
    legal_actions,
    legal_actions_mask,
    rewards,
    reward_model,
    observation_tensor_size,
    observation_tensor_shape,
    observation_tensor,
    observation_string,
    chance_mode,
    chance_outcomes,
    max_chance_outcomes,
    utility


"""
    OpenSpielEnv(name; state_type=nothing, kwargs...)

# Arguments

- `name`::`String`, you can call `OpenSpiel.registered_names()` to see all
  the supported names. Note that the name can contains parameters, like
  `"goofspiel(imp_info=True,num_cards=4,points_order=descending)"`. Because the
  parameters part is parsed by the backend C++ code, the bool variable must be
  `True` or `False` (instead of `true` or `false`). Another approach is to just
  specify parameters in `kwargs` in the Julia style.
"""
function OpenSpielEnv(name = "kuhn_poker"; kwargs...)
    game = load_game(String(name); kwargs...)
    state = new_initial_state(game)
    OpenSpielEnv(state, game)
end

Base.copy(env::OpenSpielEnv) = OpenSpielEnv(copy(env.state), env.game)

RLBase.reset!(env::OpenSpielEnv) = env.state = new_initial_state(env.game)

(env::OpenSpielEnv)(action::Integer) = apply_action(env.state, action)

RLBase.current_player(env::OpenSpielEnv) = OpenSpiel.current_player(env.state)
RLBase.chance_player(env::OpenSpielEnv) = convert(Int, OpenSpiel.CHANCE_PLAYER)

function RLBase.players(env::OpenSpielEnv)
    p = 0:(num_players(env.game)-1)
    if ChanceStyle(env) === EXPLICIT_STOCHASTIC
        (p..., RLBase.chance_player(env))
    else
        p
    end
end

function RLBase.action_space(env::OpenSpielEnv, player)
    if player == chance_player(env)
        # !!! this bug is already fixed in OpenSpiel
        # replace it with the following one later
        # ZeroTo(max_chance_outcomes(env.game)-1)
        ZeroTo(max_chance_outcomes(env.game))
    else
        ZeroTo(num_distinct_actions(env.game) - 1)
    end
end

function RLBase.legal_action_space(env::OpenSpielEnv, player)
    if player == chance_player(env)
        [k for (k, v) in chance_outcomes(env.state)]
    else
        legal_actions(env.state, player)
    end
end

function RLBase.prob(env::OpenSpielEnv, player)
    # @assert player == chance_player(env)
    p = zeros(length(action_space(env)))
    for (k, v) in chance_outcomes(env.state)
        p[k+1] = v
    end
    p
end

function RLBase.legal_action_space_mask(env::OpenSpielEnv, player)
    n =
        player == chance_player(env) ? max_chance_outcomes(env.game) :
        num_distinct_actions(env.game)
    mask = BitArray(undef, n)
    for a in legal_actions(env.state, player)
        mask[a+1] = true
    end
    mask
end

RLBase.is_terminated(env::OpenSpielEnv) = OpenSpiel.is_terminal(env.state)

function RLBase.reward(env::OpenSpielEnv, player)
    if DynamicStyle(env) === SIMULTANEOUS &&
       player == convert(Int, OpenSpiel.SIMULTANEOUS_PLAYER)
        rewards(env.state)
    elseif player < 0
        0
    else
        player_reward(env.state, player)
    end
end

function RLBase.state(env::OpenSpielEnv, ss::RLBase.AbstractStateStyle, player)
    if player < 0  # TODO: revisit this in OpenSpiel@v0.2
        @warn "unexpected player $player, falling back to default state value." maxlog = 1
        s = state_space(env)
        if s isa WorldSpace
            ""
        elseif s isa Array{<:Interval}
            rand(s)
        end
    else
        _state(env, ss, player)
    end
end

_state(env::OpenSpielEnv, ::RLBase.InformationSet{String}, player) =
    information_state_string(env.state, player)
_state(env::OpenSpielEnv, ::RLBase.InformationSet{Array}, player) = reshape(
    information_state_tensor(env.state, player),
    reverse(information_state_tensor_shape(env.game))...,
)
_state(env::OpenSpielEnv, ::Observation{String}, player) =
    observation_string(env.state, player)
_state(env::OpenSpielEnv, ::Observation{Array}, player) = reshape(
    observation_tensor(env.state, player),
    reverse(observation_tensor_shape(env.game))...,
)

RLBase.state_space(
    env::OpenSpielEnv,
    ::Union{InformationSet{String},Observation{String}},
    p,
) = WorldSpace{AbstractString}()

RLBase.state_space(env::OpenSpielEnv, ::InformationSet{Array}, p) = Space(
    fill(
        typemin(Float64)..typemax(Float64),
        reverse(information_state_tensor_shape(env.game))...,
    ),
)

RLBase.state_space(env::OpenSpielEnv, ::Observation{Array}, p) = Space(
    fill(
        typemin(Float64)..typemax(Float64),
        reverse(observation_tensor_shape(env.game))...,
    ),
)

Random.seed!(env::OpenSpielEnv, s) = @warn "seed!(OpenSpielEnv) is not supported currently."

function RLBase.ChanceStyle(env::OpenSpielEnv)
    game_type = get_type(env.game)
    if chance_mode(game_type) == OpenSpiel.DETERMINISTIC
        RLBase.DETERMINISTIC
    elseif chance_mode(game_type) == OpenSpiel.EXPLICIT_STOCHASTIC
        RLBase.EXPLICIT_STOCHASTIC
    else
        RLBase.STOCHASTIC
    end
end

function RLBase.UtilityStyle(env::OpenSpielEnv)
    game_type = get_type(env.game)
    if utility(game_type) == OpenSpiel.ZERO_SUM
        RLBase.ZERO_SUM
    elseif utility(game_type) == OpenSpiel.CONSTANT_SUM
        RLBase.CONSTANT_SUM
    elseif utility(game_type) == OpenSpiel.GENERAL_SUM
        RLBase.GENERAL_SUM
    elseif utility(game_type) == OpenSpiel.IDENTICAL_SUM
        RLBase.IDENTICAL_SUM
    end
end

RLBase.ActionStyle(env::OpenSpielEnv) = FULL_ACTION_SET
RLBase.DynamicStyle(env::OpenSpielEnv) =
    dynamics(get_type(env.game)) == OpenSpiel.SEQUENTIAL ? RLBase.SEQUENTIAL :
    RLBase.SIMULTANEOUS
RLBase.InformationStyle(env::OpenSpielEnv) =
    information(get_type(env.game)) == OpenSpiel.PERFECT_INFORMATION ?
    RLBase.PERFECT_INFORMATION : RLBase.IMPERFECT_INFORMATION
RLBase.NumAgentStyle(env::OpenSpielEnv) = MultiAgent(num_players(env.game))
RLBase.RewardStyle(env::OpenSpielEnv) =
    reward_model(get_type(env.game)) == OpenSpiel.REWARDS ? RLBase.STEP_REWARD :
    RLBase.TERMINAL_REWARD

RLBase.StateStyle(env::OpenSpielEnv) = (
    RLBase.InformationSet{String}(),
    RLBase.InformationSet{Array}(),
    RLBase.Observation{String}(),
    RLBase.Observation{Array}(),
)
