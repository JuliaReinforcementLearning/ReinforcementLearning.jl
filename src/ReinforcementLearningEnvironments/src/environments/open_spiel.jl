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
    max_chance_outcomes,
    utility
using StatsBase: sample, weights


"""
    OpenSpielEnv(name; state_type=nothing, kwargs...)

# Arguments

- `name`::`String`, you can call `ReinforcementLearningEnvironments.OpenSpiel.registered_names()` to see all the supported names. Note that the name can contains parameters, like `"goofspiel(imp_info=True,num_cards=4,points_order=descending)"`. Because the parameters part is parsed by the backend C++ code, the bool variable must be `True` or `False` (instead of `true` or `false`). Another approach is to just specify parameters in `kwargs` in the Julia style.
- `default_state_style`::`Union{AbstractStateStyle,Nothing}`, Supported values are [`Information{<:Union{String,Array}}`](https://github.com/deepmind/open_spiel/blob/1ad92a54f3b800394b2bc7f178ccdff62d8369e1/open_spiel/spiel.h#L342-L367), [`Observation{<:Union{String,Array}}`](https://github.com/deepmind/open_spiel/blob/1ad92a54f3b800394b2bc7f178ccdff62d8369e1/open_spiel/spiel.h#L397-L408) or `nothing`.
- `rng::AbstractRNG`, used to initial the `rng` for chance nodes. And the `rng` will only be used if the environment contains chance node, else it is set to `nothing`. To set the seed of inner environment, you may check the documentation of each specific game. Usually adding a keyword argument named `seed` should work.
- `is_chance_agent_required::Bool=false`, by default, no chance agent is required. An internal `rng` will be used to automatically generate actions for chance node. If set to `true`, you need to feed the action of chance agent to environment explicitly. And the `seed` will be ignored.
"""
function OpenSpielEnv(
    name;
    rng = Random.GLOBAL_RNG,
    default_state_style = nothing,
    is_chance_agent_required = false,
    kwargs...,
)
    game = load_game(String(name); kwargs...)
    game_type = get_type(game)

    if isnothing(default_state_style)
        default_state_style = if provides_information_state_string(game_type)
            RLBase.Information{String}()
        elseif provides_information_state_tensor(game_type)
            RLBase.Information{Array}()
        elseif provides_observation_tensor(game_type)
            Observation{Array}()
        elseif provides_observation_string(game_type)
            Observation{String}()
        else
            nothing
        end
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

    d =
        dynamics(game_type) == OpenSpiel.SEQUENTIAL ? RLBase.SEQUENTIAL :
        RLBase.SIMULTANEOUS

    i =
        information(game_type) == OpenSpiel.PERFECT_INFORMATION ?
        RLBase.PERFECT_INFORMATION : RLBase.IMPERFECT_INFORMATION

    n = MultiAgent(num_players(game))

    r =
        reward_model(game_type) == OpenSpiel.REWARDS ? RLBase.STEP_REWARD :
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

    env = OpenSpielEnv{
        Tuple{default_state_style,c,d,i,n,r,u},
        typeof(state),
        typeof(game),
        typeof(rng),
    }(
        state,
        game,
        rng,
    )
    reset!(env)
    env
end

RLBase.ActionStyle(env::OpenSpielEnv) = FULL_ACTION_SET
RLBase.ChanceStyle(env::OpenSpielEnv{Tuple{S,C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = C
RLBase.InformationStyle(env::OpenSpielEnv{Tuple{S,C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = I
RLBase.NumAgentStyle(env::OpenSpielEnv{Tuple{S,C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = N
RLBase.RewardStyle(env::OpenSpielEnv{Tuple{S,C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = R
RLBase.UtilityStyle(env::OpenSpielEnv{Tuple{S,C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = U
RLBase.DefaultStateStyle(env::OpenSpielEnv{Tuple{S,C,D,I,N,R,U}}) where {S,C,D,I,N,R,U} = S

Base.copy(env::OpenSpielEnv{T,ST,G,R}) where {T,ST,G,R} =
    OpenSpielEnv{T,ST,G,R}(copy(env.state), env.game, env.rng)

function RLBase.reset!(env::OpenSpielEnv)
    state = new_initial_state(env.game)
    ChanceStyle(env) === STOCHASTIC && _sample_external_events!(env.rng, state)
    env.state = state
end

_sample_external_events!(::Nothing, state) = nothing

function _sample_external_events!(rng::AbstractRNG, state)
    while is_chance_node(state)
        apply_action(
            state,
            rand(
                rng,
                reinterpret(ActionProbPair{Int,Float64}, chance_outcomes(state)),
            ).action,
        )
    end
end

function (env::OpenSpielEnv)(action::Int)
    apply_action(env.state, action)
    ChanceStyle(env) === STOCHASTIC && _sample_external_events!(env.rng, env.state)
end

RLBase.get_current_player(env::OpenSpielEnv) = current_player(env.state)
RLBase.get_chance_player(env::OpenSpielEnv) = convert(Int, OpenSpiel.CHANCE_PLAYER)
RLBase.get_players(env::OpenSpielEnv) = get_players(env, ChanceStyle(env))
RLBase.get_players(env::OpenSpielEnv, ::Any) = 0:(num_players(env.game)-1)
RLBase.get_players(env::OpenSpielEnv, ::Union{ExplicitStochastic,SampledStochastic}) =
    (get_chance_player(env), 0:(num_players(env.game)-1)...)
RLBase.get_num_players(env::OpenSpielEnv) = length(get_players(env))

function RLBase.get_actions(env::OpenSpielEnv, player)
    if player == get_chance_player(env)
        reinterpret(ActionProbPair{Int,Float64}, chance_outcomes(env.state))
    else
        0:num_distinct_actions(env.game)-1
    end
end

function RLBase.get_legal_actions(env::OpenSpielEnv, player)
    if player == get_chance_player(env)
        reinterpret(ActionProbPair{Int,Float64}, chance_outcomes(env.state))
    else
        legal_actions(env.state, player)
    end
end

function RLBase.get_legal_actions_mask(env::OpenSpielEnv, player)
    n =
        player == get_chance_player(env) ? max_chance_outcomes(env.game) :
        num_distinct_actions(env.game)
    mask = BitArray(undef, n)
    for a in legal_actions(env.state, player)
        mask[a+1] = true
    end
    mask
end

function Random.seed!(env::OpenSpielEnv, seed)
    if ChanceStyle(env) === STOCHASTIC
        Random.seed!(env.rng, seed)
    else
        @error "only environments of STOCHASTIC are supported, perhaps initialize the environment with a seed argument instead?"
    end
end

RLBase.get_terminal(env::OpenSpielEnv, player) = OpenSpiel.is_terminal(env.state)

function RLBase.get_reward(env::OpenSpielEnv, player)
    if DynamicStyle(env) === SIMULTANEOUS &&
       player == convert(Int, OpenSpiel.SIMULTANEOUS_PLAYER)
        rewards(env.state)
    elseif player == get_chance_player(env)
        0  # ??? type stable
    else
        player_reward(env.state, player)
    end
end

RLBase.get_state(env::OpenSpielEnv, player::Integer) =
    get_state(env, DefaultStateStyle(env), player)
RLBase.get_state(env::OpenSpielEnv, ::RLBase.Information{String}, player) =
    information_state_string(env.state, player)
RLBase.get_state(env::OpenSpielEnv, ::RLBase.Information{Array}, player) =
    information_state_tensor(env.state, player)
RLBase.get_state(env::OpenSpielEnv, ::Observation{String}, player) =
    observation_string(env.state, player)
RLBase.get_state(env::OpenSpielEnv, ::Observation{Array}, player) =
    observation_tensor(env.state, player)

RLBase.get_history(env::OpenSpielEnv) = history(env.state)
