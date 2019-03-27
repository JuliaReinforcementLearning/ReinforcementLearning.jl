using Hanabi

export HanabiEnv, legal_actions

@enum HANABI_OBSERVATION_ENCODER_TYPE CANONICAL
@enum COLOR R Y G W B
@enum HANABI_END_OF_GAME_TYPE NOT_FINISHED OUT_OF_LIFE_TOKENS OUT_OF_CARDS COMPLETED_FIREWORKS
@enum HANABI_MOVE_TYPE INVALID PLAY DISCARD REVEAL_COLOR REVEAL_RANK DEAL

const CHANCE_PLAYER_ID = -1

###
### moves
###

abstract type AbstractMove end

struct PlayCard <: AbstractMove
    card_idx::Int
end

function Base.convert(HanabiMove, move::PlayCard)
    m = Ref{HanabiMove}()
    get_play_move(move.card_idx, m)
    m
end

struct DiscardCard <: AbstractMove
    card_idx::Int
end

function Base.convert(HanabiMove, move::DiscardCard)
    m = Ref{HanabiMove}()
    get_discard_move(move.card_idx, m)
    m
end

struct RevealColor <: AbstractMove
    target_offset::Int
    color::Int
end

Base.show(io::IO, m::RevealColor) = print(io, "RevealColor($(m.target_offset), $(COLOR(m.color)))")

RevealColor(target_offset::Int, color::COLOR) = RevealColor(target_offset, Int(color))

function Base.convert(HanabiMove, move::RevealColor)
    m = Ref{HanabiMove}()
    get_reveal_color_move(move.target_offset, move.color, m)
    m
end

struct RevealRank <: AbstractMove
    target_offset::Int
    rank::Int
    RevealRank(t, r) = new(t, r-1)
end

Base.show(io::IO, m::RevealRank) = print(io, "RevealRank($(m.target_offset), $(m.rank+1))")

function Base.convert(HanabiMove, move::RevealRank)
    m = Ref{HanabiMove}()
    get_reveal_rank_move(move.target_offset, move.rank, m)
    m
end

function Base.convert(AbstractMove, move::Base.RefValue{Hanabi.LibHanabi.PyHanabiMove})
    move_t = move_type(move)
    if move_t == Int(PLAY)
        PlayCard(card_index(move))
    elseif move_t == Int(DISCARD)
        DiscardCard(card_index(move))
    elseif move_t == Int(REVEAL_COLOR)
        RevealColor(target_offset(move), move_color(move))
    elseif move_t == Int(REVEAL_RANK)
        RevealRank(target_offset(move), move_rank(move)+1)
    else
        error("unsupported move type: $move_t")
    end
end

"""
    HanabiEnv(;kw...)

Default game params:

random_start_player    = false,
seed                   = -1,
max_life_tokens        = 3,
hand_size              = 5,
max_information_tokens = 8,
ranks                  = 5,
colors                 = 5,
observation_type       = 1,
players                = 2
"""
mutable struct HanabiEnv
    game::Base.RefValue{Hanabi.LibHanabi.PyHanabiGame}
    state::Base.RefValue{Hanabi.LibHanabi.PyHanabiState}
    observation_encoder::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservationEncoder}
    observations::Vector{Base.RefValue{Hanabi.LibHanabi.PyHanabiObservation}}
    observation_space::MultiDiscreteSpace{Int64, 1}
    action_space::DiscreteSpace{Int64}
    reward::Dict{Int32, Int32}

    function HanabiEnv(;kw...)
        game = Ref{HanabiGame}()

        if length(kw) == 0
            new_default_game(game)
        else
            params = map(string, Iterators.flatten(kw))
            new_game(game, length(params), params)
        end

        state = Ref{HanabiState}()
        new_state(game, state)

        observation_encoder = Ref{HanabiObservationEncoder}()
        new_observation_encoder(observation_encoder, game, CANONICAL)
        observation_length = parse(Int, unsafe_string(observation_shape(observation_encoder)))
        observations = [Ref{HanabiObservation}() for _ in 1:num_players(game)]

        observation_space = MultiDiscreteSpace(ones(Int, observation_length), zeros(Int, observation_length))

        action_space = DiscreteSpace(max_moves(game) - 1, 0)  # start from 0

        env = new(game, state, observation_encoder, observations, observation_space, action_space, Dict{Int32, Int32}())
        reset!(env)  # reset immediately
        env
    end
end

observation_space(env::HanabiEnv) = env.observation_space
action_space(env::HanabiEnv) = env.action_space

line_sep(x, sep="=") = repeat(sep, 25) * x * repeat(sep, 25)

function Base.show(io::IO, env::HanabiEnv)
    print(io,"""
    $(line_sep("[HanabiEnv]"))
    $(env.game)
    $(line_sep("[State]"))
    $(env.state)
    $(line_sep("[Observations]"))
    """)
    for pid in 0:num_players(env.game)-1
        println(line_sep("[Player $pid]", "-"))
        println(env.observations[pid+1])
    end
end

Base.show(io::IO, game::Base.RefValue{Hanabi.LibHanabi.PyHanabiGame}) = print(io, unsafe_string(game_param_string(game)))
Base.show(io::IO, state::Base.RefValue{Hanabi.LibHanabi.PyHanabiState}) = print(io, unsafe_string(state_to_string(state)))
Base.show(io::IO, obs::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservation}) = print(io, unsafe_string(obs_to_string(obs)))

function reset!(env::HanabiEnv)
    state = Ref{HanabiState}()
    new_state(env.game, state)
    env.state = state
    while state_cur_player(env.state) == CHANCE_PLAYER_ID 
        state_deal_random_card(env.state)
    end
    for pid in 0:num_players(env.game)-1
        new_observation(env.state, pid, env.observations[pid+1])
    end
    nothing
end

function interact!(env::HanabiEnv, action::Integer)
    move = Ref{HanabiMove}()
    get_move_by_uid(env.game, action, move)
    _apply_move(env, move)
    nothing
end

function interact!(env::HanabiEnv, action::AbstractMove)
    move = convert(HanabiMove, action)
    _apply_move(env, move)
    nothing
end

function _apply_move(env::HanabiEnv, move)
    move_is_legal(env.state, move) || error("illegal move $(unsafe_string(move_to_string(move)))")
    player, old_score = state_cur_player(env.state), state_score(env.state)
    state_apply_move(env.state, move)
    while state_cur_player(env.state) == CHANCE_PLAYER_ID
        state_deal_random_card(env.state)
    end
    new_score = state_score(env.state)
    env.reward = Dict(player => new_score - old_score)
    for pid in 0:num_players(env.game)-1
        new_observation(env.state, pid, env.observations[pid+1])
    end
end

function observe(env::HanabiEnv, observer=state_cur_player(env.state))
    observation = Ref{HanabiObservation}()
    new_observation(env.state, observer, observation)
    (observation     = _encode_observation(observation, env.observation_encoder),
     reward          = get(env.reward, observer, zero(Int32)),
     isdone          = state_end_of_game_status(env.state) != NOT_FINISHED,
     raw_observation = env.observations[observer+1])
end

_encode_observation(observation, encoder) = [parse(Int, x) for x in split(unsafe_string(encode_observation(encoder, observation)), ',')]

###
### Some Useful APIs
###

function legal_moves(obs::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservation})
    moves = AbstractMove[]
    for i in 0:obs_num_legal_moves(obs)-1
        move = Ref{HanabiMove}()
        obs_get_legal_move(obs, i, move)
        push!(moves, convert(AbstractMove, move))
    end
    moves
end

legal_moves(env::HanabiEnv, pid=state_cur_player(env.state)) = legal_moves(env.observations[pid+1])

function legal_actions(obs, game)
    moves = Int32[]
    for i in 0:obs_num_legal_moves(obs)-1
        move = Ref{HanabiMove}()
        obs_get_legal_move(obs, i, move)
        push!(moves, get_move_uid(game, move))
    end
    moves
end

legal_actions(env::HanabiEnv, pid=state_cur_player(env.state)) = legal_actions(env.observations[pid+1], env.game)

function get_card_knowledge(obs)
    knowledges = []
    for pid in 0:obs_num_players(obs)-1
        hand_kd = []
        for i in 0:obs_get_hand_size(obs, pid) - 1
            kd = Ref{HanabiCardKnowledge}()
            obs_get_hand_card_knowledge(obs, pid, i, kd)
            push!(
                hand_kd,
                Dict{String, Any}(
                    "color" => color_was_hinted(kd) > 0 ? COLOR(known_color(kd)) : nothing,
                    "rank"  => rank_was_hinted(kd) > 0 ? known_rank(kd) : nothing))
        end
        push!(knowledges, hand_kd)
    end
    knowledges
end

function observed_hands(obs)
    hands = Vector{HanabiCard}[]
    for pid in 0:obs_num_players(obs)-1
        cards = HanabiCard[]
        for i in 0:obs_get_hand_size(obs, pid)-1
            card_ref = Ref{HanabiCard}()
            obs_get_hand_card(obs, pid, i, card_ref)
            push!(cards, card_ref[])
        end
        push!(hands, cards)
    end
    hands
end

function discard_pile(obs)
    cards = HanabiCard[]
    for i in 0:obs_discard_pile_size(obs)-1
        card_ref = Ref{HanabiCard}()
        obs_get_discard(obs, i, card_ref)
        push!(cards, card_ref[])
    end
    cards
end