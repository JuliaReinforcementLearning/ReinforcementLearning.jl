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

function PlayCard(card_idx::Int)
    m = Ref{HanabiMove}()
    get_play_move(card_idx - 1, m)
    m
end

function DiscardCard(card_idx::Int)
    m = Ref{HanabiMove}()
    get_discard_move(card_idx - 1, m)
    m
end

function RevealColor(target_offset::Int, color::COLOR)
    m = Ref{HanabiMove}()
    get_reveal_color_move(target_offset, color, m)
    m
end

function RevealRank(target_offset::Int, rank::Int)
    m = Ref{HanabiMove}()
    get_reveal_rank_move(target_offset, rank - 1, m)
    m
end

function Base.show(io::IO, move::Base.RefValue{Hanabi.LibHanabi.PyHanabiMove})
    move_t = move_type(move)
    if move_t == Int(PLAY)
        print(io, "PlayCard($(card_index(move)+1))")
    elseif move_t == Int(DISCARD)
        print(io, "DiscardCard($(card_index(move)+1))")
    elseif move_t == Int(REVEAL_COLOR)
        print(io, "RevealColor($(target_offset(move)), $(COLOR(move_color(move)))")
    elseif move_t == Int(REVEAL_RANK)
        print(io, "RevealRank($(target_offset(move)), $(move_rank(move)+1)")
    else
        print(io, "uninitialized move")
    end
end

mutable struct HanabiResult
    player::Int32
    score_gain::Int32
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
mutable struct HanabiEnv <: AbstractEnv
    game::Base.RefValue{Hanabi.LibHanabi.PyHanabiGame}
    state::Base.RefValue{Hanabi.LibHanabi.PyHanabiState}
    moves::Vector{Base.RefValue{Hanabi.LibHanabi.PyHanabiMove}}
    observation_encoder::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservationEncoder}
    observation_space::MultiDiscreteSpace{Int64, 1}
    action_space::DiscreteSpace{Int64}
    reward::HanabiResult

    function HanabiEnv(;kw...)
        game = Ref{HanabiGame}()

        if length(kw) == 0
            new_default_game(game)
        else
            params = map(string, Iterators.flatten(kw))
            new_game(game, length(params), params)
        end

        state = Ref{HanabiState}()

        observation_encoder = Ref{HanabiObservationEncoder}()
        new_observation_encoder(observation_encoder, game, CANONICAL)
        observation_length = parse(Int, unsafe_string(observation_shape(observation_encoder)))
        observation_space = MultiDiscreteSpace(ones(Int, observation_length), zeros(Int, observation_length))

        n_moves = max_moves(game)
        action_space = DiscreteSpace(Int(n_moves))
        moves = [Ref{HanabiMove}() for _ in 1:n_moves]
        for i in 1:n_moves
            get_move_by_uid(game, i-1, moves[i])
        end

        env = new(game, state, moves, observation_encoder, observation_space, action_space, HanabiResult(Int32(0), Int32(0)))
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
    """)
end

Base.show(io::IO, game::Base.RefValue{Hanabi.LibHanabi.PyHanabiGame}) = print(io, unsafe_string(game_param_string(game)))
Base.show(io::IO, state::Base.RefValue{Hanabi.LibHanabi.PyHanabiState}) = print(io, unsafe_string(state_to_string(state)))
Base.show(io::IO, obs::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservation}) = print(io, unsafe_string(obs_to_string(obs)))

function reset!(env::HanabiEnv)
    new_state(env.game, env.state)
    while state_cur_player(env.state) == CHANCE_PLAYER_ID 
        state_deal_random_card(env.state)
    end
    nothing
end

function interact!(env::HanabiEnv, action::Integer)
    interact!(env, env.moves[action])
end

function interact!(env::HanabiEnv, move::Base.RefValue{Hanabi.LibHanabi.PyHanabiMove})
    move_is_legal(env.state, move) || error("illegal move: $move")
    player, old_score = state_cur_player(env.state), state_score(env.state)
    state_apply_move(env.state, move)

    while state_cur_player(env.state) == CHANCE_PLAYER_ID
        state_deal_random_card(env.state)
    end

    new_score = state_score(env.state)
    env.reward.player = player
    env.reward.score_gain = new_score - old_score

    observation = Ref{HanabiObservation}()
    new_observation(env.state, player, observation)

    (observation = _encode_observation(observation, env),
     reward      = env.reward.score_gain,
     isdone      = state_end_of_game_status(env.state) != Int(NOT_FINISHED))
end

function observe(env::HanabiEnv, observer=state_cur_player(env.state))
    observation = Ref{HanabiObservation}()
    new_observation(env.state, observer, observation)
    (observation     = _encode_observation(observation, env),
     reward          = env.reward.player == observer ? env.reward.score_gain : Int32(0),
     isdone          = state_end_of_game_status(env.state) != Int(NOT_FINISHED),
     raw_obs         = observation)
end

function _encode_observation(obs, env)
    encoding = Vector{Int32}(undef, length(env.observation_space.low))
    encode_obs(env.observation_encoder, obs, encoding)
    encoding
end

###
### Some Useful APIs
###

function legal_actions(env::HanabiEnv)
    actions = Int32[]
    for (i, move) in enumerate(env.moves)
        if move_is_legal(env.state, move)
            push!(actions, i)
        end
    end
    actions
end

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
