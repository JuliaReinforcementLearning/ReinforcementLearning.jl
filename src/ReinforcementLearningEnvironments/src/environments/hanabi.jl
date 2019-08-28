using Hanabi

export HanabiEnv, legal_actions, observe, reset!, interact!
export PlayCard, DiscardCard, RevealColor, RevealRank, parse_move
export cur_player, get_score, get_fireworks, encode_observation, encode_observation!, legal_actions!, legal_actions, get_cur_player

@enum HANABI_OBSERVATION_ENCODER_TYPE CANONICAL
@enum COLOR R Y G W B
@enum HANABI_END_OF_GAME_TYPE NOT_FINISHED OUT_OF_LIFE_TOKENS OUT_OF_CARDS COMPLETED_FIREWORKS
@enum HANABI_MOVE_TYPE INVALID PLAY DISCARD REVEAL_COLOR REVEAL_RANK DEAL

const CHANCE_PLAYER_ID = -1
const COLORS_DICT = Dict(string(x) => x for x in instances(COLOR))

###
### finalizers
###

move_finalizer(x) = finalizer(m -> delete_move(m), x)
history_item_finalizer(x) = finalizer(h -> delete_history_item(h), x)
game_finalizer(x) = finalizer(g -> delete_game(g), x)
observation_finalizer(x) = finalizer(o -> delete_observation(o), x)
observation_encoder_finalizer(x) = finalizer(e -> delete_observation_encoder(e), x)
state_finalizer(x) = finalizer(s -> delete_state(s), x)

###
### moves
###

function PlayCard(card_idx::Int)
    m = Ref{HanabiMove}()
    get_play_move(card_idx - 1, m)
    move_finalizer(m)
    m
end

function DiscardCard(card_idx::Int)
    m = Ref{HanabiMove}()
    get_discard_move(card_idx - 1, m)
    move_finalizer(m)
    m
end

function RevealColor(target_offset::Int, color::COLOR)
    m = Ref{HanabiMove}()
    get_reveal_color_move(target_offset, color, m)
    move_finalizer(m)
    m
end

function RevealRank(target_offset::Int, rank::Int)
    m = Ref{HanabiMove}()
    get_reveal_rank_move(target_offset, rank - 1, m)
    move_finalizer(m)
    m
end

function parse_move(s::String)
    m = match(r"PlayCard\((?<card_idx>[1-5])\)", s)
    !(m === nothing) && return PlayCard(parse(Int, m[:card_idx]))
    m = match(r"DiscardCard\((?<card_idx>[1-5])\)", s)
    !(m === nothing) && return DiscardCard(parse(Int, m[:card_idx]))
    m = match(r"RevealColor\((?<target>[1-5]),(?<color>[RYGWB])\)", s)
    !(m === nothing) && return RevealColor(parse(Int, m[:target]), COLORS_DICT[m[:color]])
    m = match(r"RevealRank\((?<target>[1-5]),(?<rank>[1-5])\)", s)
    !(m === nothing) && return RevealRank(parse(Int, m[:target]), parse(Int, m[:rank]))
    return nothing
end

function Base.show(io::IO, move::Base.RefValue{Hanabi.LibHanabi.PyHanabiMove})
    move_t = move_type(move)
    if move_t == Int(PLAY)
        print(io, "PlayCard($(card_index(move)+1))")
    elseif move_t == Int(DISCARD)
        print(io, "DiscardCard($(card_index(move)+1))")
    elseif move_t == Int(REVEAL_COLOR)
        print(io, "RevealColor($(target_offset(move)), $(COLOR(move_color(move))))")
    elseif move_t == Int(REVEAL_RANK)
        print(io, "RevealRank($(target_offset(move)), $(move_rank(move)+1))")
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
mutable struct HanabiEnv
    game::Base.RefValue{Hanabi.LibHanabi.PyHanabiGame}
    state::Base.RefValue{Hanabi.LibHanabi.PyHanabiState}
    moves::Vector{Base.RefValue{Hanabi.LibHanabi.PyHanabiMove}}
    observation_encoder::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservationEncoder}
    observation_length::Int
    reward::HanabiResult

    function HanabiEnv(;kw...)
        game = Ref{HanabiGame}()

        if length(kw) == 0
            new_default_game(game)
        else
            params = map(string, Iterators.flatten(kw))
            new_game(game, length(params), params)
        end

        game_finalizer(game)

        state = Ref{HanabiState}()
        new_state(game, state)
        state_finalizer(state)

        observation_encoder = Ref{HanabiObservationEncoder}()
        new_observation_encoder(observation_encoder, game, CANONICAL)
        observation_encoder_finalizer(observation_encoder)
        observation_length = parse(Int, unsafe_string(observation_shape(observation_encoder)))

        n_moves = max_moves(game)
        moves = [Ref{HanabiMove}() for _ in 1:n_moves]
        for i in 1:n_moves
            get_move_by_uid(game, i-1, moves[i])
            move_finalizer(moves[i])
        end

        env = new(game, state, moves, observation_encoder, observation_length, HanabiResult(Int32(0), Int32(0)))
        reset!(env)  # reset immediately
        env
    end
end

line_sep(x, sep="=") = repeat(sep, 25) * x * repeat(sep, 25)

function Base.show(io::IO, env::HanabiEnv)
    print(io,"""
    $(line_sep("[HanabiEnv]"))
    $(env.game)
    $(line_sep("[State]"))
    $(env.state)
    """)
end

function highlight(s)
    s = replace(s, "R" => Base.text_colors[:red] * "R" * Base.text_colors[:default])
    s = replace(s, "G" => Base.text_colors[:green] * "G" * Base.text_colors[:default])
    s = replace(s, "B" => Base.text_colors[:blue] * "B" * Base.text_colors[:default])
    s = replace(s, "Y" => Base.text_colors[:yellow] * "Y" * Base.text_colors[:default])
    s = replace(s, "W" => Base.text_colors[:white] * "W" * Base.text_colors[:default])
    s
end

Base.show(io::IO, game::Base.RefValue{Hanabi.LibHanabi.PyHanabiGame}) = print(io, unsafe_string(game_param_string(game)))
Base.show(io::IO, state::Base.RefValue{Hanabi.LibHanabi.PyHanabiState}) = print(io, highlight("\n" * unsafe_string(state_to_string(state))))
Base.show(io::IO, obs::Base.RefValue{Hanabi.LibHanabi.PyHanabiObservation}) = print(io, highlight("\n" * unsafe_string(obs_to_string(obs))))

function reset!(env::HanabiEnv)
    env.state = Ref{HanabiState}()
    state_finalizer(env.state)
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
    nothing
end

function observe(env::HanabiEnv, observer=state_cur_player(env.state))
    raw_obs = Ref{HanabiObservation}()
    observation_finalizer(raw_obs)
    new_observation(env.state, observer, raw_obs)

    Observation(
        reward = env.reward.player == observer ? env.reward.score_gain : Int32(0),
        terminal = state_end_of_game_status(env.state) != Int(NOT_FINISHED),
        state = raw_obs,
        game = env.game
    )
end

function encode_observation(obs, env)
    encoding = Vector{Int32}(undef, env.observation_length)
    encode_obs(env.observation_encoder, obs, encoding)
    encoding
end

function encode_observation!(obs, env, encoding)
    encode_obs(env.observation_encoder, obs, encoding)
    encoding
end

###
### Some Useful APIs
###

get_score(env::HanabiEnv) = state_score(env.state)
cur_player(env::HanabiEnv) = state_cur_player(env.state)

function legal_actions(env::HanabiEnv)
    actions = Int32[]
    for (i, move) in enumerate(env.moves)
        if move_is_legal(env.state, move)
            push!(actions, i)
        end
    end
    actions
end

legal_actions!(env::HanabiEnv, actions::AbstractVector{Bool}) = legal_actions!(env, actions, true, false)
legal_actions!(env::HanabiEnv, actions::AbstractVector{T}) where T<:Number = legal_actions!(env, actions, zero(T), typemin(T))

function legal_actions!(env::HanabiEnv, actions, legal_value, illegal_value)
    for (i, move) in enumerate(env.moves)
        actions[i] = move_is_legal(env.state, move) ? legal_value : illegal_value
    end
    actions
end

function get_hand_card_knowledge(obs, pid, i)
    knowledge = Ref{HanabiCardKnowledge}()
    obs_get_hand_card_knowledge(obs, pid, i, knowledge)
    knowledge
end

function get_hand_card(obs, pid, i)
    card_ref = Ref{HanabiCard}()
    obs_get_hand_card(obs, pid, i, card_ref)
    card_ref[]
end

rank(knowledge::Base.RefValue{Hanabi.LibHanabi.PyHanabiCardKnowledge}) = rank_was_hinted(knowledge) != 0 ? known_rank(knowledge) + 1 : nothing
rank(card::Hanabi.LibHanabi.PyHanabiCard) = card.rank + 1
color(knowledge::Base.RefValue{Hanabi.LibHanabi.PyHanabiCardKnowledge}) = color_was_hinted(knowledge) != 0 ? COLOR(known_color(knowledge)) : nothing
color(card::Hanabi.LibHanabi.PyHanabiCard) = COLOR(card.color)

function get_fireworks(game, observation)
    fireworks = Dict{COLOR, Int}()
    for c in 0:(num_colors(game) - 1)
        fireworks[COLOR(c)] = obs_fireworks(observation, c) + 1
    end
    fireworks
end

get_cur_player(env) = cur_player(env) + 1  # pid is 0-based