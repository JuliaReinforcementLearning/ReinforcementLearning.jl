export TinyHanabiEnv

const TINY_HANABI_REWARD_TABLE = begin
    t = Array{Int,4}(undef, 3, 3, 2, 2)
    t[:, :, 1, 1] = [
        10 0 0
        4 8 4
        10 0 0
    ]
    t[:, :, 1, 2] = [
        0 0 10
        4 8 4
        0 0 10
    ]
    t[:, :, 2, 1] = [
        0 0 10
        4 8 4
        0 0 0
    ]
    t[:, :, 2, 2] = [
        10 0 0
        4 8 4
        10 0 0
    ]
    t
end

struct TinyHanabiEnv <: AbstractEnv
    reward_table::Array{Int,4}
    cards::Vector{Int}
    actions::Vector{Int}
end

"""
    TinyHanabiEnv()

See https://arxiv.org/abs/1902.00506.
"""
TinyHanabiEnv() = TinyHanabiEnv(TINY_HANABI_REWARD_TABLE, Int[], Int[])

function RLBase.reset!(env::TinyHanabiEnv)
    empty!(env.cards)
    empty!(env.actions)
end

RLBase.players(env::TinyHanabiEnv) = 1:2

RLBase.current_player(env::TinyHanabiEnv) =
    if length(env.cards) < 2
        CHANCE_PLAYER
    elseif length(env.actions) == 0
        1
    else
        2
    end

(env::TinyHanabiEnv)(action, ::ChancePlayer) = push!(env.cards, action)
(env::TinyHanabiEnv)(action, ::Int) = push!(env.actions, action)

RLBase.action_space(env::TinyHanabiEnv, ::Int) = Base.OneTo(3)
RLBase.action_space(env::TinyHanabiEnv, ::ChancePlayer) = Base.OneTo(2)

RLBase.legal_action_space(env::TinyHanabiEnv, ::ChancePlayer) = findall(!in(env.cards), 1:2)
RLBase.legal_action_space_mask(env::TinyHanabiEnv, ::ChancePlayer) =
    [x ∉ env.cards for x in 1:2]

function RLBase.prob(env::TinyHanabiEnv, ::ChancePlayer)
    if isempty(env.cards)
        [0.5, 0.5]
    elseif length(env.cards) == 1
        p = ones(2)
        p[env.cards[]] = 0.0
        p
    else
        @error "shouldn't reach here."
    end
end

RLBase.state_space(env::TinyHanabiEnv, ::InformationSet, ::ChancePlayer) =
    ((0,), (0, 1), (0, 2), (0, 1, 2), (0, 2, 1)) # (chance_player_id(0), chance_player's actions...)
RLBase.state(env::TinyHanabiEnv, ::InformationSet, ::ChancePlayer) = (0, env.cards...)

function RLBase.state_space(env::TinyHanabiEnv, ::InformationSet, p::Int)
    Tuple(
        (p, c..., a...) for p in 1:2 for c in ((), 1, 2) for
        a in ((), 1:3..., ((i, j) for i in 1:3 for j in 1:3)...)
    )
end

function RLBase.state(env::TinyHanabiEnv, ::InformationSet, p::Int)
    card = length(env.cards) >= p ? env.cards[p] : ()
    (p, card..., env.actions...)
end

RLBase.is_terminated(env::TinyHanabiEnv) = length(env.actions) == 2
RLBase.reward(env::TinyHanabiEnv, player) =
    is_terminated(env) ? env.reward_table[env.actions..., env.cards...] : 0

(env::TinyHanabiEnv)(action::Int, ::ChancePlayer) = push!(env.cards, action)
(env::TinyHanabiEnv)(action::Int, ::Int) = push!(env.actions, action)

RLBase.NumAgentStyle(::TinyHanabiEnv) = MultiAgent(2)
RLBase.DynamicStyle(::TinyHanabiEnv) = SEQUENTIAL
RLBase.ActionStyle(::TinyHanabiEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::TinyHanabiEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::TinyHanabiEnv) = InformationSet{Tuple{Vararg{Int}}}()
RLBase.RewardStyle(::TinyHanabiEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::TinyHanabiEnv) = IDENTICAL_UTILITY
RLBase.ChanceStyle(::TinyHanabiEnv) = EXPLICIT_STOCHASTIC
