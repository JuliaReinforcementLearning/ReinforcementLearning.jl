export KuhnPokerEnv

const KUHN_POKER_CARDS = (:J, :Q, :K)
const KUHN_POKER_CARD_COMBINATIONS =
    ((:J, :Q), (:J, :K), (:Q, :J), (:Q, :K), (:K, :J), (:K, :Q))
const KUHN_POKER_ACTIONS = (:pass, :bet)
const KUHN_POKER_STATES = (
    (),
    map(tuple, KUHN_POKER_CARDS)...,
    KUHN_POKER_CARD_COMBINATIONS...,
    (
        (cards..., actions...) for cards in ((), map(tuple, KUHN_POKER_CARDS)...) for
        actions in (
            (),
            (:bet,),
            (:bet, :bet),
            (:bet, :pass),
            (:pass,),
            (:pass, :pass),
            (:pass, :bet),
            (:pass, :bet, :pass),
            (:pass, :bet, :bet),
        )
    )...,
)

"""
![](https://upload.wikimedia.org/wikipedia/commons/a/a9/Kuhn_poker_tree.svg)
"""
const KUHN_POKER_REWARD_TABLE = Dict(
    (:J, :Q, :bet, :bet) => -2,
    (:J, :K, :bet, :bet) => -2,
    (:Q, :J, :bet, :bet) => 2,
    (:Q, :K, :bet, :bet) => -2,
    (:K, :J, :bet, :bet) => 2,
    (:K, :Q, :bet, :bet) => 2,
    (:J, :Q, :bet, :pass) => 1,
    (:J, :K, :bet, :pass) => 1,
    (:Q, :J, :bet, :pass) => 1,
    (:Q, :K, :bet, :pass) => 1,
    (:K, :J, :bet, :pass) => 1,
    (:K, :Q, :bet, :pass) => 1,
    (:J, :Q, :pass, :pass) => -1,
    (:J, :K, :pass, :pass) => -1,
    (:Q, :J, :pass, :pass) => 1,
    (:Q, :K, :pass, :pass) => -1,
    (:K, :J, :pass, :pass) => 1,
    (:K, :Q, :pass, :pass) => 1,
    (:J, :Q, :pass, :bet, :pass) => -1,
    (:J, :K, :pass, :bet, :pass) => -1,
    (:Q, :J, :pass, :bet, :pass) => -1,
    (:Q, :K, :pass, :bet, :pass) => -1,
    (:K, :J, :pass, :bet, :pass) => -1,
    (:K, :Q, :pass, :bet, :pass) => -1,
    (:J, :Q, :pass, :bet, :bet) => -2,
    (:J, :K, :pass, :bet, :bet) => -2,
    (:Q, :J, :pass, :bet, :bet) => 2,
    (:Q, :K, :pass, :bet, :bet) => -2,
    (:K, :J, :pass, :bet, :bet) => 2,
    (:K, :Q, :pass, :bet, :bet) => 2,
)

struct KuhnPokerEnv <: AbstractEnv
    cards::Vector{Symbol}
    actions::Vector{Symbol}
end

"""
    KuhnPokerEnv()

See more detailed description [here](https://en.wikipedia.org/wiki/Kuhn_poker).

Here we demonstrate how to write a typical [`ZERO_SUM`](@ref),
[`IMPERFECT_INFORMATION`](@ref) game. The implementation here has a explicit
[`CHANCE_PLAYER`](@ref).

TODO: add public state for [`SPECTOR`](@ref).
Ref: https://arxiv.org/abs/1906.11110
"""
KuhnPokerEnv() = KuhnPokerEnv(Symbol[], Symbol[])

function RLBase.reset!(env::KuhnPokerEnv)
    empty!(env.cards)
    empty!(env.actions)
end

RLBase.is_terminated(env::KuhnPokerEnv) =
    length(env.actions) == 2 && (env.actions[1] == :bet || env.actions[2] == :pass) ||
    length(env.actions) == 3
RLBase.players(env::KuhnPokerEnv) = (1, 2, CHANCE_PLAYER)

function RLBase.state(env::KuhnPokerEnv, ::InformationSet{Tuple{Vararg{Symbol}}}, p::Int)
    if length(env.cards) >= p
        (env.cards[p], env.actions...)
    else
        ()
    end
end

RLBase.state(env::KuhnPokerEnv, ::InformationSet{Tuple{Vararg{Symbol}}}, ::ChancePlayer) =
    Tuple(env.cards)
RLBase.state_space(env::KuhnPokerEnv, ::InformationSet{Tuple{Vararg{Symbol}}}, p) =
    KUHN_POKER_STATES

RLBase.action_space(env::KuhnPokerEnv, ::Int) = Base.OneTo(length(KUHN_POKER_ACTIONS))
RLBase.action_space(env::KuhnPokerEnv, ::ChancePlayer) =
    Base.OneTo(length(KUHN_POKER_CARDS))

RLBase.legal_action_space(env::KuhnPokerEnv, p::ChancePlayer) =
    [x for x in action_space(env, p) if KUHN_POKER_CARDS[x] ∉ env.cards]

function RLBase.legal_action_space_mask(env::KuhnPokerEnv, p::ChancePlayer)
    m = fill(true, 3)
    m[env.cards] .= false
    m
end

function RLBase.prob(env::KuhnPokerEnv, ::ChancePlayer)
    if length(env.cards) == 0
        fill(1 / 3, 3)
    elseif length(env.cards) == 1
        p = fill(1 / 2, 3)
        i = findfirst(==(env.cards[1]), KUHN_POKER_CARDS)
        p[i] = 0
        p
    else
        @error "it's not chance player's turn!"
    end
end

(env::KuhnPokerEnv)(action::Int, p::Int) = env(KUHN_POKER_ACTIONS[action], p)
(env::KuhnPokerEnv)(action::Int, p::ChancePlayer) = env(KUHN_POKER_CARDS[action], p)
(env::KuhnPokerEnv)(action::Symbol, ::ChancePlayer) = push!(env.cards, action)
(env::KuhnPokerEnv)(action::Symbol, ::Int) = push!(env.actions, action)

RLBase.reward(::KuhnPokerEnv, ::ChancePlayer) = 0

function RLBase.reward(env::KuhnPokerEnv, p)
    if is_terminated(env)
        v = KUHN_POKER_REWARD_TABLE[(env.cards..., env.actions...)]
        p == 1 ? v : -v
    else
        0
    end
end

RLBase.current_player(env::KuhnPokerEnv) =
    if length(env.cards) < 2
        CHANCE_PLAYER
    elseif length(env.actions) == 0
        1
    elseif length(env.actions) == 1
        2
    elseif length(env.actions) == 2
        1
    else
        2  # actually the game is over now
    end

RLBase.NumAgentStyle(::KuhnPokerEnv) = MultiAgent(2)
RLBase.DynamicStyle(::KuhnPokerEnv) = SEQUENTIAL
RLBase.ActionStyle(::KuhnPokerEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::KuhnPokerEnv) = IMPERFECT_INFORMATION
RLBase.StateStyle(::KuhnPokerEnv) = InformationSet{Tuple{Vararg{Symbol}}}()
RLBase.RewardStyle(::KuhnPokerEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::KuhnPokerEnv) = ZERO_SUM
RLBase.ChanceStyle(::KuhnPokerEnv) = EXPLICIT_STOCHASTIC
