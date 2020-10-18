export BlackjackEnv


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = vcat(collect(1:10), [10, 10, 10])

@inline draw_card(seed::AbstractRNG) = rand(seed, deck)
@inline draw_hand(seed::AbstractRNG) = [draw_card(seed), draw_card(seed)]
@inline usable_ace(hand) = 1 ∈ hand && sum(hand) + 10 ≤ 21
@inline sum_hand(hand) = usable_ace(hand) ? sum(hand) + 10 : sum(hand)
@inline is_bust(hand) = sum_hand(hand) > 21
@inline score(hand) = is_bust(hand) ? 0 : sum_hand(hand)
@inline isnatural(hand) = sort(hand) == [1, 10]

mutable struct BlackjackEnv <: AbstractEnv
    action_space::DiscreteSpace{UnitRange{Int64}}
    observation_space::MultiDiscreteSpace{Vector{Int64}}
    dealer::Array
    player::Array
    state::Array{Int64,1}
    action::Int
    reward::Int
    done::Bool
    natural::Bool
    rng::AbstractRNG
end

"""Simple blackjack environment
Blackjack is a card game where the goal is to obtain cards that sum to as
near as possible to 21 without going over.  They're playing against a fixed
dealer.
Face cards (Jack, Queen, King) have point value 10.
Aces can either count as 11 or 1, and it's called 'usable' at 11.
This game is placed with an infinite deck (or with replacement).
The game starts with each (player and dealer) having one face up and one
face down card.
The player can request additional cards (hit=1) until they decide to stop
(stick=0) or exceed 21 (bust).
After the player sticks, the dealer reveals their facedown card, and draws
until their sum is 17 or greater.  If the dealer goes bust the player wins.
If neither player nor dealer busts, the outcome (win, lose, draw) is
decided by whose sum is closer to 21.  The reward for winning is +1,
drawing is 0, and losing is -1.
The observation of a 3-tuple of: the players current sum,
the dealer's one showing card (1-10 where 1 is ace),
and whether or not the player holds a usable ace (0 or 1).
This environment corresponds to the version of the blackjack problem
described in Example 5.1 in Reinforcement Learning: An Introduction
by Sutton and Barto.
http://incompleteideas.net/book/the-book-2nd.html
"""

function BlackjackEnv(;natural=false,rng = Random.GLOBAL_RNG)
    # natural = flag to payout on a "natural" blackjack win, like casion rules
    action_space = DiscreteSpace(2)
    low = [1,1,1]
    high = [32,11,2]
    observation_space = MultiDiscreteSpace(low, high)
    
    bj = BlackjackEnv(action_space, observation_space, [], [], [0,0,0], 2, 0, false, false, rng)
    reset!(bj)
    bj
end

@inline _get_obs(env::BlackjackEnv) = [sum_hand(env.player), env.dealer[1], usable_ace(env.player)]

function RLBase.reset!(env::BlackjackEnv)
    env.dealer = draw_hand(env.rng)
    env.player = draw_hand(env.rng)
    env.state = _get_obs(env)
    env.done = false
    nothing
end


RLBase.get_actions(env::BlackjackEnv) = env.action_space
RLBase.get_reward(env::BlackjackEnv) = env.reward
RLBase.get_terminal(env::BlackjackEnv) = env.done
RLBase.get_state(env::BlackjackEnv) = env.state

function (env::BlackjackEnv)(a)
    @assert a ∈ env.action_space "Invalid action"
    env.action = a
    if a == 1 # hit: add a card to the players hand and return
        push!(env.player, draw_card(env.rng))
        env.done, env.reward = is_bust(env.player) ? (true, -1) : (false, 0)

    else # stick: play out the dealers hand, and score
        env.done = true
        while sum_hand(env.dealer) < 17
            push!(env.dealer, draw_card(env.rng))
        end
        env.reward = cmp(score(env.player),score(env.dealer))
        if env.natural && isnatural(env.player) && reward == 1
            reward = 1.5
        end 
    end
    env.state = _get_obs(env)
    nothing
end

Random.seed!(env::BlackjackEnv, seed) = Random.seed!(env.rng, seed)
