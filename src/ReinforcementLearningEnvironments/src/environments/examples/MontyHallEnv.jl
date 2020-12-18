export MontyHallEnv

const REWARD_OF_GOAT = 10.0
const REWARD_OF_CAR = 1_000.0

mutable struct MontyHallEnv <: AbstractEnv
    doors::Vector{Symbol}
    rng::AbstractRNG
    guest_action::Union{Nothing,Int}
    host_action::Union{Nothing,Int}
    reward::Union{Nothing,Float64}
end

"""
    MontyHallEnv(;rng=Random.GLOBAL_RNG)

Quoted from [wiki](https://en.wikipedia.org/wiki/Monty_Hall_problem):

> Suppose you're on a game show, and you're given the choice of three doors:
> Behind one door is a car; behind the others, goats. You pick a door, say No.
> 1, and the host, who knows what's behind the doors, opens another door, say
> No. 3, which has a goat. He then says to you, "Do you want to pick door No.
> 2?" Is it to your advantage to switch your choice?

Here we'll introduce the first environment which is of [`FULL_ACTION_SET`](@ref).
"""
function MontyHallEnv(; rng = Random.GLOBAL_RNG)
    doors = fill(:goat, 3)
    doors[rand(rng, 1:3)] = :car
    MontyHallEnv(doors, rng, nothing, nothing, nothing)
end

Random.seed!(env::MontyHallEnv, s) = Random.seed!(env.rng, s)

RLBase.action_space(::MontyHallEnv) = Base.OneTo(3)

"""
In the first round, the guest has 3 options, in the second round only two
options are valid, those different then the host's action.
"""
function RLBase.legal_action_space(env::MontyHallEnv)
    if isnothing(env.host_action)
        1:3
    else
        findall(!=(env.host_action), 1:3)
    end
end

"""
For environments of [`FULL_ACTION_SET`], this function must be implemented.
"""
function RLBase.legal_action_space_mask(env::MontyHallEnv)
    mask = BitArray(undef, 3)
    fill!(mask, true)
    if !isnothing(env.host_action)
        mask[env.host_action] = false
    end
    mask
end

function RLBase.state(env::MontyHallEnv)
    if isnothing(env.host_action)
        1
    else
        env.host_action + 1
    end
end

RLBase.state_space(env::MontyHallEnv) = 1:4

function (env::MontyHallEnv)(action)
    if isnothing(env.host_action)
        # first round
        env.guest_action = action
        if env.doors[action] == :car
            env.host_action = findall(!=(action), 1:3)[rand(env.rng, 1:2)]
        else
            for i in 1:3
                if i == action
                    continue
                elseif env.doors[i] == :goat
                    env.host_action = i
                end
            end
        end
    else
        # second round
        if action == env.host_action
            @error "Invalid action. Can not select the same door with the host."
        else
            env.guest_action = action
            env.reward = env.doors[action] == :goat ? REWARD_OF_GOAT : REWARD_OF_CAR
        end
    end
end

RLBase.reward(env::MontyHallEnv) = isnothing(env.reward) ? 0.0 : env.reward

RLBase.is_terminated(env::MontyHallEnv) = !isnothing(env.reward)

function RLBase.reset!(env::MontyHallEnv)
    env.doors .= :goat
    env.doors[rand(env.rng, 1:3)] = :car
    env.guest_action = nothing
    env.host_action = nothing
    env.reward = nothing
end

RLBase.NumAgentStyle(::MontyHallEnv) = SINGLE_AGENT
RLBase.DynamicStyle(::MontyHallEnv) = SEQUENTIAL
RLBase.ActionStyle(::MontyHallEnv) = FULL_ACTION_SET
RLBase.InformationStyle(::MontyHallEnv) = IMPERFECT_INFORMATION  # the distribution of noise and original reward is unknown to the agent
RLBase.StateStyle(::MontyHallEnv) = Observation{Int}()
RLBase.RewardStyle(::MontyHallEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::MontyHallEnv) = GENERAL_SUM
RLBase.ChanceStyle(::MontyHallEnv) = STOCHASTIC  # the same action lead to different reward each time.
