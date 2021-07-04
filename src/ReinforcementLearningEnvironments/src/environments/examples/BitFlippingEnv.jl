export BitFlippingEnv

"""
In Bit Flipping Environment we have n bits. The actions are 1 to n where executing i-th action flips the i-th bit of the state. For every episode we sample uniformly and initial state as well as the target state.

Refer [Hindsight Experience Replay paper](https://arxiv.org/pdf/1707.01495.pdf) for the motivation behind the environment.
"""

mutable struct BitFlippingEnv <: AbstractEnv
    N::Int
    rng::AbstractRNG
    state::BitArray{1}
    goal_state::BitArray{1}
    max_steps::Int
    t::Int
end

function BitFlippingEnv(; N = 8, T = N, rng = Random.GLOBAL_RNG)
    state = bitrand(rng, N)
    goal_state = bitrand(rng, N)
    max_steps = T
    BitFlippingEnv(N, rng, state, goal_state, max_steps, 0)
end

Random.seed!(env::BitFlippingEnv, s) = Random.seed!(env.rng, s)

RLBase.action_space(env::BitFlippingEnv) = Base.OneTo(env.N)

RLBase.legal_action_space(env::BitFlippingEnv) = Base.OneTo(env.N)

function (env::BitFlippingEnv)(action::Int)
    env.t += 1
    if 1 <= action <= env.N
        env.state[action] = !env.state[action]
        nothing
    else
        @error "Invalid Action"
    end
end

RLBase.state(env::BitFlippingEnv) = state(env::BitFlippingEnv, Observation{BitArray{1}}())
RLBase.state(env::BitFlippingEnv, ::Observation) = env.state
RLBase.state(env::BitFlippingEnv, ::GoalState) = env.goal_state
RLBase.state_space(env::BitFlippingEnv, ::Observation) = Space(fill(false..true, env.N))
RLBase.state_space(env::BitFlippingEnv, ::GoalState) = Space(fill(false..true, env.N))
RLBase.is_terminated(env::BitFlippingEnv) =
    (env.state == env.goal_state) || (env.t >= env.max_steps)

function RLBase.reset!(env::BitFlippingEnv)
    env.t = 0
    env.state .= bitrand(env.rng, env.N)
    env.goal_state .= bitrand(env.rng, env.N)
end

function RLBase.reward(env::BitFlippingEnv)
    if env.state == env.goal_state
        0.0
    else
        -1.0
    end
end

RLBase.NumAgentStyle(::BitFlippingEnv) = SINGLE_AGENT
RLBase.DynamicStyle(::BitFlippingEnv) = SEQUENTIAL
RLBase.ActionStyle(::BitFlippingEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::BitFlippingEnv) = PERFECT_INFORMATION
RLBase.StateStyle(::BitFlippingEnv) = (Observation{BitArray{1}}(), GoalState{BitArray{1}}())
RLBase.RewardStyle(::BitFlippingEnv) = STEP_REWARD
RLBase.UtilityStyle(::BitFlippingEnv) = GENERAL_SUM
RLBase.ChanceStyle(::BitFlippingEnv) = STOCHASTIC
