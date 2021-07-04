export MultiArmBanditsEnv

mutable struct MultiArmBanditsEnv <: AbstractEnv
    true_reward::Float64
    true_values::Vector{Float64}
    rng::AbstractRNG
    # cache
    reward::Float64
    is_terminated::Bool
end

"""
    MultiArmBanditsEnv(;true_reward=0., k = 10,rng=Random.GLOBAL_RNG)

`true_reward` is the expected reward. `k` is the number of arms. See
[multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) for more
detailed explanation.

This is a **one-shot** game. The environment terminates immediately after taking
in an action. Here we use it to demonstrate how to write a customized
environment with only minimal interfaces defined.
"""
function MultiArmBanditsEnv(; true_reward = 0.0, k = 10, rng = Random.GLOBAL_RNG)
    true_values = true_reward .+ randn(rng, k)
    MultiArmBanditsEnv(true_reward, true_values, rng, 0.0, false)
end

"""
First we need to define the action space. In the [`MultiArmBanditsEnv`](@ref)
environment, the possible actions are `1` to `k` (which equals to
`length(env.true_values)`).

!!! note
    Although we decide to return an action space of `Base.OneTo`  here, it is
    not a hard requirement. You can return anything else (`Tuple`,
    `Distribution`, etc) that is more suitable to describe your problem and
    handle it correctly in the `you_env(action)` function. Some algorithms may
    require that the action space must be of `Base.OneTo`. However, it's the
    algorithm designer's job to do the checking and conversion.
"""
RLBase.action_space(env::MultiArmBanditsEnv) = Base.OneTo(length(env.true_values))

"""
In our design, the return of taking an action in `env` is **undefined**. This is
the main difference compared to those interfaces defined in
[OpenAI/Gym](https://github.com/openai/gym). We find that the async manner is
more suitable to describe many complicated environments. However, one of the
inconveniences is that we have to cache some intermediate data for future
queries. Here we have to store `reward` and `is_terminated` in the instance of
`env` for future queries.
"""
function (env::MultiArmBanditsEnv)(action)
    env.reward = randn(env.rng) + env.true_values[action]
    env.is_terminated = true
end

RLBase.is_terminated(env::MultiArmBanditsEnv) = env.is_terminated

"""
!!! warn
    If the `env` is not started yet, the returned value is meaningless. The
    reason why we don't throw an exception here is to simplify the code logic to
    keep type consistency when storing the value in buffers.
"""
RLBase.reward(env::MultiArmBanditsEnv) = env.reward

"""
Since `MultiArmBanditsEnv` is just a one-shot game, it doesn't matter what the
state is after each action. So here we can simply set it to a constant `1`.
"""
RLBase.state(env::MultiArmBanditsEnv) = 1

RLBase.state_space(env::MultiArmBanditsEnv) = Base.OneTo(1)

function RLBase.reset!(env::MultiArmBanditsEnv)
    env.is_terminated = false
    # since the reward is meaningless if the game is not started yet,
    # we don't need to reset the reward here.
end

"""
The multi-arm bandits environment is a stochastic environment. The resulted
reward may be different even after taking the same actions each time. So for
this kind of environments, the `Random.seed!(env)` must be implemented to help
increase reproducibility without creating a new instance of the same `rng`.
"""
Random.seed!(env::MultiArmBanditsEnv, x) = seed!(env.rng, x)

# For this simple one-shot environment, the default definitions are enough.
# Here we redefined them to help you compare the traits across different
# environments to gain a better understanding.

RLBase.NumAgentStyle(::MultiArmBanditsEnv) = SINGLE_AGENT
RLBase.DynamicStyle(::MultiArmBanditsEnv) = SEQUENTIAL
RLBase.ActionStyle(::MultiArmBanditsEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::MultiArmBanditsEnv) = IMPERFECT_INFORMATION  # the distribution of noise and original reward is unknown to the agent
RLBase.StateStyle(::MultiArmBanditsEnv) = Observation{Int}()
RLBase.RewardStyle(::MultiArmBanditsEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::MultiArmBanditsEnv) = GENERAL_SUM
RLBase.ChanceStyle(::MultiArmBanditsEnv) = STOCHASTIC  # the same action lead to different reward each time.
