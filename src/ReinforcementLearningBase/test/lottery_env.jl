using Random

#####
# LotteryEnv
#####

"""
    LotteryEnv()

Here we use an example introduced in [Monte Carlo Tree Search: A Tutorial](https://www.informs-sim.org/wsc18papers/includes/files/021.pdf) to demenstrate how to write an environment.

Assume you have \$10 in your pocket, and you are faced with the following three choices:

1. buy a PowerRich lottery ticket (win \$100M w.p. 0.01; nothing otherwise);
2. buy a MegaHaul lottery ticket (win \$1M w.p. 0.05; nothing otherwise);
3. do not buy a lottery ticket.
"""
mutable struct LotteryEnv <: AbstractEnv
    reward::Int
    is_done::Bool
    rng::MersenneTwister
end

LotteryEnv(; seed = nothing) = LotteryEnv(0, false, MersenneTwister(seed))

RLBase.get_action_space(env::LotteryEnv) = DiscreteSpace((:PowerRich, :MegaHaul, nothing))

function (env::LotteryEnv)(action::Union{Symbol,Nothing})
    if action == :PowerRich
        env.reward = rand(env.rng) < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        env.reward = rand(env.rng) < 0.05 ? 1_000_000 : -10
    else
        env.reward = 0
    end
    env.is_done = true
end

RLBase.observe(env::LotteryEnv) = (reward = env.reward, terminal = env.is_done)

RLBase.reset!(env::LotteryEnv) = env.is_done = false
