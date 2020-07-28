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
mutable struct LotteryEnv{R<:AbstractRNG} <: AbstractEnv
    reward::Int
    terminal::Bool
    rng::R
end

LotteryEnv() = LotteryEnv(0, false, Random.GLOBAL_RNG)

RLBase.get_actions(env::LotteryEnv) = (:PowerRich, :MegaHaul, nothing)
RLBase.get_state(env::LotteryEnv) = env.terminal ? 2 : 1
Random.seed!(env::LotteryEnv, seed) = Random.seed!(env.rng, seed)

function (env::LotteryEnv)(action::Union{Symbol,Nothing})
    if action == :PowerRich
        env.reward = rand(env.rng) < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        env.reward = rand(env.rng) < 0.05 ? 1_000_000 : -10
    else
        env.reward = 0
    end
    env.terminal = true
end

RLBase.reset!(env::LotteryEnv) = env.terminal = false
