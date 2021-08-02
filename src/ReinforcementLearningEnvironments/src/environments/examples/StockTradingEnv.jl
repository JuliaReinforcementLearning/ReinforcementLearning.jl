export StockTradingEnv

using Pkg.Artifacts
using DelimitedFiles
using LinearAlgebra:dot

mutable struct StockTradingEnv <: AbstractEnv
    turbulence_threshold::Float64
    is_hard_reset::Bool
    features::Matrix{Float64}
    prices::Matrix{Float64}
    turbulence::Vector{Float64}
    initial_account_balance::Float64
    day::Int
    last_day::Int
    daily_reward::Float32
    state::Vector{Float32}
    HMAX_NORMALIZE::Float32
    TRANSACTION_FEE_PERCENT::Float32
    REWARD_SCALING::Float32
    total_cost::Float32
end

function load_default_stock_trading_data()
    turbulence = readdlm(joinpath(artifact"stock_trading_data", "turbulence.csv")) |> vec
    prices, _ = readdlm(joinpath(artifact"stock_trading_data", "prices.csv"), ',', header=true)
    features, _ = readdlm(joinpath(artifact"stock_trading_data", "features.csv"), ',', header=true)
    # column major
    collect(prices'), collect(features'), turbulence
end

"""
    StockTradingEnv(;kw...)

This environment is originally provided in [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020)

# Keyword Arguments

- `turbulence_threshold=140`, when turbulence is higher than a threshold, which
  indicates extreme market conditions, simply halt buying and the trading agent
  sells all shares.
- `is_hard_reset::Bool=true`, if set to `true`, each `reset!` call will reset
  the `asset` to the `initial_account_balance`.
- `initial_account_balance=1_000_000`.
"""
function StockTradingEnv(;
    turbulence_threshold=140,
    is_hard_reset=true,
    initial_account_balance=1_000_000,
    features=nothing,
    prices=nothing,
    day=1,
    last_day=nothing,
    state=nothing,
    HMAX_NORMALIZE = 100,
    TRANSACTION_FEE_PERCENT = 0.001,
    REWARD_SCALING = 1f-4
)
    if isnothing(features) && isnothing(prices)
        prices, features, turbulence = load_default_stock_trading_data()
    end

    if isnothing(state)
        state = zeros(Float32, 1 + size(prices, 1) * 2, size(features, 1))
    end

    if isnothing(last_day)
        last_day = length(turbulence)
    end

    StockTradingEnv(
        turbulence_threshold,
        is_hard_reset,
        features,
        prices,
        turbulence,
        initial_account_balance,
        day,
        last_day,
        0.0,
        state,
        HMAX_NORMALIZE,
        TRANSACTION_FEE_PERCENT,
        0f0
    )
end

n_stocks(env::StockTradingEnv) = size(env.prices, 1)
prices(env::StockTradingEnv) = @view(env.state[2:1+n_stocks(env)])
holds(env::StockTradingEnv) = @view(env.state[2+n_stocks(env):n_stocks(env)*2+1])
features(env::StockTradingEnv) = @view(env.state[n_stocks(env)*2+2:end])
asset(env::StockTradingEnv) = @view env.state[1]
total_asset(env::StockTradingEnv) = env.state[1] + dot(prices(env), holds(env))


function (env::StockTradingEnv)(actions)
    init_asset = total_asset(env)

    # sell first
    sell = clamp.(env.HMAX_NORMALIZE .* actions, .-(holds(env)), 0f0)
    holds(env) .-= sell
    gain = dot(sell, prices(env))
    cost = gain * env.TRANSACTION_FEE_PERCENT
    asset(env)[] += gain - cost
    env.total_cost += cost

    # then buy
    # better to shuffle?
    for i,b in enumerate(actions)
        if b > 0
            A = asset(env)
            max_buy = div(A[], P)
            buy = min(b*env.HMAX_NORMALIZE, max_buy)
            holds(env)[i] += buy
            deduction = buy * prices(env)[i]
            cost = deduction * env.TRANSACTION_FEE_PERCENT
            A[] -= deduction + cost
            env.cost += cost
        end
    end

    env.day += 1
    prices(env) .= @view env.prices[:, env.day]
    features(env) .= @view env.features[:, env.day]

    env.daily_reward = total_asset(env) - init_asset
end

RLBase.reward(env::StockTradingEnv) = env.daily_reward * env.REWARD_SCALING
RLBase.is_terminated(env::StockTradingEnv) = env.day > env.last_day
RLBase.state(env::StockTradingEnv) = env.state
