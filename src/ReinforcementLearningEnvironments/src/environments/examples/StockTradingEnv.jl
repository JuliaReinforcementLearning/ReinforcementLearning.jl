export StockTradingEnv, StockTradingEnvWithTurbulence

using Pkg.Artifacts
using DelimitedFiles
using LinearAlgebra:dot
using IntervalSets

function load_default_stock_data(s)
    if s == "prices.csv" || s == "features.csv"
        data, _ = readdlm(joinpath(artifact"stock_trading_data", s), ',', header=true)
        collect(data')
    elseif s == "turbulence.csv"
        readdlm(joinpath(artifact"stock_trading_data", "turbulence.csv")) |> vec
    else
        @error "unknown dataset $s"
    end
end

mutable struct StockTradingEnv{F<:AbstractMatrix{Float64}, P<:AbstractMatrix{Float64}} <: AbstractEnv
    features::F
    prices::P
    HMAX_NORMALIZE::Float32
    TRANSACTION_FEE_PERCENT::Float32
    REWARD_SCALING::Float32
    initial_account_balance::Float32
    state::Vector{Float32}
    total_cost::Float32
    day::Int
    first_day::Int
    last_day::Int
    daily_reward::Float32
end

_n_stocks(env::StockTradingEnv) = size(env.prices, 1)
_prices(env::StockTradingEnv) = @view(env.state[2:1+_n_stocks(env)])
_holds(env::StockTradingEnv) = @view(env.state[2+_n_stocks(env):_n_stocks(env)*2+1])
_features(env::StockTradingEnv) = @view(env.state[_n_stocks(env)*2+2:end])
_balance(env::StockTradingEnv) = @view env.state[1]
_total_asset(env::StockTradingEnv) = env.state[1] + dot(_prices(env), _holds(env))

"""
    StockTradingEnv(;kw...)

This environment is originally provided in [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020)

# Keyword Arguments

- `initial_account_balance=1_000_000`.
"""
function StockTradingEnv(;
    initial_account_balance=1_000_000f0,
    features=nothing,
    prices=nothing,
    first_day=nothing,
    last_day=nothing,
    HMAX_NORMALIZE = 100f0,
    TRANSACTION_FEE_PERCENT = 0.001f0,
    REWARD_SCALING = 1f-4
)
    prices = isnothing(prices) ? load_default_stock_data("prices.csv") : prices
    features = isnothing(features) ? load_default_stock_data("features.csv") : features

    @assert size(prices, 2) == size(features, 2)

    first_day = isnothing(first_day) ? 1 : first_day
    last_day = isnothing(last_day) ? size(prices, 2) : last_day
    day = first_day

    # [balance, stock_prices..., stock_holds..., features...]
    state = zeros(Float32, 1 + size(prices, 1) * 2 + size(features, 1))

    env = StockTradingEnv(
        features,
        prices,
        HMAX_NORMALIZE,
        TRANSACTION_FEE_PERCENT,
        REWARD_SCALING,
        initial_account_balance,
        state,
        0f0,
        day,
        first_day,
        last_day,
        0f0
    )

    _balance(env)[] = initial_account_balance
    _prices(env) .= @view prices[:, day]
    _features(env) .= @view features[:, day]

    env
end

function (env::StockTradingEnv)(actions)
    init_asset = _total_asset(env)

    # sell first
    for (i, s) in enumerate(actions)
        if s < 0
            sell = min(-env.HMAX_NORMALIZE * s, _holds(env)[i])
            _holds(env)[i] -= sell
            gain = _prices(env)[i] * sell
            cost = gain * env.TRANSACTION_FEE_PERCENT
            _balance(env)[] += gain - cost
            env.total_cost += cost
        end
    end

    # then buy
    # better to shuffle?
    for (i,b) in enumerate(actions)
        if b > 0
            max_buy = div(_balance(env)[], _prices(env)[i])
            buy = min(b*env.HMAX_NORMALIZE, max_buy)
            _holds(env)[i] += buy
            deduction = buy * _prices(env)[i]
            cost = deduction * env.TRANSACTION_FEE_PERCENT
            _balance(env)[] -= deduction + cost
            env.total_cost += cost
        end
    end

    env.day += 1
    _prices(env) .= @view env.prices[:, env.day]
    _features(env) .= @view env.features[:, env.day]

    env.daily_reward = _total_asset(env) - init_asset
end

RLBase.reward(env::StockTradingEnv) = env.daily_reward * env.REWARD_SCALING
RLBase.is_terminated(env::StockTradingEnv) = env.day >= env.last_day
RLBase.state(env::StockTradingEnv) = env.state

function RLBase.reset!(env::StockTradingEnv)
    env.day = env.first_day
    _balance(env)[] = env.initial_account_balance
    _prices(env) .= @view env.prices[:, env.day]
    _features(env) .= @view env.features[:, env.day]
    env.total_cost = 0.
    env.daily_reward = 0.
end

RLBase.state_space(env::StockTradingEnv) = Space(fill(-Inf32..Inf32, length(state(env))))
RLBase.action_space(env::StockTradingEnv) = Space(fill(-1f0..1f0, length(_holds(env))))

RLBase.ChanceStyle(::StockTradingEnv) = DETERMINISTIC

# wrapper

struct StockTradingEnvWithTurbulence{E<:StockTradingEnv} <: AbstractEnvWrapper
    env::E
    turbulences::Vector{Float64}
    turbulence_threshold::Float64
end

function StockTradingEnvWithTurbulence(;
    turbulence_threshold=140.,
    turbulences=nothing,
    kw...
)
    turbulences = isnothing(turbulences) && load_default_stock_data("turbulence.csv")

    StockTradingEnvWithTurbulence(
        StockTradingEnv(;kw...),
        turbulences,
        turbulence_threshold
    )
end

function (w::StockTradingEnvWithTurbulence)(actions)
    if w.turbulences[w.env.day] >= w.turbulence_threshold
        actions .= ifelse.(actions .< 0, -Inf32, 0)
    end
    w.env(actions)
end