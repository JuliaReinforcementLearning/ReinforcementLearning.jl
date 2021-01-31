export TabularRandomPolicy

"""
    TabularRandomPolicy(;table=Dict{Int, Float32}(), rng=Random.GLOBAL_RNG)

Use a `Dict` to store action distribution.
"""
Base.@kwdef struct TabularRandomPolicy{S,T,R} <: AbstractPolicy
    table::Dict{S,T} = Dict{Any,Vector{Float32}}()
    rng::R = Random.GLOBAL_RNG
end

TabularRandomPolicy{S}(; rng = Random.GLOBAL_RNG) where {S} =
    TabularRandomPolicy{S,Vector{Float32}}(; rng = rng)
TabularRandomPolicy{S,T}(; rng = Random.GLOBAL_RNG) where {S,T} =
    TabularRandomPolicy(Dict{S,T}(), rng)

RLBase.prob(p::TabularRandomPolicy, env::AbstractEnv) = prob(p, ChanceStyle(env), env)

function RLBase.prob(p::TabularRandomPolicy, ::ExplicitStochastic, env::AbstractEnv)
    if current_player(env) == chance_player(env)
        prob(env)
    else
        prob(p, DETERMINISTIC, env)  # treat it just like a normal one
    end
end

function RLBase.prob(t::TabularRandomPolicy, ::RLBase.AbstractChanceStyle, env::AbstractEnv)
    prob(t, ActionStyle(env), env)
end

function RLBase.prob(t::TabularRandomPolicy, ::FullActionSet, env::AbstractEnv)
    get!(t.table, state(env)) do
        m = legal_action_space_mask(env)
        m ./ sum(m)
    end
end

function RLBase.prob(t::TabularRandomPolicy, ::MinimalActionSet, env::AbstractEnv)
    get!(t.table, state(env)) do
        n = length(action_space(env))
        fill(1 / n, n)
    end
end

function RLBase.prob(t::TabularRandomPolicy, env::AbstractEnv, action)
    prob(t, env, action_space(env), action)
end

function RLBase.prob(t::TabularRandomPolicy, env::AbstractEnv, action_space, action)
    prob(t, env)[findfirst(==(action), action_space)]
end

function RLBase.prob(
    t::TabularRandomPolicy,
    env::AbstractEnv,
    action_space::Base.OneTo,
    action,
)
    prob(t, env)[action]
end

# function RLBase.prob(t::TabularRandomPolicy, env::AbstractEnv, action_space::ZeroTo, action)
#     prob(t, env)[action+1]
# end

function RLBase.prob(t::TabularRandomPolicy, state, action)
    # assume table is already initialized
    t.table[state][action]
end

(p::TabularRandomPolicy)(env::AbstractEnv) =
    sample(p.rng, action_space(env), Weights(prob(p, env), 1.0))

# !!! Assumeing table is already initialized
(p::TabularRandomPolicy{S})(state::S) where {S} =
    sample(p.rng, Weights(p.table[state], 1.0))

"""
    update!(p::TabularRandomPolicy, state => value)

You should manually check `value` sum to `1.0`.
"""
function RLBase.update!(p::TabularRandomPolicy, experience::Pair)
    s, dist = experience
    if haskey(p.table, s)
        p.table[s] .= dist
    else
        p.table[s] = dist
    end
end
