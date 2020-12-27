export TabularLearner

"""
    TabularLearner{S, T}

Use a `Dict{S,Vector{T}}` to store action probabilities. `S` is the type of
state. `T` is the element type of probabilities.
"""
struct TabularLearner{S,T} <: AbstractLearner
    table::Dict{S,Vector{T}}
end

TabularLearner() = TabularLearner{Int,Float32}()
TabularLearner{S}() where {S} = TabularLearner{S,Float32}()
TabularLearner{S,T}() where {S,T} = TabularLearner(Dict{S,Vector{T}}())

(p::TabularLearner)(env::AbstractEnv) = p(ChanceStyle(env), env)

function (p::TabularLearner)(::ExplicitStochastic, env::AbstractEnv)
    if current_player(env) == chance_player(env)
        prob(env)
    else
        p(DETERMINISTIC, env)  # treat it just like a normal one
    end
end

function (t::TabularLearner)(::RLBase.AbstractChanceStyle, env::AbstractEnv)
    t(ActionStyle(env), env)
end

function (t::TabularLearner)(::FullActionSet, env::AbstractEnv)
    get!(t.table, state(env)) do
        m = legal_action_space_mask(env)
        m ./ sum(m)
    end
end

function (t::TabularLearner)(::MinimalActionSet, env::AbstractEnv)
    get!(t.table, state(env)) do
        n = length(action_space(env))
        fill(1 / n, n)
    end
end

"""
    update!(p::TabularLearner, state => prob)

!!! warn
    For environments of `FULL_ACTION_SET`, `prob` represents the probability
    distribution of `legal_action_space(env)`. For environments of
    `MINIMAL_ACTION_SET`, `prob` should represent the probability distribution
    of `action_space(env)`.
"""
RLBase.update!(p::TabularLearner, experience::Pair) =
    p.table[first(experience)] = last(experience)
