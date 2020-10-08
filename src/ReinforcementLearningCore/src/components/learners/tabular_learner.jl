export TabularLearner

"""
    TabularLearner{S, T}

Use a `Dict{S,Vector{T}}` to store action probabilities.
"""
struct TabularLearner{S,T} <: AbstractLearner
    table::Dict{S,Vector{T}}
end

TabularLearner() = TabularLearner{Int,Float32}()
TabularLearner{S}() where {S} = TabularLearner{S,Float32}()
TabularLearner{S,T}() where {S,T} = TabularLearner(Dict{S,Vector{T}}())

(p::TabularLearner)(env::AbstractEnv) = p(ChanceStyle(env), env)

function (p::TabularLearner)(::ExplicitStochastic, env::AbstractEnv)
    if get_current_player(env) == get_chance_player(env)
        [a.prob for a::ActionProbPair in get_actions(env)]
    else
        p(DETERMINISTIC, env)  # treat it just like a normal one
    end
end

function (p::TabularLearner)(::RLBase.AbstractChanceStyle, env::AbstractEnv)
    get!(p.table, get_state(env)) do 
        n = length(get_legal_actions(env))
        fill(1/n, n)
    end
end

RLBase.update!(p::TabularLearner, experience::Pair) =
    p.table[first(experience)] = last(experience)
