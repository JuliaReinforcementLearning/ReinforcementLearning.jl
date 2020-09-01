export TabularLearner

"""
    TabularLearner{S, T}

Use a `Dict{S,Vector{T}}` to store action probabilities.
"""
struct TabularLearner{S,T} <: AbstractPolicy
    table::Dict{S,Vector{T}}
end

TabularLearner() = TabularLearner{Int,Float32}()
TabularLearner{S}() where S = TabularLearner{S,Float32}()
TabularLearner{S,T}() where {S,T} = TabularLearner(Dict{S,Vector{T}}())

function (p::TabularLearner)(env::AbstractEnv)
    s = get_state(env)
    if haskey(p.table, s)
        p.table[s]
    elseif ActionStyle(env) === FULL_ACTION_SET
        mask = get_legal_actions_mask(env)
        prob = mask ./ sum(mask)
        p.table[s] = prob
        prob
    elseif ActionStyle(env) === MINIMAL_ACTION_SET
        n = length(get_actions(env))
        prob = fill(1 / n, n)
        p.table[s] = prob
        prob
    end
end

RLBase.update!(p::TabularLearner, experience::Pair) = p.table[first(experience)] = last(experience)

