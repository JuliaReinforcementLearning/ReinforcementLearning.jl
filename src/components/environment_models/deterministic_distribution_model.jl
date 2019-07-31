export DeterministicDistributionModel, get_states, get_actions

"""
Store all the transformations in the `table` field.
"""
 struct DeterministicDistributionModel <: AbstractDistributionBasedModel
    table::Array{Vector{NamedTuple{(:nextstate, :reward, :prob), Tuple{Int, Float64, Float64}}}, 2}
 end

get_states(m::DeterministicDistributionModel) = axes(m.table, 1)
get_actions(m::DeterministicDistributionModel) = axes(m.table, 2)

(m::DeterministicDistributionModel)(s::Int, a::Int) =  m.table[s, a]