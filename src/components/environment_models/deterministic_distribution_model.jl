export DeterministicDistributionModel, get_states, get_actions

import ReinforcementLearningEnvironments: observation_space, action_space

"""
    DeterministicDistributionModel(table::Array{Vector{NamedTuple{(:nextstate, :reward, :prob),Tuple{Int,Float64,Float64}}}, 2})

Store all the transformations in the `table` field.
"""
struct DeterministicDistributionModel <: AbstractDistributionBasedModel
    table::Array{
        Vector{NamedTuple{(:nextstate, :reward, :prob),Tuple{Int,Float64,Float64}}},
        2,
    }
end

observation_space(m::DeterministicDistributionModel) = DiscreteSpace(size(m.table, 1))
action_space(m::DeterministicDistributionModel) = DiscreteSpace(size(m.table, 2))

(m::DeterministicDistributionModel)(s::Int, a::Int) = m.table[s, a]