export DynamicDistributionModel, get_states, get_actions

import ReinforcementLearningEnvironments: observation_space, action_space

"""
    DynamicDistributionModel(f::Tf, ns::Int, na::Int) -> DynamicDistributionModel{Tf}

Use a general function `f` to store the transformations. `ns` and `na` are the number of states and actions.
"""
struct DynamicDistributionModel{Tf<:Function} <: AbstractDistributionBasedModel
    f::Tf
    ns::Int
    na::Int
end

observation_space(m::DynamicDistributionModel) = DiscreteSpace(m.ns)
action_space(m::DynamicDistributionModel) = DiscreteSpace(m.na)

(m::DynamicDistributionModel)(s, a) = m.f(s, a)