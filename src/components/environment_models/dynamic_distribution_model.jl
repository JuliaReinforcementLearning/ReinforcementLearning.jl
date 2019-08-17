export DynamicDistributionModel, get_states, get_actions

"""
Using a general function `f` to store the transformations.
"""
struct DynamicDistributionModel{Tf<:Function} <: AbstractDistributionBasedModel
    f::Tf
    ns::Int
    na::Int
end

get_states(m::DynamicDistributionModel) = 1:m.ns
get_actions(m::DynamicDistributionModel) = 1:m.na

(m::DynamicDistributionModel)(s, a) = m.f(s, a)