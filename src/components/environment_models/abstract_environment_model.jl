export AbstractEnvironmentModel, AbstractSampleBasedModel, AbstractDistributionBasedModel

"""
Describe how to model a reinforcement learning environment.

See also [`AbstractDistributionBasedModel`](@ref), [`AbstractSampleBasedModel`](@ref).
"""
abstract type AbstractEnvironmentModel end

"""
A collection of models that can be used to sample transitions.
"""
abstract type AbstractSampleBasedModel <: AbstractEnvironmentModel end

"""
A collection of models that can get the distribution given a state and an actioi.
"""
abstract type AbstractDistributionBasedModel <: AbstractEnvironmentModel end
