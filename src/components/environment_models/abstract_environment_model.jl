export AbstractEnvironmentModel, AbstractSampleBasedModel, AbstractDistributionBasedModel

abstract type AbstractEnvironmentModel end

abstract type AbstractSampleBasedModel <: AbstractEnvironmentModel end

abstract type AbstractDistributionBasedModel <: AbstractEnvironmentModel end
