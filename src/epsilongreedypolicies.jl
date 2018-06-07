abstract type AbstractEpsilonGreedyPolicy end
@subtypes(AbstractEpsilonGreedyPolicy,
          begin; ϵ::Float64; end,
          OptimisticEpsilonGreedyPolicy,
          VeryOptimisticEpsilonGreedyPolicy,
          PesimisticEpsilonGreedyPolicy)
          

for (typ, max, rel) in ((OptimisticEpsilonGreedyPolicy, maximumbelowInf, :(>=)),
                        (VeryOptimisticEpsilonGreedyPolicy, maximum, :(==)),
                        (PesimisticEpsilonGreedyPolicy, maximumbelowInf, :(==)))
    @eval function getgreedystates(policy::$typ, values)
        a = Int64[]
        vmax = $max(values)
        if isnan(vmax)
            error("NaN encountered in getgreedystates: $values")
        end
        for (i, v) in enumerate(values)
            if ($rel)(v, vmax)
                push!(a, i)
            end
        end
        a
    end
end

const EpsilonGreedyPolicy = VeryOptimisticEpsilonGreedyPolicy
export EpsilonGreedyPolicy

@doc """
    mutable struct EpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy
        ϵ::Float64

Chooses the action with the highest value with probability 1 - ϵ and selects an
action uniformly random with probability ϵ. For states with actions that where
never performed before, the behavior of the
[`VeryOptimisticEpsilonGreedyPolicy`](@ref) is followed.
""" EpsilonGreedyPolicy

@doc """
    mutable struct VeryOptimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy
        ϵ::Float64

[`EpsilonGreedyPolicy`](@ref) that samples uniformly from novel actions in each
state where actions are available that where never chosen before. See also 
[Initial values, novel actions and unseen values](@ref initunseen).
""" VeryOptimisticEpsilonGreedyPolicy

@doc """
    mutable struct OptimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy
        ϵ::Float64

[`EpsilonGreedyPolicy`](@ref) that samples uniformly from the actions with the
highest Q-value and novel actions in each state where actions are available that
where never chosen before. 
""" OptimisticEpsilonGreedyPolicy

@doc """
    mutable struct PesimisticEpsilonGreedyPolicy <: AbstractEpsilonGreedyPolicy
        ϵ::Float64

[`EpsilonGreedyPolicy`](@ref) that does not handle novel actions differently.
""" PesimisticEpsilonGreedyPolicy


function selectaction(policy::AbstractEpsilonGreedyPolicy, values)
    if rand() < policy.ϵ
        rand(1:length(values))
    else
        rand(getgreedystates(policy, values))
    end
end

function getactionprobabilities(policy::AbstractEpsilonGreedyPolicy, values)
    p = ones(length(values))/length(values) * policy.ϵ
    a = getgreedystates(policy, values)
    p2 = (1. - policy.ϵ)/length(a)
    for i in a
        p[i] =+ p2
    end
    p
end

