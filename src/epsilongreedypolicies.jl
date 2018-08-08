abstract type AbstractEpsilonGreedyPolicy end
@subtypes(AbstractEpsilonGreedyPolicy,
          begin; ϵ::Float64; end,
          OptimisticEpsilonGreedyPolicy,
          VeryOptimisticEpsilonGreedyPolicy,
          PesimisticEpsilonGreedyPolicy)
          
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


for (typ, max, rel) in ((OptimisticEpsilonGreedyPolicy, maximumbelowInf, :(>=)),
                        (VeryOptimisticEpsilonGreedyPolicy, maximum, :(==)),
                        (PesimisticEpsilonGreedyPolicy, maximumbelowInf, :(==)))
    @eval function selectaction(policy::$typ, values)
        if rand() < policy.ϵ
            rand(1:length(values))
        else
            vmax = $max(values)
            c = 1
            a = 1
            for (i, v) in enumerate(values)
                if ($rel)(v, vmax)
                    if rand() < 1/c
                        a = i
                    end
                    c += 1
                end
            end
            a
        end
    end
    @eval function getactionprobabilities(policy::$typ, values)
        p = ones(length(values))/length(values) * policy.ϵ
        vmax = $max(values)
        c = 0
        for v in values
            if ($rel)(v, vmax)
                c += 1
            end
        end
        p2 = (1. - policy.ϵ)/c
        for (i, v) in enumerate(values)
            if ($rel)(v, vmax)
                p[i] += p2
            end
        end
        p
    end
end

