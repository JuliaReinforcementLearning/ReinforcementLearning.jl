export OffPolicy

using MacroTools: @forward

"""
    OffPolicy(π_target::P, π_behavior::B) -> OffPolicy{P,B}
"""
Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
    π_target::P
    π_behavior::B
end

(π::OffPolicy)(env) = π.π_behavior(env)

@forward OffPolicy.π_behavior RLBase.get_priority, RLBase.get_prob
