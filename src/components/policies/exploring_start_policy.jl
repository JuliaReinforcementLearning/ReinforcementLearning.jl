export ExploringStartPolicy

Base.@kwdef mutable struct ExploringStartPolicy{P,A} <: AbstractPolicy
    π::P
    actions::A
end

(π::ExploringStartPolicy)(obs::Observation) = get_terminal(obs) ? rand(π.actions) : π.π(obs)

learner(π::ExploringStartPolicy) = learner(π.π)

extract_transitions(buffer, π::ExploringStartPolicy) = extract_transitions(buffer, π.π)