export AbstractPolicy

"""
A policy is a functional object to generate an action given a state.
"""
abstract type AbstractPolicy end

learner(π::AbstractPolicy) = π.learner

function update!(π::AbstractPolicy, buffer::AbstractTurnBuffer; kw...)
    transitions = extract_transitions(buffer, π)
    if !isnothing(transitions)
        update!(learner(π), transitions; kw...)
    end
end