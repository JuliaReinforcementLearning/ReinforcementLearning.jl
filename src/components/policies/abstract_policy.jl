export AbstractPolicy

abstract type AbstractPolicy end

learner(π::AbstractPolicy) = π.learner

function update!(π::AbstractPolicy, buffer::AbstractTurnBuffer; kw...)
    transitions = extract_transitions(buffer, π)
    if !isnothing(transitions)
        update!(learner(π), transitions; kw...)
    end
end