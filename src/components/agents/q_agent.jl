export QAgent

struct QAgent{Tl<:AbstractLearner, Tb<:AbstractTurnBuffer} <: AbstractAgent
    role::String
    learner::Tl
    buffer::Tb
end

function (agent::QAgent)(s)
end