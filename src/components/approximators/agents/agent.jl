export Agent

struct Agent{Tl<:AbstractLearner, Tb<:AbstractTurnBuffer} <: AbstractAgent
    role::String
    learner::Tl
    buffer::Tb
end