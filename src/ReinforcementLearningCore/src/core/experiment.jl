export @experiment_cmd, @E_cmd, Experiment

using BSON
using Markdown

Base.@kwdef mutable struct Experiment
    agent::Any
    env::Any
    stop_condition::Any
    hook::Any
    description::String
end

macro experiment_cmd(s)
    Experiment(s)
end

# alias for experiment_cmd
macro E_cmd(s)
    Experiment(s)
end

function Experiment(s::String)
    m = match(r"(?<source>\w+)_(?<method>\w+)_(?<env>\w+)(\((?<game>\w*)\))?", s)
    isnothing(m) &&
        throw(ArgumentError("invalid format, got $s, expected format is a local dir or a predefined experiment like dopamine_dqn_atari(pong)`"))
    Experiment(
        Val(Symbol(m[:source])),
        Val(Symbol(m[:method])),
        Val(Symbol(m[:env])),
        m[:game],
    )
end

function Base.run(x::Experiment)
    display(Markdown.parse(x.description))
    run(x.agent, x.env, x.stop_condition, x.hook)
    x
end
