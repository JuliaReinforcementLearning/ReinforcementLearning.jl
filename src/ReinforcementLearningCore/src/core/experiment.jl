export @experiment_cmd, @E_cmd, Experiment

using Markdown
using Dates

"""
    Experiment(policy, env, stop_condition, hook, description)

These are the four essential components in a typical reinforcement learning experiment:

- `policy`, generates an action during the interaction with the `env`. It may update its strategy in the meanwhile.
- `env`, the environment we're going to experiment with.
- `stop_condition`, defines the when the experiment terminates.
- `hook`, collects some intermediate data during the experiment.
- `description`, displays some useful information for logging.
"""
Base.@kwdef mutable struct Experiment
    policy::Any
    env::Any
    stop_condition::Any
    hook::Any
    description::String = "Experiment created at $(now())"
end

function Base.show(io::IO, x::Experiment)
    display(Markdown.parse(x.description))
    AbstractTrees.print_tree(io, StructTree(x), maxdepth=get(io, :max_depth, 10))
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
    isnothing(m) && throw(
        ArgumentError(
            "invalid format, got $s, expected format is a local dir or a predefined experiment like dopamine_dqn_atari(pong)`",
        ),
    )
    Experiment(
        Val(Symbol(m[:source])),
        Val(Symbol(m[:method])),
        Val(Symbol(m[:env])),
        m[:game],
    )
end

function Base.run(x::Experiment; describe::Bool=true)
    describe && display(Markdown.parse(x.description))
    run(x.policy, x.env, x.stop_condition, x.hook)
    x
end
