export BehaviorCloningPolicy

"""
    BehaviorCloningPolicy(;kw...)

# Keyword Arguments

- `approximator`: calculate the logits of possible actions directly
- `explorer=GreedyExplorer()` 

"""
Base.@kwdef struct BehaviorCloningPolicy{A} <: AbstractPolicy
    approximator::A
    explorer::Any = GreedyExplorer()
end

function (p::BehaviorCloningPolicy)(env::AbstractEnv)
    s = state(env)
    s_batch = Flux.unsqueeze(s, ndims(s) + 1)
    logits = p.approximator(s_batch) |> vec  # drop dimension
    p.explorer(logits)
end

function RLBase.update!(p::BehaviorCloningPolicy, batch::NamedTuple{(:state, :action)})
    s, a = batch.state, batch.action
    m = p.approximator
    gs = gradient(params(m)) do
        ŷ = m(s)
        y = Flux.onehotbatch(a, axes(ŷ, 1))
        logitcrossentropy(ŷ, y)
    end
    update!(m, gs)
end