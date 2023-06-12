export BehaviorCloningPolicy

using Flux
using Flux: gradient, params

mutable struct BehaviorCloningPolicy{A} <: AbstractPolicy
    approximator::A
    explorer::AbstractExplorer
    sampler::BatchSampler{(:state, :action)}
    min_reservoir_history::Int
end

"""
    BehaviorCloningPolicy(;kw...)

# Keyword Arguments

- `approximator`: calculate the logits of possible actions directly
- `explorer=GreedyExplorer()` 
- `batch_size::Int = 32`
- `min_reservoir_history::Int = 100`, number of transitions that should be experienced before updating the `approximator`. 
- `rng = Random.default_rng()`
"""
function BehaviorCloningPolicy(;
    approximator::A,
    explorer::AbstractExplorer=GreedyExplorer(),
    batch_size::Int=32,
    min_reservoir_history::Int=100,
    rng=Random.default_rng()
) where {A}
    sampler = BatchSampler{(:state, :action)}(batch_size; rng=rng)
    BehaviorCloningPolicy(approximator, explorer, sampler, min_reservoir_history)
end

function RLBase.plan!(p::BehaviorCloningPolicy, env::AbstractEnv)
    s = state(env)
    s_batch = Flux.unsqueeze(s, dims=ndims(s) + 1)
    s_batch = send_to_device(device(p.approximator), s_batch)
    logits = p.approximator(s_batch) |> vec |> send_to_host # drop dimension
    typeof(ActionStyle(env)) == MinimalActionSet ? p.explorer(logits) :
    p.explorer(logits, legal_action_space_mask(env))
end

function RLCore.update!(p::BehaviorCloningPolicy, batch::NamedTuple{(:state, :action)})
    s = send_to_device(device(p.approximator), batch.state)
    a = send_to_device(device(p.approximator), batch.action)
    m = p.approximator
    gs = gradient(params(m)) do
        ŷ = m(s)
        y = Flux.OneHotMatrix(a, size(ŷ, 1))
        logitcrossentropy(ŷ, y)
    end
    update!(m, gs)
end

function RLCore.update!(p::BehaviorCloningPolicy, t::Any)
    (length(t) <= p.min_reservoir_history || length(t) <= p.sampler.batch_size) && return

    _, batch = p.sampler(t)
    update!(p, batch)
end

function RLBase.prob(p::BehaviorCloningPolicy, env::AbstractEnv)
    s = state(env)
    m = p.approximator
    s_batch = send_to_device(device(m), Flux.unsqueeze(s, dims=ndims(s) + 1))
    values = m(s_batch) |> vec |> send_to_host
    typeof(ActionStyle(env)) == MinimalActionSet ? prob(p.explorer, values) :
    prob(p.explorer, values, legal_action_space_mask(env))
end

function RLBase.prob(p::BehaviorCloningPolicy, env::AbstractEnv, action)
    A = action_space(env)
    P = prob(p, env)
    @assert length(A) == length(P)
    if A isa Base.OneTo
        P[action]
    else
        for (a, p) in zip(A, P)
            if a == action
                return p
            end
        end
        @error "action[$action] is not found in action space[$(action_space(env))]"
    end
end
