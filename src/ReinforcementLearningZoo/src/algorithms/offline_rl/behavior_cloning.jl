export BehaviorCloningPolicy

mutable struct BehaviorCloningPolicy{A} <: AbstractPolicy
    approximator::A
    explorer::Any
    sampler::BatchSampler{(:state, :action)}
    min_reservoir_history::Int
    update_freq::Int
    update_step::Int
end

"""
    BehaviorCloningPolicy(;kw...)

# Keyword Arguments

- `approximator`: calculate the logits of possible actions directly
- `explorer=GreedyExplorer()` 
- `batch_size::Int = 32`
- `min_reservoir_history::Int = 100`, number of transitions that should be experienced before updating the `approximator`. 
- `update_freq::Int = 1`: the frequency of updating the `approximator`.
- `rng = Random.GLOBAL_RNG`
"""
function BehaviorCloningPolicy(;
        approximator::A,
        explorer::Any = GreedyExplorer(),
        batch_size::Int = 32,
        min_reservoir_history::Int = 100,
        update_freq::Int = 1,
        rng = Random.GLOBAL_RNG
) where {A}
    sampler = BatchSampler{(:state, :action)}(batch_size; rng = rng)
    BehaviorCloningPolicy(
        approximator,
        explorer,
        sampler,
        min_reservoir_history,
        update_freq,
        0,
    )
end

function (p::BehaviorCloningPolicy)(env::AbstractEnv)
    s = state(env)
    s_batch = Flux.unsqueeze(s, ndims(s) + 1)
    s_batch = send_to_device(device(p.approximator), s_batch)
    logits = p.approximator(s_batch) |> vec |> send_to_host # drop dimension
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

function RLBase.update!(p::BehaviorCloningPolicy, t::AbstractTrajectory)
    length(t) <= p.min_reservoir_history && return

    p.update_step += 1
    p.update_step % p.update_freq == 0 || return

    _, batch = p.sampler(t)
    RLBase.update!(p, send_to_device(device(p.approximator), batch))
end