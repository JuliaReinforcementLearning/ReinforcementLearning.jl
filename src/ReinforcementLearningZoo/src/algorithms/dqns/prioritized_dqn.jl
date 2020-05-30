export PrioritizedDQNLearner

using Random
using Flux
using Zygote
using StatsBase: mean
using LinearAlgebra: dot

"""
    PrioritizedDQNLearner(;kwargs...)

See paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
And also https://danieltakeshi.github.io/2019/07/14/per/

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_func`: the loss function.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `default_priority::Float64=100.`: the default priority for newly added transitions.
- `seed = nothing`
"""
mutable struct PrioritizedDQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    S<:Union{Int,Nothing},
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_func::Tf
    stack_size::S
    γ::Float32
    batch_size::Int
    update_horizon::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    default_priority::Float32
    β_priority::Float32
    rng::R
    loss::Float32
end

function PrioritizedDQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
    loss_func::Tf,
    stack_size::Union{Int,Nothing} = 4,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    target_update_freq::Int = 100,
    update_step::Int = 0,
    default_priority::Float32 = 100f0,
    β_priority::Float32 = 0.5f0,
    seed = nothing,
) where {Tq,Tt,Tf}
    copyto!(approximator, target_approximator)
    rng = MersenneTwister(seed)
    PrioritizedDQNLearner(
        approximator,
        target_approximator,
        loss_func,
        stack_size,
        γ,
        batch_size,
        update_horizon,
        min_replay_history,
        update_freq,
        target_update_freq,
        update_step,
        default_priority,
        β_priority,
        rng,
        0.f0,
    )
end


Flux.functor(x::PrioritizedDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

"""

!!! note
    The state of the observation is assumed to have been stacked,
    if `!isnothing(stack_size)`.
"""
(learner::PrioritizedDQNLearner)(obs) =
    obs |>
    get_state |>
    x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>
        x ->
            send_to_device(device(learner.approximator), x) |>
            learner.approximator |>
            send_to_host |>
            Flux.squeezebatch

function RLBase.update!(learner::PrioritizedDQNLearner, batch::NamedTuple)
    Q, Qₜ, γ, β, loss_func, update_horizon, batch_size = learner.approximator,
    learner.target_approximator,
    learner.γ,
    learner.β_priority,
    learner.loss_func,
    learner.update_horizon,
    learner.batch_size
    D = device(Q)
    states, rewards, terminals, next_states = map(
        x -> send_to_device(D, x),
        (batch.states, batch.rewards, batch.terminals, batch.next_states),
    )
    actions = CartesianIndex.(batch.actions, 1:batch_size)

    updated_priorities = Vector{Float32}(undef, batch_size)
    weights = 1f0 ./ ((batch.priorities .+ 1f-10) .^ β)
    weights ./= maximum(weights)
    weights = send_to_device(D, weights)

    gs = gradient(params(Q)) do
        q = Q(states)[actions]
        q′ = dropdims(maximum(Qₜ(next_states); dims = 1), dims = 1)
        G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

        batch_losses = loss_func(G, q)
        loss = dot(vec(weights), vec(batch_losses)) / batch_size
        ignore() do
            updated_priorities .= send_to_host(vec((batch_losses .+ 1f-10) .^ β))
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
    updated_priorities
end
