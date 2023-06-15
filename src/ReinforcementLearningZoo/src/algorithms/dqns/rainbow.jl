export RainbowLearner

import Random
using Flux: params, unsqueeze, softmax, gradient
using Flux.Losses: logitcrossentropy
using Functors: @functor

Base.@kwdef mutable struct RainbowLearner{A<:Approximator{<:TwinNetwork}, F, R} <: AbstractLearner
    approximator::A
    Vₘₐₓ::Float32
    Vₘᵢₙ::Float32
    n_actions::Int
    γ::Float32
    update_horizon::Int = 1
    n_atoms::Int = 51
    support::AbstractVector = range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length=n_atoms)
    delta_z::Float32 = convert(Float32, support.step)
    default_priority::Float32 = 1.0f2
    β_priority::Float32 = 0.5f0
    loss_func::F = (ŷ, y) -> logitcrossentropy(ŷ, y; agg=identity)
    rng::R = Random.default_rng()
    # for logging
    loss::Float32 = 0.0f0
end

@functor RainbowLearner (support, approximator)

function RLCore.forward(L::RainbowLearner, s::A) where {A<:AbstractArray}
    logits = RLCore.forward(L.approximator, s)
    q = L.support .* softmax(reshape(logits, :, L.n_actions))
    sum(q, dims=1) |> vec
end

function RLBase.plan!(learner::RainbowLearner, env::AbstractEnv)
    s = send_to_device(device(learner.approximator), state(env))
    s = unsqueeze(s, dims=ndims(s) + 1)
    s |> learner |> vec |> send_to_host
end

function RLBase.optimise!(learner::RainbowLearner, batch::NamedTuple)
    A = learner.approximator
    Q = A.model.source
    Qₜ = A.model.target
    γ = learner.γ
    β = learner.β_priority
    loss_func = learner.loss_func
    n_atoms = learner.n_atoms
    n_actions = learner.n_actions
    support = learner.support
    delta_z = learner.delta_z
    update_horizon = learner.update_horizon

    D = device(Q)
    states = send_to_device(D, batch.state)
    rewards = send_to_device(D, batch.reward)
    terminals = send_to_device(D, batch.terminal)
    next_states = send_to_device(D, batch.next_state)

    batch_size = length(terminals)
    actions = CartesianIndex.(batch.action, 1:batch_size)

    target_support =
        reshape(rewards, 1, :) .+
        (reshape(support, :, 1) * reshape((γ^update_horizon) .* (1 .- terminals), 1, :))

    next_logits = Qₜ(next_states)
    next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)

    next_q = reshape(sum(support .* next_probs, dims=1), n_actions, :)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        next_q .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    next_prob_select = select_best_probs(next_probs, next_q)

    target_distribution = project_distribution(
        target_support,
        next_prob_select,
        support,
        delta_z,
        learner.Vₘᵢₙ,
        learner.Vₘₐₓ,
    )

    is_use_PER = haskey(batch, :priority)  # is use Prioritized Experience Replay

    if is_use_PER
        updated_priorities = Vector{Float32}(undef, batch_size)
        weights = 1.0f0 ./ ((batch.priority .+ 1.0f-10) .^ β)
        weights ./= maximum(weights)
        weights = send_to_device(D, weights)
        # TODO: init on device directly
    end

    gs = gradient(params(A)) do
        logits = reshape(Q(states), n_atoms, n_actions, :)
        select_logits = logits[:, actions]
        # The original paper normalized logits, but using normalization and Flux.Losses.crossentropy is not as stable as using Flux.Losses.logitcrossentropy.
        batch_losses = loss_func(select_logits, target_distribution)
        loss = is_use_PER ? dot(vec(weights), vec(batch_losses)) * 1 // batch_size : mean(batch_losses)
        ignore_derivatives() do
            if is_use_PER
                updated_priorities .= send_to_host(vec((batch_losses .+ 1.0f-10) .^ β))
            end
            learner.loss = loss
        end
        loss
    end

    RLBase.optimise!(A, gs)

    is_use_PER ? batch.key => updated_priorities : nothing
end

@inline function select_best_probs(probs, q)
    q_argmax = argmax(q, dims=1)
    prob_select = @inbounds probs[:, q_argmax] # !!! without @inbounds it would be really slow
    reshape(prob_select, :, length(q_argmax))
end

function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(
        repeat(clampped_support; outer=(n_atoms, 1)),
        n_atoms,
        n_atoms,
        batch_size,
    )

    projection =
        clamp.(
            1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z,
            0,
            1,
        ) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims=1), n_atoms, batch_size)
end

function RLBase.optimise!(policy::QBasedPolicy{<:RainbowLearner}, ::PostActStage, trajectory::Trajectory)
    for batch in trajectory
        res = RLBase.optimise!(policy, PostActStage(), batch) |> send_to_host
        if !isnothing(res)
            k, p = res
            trajectory[:priority, k] = p
        end
    end
end
