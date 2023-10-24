export TRPO

using Random: GLOBAL_RNG, shuffle, AbstractRNG
using Functors: @functor
using Flux: Flux, destructure, params, gradient
using StatsBase: mean
using ChainRulesCore: ignore_derivatives

"""
Trust Region Policy Optimization
"""
Base.@kwdef struct TRPO{A,B,D} <: AbstractPolicy
    approximator::A
    baseline::B
    "only a discrete action space is supported for now"
    dist::D = Distributions.Categorical
    γ::Float32 = 0.99f0
    batchsize::Int = 1024
    rng::AbstractRNG = GLOBAL_RNG
    max_backtrack_step::Int = 10
    kldivergence_limit::Float32 = 1f-2
    backtrack_coeff::Float32 = 0.8f0
    cg_max_iter::Int = 10
    cg_ϵ::Float32 = 1f-10
    damping_coeff = 1f-2
end

IsPolicyGradient(::Type{<:TRPO}) = IsPolicyGradient()
@functor TRPO (approximator, baseline)

function RLBase.plan!(π::TRPO, env::AbstractEnv)
    res = env |> state |> send_to_device(π) |> π.approximator |> send_to_host
    rand(π.rng, action_distribution(π.dist, res)[1])
end

function Base.push!(p::Agent{<:TRPO}, ::PostEpisodeStage, env::AbstractEnv)
    p.trajectory.container[] = true
    RLBase.optimise!(p.policy, p.trajectory.container)
    empty!(p.trajectory.container)
end

function RLBase.optimise!(p::TRPO, ::PostEpisodeStage, trajectory::Trajectory)
    has_optimized = false
    for batch in trajectory #batch is a vector of Episode
        gains = vcat(discount_rewards(ep[:reward], p.γ) for ep in batch)
        states = reduce(ep[:state] for ep in batch) do s, s2 
            cat(s,s2, dims = ndims(first(batch[:state])))
        end
        actions = reduce(ep[:action] for ep in batch) do s, s2
            cat(s, s2, dims = ndims(first(batch[:action])))
        end
        for inds in Iterators.partition(shuffle(p.rng, eachindex(gains)), p.batchsize)
            RLBase.optimise!(p, (state=selectdim(states,ndims(states),inds), action=selectdim(actions,ndims(actions),inds), gain=gains[inds]))
        end
        has_optimized = true
    end
    has_optimized && empty!(trajectory.container)
    return nothing
end

function RLBase.optimise!(p::TRPO, ::PostActStage, batch::NamedTuple{(:state, :action, :gain)})
    A = p.approximator
    B = p.baseline
    s, a, g = map(Array, batch) # !!! FIXME

    # fit value network on mean-square
    # store δ for advantage estimate
    if isnothing(B)
        δ = normalise(g)
    else
        gs = gradient(params(B)) do
            δ = g - vec(B(s))
            loss = mean(δ .^ 2)
            ignore_derivatives() do
                # @info "TRPO/baseline" loss = loss δ
            end
            loss
        end
        RLBase.optimise!(B, gs)
    end
    
    # store logits as intermediate value
    old_logits = Ref{Matrix{Float32}}()

    gps = gradient(params(A.model)) do
        old_logits[] = A.model(s)
        total_loss = map(eachcol(softmax(old_logits[])), a) do x, y
            x[y]
        end .* δ
        loss = mean(total_loss)
        loss
    end

    ĝₖ = mapreduce(vec, vcat, gps)
    θₖ, re = Flux.destructure(A.model)

    # solve Hx̂ₖ = ĝₖ approx. with conjugate gradient
    hes(x) = hvp(A.model, s, old_logits[], x; damping_coeff = p.damping_coeff)
    x̂ₖ = conjugate_gradient(hes, ĝₖ; max_iter = p.cg_max_iter, ϵ = p.cg_ϵ)

    # backtracking line search

    # set-up
    search_direction = sqrt(2*p.kldivergence_limit / (x̂ₖ' * ĝₖ)) .* x̂ₖ
    Δ = copy(search_direction)
    local θ

    # search
    for _ in 1:p.max_backtrack_step
        θ = θₖ + Δ
        model_θ = re(θ)
        sur_adv = surrogate_advantage(model_θ, s, a, δ, old_logits[]) - mean(δ)
        kld_excess = kld(model_θ, s, old_logits[]) - p.kldivergence_limit
        sur_adv > 0 && kld_excess <= 0 && break
        Δ = Δ * p.backtrack_coeff
    end

    # update policy approximator
    copy!(params(A.model), θ)
end
