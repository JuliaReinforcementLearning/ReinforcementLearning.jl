export VPG

using Random: Random, shuffle
using Distributions: ContinuousDistribution, DiscreteDistribution, logpdf
using Functors: @functor
using Flux: params, softmax, gradient, logsoftmax
using StatsBase: mean
using ChainRulesCore: ignore_derivatives

"""
Vanilla Policy Gradient
"""
Base.@kwdef struct VPG{A,B,D, R} <: AbstractPolicy
    "For discrete actions, logits before softmax is expected. For continuous actions, a `Tuple` of arguments are expected to initialize `dist`"
    approximator::A
    baseline::B = nothing
    "`ContinuousDistribution` or `DiscreteDistribution`"
    dist::D
    "discount ratio"
    γ::Float32 = 0.99f0
    batch_size::Int = 1024
    rng::R = Random.default_rng()
end

IsPolicyGradient(::Type{<:VPG}) = IsPolicyGradient()
@functor VPG (approximator, baseline)

function RLBase.plan!(π::VPG, env::AbstractEnv)
    res = env |> state |> send_to_device(π) |> x -> RLCore.forward(π.approximator, x) |> send_to_host
    rand(π.rng, action_distribution(π.dist, res)[1])
end

function RLBase.optimise!(p::VPG, ::PostEpisodeStage, trajectory::Trajectory)
    has_optimized = false
    for batch in trajectory #batch is a vector of Episode
        gains = stack(discount_rewards(ep[:reward], p.γ) for ep in batch)
        states = stack(ep[:state] for ep in batch)
        actions = stack(ep[:action] for ep in batch)
        for inds in Iterators.partition(shuffle(p.rng, eachindex(gains)), p.batch_size)
            RLBase.optimise!(p, (state=selectdim(states,ndims(states),inds), action=selectdim(actions,ndims(actions),inds), gain=selectdim(gainss,ndims(gains),inds)))
        end
        has_optimized = true
    end
    has_optimized && empty!(trajectory.container)
    return nothing
end

function RLBase.optimise!(p::VPG, batch::NamedTuple{(:state, :action, :gain)})
    A = p.approximator
    B = p.baseline
    s, a, g = batch[:state], batch[:action], batch[:gain]
    local δ
    println(s)
    if isnothing(B)
        δ = normalise(g)
        loss = 0
    else
        gs = gradient(params(B)) do
            δ = g - vec(RLCore.forward(B, s))
            loss = mean(δ .^ 2)
            ignore_derivatives() do
                # @info "VPG/baseline" loss = loss δ
            end
            loss
        end
        RLBase.optimise!(B, gs)
    end
    
    gs = policy_gradient_estimate(p, s, a, δ)
    RLBase.optimise!(A, gs)
end
