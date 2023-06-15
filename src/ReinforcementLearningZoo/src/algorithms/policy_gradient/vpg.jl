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

function optimise!(p::VPG, ::PostEpisodeStage, trajectory::Trajectory)
    trajectory.container[] = true
    for batch in trajectory
        optimise!(p, batch)
    end
    empty!(trajectory.container)
end

function RLBase.optimise!(π::VPG, ::PostActStage, episode::Episode)
    gain = discount_rewards(episode[:reward][:], π.γ)
    for inds in Iterators.partition(shuffle(π.rng, 1:length(episode)), π.batch_size)
        optimise!(π, PostActStage(), (state=episode[:state][inds], action=episode[:action][inds], gain=gain[inds]))
    end
end

function RLBase.optimise!(p::VPG, batch::NamedTuple{(:state, :action, :gain)})
    A = p.approximator
    B = p.baseline
    s, a, g = map(Array, batch) # !!! FIXME
    local δ

    if isnothing(B)
        δ = normalise(g)
        loss = 0
    else
        gs = gradient(params(B)) do
            δ = g - vec(B(s))
            loss = mean(δ .^ 2)
            ignore_derivatives() do
                # @info "VPG/baseline" loss = loss δ
            end
            loss
        end
        optimise!(B, gs)
    end
    
    gs = policy_gradient_estimate(p, s, a, δ)
    optimise!(A, gs)
end
